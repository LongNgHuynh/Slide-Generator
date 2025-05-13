from models.LLMs import GPT_4o
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from utils.tools import image_search_tool, web_search_tool, crawl_tool, presentation_tool, generate_presentation_outline_tool
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Annotated, TypedDict
from langgraph.graph import StateGraph, add_messages
from utils.custom_output_parser import CustomOutputParser
import uuid
import json
import os

checkpointer = MemorySaver()

llm = GPT_4o()
research_tools = [image_search_tool, web_search_tool, crawl_tool, generate_presentation_outline_tool]

class State(TypedDict):
    user_input: str
    recommend_outline: Annotated[list[str], add_messages]
    slides: Annotated[list[str], add_messages]
    evaluation: Annotated[list[str], add_messages]
    approved: bool
    next: str


def task_planning_agent(state: State):
    print("\n[task_planning_agent] Starting task planning process...")
    topic = state["user_input"]
    
    previous_evaluation = state["evaluation"][-1] if "evaluation" in state and state["evaluation"] else "No previous evaluation"
    has_previous_evaluation = previous_evaluation != "No previous evaluation"
    
    evaluation_section = ""
    if has_previous_evaluation:
        evaluation_section = f"""
    Previous Evaluation Feedback:
    {previous_evaluation}
    """
    
    improvements_point = ""
    if has_previous_evaluation:
        improvements_point = "5. Improvements based on the previous evaluation feedback"
    
    # Check if we have slides already, which would indicate we're in a refinement loop
    has_slides = "slides" in state and state["slides"]
    
    # Prepare conditional sections
    refinement_intro = "We are currently in a refinement phase. Previous slides have been created and evaluated. Your task is to decide the next step:" if has_slides else "Please create a detailed plan for developing a presentation on this topic. Include:"
    
    critical_decision = ""
    if has_previous_evaluation:
        critical_decision = """
    CRITICAL DECISION: Based on the evaluation feedback, you must decide whether to:
    1. ENHANCE_SLIDES: If the evaluation feedback suggests making improvements to the existing slides without changing the outline
    2. NEW_OUTLINE: If the evaluation feedback suggests that a completely new outline is needed
    
    At the end of your response, add EXACTLY ONE of these decision markers:
    DECISION: ENHANCE_SLIDES
    or
    DECISION: NEW_OUTLINE
    """
    
    planning_prompt = f"""
    You are a Task Planning Agent responsible for organizing a presentation creation process.
    
    Research Topic: {topic}
    
    {evaluation_section}
    
    {refinement_intro}
    1. Key areas to research
    2. Recommended structure for the presentation
    3. Types of visuals/illustrations that might be useful
    4. Potential challenges and how to address them
    {improvements_point}
    
    Format your response as a structured task plan.
    {critical_decision}
    """
    
    try:
        response = llm.invoke(planning_prompt)
        task_plan = response.content
        print(f"[task_planning_agent] Generated task plan")
        print(task_plan)
        
        # Determine the next step based on the agent's decision
        if has_previous_evaluation:
            if "DECISION: ENHANCE_SLIDES" in task_plan or "RECOMMENDATION: ENHANCE_SLIDES" in previous_evaluation:
                next_agent = "slide_creating_agent"
                # Remove the decision text from the task plan
                task_plan = task_plan.replace("DECISION: ENHANCE_SLIDES", "").strip()
            else:
                next_agent = "recommend_outline_agent"
                # Remove the decision text if present
                task_plan = task_plan.replace("DECISION: NEW_OUTLINE", "").strip()
        elif "NEEDS_RESEARCH: NO" in task_plan:
            next_agent = "slide_creating_agent"
            # Remove the decision text from the task plan
            task_plan = task_plan.replace("NEEDS_RESEARCH: NO", "").strip()
        else:
            next_agent = "recommend_outline_agent"
            # Remove the decision text if present
            task_plan = task_plan.replace("NEEDS_RESEARCH: YES", "").strip()
            
        return {"recommend_outline": [task_plan], "next": next_agent}
    except Exception as e:
        print(f"Error in task planning: {e}")
        return {"recommend_outline": ["Error in task planning"], "next": "recommend_outline_agent"}


def recommend_outline_agent(state: State):
    print("\n[recommend_outline_agent] Generating presentation outline...")
    topic = state["user_input"]
    task_plan = state["recommend_outline"][-1] if state["recommend_outline"] else "No task plan available"

    agent_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: Based on the task plan and research topic, create a comprehensive presentation outline: {input}
    Task Plan: {task_plan}

    {agent_scratchpad}""")       

    agent = create_react_agent(
        llm=llm,
        tools=research_tools,
        prompt=agent_prompt,
        output_parser=CustomOutputParser()
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=research_tools,
        verbose=True
    )
    
    try:
        result = agent_executor.invoke({
            "guide": f"Generate a comprehensive presentation outline. Use the available tools to gather information and images.",
            "input": topic,
            "task_plan": task_plan,
        })
        recommend_outline = result.get("output", "No output generated")
        print(f"[recommend_outline_agent] Generated outline:\n{recommend_outline}\n")
        return {"recommend_outline": state["recommend_outline"] + [recommend_outline], "next": "slide_creating_agent"}
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return {"recommend_outline": state["recommend_outline"] + ["Error generating outline"], "next": "end_node"}


def slide_creating_agent(state: State):
    print("\n[slide_creating_agent] Creating presentation slides...")
    
    current_outline = state["recommend_outline"][-1] if state["recommend_outline"] else ""
    evaluation_feedback = state["evaluation"][-1] if "evaluation" in state and state["evaluation"] else "No feedback available"
    has_feedback = evaluation_feedback != "No feedback available"
    user_input = state["user_input"]
    
    slide_tools = [presentation_tool]
    
    # Modified intro prompt to include evaluation feedback
    feedback_section = ""
    if has_feedback:
        feedback_section = f"""
    IMPORTANT - Previous Evaluation Feedback:
    {evaluation_feedback}
    
    You MUST address the feedback points above in your slide creation. Focus especially on:
    1. Enhancing visual elements and design consistency
    2. Improving content organization and flow
    3. Addressing any content gaps identified
    4. Following the specific improvement recommendations
    """
    
    intro_prompt = f"""
    You are tasked with creating a comprehensive presentation on "{user_input}".
    
    Based on the outline provided, create a series of detailed and engaging presentation slides.
    For each slide, you should provide:
    1. A clear, concise title
    2. Content that is informative and well-structured
    3. Each slide should focus on a specific aspect of the topic
    
    {feedback_section}
    
    IMPORTANT: You must create ALL slides needed to cover the entire outline, calling the generate_presentation tool separately for EACH slide.
    
    To create a slide, use the generate_presentation tool with these exact parameters:
    - slide_number (integer): the position of the slide in the presentation
    - title (string): a concise title for the slide
    - content (string): the main content for the slide
    - style (string): the design style for the slide using tailwind css (instructions for the style format, NOTICE: the style should be related to the content of the slide, and inherit the style of the previous slides)
    - layout (string): the layout for the slide (instructions for the layout format, NOTICE: the layout should be related to the content of the slide, and inherit the style of the previous slides)
    
    Create the slides in sequential order, starting with slide 1 (introduction) and
    ending with a conclusion slide. Make a separate tool call for each slide in the presentation.
    """
    
    agent_prompt = PromptTemplate.from_template("""Answer the following questions as best you can. You have access to the following tools:

    {tools}

    Use the following format:

    Question: the input question you must answer
    Thought: you should always think about what to do
    Action: the action to take, should be one of [{tool_names}]
    Action Input: the input to the action
    Observation: the result of the action
    ... (this Thought/Action/Action Input/Observation can repeat N times)
    Thought: I now know the final answer
    Final Answer: the final answer to the original input question

    Begin!

    Question: Create detailed presentation slides based on the following information:
    
    Topic: {input}
    Outline: {outline}
    Presentation Guidelines: {intro}

    CRITICAL INSTRUCTION FOR USING generate_presentation TOOL:
    When using the generate_presentation tool, your Action Input must include 5 parameters:
    
    Action: generate_presentation
    Action Input: {{"slide_number": 1, "title": "Title", "content": "Content", "layout": "Layout Description", "style": "Design style using tailwind CSS"}}
    
    Make sure that:
    1. slide_number is a plain integer, not in quotes
    2. title and content are strings in quotes
    3. layout and style are strings in quotes
    4. You format the JSON properly with correct quotes and braces
    5. YOU MUST MAKE SEPARATE TOOL CALLS for each slide in the presentation
    
    YOUR TASK REQUIRES MULTIPLE TOOL CALLS:
    - Create slides in sequential order
    - First, create slide 1 (introduction)
    - Then create additional slides based on the outline, one at a time
    - End with a conclusion slide
    
    Example of correct sequence (make each call separately after receiving the observation from the previous call):
    
    Action: generate_presentation
    Action Input: {{"slide_number": 1, "title": "Introduction", "content": "First slide content...", "layout": "title-and-content", "style": "bg-blue-100 text-gray-800"}}
    
    ... wait for observation ...
    
    Action: generate_presentation
    Action Input: {{"slide_number": 2, "title": "Main Point 1", "content": "Second slide content...", "layout": "two-column", "style": "bg-blue-200 text-gray-800"}}
    
    ... and so on for all slides in the outline ...
    
    Make these tool calls one after another. Wait for each observation before making the next tool call.
    
    {agent_scratchpad}""")       

    agent = create_react_agent(
        llm=llm,
        tools=slide_tools,  
        prompt=agent_prompt,
        output_parser=CustomOutputParser()
    )
    
    agent_executor = AgentExecutor.from_agent_and_tools(
        agent=agent,
        tools=slide_tools,
        verbose=True
    )
    
    try:
        result = agent_executor.invoke({
            "guide": "Create detailed presentation slides. Include slide titles, content points, visual elements, and speaker notes.",
            "input": user_input,
            "outline": current_outline,
            "intro": intro_prompt
        })
        slide_content = result.get("output", "No output generated")
        print(f"[slide_creating_agent] Generated slides")
        
        complete_presentation_path = os.path.join(os.getcwd(), "generated_slides", "complete_presentation.html")
        if os.path.exists(complete_presentation_path):
            return {"slides": [f"Complete presentation generated successfully at: {complete_presentation_path}\n\n{slide_content}"], "next": "slide_evaluating_agent"}
        
        return {"slides": [slide_content], "next": "slide_evaluating_agent"}
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return {"slides": [f"Error creating slides: {str(e)}"], "next": "end_node"}


def slide_evaluating_agent(state: State):
    print("\n[slide_evaluating_agent] Evaluating presentation slides...")
    
    current_slides = state["slides"][-1] if state["slides"] else ""
    original_outline = state["recommend_outline"][-1] if state["recommend_outline"] else ""
    
    evaluation_prompt = f"""
    As a Slide Evaluating Agent, review these presentation slides against the original outline.
    
    Original Outline:
    {original_outline}
    
    Slides Created:
    {current_slides}
    
    Please evaluate:
    1. Alignment with original outline goals
    2. Visual coherence and effectiveness
    3. Content quality and completeness
    4. Flow and narrative structure
    5. Specific improvement recommendations
    
    Format your response with these clear sections:
    - STRENGTHS: List the strong points of the presentation
    - WEAKNESSES: List areas that need improvement
    - RECOMMENDATIONS: Provide specific, actionable recommendations for the next iteration
    - OVERALL ASSESSMENT: Give a brief overall assessment
    
    CRITICAL DECISION GUIDANCE: At the end of your evaluation, provide guidance on the next steps.
    1. If the presentation requires a completely new outline, state "RECOMMENDATION: NEW_OUTLINE"
    2. If the presentation can be improved with the current outline, state "RECOMMENDATION: ENHANCE_SLIDES"
    
    Provide a detailed evaluation with actionable feedback that can be used in the next planning cycle.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        evaluation = response.content
        print(f"[slide_evaluating_agent] Completed evaluation")
        
        return {"evaluation": [evaluation], "next": "task_planning_agent"}
    except Exception as e:
        print(f"Error in slide evaluation: {e}")
        return {"evaluation": ["Error evaluating slides"], "next": "end_node"}


def end_node(state: State):
    print("\n[end_node] Process finished.")
    print("Final Slides:", state["slides"][-1] if "slides" in state and state["slides"] else "No slides generated")
    print("Final Evaluation:", state["evaluation"][-1] if "evaluation" in state and state["evaluation"] else "No evaluation available")
    return {"status": "completed", "final_state": state}


def router(state: State):
    """Route to the next node based on the next field in the state."""
    return state["next"]


graph_builder = StateGraph(State)
graph_builder.add_node("task_planning_agent", task_planning_agent)
graph_builder.add_node("recommend_outline_agent", recommend_outline_agent)
graph_builder.add_node("slide_creating_agent", slide_creating_agent)
graph_builder.add_node("slide_evaluating_agent", slide_evaluating_agent)
graph_builder.add_node("end_node", end_node)

graph_builder.set_entry_point("task_planning_agent")

# Dynamic routing from each agent based on the "next" field
graph_builder.add_conditional_edges(
    "task_planning_agent",
    router,
    {
        "recommend_outline_agent": "recommend_outline_agent",
        "slide_creating_agent": "slide_creating_agent",
        "end_node": "end_node"
    }
)

graph_builder.add_edge("recommend_outline_agent", "slide_creating_agent")
graph_builder.add_edge("slide_creating_agent", "slide_evaluating_agent")
graph_builder.add_edge("slide_evaluating_agent", "task_planning_agent")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}

try:
    print("Starting workflow execution...")
    user_input = input("Enter your research topic: ")
    initial_state = {
        "user_input": user_input, 
        "recommend_outline": [], 
        "slides": [], 
        "evaluation": [],
        "approved": False,
        "next": "task_planning_agent"
    }
    
    print("Initializing the graph with user input:", user_input)
    try:
        for event in graph.stream(initial_state, config=thread_config):
            print(f"Event type: {type(event)}")
    except Exception as e:
        import traceback
        print(f"Error during graph execution: {str(e)}")
        traceback.print_exc()
except Exception as e:
    import traceback
    print(f"Initialization error: {str(e)}")
    traceback.print_exc()