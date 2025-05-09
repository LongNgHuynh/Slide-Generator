from models.LLMs import GPT_4o
from langchain.agents import AgentExecutor, create_react_agent, create_json_chat_agent
from langchain.prompts import PromptTemplate
from utils.tools import image_search_tool, web_search_tool, crawl_tool, presentation_tool
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import interrupt, Command
from typing import Literal, Annotated, TypedDict
from langgraph.constants import START, END
from langgraph.graph import StateGraph, add_messages
from utils.custom_output_parser import CustomOutputParser
import uuid
import json
import os

checkpointer = MemorySaver()

llm = GPT_4o()
research_tools = [image_search_tool, web_search_tool, crawl_tool]

class State(TypedDict):
    """Graph state containing research topic, generated content, and human feedback."""
    research_topic: str
    recommend_outline: Annotated[list[str], add_messages]
    human_feedback: Annotated[list[str], add_messages]
    slides: Annotated[list[str], add_messages]
    evaluation: Annotated[list[str], add_messages]
    approved: bool


def task_planning_agent(state: State):
    """Initial task planning agent that sets up the research process."""
    print("\n[task_planning_agent] Starting task planning process...")
    topic = state["research_topic"]
    
    # Check if there's previous evaluation feedback to incorporate
    previous_evaluation = state["evaluation"][-1] if "evaluation" in state and state["evaluation"] else "No previous evaluation"
    has_previous_evaluation = previous_evaluation != "No previous evaluation"
    
    # Prepare evaluation section conditionally
    evaluation_section = ""
    if has_previous_evaluation:
        evaluation_section = f"""
    Previous Evaluation Feedback:
    {previous_evaluation}
    """
    
    # Prepare improvements point conditionally
    improvements_point = ""
    if has_previous_evaluation:
        improvements_point = "5. Improvements based on the previous evaluation feedback"
    
    planning_prompt = f"""
    You are a Task Planning Agent responsible for organizing a presentation creation process.
    
    Research Topic: {topic}
    
    {evaluation_section}
    
    Please create a detailed plan for developing a presentation on this topic. Include:
    1. Key areas to research
    2. Recommended structure for the presentation
    3. Types of visuals/illustrations that might be useful
    4. Potential challenges and how to address them
    {improvements_point}
    
    Format your response as a structured task plan.
    """
    
    try:
        response = llm.invoke(planning_prompt)
        task_plan = response.content
        print(f"[task_planning_agent] Generated task plan")
        print(task_plan)
        
        # Move to outline recommendation
        return {"recommend_outline": [task_plan]}
    except Exception as e:
        print(f"Error in task planning: {e}")
        return {"recommend_outline": ["Error in task planning"]}


def recommend_outline_agent(state: State):
    """Uses GPT-4o to generate a presentation outline based on a research topic with human feedback incorporated."""
    print("\n[recommend_outline_agent] Generating presentation outline...")
    topic = state["research_topic"]
    task_plan = state["recommend_outline"][-1] if state["recommend_outline"] else "No task plan available"
    feedback = state["human_feedback"] if "human_feedback" in state and state["human_feedback"] else ["No feedback yet"]

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
    Previous feedback: {feedback}

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
            "feedback": feedback
        })
        recommend_outline = result.get("output", "No output generated")
        print(f"[recommend_outline_agent] Generated outline:\n{recommend_outline}\n")
        return {"recommend_outline": state["recommend_outline"] + [recommend_outline]}
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return {"recommend_outline": state["recommend_outline"] + ["Error generating outline"]}


def slide_creating_agent(state: State):
    """Creates actual presentation slides based on the approved outline."""
    print("\n[slide_creating_agent] Creating presentation slides...")
    
    # Get the latest outline from the state
    current_outline = state["recommend_outline"][-1] if state["recommend_outline"] else ""
    feedback = state["human_feedback"][-1] if "human_feedback" in state and state["human_feedback"] else "No feedback available"
    research_topic = state["research_topic"]
    
    # Create list of tools including presentation_tool
    slide_tools = [presentation_tool]
    
    # First, create an introduction to the presentation task for the agent
    intro_prompt = f"""
    You are tasked with creating a comprehensive presentation on "{research_topic}".
    You will need to create approximately 10-15 slides covering this topic.
    
    Based on the outline and feedback provided, create a series of detailed and engaging presentation slides.
    For each slide, you should provide:
    1. A clear, concise title
    2. Content that is informative and well-structured
    3. Each slide should focus on a specific aspect of the topic
    
    To create a slide, use the generate_presentation tool with these exact parameters:
    - slide_number (integer): the position of the slide in the presentation
    - title (string): a concise title for the slide
    - content (string): the main content for the slide
    
    Create the slides in sequential order, starting with slide 1 (introduction) and
    ending with a conclusion slide.
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
    Previous feedback: {feedback}
    Presentation Guidelines: {intro}

    CRITICAL INSTRUCTION FOR USING generate_presentation TOOL:
    When using the generate_presentation tool, your Action Input must include 3 parameters:
    
    Action: generate_presentation
    Action Input: {{"slide_number": 1, "title": "Title", "content": "Content"}}
    
    Make sure that:
    1. slide_number is a plain integer, not in quotes
    2. title and content are strings in quotes
    3. You format the JSON properly with correct quotes and braces
    
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
            "input": research_topic,
            "outline": current_outline,
            "feedback": feedback,
            "intro": intro_prompt
        })
        slide_content = result.get("output", "No output generated")
        print(f"[slide_creating_agent] Generated slides")
        
        # Return the path to the complete presentation
        complete_presentation_path = os.path.join(os.getcwd(), "generated_slides", "complete_presentation.html")
        if os.path.exists(complete_presentation_path):
            return {"slides": [f"Complete presentation generated successfully at: {complete_presentation_path}\n\n{slide_content}"]}
        
        return {"slides": [slide_content]}
    except Exception as e:
        print(f"Error in agent execution: {e}")
        return {"slides": [f"Error creating slides: {str(e)}"]}


def slide_evaluating_agent(state: State):
    """Evaluates the quality of the created slides and provides improvement suggestions."""
    print("\n[slide_evaluating_agent] Evaluating presentation slides...")
    
    # Get the latest slides from the state
    current_slides = state["slides"][-1] if state["slides"] else ""
    original_outline = state["recommend_outline"][-1] if state["recommend_outline"] else ""
    
    # Create prompt for slide evaluation
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
    
    Provide a detailed evaluation with actionable feedback that can be used in the next planning cycle.
    """
    
    try:
        response = llm.invoke(evaluation_prompt)
        evaluation = response.content
        print(f"[slide_evaluating_agent] Completed evaluation")
        
        return {"evaluation": [evaluation]}
    except Exception as e:
        print(f"Error in slide evaluation: {e}")
        return {"evaluation": ["Error evaluating slides"]}


def is_completion_command(text):
    """Use LLM to determine if the user input indicates they're done providing feedback."""
    prompt = f"""
    I need to determine if the following user input indicates they are finished providing feedback 
    and want to complete the current process. The user has been asked to provide feedback on a
    generated text, and we need to know if they're done giving feedback.
    
    User input: "{text}"
    
    Answer with ONLY "YES" if the user is clearly indicating they're done providing feedback
    (e.g., "looks good", "I'm finished", "that works", "no more changes needed", "that's ok", "ok", "okay", "fine", "good", "yes", etc.)
    
    Answer with ONLY "NO" if the user is providing more feedback or if it's unclear.
    """
    
    response = llm.invoke(prompt)
    result = response.content.strip().upper()
    print(f"[is_completion_command] Checking completion command: {text} -> {result}")
    
    return "YES" in result


def human_feedback_node(state: State):
    """Human intervention node for providing feedback on the outline."""
    print("\n" + "="*50)
    print("[human_feedback_node] Awaiting human feedback...")
    print("="*50)

    current_outline = state["recommend_outline"][-1] if state["recommend_outline"] else "No outline available"
    
    # Print the outline for the user to see clearly
    print(f"\nCurrent Outline:\n{current_outline}\n")

    # Get direct input from the user instead of using interrupt
    print("\n" + "-"*50)
    print("FEEDBACK REQUIRED")
    print("-"*50)
    user_feedback = input("Your feedback: ")
    print("-"*50 + "\n")
    
    print(f"[human_feedback_node] Received human feedback: {user_feedback}")

    # Check if user indicates they're done
    if is_completion_command(user_feedback):
        print("[human_feedback_node] User indicated completion. Moving to slide creation.")
        # Set a flag in state to force slide creation
        state["approved"] = True
        return {"human_feedback": state.get("human_feedback", []) + ["Approved outline"], "approved": True}

    # Add user feedback with a clear marker that it's not an approval
    print("[human_feedback_node] User provided additional feedback. Returning to outline creation.")
    return {"human_feedback": state.get("human_feedback", []) + [f"Additional feedback: {user_feedback}"]}


def end_node(state: State):
    """Final node that terminates the process"""
    print("\n[end_node] Process finished.")
    print("Final Slides:", state["slides"][-1] if "slides" in state and state["slides"] else "No slides generated")
    print("Final Evaluation:", state["evaluation"][-1] if "evaluation" in state and state["evaluation"] else "No evaluation available")
    return {"status": "completed", "final_state": state}


graph_builder = StateGraph(State)
graph_builder.add_node("task_planning_agent", task_planning_agent)
graph_builder.add_node("recommend_outline_agent", recommend_outline_agent)
graph_builder.add_node("human_feedback_node", human_feedback_node)
graph_builder.add_node("slide_creating_agent", slide_creating_agent)
graph_builder.add_node("slide_evaluating_agent", slide_evaluating_agent)
graph_builder.add_node("end_node", end_node)

graph_builder.set_entry_point("task_planning_agent")
graph_builder.add_edge("task_planning_agent", "recommend_outline_agent")

# Replace direct edges with conditional logic
def route_after_human_feedback(state):
    # Print full state for debugging
    print(f"\n[DEBUG] Full state keys: {state.keys()}")
    
    # First, explicitly check for "approved" flag which will override other checks
    if "approved" in state and state["approved"]:
        print("[route_after_human_feedback] Explicitly approved flag found. Routing to slide creation.")
        return "slide_creating_agent"
    
    # Check the latest feedback
    if "human_feedback" in state and state["human_feedback"]:
        latest_feedback = state["human_feedback"][-1]
        # Check if this is a message object with a content field
        if hasattr(latest_feedback, 'content'):
            feedback_content = latest_feedback.content
        else:
            feedback_content = str(latest_feedback)
            
        print(f"[route_after_human_feedback] Processing feedback: {feedback_content}")
        
        # If the feedback was "Approved outline", route to slide creation
        if "Approved outline" in feedback_content:
            print("[route_after_human_feedback] Routing to slide creation")
            return "slide_creating_agent"
        
        # Otherwise, route back to outline recommendation (this always happens if not approved)
        print(f"[route_after_human_feedback] User did not approve. Routing back to outline recommendation")
        return "recommend_outline_agent"
    
    # Default case - route back to outline recommendation
    print("[route_after_human_feedback] No feedback found, routing to outline recommendation")
    return "recommend_outline_agent"

# Connect nodes with proper conditional routing
graph_builder.add_edge("recommend_outline_agent", "human_feedback_node")
graph_builder.add_conditional_edges("human_feedback_node", route_after_human_feedback)
graph_builder.add_edge("slide_creating_agent", "slide_evaluating_agent")
graph_builder.add_edge("slide_evaluating_agent", "task_planning_agent")

checkpointer = MemorySaver()
graph = graph_builder.compile(checkpointer=checkpointer)

thread_config = {"configurable": {"thread_id": uuid.uuid4()}}
research_topic = input("Enter your research topic: ")
initial_state = {
    "research_topic": research_topic, 
    "recommend_outline": [], 
    "human_feedback": [], 
    "slides": [], 
    "evaluation": [],
    "approved": False  # Initialize as not approved
}

try:
    print("\n========== PRESENTATION GENERATOR ==========")
    print("Starting workflow process. You will be prompted for input when needed.\n")
    
    # Direct execution without using stream
    result = graph.invoke(initial_state, config=thread_config)
    
    print("\n" + "="*50)
    print("WORKFLOW COMPLETED SUCCESSFULLY")
    print("="*50)
    
    print("\nFinal presentation has been generated.")
    
except Exception as e:
    print(f"\n[ERROR] An error occurred during execution: {str(e)}")
    import traceback
    traceback.print_exc()
    
    # Exit with error code
    import os
    os._exit(1)