from models.LLMs import GPT_4o, GPT_o3, Claude_3_7_Sonnet, Gemini
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from utils.tools import Searxng
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Annotated, List
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, add_messages
from utils.custom_output_parser import CustomOutputParser
from langgraph.prebuilt import create_react_agent

import uuid, json, os, logging
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage

from utils.tools import image_search, web_search, crawl_url, generate_slide
from prompt import prompt_system_outline, prompt_system_slide, planning_instructions, prompt_system_layout
from utils.utils import get_research_topic
from utils.schemas import SearchQueryList, Plan

from langfuse import Langfuse
from langfuse.callback import CallbackHandler
import pprint
import re

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM_4o = GPT_4o()
LLM_o3 = GPT_o3()
LLM_claude = Claude_3_7_Sonnet()
LLM = Gemini()

memory = MemorySaver()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

trace_id = str(uuid.uuid4())
# Base configuration
config = {
    "recursion_limit": 100,
    "configurable": {
        "trace_id": trace_id
    },
    "callbacks": [langfuse_handler],
    "run_id": trace_id
}

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('workflow.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    is_outline_generated: bool
    images: list[dict]
    found_information: list[dict]
    slides: list[dict]
    summary: list[dict]
    outline_content: str
    layout_instructions: str
    outline_attempts: int
    plan: dict

members = ["planner", "outline_agent", "slide_agent", "artist_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal[*options]

outline_agent = create_react_agent(
    model=LLM_4o,  # Use GPT-4o for better ReAct agent compatibility
    tools=[image_search, crawl_url, web_search],
    prompt=prompt_system_outline
)

slide_agent = create_react_agent(
    model=LLM_4o,  # Use GPT-4o for better ReAct agent compatibility
    tools=[generate_slide], # generate_slide tool is now globally defined
    prompt=prompt_system_slide
)

def _detect_user_provided_content(messages: List[BaseMessage]) -> dict:
    """
    Detect if user has provided outline or layout content in their messages.
    Returns a dict with flags for what content was provided.
    """
    user_content = ""
    for msg in messages:
        if isinstance(msg, HumanMessage):
            user_content += msg.content.lower()
    
    # Simple keyword detection for outline content
    outline_keywords = ["slide 1:", "slide 2:", "outline:", "presentation outline", "table of contents"]
    has_outline = any(keyword in user_content for keyword in outline_keywords)
    
    # Simple keyword detection for layout content  
    layout_keywords = ["layout:", "design:", "positioning:", "visual hierarchy", "slide design"]
    has_layout = any(keyword in user_content for keyword in layout_keywords)
    
    return {
        "has_outline": has_outline,
        "has_layout": has_layout,
        "user_content": user_content[:200]  # First 200 chars for debugging
    }

def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
    """
    Router function that decides which agent should run next based on the planner's logic.
    Routes according to: planner → outline_agent → artist_agent → slide_agent → FINISH
    Skips agents if their content is already provided by the user.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command indicating which node to go to next
    """
    logger.info("Supervisor node: Starting workflow routing based on planner logic")
    
    outline_attempts = state.get("outline_attempts", 0)
    is_outline_generated_flag = state.get("is_outline_generated", False)
    current_slides = state.get("slides")
    layout_instructions = state.get("layout_instructions", "")
    last_message_obj = state["messages"][-1] if state["messages"] else None
    last_message_content = last_message_obj.content if last_message_obj else ""
    last_message_sender = getattr(last_message_obj, 'name', None) or (last_message_obj.type if last_message_obj else "")

    logger.debug(f"Current state: is_outline_generated={is_outline_generated_flag}, slides_present={bool(current_slides)}, outline_attempts={outline_attempts}, last_sender='{last_message_sender}'")
    logger.debug(f"State details: outline_content_length={len(state.get('outline_content', ''))}, layout_instructions_length={len(layout_instructions)}, slides_count={len(current_slides) if current_slides else 0}")

    # Check if this is a completely new topic/request vs a continuation/modification
    human_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    agent_messages = [msg for msg in state["messages"] if hasattr(msg, 'name') and msg.name in members]
    
    # Only reset if we detect a DIFFERENT topic, not just any new user message
    is_completely_new_topic = False
    if (isinstance(last_message_obj, HumanMessage) and 
        is_outline_generated_flag and 
        layout_instructions and 
        current_slides and
        len(human_messages) > 1 and 
        len(agent_messages) > 0):
        
        # Analyze if this is a new topic by comparing current request with previous topics
        current_user_request = last_message_obj.content.lower()
        
        # Keywords that indicate a completely new presentation topic
        new_topic_indicators = [
            "new presentation", "different topic", "another presentation",
            "new slides", "different slides", "create presentation about",
            "make slides about", "generate presentation on"
        ]
        
        # Check if the current request is about a significantly different topic
        has_new_topic_keywords = any(indicator in current_user_request for indicator in new_topic_indicators)
        
        # Only reset if explicit keywords are found - remove the word overlap check for now
        is_completely_new_topic = has_new_topic_keywords
        
        logger.debug(f"Topic analysis - Current: '{current_user_request[:50]}...', New topic indicators: {has_new_topic_keywords}")
    
    if is_completely_new_topic:
        logger.info(f"Supervisor: Detected completely new topic after previous completion. Resetting state for new workflow.")
        # Reset all workflow state for the new request but keep the message history
        return Command(
            update={
                "is_outline_generated": False,
                "outline_content": "",
                "layout_instructions": "",
                "slides": [],
                "images": [],
                "found_information": [],
                "outline_attempts": 0,
                "plan": {},
                "next": "planner"
            },
            goto="planner"
        )

    MAX_OUTLINE_ATTEMPTS = 3

    # Check for specific completion signals - only from slide_agent and only with actual completion phrases
    completion_phrases = ["presentation completed", "slide generation completed", "workflow completed", "all slides generated"]
    has_completion_signal = any(phrase in last_message_content.lower() for phrase in completion_phrases)
    
    if has_completion_signal and last_message_sender == "slide_agent" and current_slides:
        logger.info("Supervisor: Detected specific completion signal from slide_agent. Routing to FINISH.")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Check if we've reached max attempts for outline generation
    if not is_outline_generated_flag and outline_attempts >= MAX_OUTLINE_ATTEMPTS:
        logger.warning(f"Supervisor: Max outline attempts ({MAX_OUTLINE_ATTEMPTS}) reached. Routing to FINISH.")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Detect if user has provided outline or layout content
    user_content_detection = _detect_user_provided_content(state["messages"])
    logger.debug(f"User content detection: {user_content_detection}")
    
    # Follow planner logic: Check what's available and route to next needed agent
    
    # 1. If no plan exists yet, start with planner (only for initial message)
    if not any(hasattr(msg, 'name') and msg.name == "planner" for msg in state["messages"]):
        logger.info("Supervisor: No plan found. Routing to planner.")
        return Command(goto="planner", update={"next": "planner", "outline_attempts": outline_attempts})
    
    # 2. If no outline is generated yet, check if user provided it
    if not is_outline_generated_flag:
        if user_content_detection["has_outline"]:
            logger.info("Supervisor: User provided outline content. Marking as generated and proceeding.")
            # Extract outline from user messages
            user_outline = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage):
                    user_outline += msg.content + "\n"
            
            # Update state to mark outline as generated and proceed to artist_agent
            return Command(
                update={
                    "is_outline_generated": True,
                    "outline_content": user_outline,
                    "messages": [AIMessage(content=f"Using user-provided outline: {user_outline[:100]}...", name="outline_agent")],
                    "next": "artist_agent",
                    "outline_attempts": outline_attempts
                },
                goto="artist_agent"
            )
        else:
            logger.info(f"Supervisor: No outline generated. Routing to outline_agent (attempt {outline_attempts + 1}).")
            return Command(goto="outline_agent", update={"next": "outline_agent", "outline_attempts": outline_attempts + 1})
    
    # 3. If outline exists but no layout instructions, check if user provided layout
    if is_outline_generated_flag and not layout_instructions:
        if user_content_detection["has_layout"]:
            logger.info("Supervisor: User provided layout content. Using it and proceeding to slide_agent.")
            # Extract layout from user messages
            user_layout = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage) and any(keyword in msg.content.lower() for keyword in ["layout:", "design:", "positioning:"]):
                    user_layout += msg.content + "\n"
            
            return Command(
                update={
                    "layout_instructions": user_layout,
                    "messages": [AIMessage(content=f"Using user-provided layout: {user_layout[:100]}...", name="artist_agent")],
                    "next": "slide_agent",
                    "outline_attempts": outline_attempts
                },
                goto="slide_agent"
            )
        else:
            # Check if artist_agent previously errored to avoid loops
            if last_message_sender == "artist_agent" and ("Error" in last_message_content or "error" in last_message_content.lower()):
                logger.warning("Supervisor: Artist agent previously errored. Routing to FINISH to avoid loop.")
                return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
            
            logger.info("Supervisor: Outline generated, no layout instructions. Routing to artist_agent.")
            return Command(goto="artist_agent", update={"next": "artist_agent", "outline_attempts": outline_attempts})
    
    # 4. If outline and layout exist but no slides, go to slide_agent
    if is_outline_generated_flag and layout_instructions and not current_slides:
        # Check if slide_agent previously errored to avoid loops
        if last_message_sender == "slide_agent" and ("Error generating slides" in last_message_content or "error" in last_message_content.lower()):
            logger.warning("Supervisor: Slide agent previously errored. Routing to FINISH to avoid loop.")
            return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
            
        logger.info("Supervisor: Outline and layout generated, no slides. Routing to slide_agent.")
        return Command(goto="slide_agent", update={"next": "slide_agent", "outline_attempts": outline_attempts})

    # 5. If all components are ready (outline, layout, slides), finish
    if is_outline_generated_flag and layout_instructions and current_slides:
        logger.info("Supervisor: All components ready (outline, layout, slides). Routing to FINISH.")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
    
    # Fallback: if we reach here, something unexpected happened
    logger.warning("Supervisor: Unexpected state reached. Routing to FINISH as fallback.")
    return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
    
def planner_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Planner node that generates a plan based on the current state.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("Planner node: Starting planning process")
    
    prompt_system_planner = f"""You are a planning agent. Your task is to create a detailed plan for the next steps in the workflow based on the current state and conversation history.
    
    Available agents: {', '.join(members)}
    
    Agent roles:
    - outline_agent: Researches information and generates a presentation outline with numbered slides
    - artist_agent: Creates layout and design instructions for the presentation slides  
    - slide_agent: Generates the actual presentation slides based on outline and layout
    
    Routing logic:
    1. If user provided only a topic: outline_agent → artist_agent → slide_agent
    2. If user provided topic + outline: artist_agent → slide_agent (skip outline_agent)
    3. If user provided topic + outline + layout: slide_agent (skip outline_agent and artist_agent)
    
    Your task is to analyze what the user has provided and create a plan with the appropriate agent tasks.
    
    The plan should specify which agent will handle each step. Use these exact agent IDs in your tasks:
    - "outline_agent" for research and outline generation
    - "artist_agent" for layout and design instructions
    - "slide_agent" for final slide generation
    
    Example plan format:
    {{
        "tasks": [
            {{
                "id": "outline_agent", 
                "description": "Research [topic] and generate a detailed presentation outline with numbered slides"
            }},
            {{
                "id": "artist_agent",
                "description": "Create layout and design instructions for the presentation slides based on the outline"
            }},
            {{
                "id": "slide_agent", 
                "description": "Generate the final presentation slides using the outline, layout instructions, and research data"
            }}
        ]
    }}
    
    Return the plan in the expected JSON format with 'tasks' field containing an array of task objects."""
    
    structured_llm = LLM_4o.with_structured_output(Plan)
    
    user_query = get_research_topic(state["messages"])
    
    formatted_prompt = planning_instructions.format(user_query=user_query)
    
    # Create the full prompt combining system and task instructions
    full_prompt = f"{prompt_system_planner}\n\n{formatted_prompt}"
    
    current_config = RunnableConfig()
    current_config.update(config)
    if langfuse_handler and hasattr(langfuse_handler, 'current_run_tree') and langfuse_handler.current_run_tree:
        current_config["configurable"]["run_tree"] = langfuse_handler.current_run_tree
    if not current_config.get("callbacks") and langfuse_handler:
        current_config["callbacks"] = [langfuse_handler]
    elif langfuse_handler not in current_config.get("callbacks",[]):
        current_config["callbacks"] = current_config.get("callbacks", []) + [langfuse_handler]
    
    plan = structured_llm.invoke(full_prompt, config=current_config)
    
    # Convert plan to string for message content - avoid completion keywords
    plan_content = f"Created workflow plan with {len(plan.tasks)} tasks: {[task.description for task in plan.tasks]}"
    
    # Update the state with the generated plan
    return Command(
        update={
            "messages": [AIMessage(content=plan_content, name="planner")],
            "plan": plan.dict()
        },
        goto="supervisor"
   )

def outline_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    logger.info("Outline agent: Starting outline generation")
    logger.debug(f"Input state for outline_agent: {state['messages']}") 
    
    agent_input = {"messages": state["messages"]}
    
    current_config = RunnableConfig()
    current_config.update(config) 
    if langfuse_handler and hasattr(langfuse_handler, 'current_run_tree') and langfuse_handler.current_run_tree:
        current_config["configurable"]["run_tree"] = langfuse_handler.current_run_tree
    if not current_config.get("callbacks") and langfuse_handler:
        current_config["callbacks"] = [langfuse_handler]
    elif langfuse_handler not in current_config.get("callbacks",[]):
        current_config["callbacks"] = current_config.get("callbacks", []) + [langfuse_handler]

    result = outline_agent.invoke(agent_input, config=current_config)
    
    logger.info("Outline agent: React agent invocation completed.")
    logger.debug(f"Raw result from outline_agent: {str(result)[:500]}...")

    extracted_outline_content = "" # Renamed for clarity
    images = []
    found_info = []
    # Default message if no outline is extracted but agent runs.
    agent_output_message_content = "Outline agent completed its turn. No specific outline content was extracted."
    
    returned_messages = result.get("messages", [])

    if returned_messages:
        for msg in returned_messages:
            if isinstance(msg, ToolMessage):
                logger.debug(f"Processing tool message: {msg.name} - {str(msg.content)[:100]}...")
                try:
                    content = json.loads(msg.content)
                    if msg.name == "image_search":
                        results_data = content.get("results", content) 
                        if isinstance(results_data, list):
                            images.extend([item if isinstance(item, dict) and 'url' in item else {"url": item, "source": msg.name} for item in results_data])
                            logger.info(f"Processed {len(results_data)} items from image_search")
                        elif isinstance(results_data, str): 
                            images.append({"url": results_data, "source": msg.name})
                    elif msg.name in ["web_search", "crawl_url"]:
                        results_data = content.get("results", content)
                        if isinstance(results_data, list):
                            for res_item in results_data:
                                found_info.append({
                                    "source": msg.name,
                                    "content": res_item.get("content", "") if isinstance(res_item, dict) else res_item,
                                    "title": res_item.get("title", "N/A") if isinstance(res_item, dict) else "N/A"
                                })
                            logger.info(f"Processed {len(results_data)} results from {msg.name}")
                        elif isinstance(results_data, dict): 
                             found_info.append({
                                "source": msg.name,
                                "content": results_data.get("content", ""),
                                "title": results_data.get("title", "N/A")
                            })
                        else: 
                           found_info.append({"source": msg.name, "content": str(results_data)})
                except json.JSONDecodeError:
                    logger.warning(f"Failed to parse JSON from tool {msg.name}: {str(msg.content)[:100]}...")
                    if msg.name == "image_search": images.append({"url": str(msg.content), "source": "image_search_json_error"})
                    else: found_info.append({"source": msg.name, "content": str(msg.content), "error": "JSONDecodeError"})
            
        final_ai_messages = [m for m in returned_messages if isinstance(m, AIMessage)] # Check for AIMessage
        if final_ai_messages:
            last_ai_message_text = final_ai_messages[-1].content
            if isinstance(last_ai_message_text, str) and last_ai_message_text.strip():
                 extracted_outline_content = last_ai_message_text
                 agent_output_message_content = extracted_outline_content # This content goes into the HumanMessage for history
                 logger.info("Extracted outline content from agent's last AIMessage.")

    # Fallback extraction methods
    if not extracted_outline_content.strip(): # Check if it's still empty/whitespace
        if isinstance(result, dict) and result.get("output") and isinstance(result["output"], str) and result["output"].strip():
            extracted_outline_content = result["output"].strip()
            agent_output_message_content = extracted_outline_content
            logger.info("Extracted outline content from agent result's 'output' field.")
        elif isinstance(result, str) and result.strip(): 
            extracted_outline_content = result.strip()
            agent_output_message_content = extracted_outline_content
            logger.info("Extracted outline content from agent result (plain string).")
        else:
            # Final fallback - use any meaningful content from the conversation
            logger.warning("Outline agent: No content extracted from agent result. Using fallback extraction.")
            user_query = get_research_topic(state["messages"])
            if user_query.strip():
                extracted_outline_content = f"Basic outline for: {user_query.strip()}\n\nSlide 1: Introduction\nSlide 2: Main Content\nSlide 3: Conclusion"
                agent_output_message_content = extracted_outline_content
                logger.info("Created basic outline as fallback.")

    is_generated = bool(extracted_outline_content.strip()) # Determine the boolean flag

    logger.info(f"Outline agent: Completed. is_outline_generated: {is_generated}. Images: {len(images)}. Info: {len(found_info)}.")
    logger.debug(f"Outline content for history message (first 100 chars): '{agent_output_message_content[:100]}'")
    
    return Command(
        update={
            "messages": [AIMessage(content=agent_output_message_content, name="outline_agent")],
            "is_outline_generated": is_generated, # Set the boolean flag
            "outline_content": extracted_outline_content,
            "images": images,
            "found_information": found_info,
            "outline_attempts": state.get("outline_attempts", 0) 
        },
        goto="supervisor"
    )
    
def artist_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that instructs the slide_agent positioning items in the presentation slides.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("Artist agent: Starting artist agent")
    
    outline_content = state.get("outline_content", "")
    prompt_system = prompt_system_layout.format(outline_content=outline_content)
    
    current_config = RunnableConfig()
    current_config.update(config)
    if langfuse_handler and hasattr(langfuse_handler, 'current_run_tree') and langfuse_handler.current_run_tree:
        current_config["configurable"]["run_tree"] = langfuse_handler.current_run_tree
    if not current_config.get("callbacks") and langfuse_handler:
        current_config["callbacks"] = [langfuse_handler]
    elif langfuse_handler not in current_config.get("callbacks",[]):
        current_config["callbacks"] = current_config.get("callbacks", []) + [langfuse_handler]
    
    response = LLM_4o.invoke(prompt_system, config=current_config)
    
    # Extract string content from the LLM response
    if hasattr(response, 'content'):
        instructions_content = response.content
    else:
        instructions_content = str(response)
    
    logger.info(f"Artist agent: Generated layout instructions (length: {len(instructions_content)})")
    
    return Command(
        update={
            "messages": [AIMessage(content=instructions_content, name="artist_agent")],
            "layout_instructions": instructions_content
        },
        goto="supervisor"
    )

    
    
    
    
    
def slide_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that generates the presentation slides.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("Slide agent: Starting slide generation")
    logger.debug(f"Input state for slide_agent: images_count={len(state.get('images',[]))}, info_count={len(state.get('found_information',[]))}")

    # Retrieve the outline content from the last message from outline_agent
    actual_outline_str = ""
    if state.get("messages"):
        for msg in reversed(state["messages"]):
            if hasattr(msg, 'name') and msg.name == "outline_agent" and isinstance(msg, AIMessage):
                if isinstance(msg.content, str) and msg.content.strip():
                    actual_outline_str = msg.content
                    logger.info(f"Retrieved outline content from outline_agent's last message (length: {len(actual_outline_str)}).")
                    break
    
    if not actual_outline_str:
        logger.warning("Slide agent: Outline content not found in message history or is empty. Proceeding might result in poor slides.")
        # This situation should ideally be prevented by the supervisor if is_outline_generated is False.
        actual_outline_str = "Outline not available. Please ensure the outline agent ran successfully and provided content."

    images = state.get("images", [])
    found_info = state.get("found_information", [])
    layout_instructions = state.get("layout_instructions", "")
    
    logger.debug(f"Images for slides: {images}")
    logger.debug(f"Found information for slides: {found_info}")
    
    instruction_file_path = os.path.join(os.getcwd(), "rules", "instruction.txt")
    slide_gen_instructions = ""
    try:
        with open(instruction_file_path, "r") as f:
            slide_gen_instructions = f.read()
    except FileNotFoundError:
        logger.warning(f"Slide generation instruction file not found at {instruction_file_path}. Using default.")
        slide_gen_instructions = "Create informative and visually appealing slides."
        
    # Construct the prompt for the slide_agent (ReAct agent)
    # using the retrieved actual_outline_str
    react_agent_prompt_str = f"""You are a presentation slide generator. 
Your task is to create multiple slides based on the provided presentation outline.

PRESENTATION OUTLINE:
{actual_outline_str}

LAYOUT INSTRUCTIONS:
{layout_instructions}

AVAILABLE IMAGES (use discretion, pick relevant ones per slide instruction):
{images}

AVAILABLE RESEARCH INFORMATION (use relevant snippets per slide instruction):
{found_info}

GENERAL INSTRUCTIONS FOR SLIDE GENERATION:
{slide_gen_instructions}

Strategy:
1. Analyze the outline and plan the slides one by one.
2. For each slide, determine its content based on the outline, research info, and suitable images.
3. Formulate detailed instructions for the 'generate_slide' tool for that specific slide, including desired style, color scheme, design language, image URLs to use (if any from the list), and the textual content.
4. Call the 'generate_slide' tool for each slide.
5. Repeat until all necessary slides based on the outline are generated.
IMPORTANT: Generate slides one at a time, starting with slide 1.
IMPORTANT: When calling `generate_slide`, the `instructions` argument to the tool should be very specific for THAT slide's content, drawing from the outline and research info.
"""
    
    # Prepare state for invoking the react agent
    current_task_messages = [
        AIMessage(content=react_agent_prompt_str),
        HumanMessage(content=get_research_topic(state["messages"]))
    ]
    # We can also include previous messages if they are relevant for the react agent beyond the prompt above
    # agent_invoke_state = {"messages": state["messages"] + current_task_messages} 
    # Or, more simply, just the task: 
    agent_invoke_state = {"messages": current_task_messages}

    logger.info("Invoking slide_agent (ReAct agent)... This may take some time.")
    
    current_config = RunnableConfig()
    current_config.update(config)
    if langfuse_handler and hasattr(langfuse_handler, 'current_run_tree') and langfuse_handler.current_run_tree:
        current_config["configurable"]["run_tree"] = langfuse_handler.current_run_tree
    if not current_config.get("callbacks") and langfuse_handler:
        current_config["callbacks"] = [langfuse_handler]
    elif langfuse_handler not in current_config.get("callbacks",[]):
        current_config["callbacks"] = current_config.get("callbacks", []) + [langfuse_handler]

    # Assuming slide_agent is a globally defined ReAct agent similar to outline_agent
    result = slide_agent.invoke(agent_invoke_state, config=current_config)
    logger.info(f"Slide agent (ReAct agent) invocation completed.")
    logger.debug(f"Raw result from slide_agent: {str(result)[:500]}...")
        
    generated_slides_info = []
    agent_final_response_content = "Slide generation process completed."

    returned_messages_from_slide_agent = result.get("messages", [])
    if returned_messages_from_slide_agent:
        for msg in returned_messages_from_slide_agent:
            if isinstance(msg, ToolMessage) and msg.name == "generate_slide":
                logger.info(f"Slide generation tool was called. Output: {msg.content}")
                # Try to extract slide number if possible from content, for more structured info
                slide_num_match = re.search(r"Slide #(\d+)", str(msg.content))
                s_num = int(slide_num_match.group(1)) if slide_num_match else 0
                generated_slides_info.append({"content": msg.content, "slide_number": s_num})
        
        final_ai_messages = [m for m in returned_messages_from_slide_agent if isinstance(m, AIMessage)]
        if final_ai_messages:
            agent_final_response_content = final_ai_messages[-1].content

    logger.info(f"Slide agent node: Generated {len(generated_slides_info)} slides. Final agent response: {agent_final_response_content[:100]}")
    return Command(
        update={
            "messages": [AIMessage(content=agent_final_response_content, name="slide_agent")],
            "slides": generated_slides_info # Update with info about generated slides
            # is_outline_generated and outline_attempts are not modified by this node directly
        },
        goto="supervisor"
    )

# Create and configure the workflow graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("planner", planner_node)
graph.add_node("outline_agent", outline_agent_node)
graph.add_node("artist_agent", artist_agent_node)
graph.add_node("slide_agent", slide_agent_node)
# graph.add_node("summarizer", summarizer_node)
graph.add_edge(START, "supervisor")

# Compile the graph and create configuration
app = graph.compile().with_config(config)