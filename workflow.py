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

import uuid, json, os
import requests, datetime, logging
from langchain_core.tools import tool
from langgraph.prebuilt import InjectedState
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage
from utils.tools import image_search, web_search, crawl_url, generate_slide
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
    input: str
    slides: list[dict]
    summary: list[dict]
    outline_attempts: int

members = ["outline_agent", "slide_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal[*options]

prompt_template_outline = """You are a research assistant helping to create a numbered presentation outline on a given topic. Follow these steps carefully:
Use web_search to gather recent and relevant information about the topic provided in the user message.
If relevant URLs are found, use crawl_url to extract detailed content from them.
Use image_search to locate high-quality and relevant images (include image URLs or descriptions for slide recommendations).

Store all collected facts, data, quotes, image links, and relevant insights in a data bank — a structured internal reference you will use to build the outline.
Using the data bank, generate a comprehensive, well-structured, and slide-numbered presentation outline.

Data Bank (Internal Use)
Store all gathered information here before building the outline. Include:
Key facts and figures
Quotes or excerpts from crawled URLs
Image URLs and their captions
Noteworthy charts/graphs or statistics
Sources and links for the reference slide

Final Output
Your final output should consist only of the presentation outline text (no commentary or metadata).
Each slide must be clearly numbered, with a title and bullet points or descriptions.

While the total number of slides depends on the topic and user needs, the outline should generally include:
Cover Slide (Title, Subtitle, optional image)
Table of Contents
Introduction Slide
Main Content Slides (organized by subtopic)
Key Points / Summary Slide
Graphs and Charts Slide(s) (if applicable; describe what should be visualized)
Conclusion Slide
References Slide (sources used, include URLs)

IMPORTANT:
You MUST use web_search and image_search before generating the final outline.
The data bank must be built first and used as a foundation.
Slide numbers must be clearly included (e.g., “Slide 4: Economic Impact of Renewable Energy”)."""

outline_agent = create_react_agent(
    model=LLM,
    tools=[image_search, crawl_url, web_search],
    prompt=prompt_template_outline
)

slide_agent_system_prompt = """You are an expert presentation slide generator. 
Your task is to create multiple slides based on a provided presentation outline and other context like available images and research information. 
You will be given a detailed task message which includes:
1. The full PRESENTATION OUTLINE.
2. A list of AVAILABLE IMAGES (URLs).
3. A list of AVAILABLE RESEARCH INFORMATION (text snippets).
4. GENERAL INSTRUCTIONS for slide generation (e.g., from rules/instruction.txt).

Your strategy MUST be:
1. Carefully analyze the entire PRESENTATION OUTLINE to understand the flow and content of all slides.
2. For EACH section or point in the outline that should become a slide, you MUST call the `generate_slide` tool ONCE.
3. When calling `generate_slide`:
    a. `slide_number`: Assign a sequential number.
    b. `instructions`: Provide VERY SPECIFIC instructions for THIS slide. This should include the exact text content for the slide (derived from the outline and research info), any data for charts, and guidance on layout or emphasis. Crucially, pass relevant parts of the AVAILABLE RESEARCH INFORMATION here.
    c. `images_urls`: Select RELEVANT image URL(s) from the AVAILABLE IMAGES list for this specific slide, if any are appropriate. Pass as a JSON string like '[{"url":"..."}]'.
    d. `style`: Define the CSS style for the slide.
    e. `content`: Provide the content for the slide.
4. Continue this process until all parts of the outline are covered by generated slides.
Your final response after all tool calls should be a summary of the slides generated."""

slide_agent = create_react_agent(
    model=LLM,
    tools=[generate_slide], # generate_slide tool is now globally defined
    prompt=slide_agent_system_prompt
)

def supervisor_node(state: AgentState) -> Command[Literal[*members, "__end__"]]:
    """
    Router function that decides which agent should run next based on the current state.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command indicating which node to go to next
    """
    logger.info("Supervisor node: Starting workflow routing")
    
    outline_attempts = state.get("outline_attempts", 0)
    is_outline_generated_flag = state.get("is_outline_generated", False)
    current_slides = state.get("slides")
    last_message_obj = state["messages"][-1] if state["messages"] else None
    last_message_content = last_message_obj.content if last_message_obj else ""
    last_message_sender = getattr(last_message_obj, 'name', None) or (last_message_obj.type if last_message_obj else "")

    logger.debug(f"Current state: is_outline_generated={is_outline_generated_flag}, slides_present={bool(current_slides)}, outline_attempts={outline_attempts}, last_sender='{last_message_sender}'")

    MAX_OUTLINE_ATTEMPTS = 3

    if "finish" in last_message_content.lower() or "completed" in last_message_content.lower():
        if last_message_sender == "slide_agent" or current_slides:
             logger.info("Supervisor: Detected completion signal. Routing to FINISH.")
             return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    if not is_outline_generated_flag:
        if outline_attempts >= MAX_OUTLINE_ATTEMPTS:
            logger.warning(f"Supervisor: Max outline attempts ({MAX_OUTLINE_ATTEMPTS}) reached. Outline still not generated. Routing to FINISH.")
            return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts + 1})
        
        logger.info("Supervisor: Outline is not generated.")
        outline_attempts += 1
        if last_message_sender != "outline_agent" or outline_attempts <= 1:
            logger.info(f"Routing to outline_agent (attempt {outline_attempts}).")
            return Command(goto="outline_agent", update={"next": "outline_agent", "outline_attempts": outline_attempts})

    if is_outline_generated_flag and not current_slides:
        if last_message_sender == "slide_agent" and "Error generating slides" in last_message_content:
             logger.warning("Supervisor: Slide agent previously errored. Routing to FINISH to avoid loop.")
             return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
        else:
            logger.info("Supervisor: Outline generated, no slides. Routing to slide_agent.")
            return Command(goto="slide_agent", update={"next": "slide_agent", "outline_attempts": outline_attempts})

    routing_prompt_template = PromptTemplate.from_template(
        f"""You are a supervisor managing a team of agents: {', '.join(members)}.
Given the current state and conversation history, decide which agent should act next or if the task is complete.
The available options are: {', '.join(options)}.

Conversation History (last few messages):
{{chat_history}}

User's initial request: {{input_request}}
Outline Status: {{outline_status}} (Attempts: {{outline_attempts}})
Slides Status: {{slides_status}}

Key information:
- If Outline Status is 'Not Generated' and attempts are low, 'outline_agent' is preferred.
- If Outline Status is 'Not Generated' and attempts are high (e.g., >= {MAX_OUTLINE_ATTEMPTS}), consider 'FINISH' if it seems stuck.
- If Outline Status is 'Generated' and Slides Status is 'None generated', 'slide_agent' is preferred.
- If all tasks seem done or an agent is stuck, consider 'FINISH'.

Based on this, which of the following should act next? Choose exactly one: {', '.join(options)}"""
    )

    chat_history_str = "\n".join([f"{msg.type} ({getattr(msg, 'name', 'user') if hasattr(msg, 'name') else msg.type}): {msg.content}" for msg in state["messages"][-5:]])
    
    prompt_input = {
        "chat_history": chat_history_str,
        "input_request": state.get("input", "N/A"),
        "outline_status": "Generated" if is_outline_generated_flag else "Not Generated",
        "slides_status": f"{len(current_slides) if current_slides else 0} slides generated" if current_slides is not None else "None generated",
        "outline_attempts": outline_attempts
    }
    
    formatted_prompt = routing_prompt_template.format(**prompt_input)
    logger.info(f"Supervisor LLM routing prompt:\n{formatted_prompt}")
    
    try:
        response = LLM.invoke(formatted_prompt, config=config)
        next_node_str = response.content if hasattr(response, 'content') else str(response)
        logger.info(f"Supervisor LLM raw response for routing: {next_node_str}")

        cleaned_response = next_node_str.strip().replace("'", "").replace("\"", "")
        chosen_option = None
        for opt in options:
            if opt.lower() in cleaned_response.lower():
                chosen_option = opt
                break
        
        if chosen_option:
            logger.info(f"Supervisor LLM decided to route to: {chosen_option}")
            return Command(goto=chosen_option, update={"next": chosen_option, "outline_attempts": outline_attempts})
        else:
            logger.warning(f"Supervisor LLM response '{cleaned_response}' didn't directly match options. Fallback needed.")
            if not is_outline_generated_flag and outline_attempts < MAX_OUTLINE_ATTEMPTS:
                 logger.info(f"Fallback: Routing to outline_agent (attempt {outline_attempts + 1})")
                 return Command(goto="outline_agent", update={"next": "outline_agent", "outline_attempts": outline_attempts + 1})
            elif is_outline_generated_flag and not current_slides:
                logger.info("Fallback: Routing to slide_agent")
                return Command(goto="slide_agent", update={"next": "slide_agent", "outline_attempts": outline_attempts})
            else:
                logger.error("Supervisor fallback failed or max attempts reached. Routing to FINISH.")
                return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    except Exception as e:
        logger.error(f"Error during supervisor LLM call: {e}. Routing to FINISH.")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

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

    is_generated = bool(extracted_outline_content.strip()) # Determine the boolean flag

    logger.info(f"Outline agent: Completed. is_outline_generated: {is_generated}. Images: {len(images)}. Info: {len(found_info)}.")
    logger.debug(f"Outline content for history message (first 100 chars): '{agent_output_message_content[:100]}'")
    
    return Command(
        update={
            "messages": [HumanMessage(content=agent_output_message_content, name="outline_agent")],
            "is_outline_generated": is_generated, # Set the boolean flag
            # The actual outline string is NOT directly in state, only in the message above.
            "images": images,
            "found_information": found_info,
            "outline_attempts": state.get("outline_attempts", 0) 
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
            if hasattr(msg, 'name') and msg.name == "outline_agent" and isinstance(msg, HumanMessage):
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
    
    # Define the slide_agent (ReAct agent) locally or ensure it's passed/accessible
    # For this example, let's assume slide_agent is defined globally like outline_agent
    # If not, it would be: slide_agent = create_react_agent(model=LLM, tools=[generate_slide], prompt=react_agent_prompt_str)
    
    # Prepare state for invoking the react agent. It primarily uses messages for context.
    # The prompt is now part of how the agent is created/configured, or passed via messages if it's a generic agent.
    # For create_react_agent, the system prompt is usually set at creation.
    # We will pass the react_agent_prompt_str as the main human message to kick off this specific task.
    
    current_task_messages = [
        HumanMessage(content=react_agent_prompt_str) 
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
            "messages": [HumanMessage(content=agent_final_response_content, name="slide_agent")],
            "slides": generated_slides_info # Update with info about generated slides
            # is_outline_generated and outline_attempts are not modified by this node directly
        },
        goto="supervisor"
    )
        
# def summarizer_node(state: AgentState) -> Command[Literal["supervisor"]]:
#     """
#     Agent that summarizes the presentation slides.
    
#     Args:
#         state: The current state of the workflow
#     """
#     logger.info("Summarizer: Starting slide summarization")
#     logger.debug(f"Input state: {state}")
    
#     slides = state["slides"]
#     prompt = f"""
#     You are a summarizer, tasked with summarizing the presentation slides.
    
#     This is the list of slides: {slides}
#     Please summarize that which slides are need to enhance visual and which slides is not.
#     """
#     response = LLM.with_structured_output(Summarize).invoke(prompt)
#     logger.info("Summarizer: Completed slide analysis")
    
#     summary_text = ""
#     if isinstance(response, dict) and "slides" in response:
#         for slide in response["slides"]:
#             summary_text += f"Slide {slide['slide_number']}: {slide['summary']} - {'Needs visual enhancement' if slide['need_enhance_visual'] else 'Visuals are good'}\n"
#             logger.info(f"Analyzed slide {slide['slide_number']}")
#     else:
#         summary_text = str(response)
#         logger.warning("Received unexpected response format from summarizer")
    
#     logger.info("Summarizer: Completed all slide summaries")
#     return Command(
#         update={
#             "messages": [HumanMessage(content=summary_text, name="summarizer")],
#             "summary": summary_text
#         },
#         goto="supervisor"
#     )

# Create and configure the workflow graph
graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("outline_agent", outline_agent_node)
graph.add_node("slide_agent", slide_agent_node)
# graph.add_node("summarizer", summarizer_node)
graph.add_edge(START, "supervisor")

# Compile the graph and create configuration
app = graph.compile().with_config(config)