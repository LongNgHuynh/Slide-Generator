from models.LLMs import GPT_4o
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from utils.tools import Searxng
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Annotated, TypedDict, List
from langgraph.graph import StateGraph, add_messages
from utils.custom_output_parser import CustomOutputParser
from langgraph.prebuilt import create_react_agent
from langgraph_supervisor import create_supervisor, create_handoff_tool
import uuid, json, os
import requests, datetime, logging
from langchain_core.tools import tool, InjectedToolCallId
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage
from langchain_core.messages import ToolMessage
from utils.tools import image_search, web_search, crawl_url, generate_presentation_outline, generate_presentation
import operator
import pprint

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM = GPT_4o()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AgentState(TypedDict):
    messages: List[BaseMessage]
    remaining_steps: int
    is_last_step: bool
    outline: str
    images: list[dict]
    found_information: list[dict]
    input: str
    next_step: str
    slides: list[dict]

members = ["outline_agent", "slide_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal["outline_agent", "slide_agent", "FINISH"]

def supervisor_node(state: AgentState) -> Command[Literal["outline_agent", "slide_agent", "__end__"]]:
    """
    Router function that decides which agent should run next based on the current state.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command indicating which node to go to next
    """
    system_prompt = f"""
    You are a supervisor, tasked with managing a conversation between the following workers: {members}. 
    Given the user's request, respond with the worker to act next. 
    Each worker will perform a task and respond with results and status. 
    When finished, respond with FINISH.
    """
    messages = [{"role": "system", "content": system_prompt}] + state["messages"]
    response = LLM.with_structured_output(Router).invoke(messages)
    goto = response["next"]
    if goto == "FINISH":
        goto = END
    return Command(goto=goto, update={"next": goto})

def outline_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that generates the presentation outline.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    # Create a prompt that encourages tool usage
    prompt = """You are a research assistant helping to create a presentation. Follow these steps:

1. First, use web_search to gather information about the topic
2. Then, use crawl_url to get detailed information from specific websites
3. Use image_search to find relevant images
4. Finally, use generate_presentation_outline to create the outline

IMPORTANT: You MUST use at least web_search and image_search before generating the outline.
"""
    
    outline_agent = create_react_agent(
        model=LLM,
        tools=[image_search, crawl_url, web_search, generate_presentation_outline],
        prompt=prompt
    )
    result = outline_agent.invoke(state)
    
    # Extract tool outputs from messages
    outline = ""
    images = []
    found_info = []
    
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            try:
                # Try to parse JSON response
                content = json.loads(message.content)
                
                if message.name == "generate_presentation_outline":
                    outline = content if isinstance(content, str) else json.dumps(content)
                elif message.name == "image_search":
                    if isinstance(content, dict) and "results" in content:
                        images.extend(content["results"])
                    else:
                        images.append({"url": message.content})
                elif message.name in ["web_search", "crawl_url"]:
                    if isinstance(content, dict):
                        found_info.append({
                            "source": message.name,
                            "content": content.get("content", ""),
                            "url": content.get("url", ""),
                        })
                    else:
                        found_info.append({
                            "source": message.name,
                            "content": message.content,
                        })
            except json.JSONDecodeError:
                # If not JSON, handle as plain text
                if message.name == "generate_presentation_outline":
                    outline = message.content
                elif message.name == "image_search":
                    images.append({"url": message.content})
                elif message.name in ["web_search", "crawl_url"]:
                    found_info.append({
                        "source": message.name,
                        "content": message.content,
                        "timestamp": datetime.datetime.now().isoformat()
                    })
    
    logger.info(f"Outline: {outline}")
    logger.info(f"Images: {images}")
    logger.info(f"Found information: {found_info}")
    
    return Command(
        update={
            "messages": [HumanMessage(content=result["messages"][-1].content, name="outline_agent")],
            "outline": outline,
            "images": images,
            "found_information": found_info
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
    images = state["images"]
    found_info = state["found_information"]
    # Create a prompt that guides slide generation
    prompt = f"""You are a presentation slide generator. Your task is to create slides based on the outline.
    
    This is list of images that you can use for the slides: {images}
    This is the list of information that you can use for the slides: {found_info}
    
    Remember to generate enough slides to cover the outline.
    For each slide:

    1. Use the generate_presentation tool with appropriate parameters
    2. Include relevant images from the found images if available
    3. Use a consistent style and color scheme
    4. Make sure content is clear and concise

IMPORTANT: Generate slides one at a time, starting with slide 1.
"""
    
    slide_agent = create_react_agent(
        model=LLM,
        tools=[generate_presentation],
        prompt=prompt
    )
    
    try:
        result = slide_agent.invoke(state)
        
        # Extract tool outputs from messages
        slides = []
        for message in result["messages"]:
            if isinstance(message, ToolMessage):
                if message.name == "generate_presentation":
                    try:
                        # Try to parse slide number from the message
                        slide_number = message.additional_kwargs.get("slide_number", 0)
                        if not slide_number and "Slide #" in message.content:
                            # Try to extract slide number from content
                            import re
                            match = re.search(r"Slide #(\d+)", message.content)
                            if match:
                                slide_number = int(match.group(1))
                        
                        slides.append({
                            "content": message.content,
                            "slide_number": slide_number,
                            "timestamp": datetime.datetime.now().isoformat()
                        })
                        logger.info(f"Generated slide #{slide_number}")
                    except Exception as e:
                        logger.error(f"Error processing slide: {str(e)}")
                        slides.append({
                            "content": message.content,
                            "error": str(e),
                            "timestamp": datetime.datetime.now().isoformat()
                        })
        
        return Command(
            update={
                "messages": [HumanMessage(content=result["messages"][-1].content, name="slide_agent")],
                "slides": slides
            },
            goto="supervisor"
        )
    except Exception as e:
        logger.error(f"Error in slide generation: {str(e)}")
        return Command(
            update={
                "messages": [HumanMessage(content=f"Error generating slides: {str(e)}", name="slide_agent")],
                "slides": []
            },
            goto="supervisor"
        )

def custom_tool_node(state):
    
    return {
        "outline": generate_presentation_outline(state["input"]), 
        "images": image_search(state["input"])
    }

graph = StateGraph(AgentState)
graph.add_node("supervisor", supervisor_node)
graph.add_node("outline_agent", outline_agent_node)
graph.add_node("slide_agent", slide_agent_node)

graph.add_edge(START, "supervisor")

app = graph.compile()
config = {"configurable": {"thread_id": "1"}}

while True:
    user_input = input("\nEnter your query (or 'exit' to quit): ")

    if user_input.lower() == 'exit':
        print("Goodbye!")
        break

    # Create initial state
    initial_state = {
        "messages": [HumanMessage(content=user_input)],
        "remaining_steps": 5,
        "is_last_step": False,
        "outline": "",
        "images": [],
        "found_information": [],
        "input": user_input,
        "next_step": "",
        "slides": []
    }

    try:
        result = app.invoke(initial_state, config=config)
        print("\n[DEBUG] Full result:")
        pprint.pprint(result)

        for m in result["messages"]:
            if isinstance(m, ToolMessage):
                print(f"ToolMessage: {m.content}")
    except Exception as e:
        print(f"Error occurred: {str(e)}")