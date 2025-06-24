from models.LLMs import GPT_4o, GPT_o3, Claude_3_7_Sonnet, Gemini, Gemini_2_5_Flash
from langchain.agents import AgentExecutor, create_react_agent
from langchain.prompts import PromptTemplate
from utils.tools import Searxng, image_search, web_search, crawl_url, generate_slide, generate_cover_slide
from langgraph.checkpoint.memory import MemorySaver
from typing import Literal, Annotated, List, Callable, Optional
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

from prompt import prompt_system_outline, prompt_system_slide, planning_instructions, prompt_system_layout
from utils.utils import get_research_topic
from utils.schemas import SearchQueryList, Plan, Outline, Slide_Content

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
LLM_2_5_Flash = Gemini_2_5_Flash()

memory = MemorySaver()

langfuse = Langfuse(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

# Enable Langfuse tracing
langfuse_handler = CallbackHandler(
    secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
    public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
    host="https://cloud.langfuse.com"
)

trace_id = str(uuid.uuid4())
# Base configuration with Langfuse enabled
config = {
    "recursion_limit": 100,
    "configurable": {
        "trace_id": trace_id
    },
    "callbacks": [langfuse_handler],  # Re-enabled langfuse_handler
    "run_id": trace_id
}

# Global trace for the entire workflow session
workflow_trace = None

def create_workflow_trace(session_id: str, user_query: str):
    """Create a new Langfuse trace for the entire workflow session"""
    global workflow_trace
    workflow_trace = langfuse.trace(
        name="slide_generation_workflow",
        session_id=session_id,
        user_id=session_id,  # Using session_id as user_id for now
        input={"user_query": user_query},
        metadata={
            "workflow_version": "v2.0",
            "trace_id": trace_id,
            "timestamp": str(uuid.uuid4())
        }
    )
    return workflow_trace

def get_current_trace():
    """Get the current workflow trace"""
    global workflow_trace
    return workflow_trace

def complete_supervisor_span(supervisor_span, next_node: str, reason: str, additional_output: dict = None):
    """Helper function to complete supervisor span with consistent output"""
    if supervisor_span:
        output = {"next_node": next_node, "reason": reason}
        if additional_output:
            output.update(additional_output)
        supervisor_span.end(output=output)

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

# Global streaming callback - will be set by the Flask app
streaming_callback: Optional[Callable] = None

def set_streaming_callback(callback: Callable):
    """Set the streaming callback function for real-time updates"""
    global streaming_callback
    streaming_callback = callback

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    is_outline_generated: bool
    is_outline_approved: bool
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
    model=LLM_4o,  # Use GPT-4o for better ReAct agent compatibility and reliability
    tools=[image_search, crawl_url, web_search],
    prompt=prompt_system_outline,
)

slide_agent = create_react_agent(
    model=LLM_4o,  # Use GPT-4o for better ReAct agent compatibility and reliability
    tools=[generate_slide, generate_cover_slide], # generate_slide tool is now globally defined
    prompt=prompt_system_slide
)

def stream_llm_response(llm, prompt: str, agent_name: str, config=None) -> str:
    """
    Stream LLM response token by token and return the complete response
    """
    if not streaming_callback:
        # Fallback to non-streaming if no callback is set
        response = llm.invoke(prompt, config=config)
        return response.content if hasattr(response, 'content') else str(response)
    
    # Start streaming
    streaming_callback('agent_stream_start', {'agent': agent_name})
    
    accumulated_content = ""
    
    try:
        # Use streaming for supported LLMs
        for chunk in llm.stream(prompt, config=config):
            if hasattr(chunk, 'content') and chunk.content:
                token = chunk.content
                accumulated_content += token
                
                # Emit token update
                streaming_callback('agent_stream_token', {
                    'agent': agent_name,
                    'token': token,
                    'accumulated': accumulated_content
                })
        
        # End streaming
        streaming_callback('agent_stream_end', {
            'agent': agent_name,
            'final_content': accumulated_content
        })
        
        return accumulated_content
        
    except Exception as e:
        logger.error(f"Error during streaming for {agent_name}: {str(e)}")
        # Fallback to non-streaming
        response = llm.invoke(prompt, config=config)
        final_content = response.content if hasattr(response, 'content') else str(response)
        
        streaming_callback('agent_stream_end', {
            'agent': agent_name,
            'final_content': final_content
        })
        
        return final_content

def stream_react_agent_response(agent, agent_input: dict, agent_name: str, config=None) -> dict:
    """
    Stream ReAct agent response with tool call notifications
    """
    if not streaming_callback:
        # Fallback to non-streaming
        return agent.invoke(agent_input, config=config)
    
    # Start streaming
    streaming_callback('agent_stream_start', {'agent': agent_name})
    
    accumulated_content = ""
    result = None
    seen_content = set()
    
    try:
        logger.info(f"Starting streaming for ReAct agent: {agent_name}")
        
        # Try streaming first, but catch any streaming issues
        try:
            for chunk in agent.stream(agent_input, config=config):
                logger.debug(f"ReAct agent chunk: {str(chunk)[:200]}...")
                
                if 'messages' in chunk:
                    for message in chunk['messages']:
                        if isinstance(message, AIMessage) and hasattr(message, 'content') and message.content:
                            content = message.content.strip()
                            # Only process new content
                            if content and content not in seen_content:
                                seen_content.add(content)
                                
                                if accumulated_content:
                                    accumulated_content += "\n\n"
                                accumulated_content += content
                                
                                # Stream thinking process
                                streaming_callback('agent_stream_token', {
                                    'agent': agent_name,
                                    'token': content,
                                    'accumulated': accumulated_content
                                })
                                
                        elif isinstance(message, ToolMessage):
                            tool_name = message.name
                            tool_content = str(message.content)
                            
                            # Add tool notification to stream
                            tool_summary = f"🔧 Using {tool_name} tool"
                            if accumulated_content:
                                accumulated_content += "\n\n"
                            accumulated_content += tool_summary
                            
                            # Emit tool call notification
                            streaming_callback('agent_tool_call', {
                                'agent': agent_name,
                                'tool': tool_name,
                                'content': tool_content[:200] + "..." if len(tool_content) > 200 else tool_content
                            })
                            
                            # Stream tool usage
                            streaming_callback('agent_stream_token', {
                                'agent': agent_name,
                                'token': tool_summary,
                                'accumulated': accumulated_content
                            })
                
                result = chunk
                
        except Exception as stream_error:
            logger.warning(f"Streaming failed for {agent_name}, falling back to invoke: {stream_error}")
            # If streaming fails, fall back to regular invoke but still try to show progress
            result = agent.invoke(agent_input, config=config)
        
        # Process the final result to extract meaningful content
        final_content = accumulated_content
        
        if result and 'messages' in result:
            # Extract all messages from the result
            messages = result['messages']
            
            # Build a comprehensive view of what happened
            process_summary = []
            tools_used = []
            final_ai_content = ""
            
            for msg in messages:
                if isinstance(msg, ToolMessage):
                    tools_used.append(f"🔧 Used {msg.name} tool")
                    # Add tool result summary if not already streamed
                    if not accumulated_content or msg.name not in accumulated_content:
                        tool_result_summary = f"✅ {msg.name} completed"
                        process_summary.append(tool_result_summary)
                        
                elif isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
                    content = msg.content.strip()
                    if content and len(content) > 50:  # Substantial content
                        final_ai_content = content
                        if not accumulated_content or content not in accumulated_content:
                            process_summary.append(f"💭 {content[:100]}...")
            
            # If we didn't capture much during streaming, build a summary
            if not accumulated_content or len(accumulated_content) < 100:
                summary_parts = []
                
                if tools_used:
                    summary_parts.extend(tools_used)
                    
                if process_summary:
                    summary_parts.extend(process_summary[:3])  # Limit to 3 items
                    
                if final_ai_content:
                    summary_parts.append(f"📋 Generated outline: {final_ai_content[:200]}...")
                
                if summary_parts:
                    final_content = "\n\n".join(summary_parts)
                    
                    # Stream this summary if we didn't stream much before
                    if not accumulated_content:
                        for part in summary_parts:
                            streaming_callback('agent_stream_token', {
                                'agent': agent_name,
                                'token': part,
                                'accumulated': final_content
                            })
                            # Small delay to make it visible
                            import time
                            time.sleep(0.5)
                else:
                    final_content = final_ai_content if final_ai_content else "Agent completed execution"
            
        logger.info(f"Final content length for {agent_name}: {len(final_content)}")
        logger.info(f"Final content preview for {agent_name}: {final_content[:300]}...")
        
        # End streaming
        streaming_callback('agent_stream_end', {
            'agent': agent_name,
            'final_content': final_content if final_content else "Agent completed execution"
        })
        
        return result
        
    except Exception as e:
        logger.error(f"Error in ReAct agent streaming for {agent_name}: {str(e)}")
        
        # Emergency fallback
        try:
            result = agent.invoke(agent_input, config=config)
            
            # Try to extract any meaningful content from the result
            fallback_content = "Agent completed execution"
            if result and 'messages' in result:
                ai_messages = [m for m in result['messages'] if isinstance(m, AIMessage) and hasattr(m, 'content')]
                tool_messages = [m for m in result['messages'] if isinstance(m, ToolMessage)]
                
                if tool_messages:
                    tool_list = [f"🔧 Used {m.name}" for m in tool_messages]
                    fallback_content = "\n".join(tool_list)
                    
                if ai_messages and ai_messages[-1].content:
                    if len(ai_messages[-1].content.strip()) > 50:
                        fallback_content += f"\n\n📋 Result: {ai_messages[-1].content[:300]}..."
            
            streaming_callback('agent_stream_end', {
                'agent': agent_name,
                'final_content': fallback_content
            })
            
            return result
            
        except Exception as fallback_error:
            logger.error(f"Complete failure for {agent_name}: {fallback_error}")
            streaming_callback('agent_stream_end', {
                'agent': agent_name,
                'final_content': f"Error: {str(e)}"
            })
            return {"messages": []}

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
    
    # Create Langfuse span for supervisor decision
    trace = get_current_trace()
    supervisor_span = None
    if trace:
        supervisor_span = trace.span(
            name="supervisor_routing",
            input={
                "is_outline_generated": state.get("is_outline_generated", False),
                "is_outline_approved": state.get("is_outline_approved", False),
                "has_layout": bool(state.get("layout_instructions")),
                "has_slides": bool(state.get("slides")),
                "outline_attempts": state.get("outline_attempts", 0)
            },
            metadata={"agent": "supervisor"}
        )
    
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
                "is_outline_approved": False,
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

    # Check for outline approval responses
    is_outline_approved = state.get("is_outline_approved", False)
    outline_approval_keywords = ["approve", "yes", "proceed", "continue", "looks good", "ok", "accept"]
    outline_rejection_keywords = ["reject", "no", "redo", "regenerate", "change", "modify"]
    
    # Check if user is responding to outline approval request
    if (is_outline_generated_flag and not is_outline_approved and 
        isinstance(last_message_obj, HumanMessage)):
        
        user_response = last_message_obj.content.lower().strip()
        
        # Check for approval
        if any(keyword in user_response for keyword in outline_approval_keywords):
            logger.info("Supervisor: User approved the outline. Proceeding to artist_agent.")
            complete_supervisor_span(supervisor_span, "artist_agent", "outline_approved")
            return Command(
                update={
                    "is_outline_approved": True,
                    "messages": [AIMessage(content="Outline approved! Proceeding to layout design...", name="supervisor")],
                    "next": "artist_agent",
                    "outline_attempts": outline_attempts
                },
                goto="artist_agent"
            )
        
        # Check for rejection
        elif any(keyword in user_response for keyword in outline_rejection_keywords):
            logger.info("Supervisor: User rejected the outline. Resetting to regenerate.")
            complete_supervisor_span(supervisor_span, "outline_agent", "outline_rejected")
            return Command(
                update={
                    "is_outline_generated": False,
                    "is_outline_approved": False,
                    "outline_content": "",
                    "messages": [AIMessage(content="Regenerating outline based on your feedback...", name="supervisor")],
                    "next": "outline_agent",
                    "outline_attempts": outline_attempts
                },
                goto="outline_agent"
            )
        
        # If response is unclear, ask for clarification
        else:
            logger.info("Supervisor: Unclear response to outline. Asking for clarification.")
            complete_supervisor_span(supervisor_span, "FINISH", "unclear_approval_response")
            return Command(
                update={
                    "messages": [AIMessage(content="Please respond with 'approve' to proceed with the outline, or 'reject' to regenerate it.", name="supervisor")],
                    "next": "supervisor"
                },
                goto="FINISH"
            )

    # Check for specific completion signals - only from slide_agent and only with actual completion phrases
    completion_phrases = ["presentation completed", "slide generation completed", "workflow completed", "all slides generated"]
    has_completion_signal = any(phrase in last_message_content.lower() for phrase in completion_phrases)
    
    if has_completion_signal and last_message_sender == "slide_agent" and current_slides:
        logger.info("Supervisor: Detected specific completion signal from slide_agent. Routing to FINISH.")
        complete_supervisor_span(supervisor_span, "FINISH", "slide_agent_completion_signal")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Check if we've reached max attempts for outline generation
    if not is_outline_generated_flag and outline_attempts >= MAX_OUTLINE_ATTEMPTS:
        logger.warning(f"Supervisor: Max outline attempts ({MAX_OUTLINE_ATTEMPTS}) reached. Routing to FINISH.")
        complete_supervisor_span(supervisor_span, "FINISH", f"max_outline_attempts_reached_{MAX_OUTLINE_ATTEMPTS}")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Detect if user has provided outline or layout content
    user_content_detection = _detect_user_provided_content(state["messages"])
    logger.debug(f"User content detection: {user_content_detection}")
    
    # Follow planner logic: Check what's available and route to next needed agent
    
    # 1. If no plan exists yet, start with planner (only for initial message)
    current_plan = state.get("plan", {})
    if not current_plan and not any(hasattr(msg, 'name') and msg.name == "planner" for msg in state["messages"]):
        logger.info("Supervisor: No plan found. Routing to planner.")
        complete_supervisor_span(supervisor_span, "planner", "no_plan_found")
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
            
            # Update state to mark outline as generated and approved (since user provided it)
            complete_supervisor_span(supervisor_span, "artist_agent", "user_provided_outline")
            return Command(
                update={
                    "is_outline_generated": True,
                    "is_outline_approved": True,
                    "outline_content": user_outline,
                    "messages": [AIMessage(content=f"Using user-provided outline: {user_outline[:100]}...", name="outline_agent")],
                    "next": "artist_agent",
                    "outline_attempts": outline_attempts
                },
                goto="artist_agent"
            )
        else:
            logger.info(f"Supervisor: No outline generated. Routing to outline_agent (attempt {outline_attempts + 1}).")
            complete_supervisor_span(supervisor_span, "outline_agent", f"outline_generation_attempt_{outline_attempts + 1}")
            return Command(goto="outline_agent", update={"next": "outline_agent", "outline_attempts": outline_attempts + 1})
    
    # 3. If outline exists but not approved yet, wait for approval
    if is_outline_generated_flag and not is_outline_approved:
        logger.info("Supervisor: Outline generated but not approved yet. Waiting for user approval.")
        approval_message = "📋 I've generated an outline for your presentation. Please review it above and respond with:\n• 'approve' or 'yes' to proceed with slide generation\n• 'reject' or 'no' to regenerate the outline\n\nWhat would you like to do?"
        logger.debug(f"Supervisor: Sending approval request message: {approval_message}")
        complete_supervisor_span(supervisor_span, "FINISH", "waiting_for_outline_approval")
        return Command(
            update={
                "messages": [AIMessage(content=approval_message, name="supervisor")],
                "next": "supervisor"
            },
            goto="FINISH"
        )

    # 4. If outline exists and is approved but no layout instructions, check if user provided layout
    if is_outline_generated_flag and is_outline_approved and not layout_instructions:
        if user_content_detection["has_layout"]:
            logger.info("Supervisor: User provided layout content. Using it and proceeding to slide_agent.")
            # Extract layout from user messages
            user_layout = ""
            for msg in state["messages"]:
                if isinstance(msg, HumanMessage) and any(keyword in msg.content.lower() for keyword in ["layout:", "design:", "positioning:"]):
                    user_layout += msg.content + "\n"
            
            complete_supervisor_span(supervisor_span, "slide_agent", "user_provided_layout")
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
                complete_supervisor_span(supervisor_span, "FINISH", "artist_agent_error_loop_prevention")
                return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
            
            logger.info("Supervisor: Outline approved, no layout instructions. Routing to artist_agent.")
            complete_supervisor_span(supervisor_span, "artist_agent", "need_layout_instructions")
            return Command(goto="artist_agent", update={"next": "artist_agent", "outline_attempts": outline_attempts})
    
    # 5. If outline approved, layout exists, but no slides, go to slide_agent
    if is_outline_generated_flag and is_outline_approved and layout_instructions and not current_slides:
        # Check if slide_agent previously errored to avoid loops
        if last_message_sender == "slide_agent" and ("Error generating slides" in last_message_content or "error" in last_message_content.lower()):
            logger.warning("Supervisor: Slide agent previously errored. Routing to FINISH to avoid loop.")
            complete_supervisor_span(supervisor_span, "FINISH", "slide_agent_error_loop_prevention")
            return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
            
        logger.info("Supervisor: Outline approved and layout generated, no slides. Routing to slide_agent.")
        complete_supervisor_span(supervisor_span, "slide_agent", "ready_for_slide_generation")
        return Command(goto="slide_agent", update={"next": "slide_agent", "outline_attempts": outline_attempts})

    # 6. If all components are ready (outline approved, layout, slides), finish
    if is_outline_generated_flag and is_outline_approved and layout_instructions and current_slides:
        logger.info("Supervisor: All components ready (outline approved, layout, slides). Routing to FINISH.")
        complete_supervisor_span(supervisor_span, "FINISH", "workflow_complete_all_components_ready")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})
    
    # Fallback: if we reach here, something unexpected happened
    logger.warning("Supervisor: Unexpected state reached. Routing to FINISH as fallback.")
    
    # Complete supervisor span
    if supervisor_span:
        supervisor_span.end(output={"next_node": "FINISH", "reason": "unexpected_state"})
    
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
    
    # Create Langfuse span for planner
    trace = get_current_trace()
    planner_span = None
    if trace:
        user_query = get_research_topic(state["messages"])
        planner_span = trace.span(
            name="planner_agent",
            input={"user_query": user_query},
            metadata={"agent": "planner"}
        )
    
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
    
    user_query = get_research_topic(state["messages"])
    formatted_prompt = planning_instructions.format(user_query=user_query)
    full_prompt = f"{prompt_system_planner}\n\n{formatted_prompt}"
    
    current_config = RunnableConfig()
    current_config.update(config)
    current_config["callbacks"] = []
    
    # Stream the planning response
    response_content = stream_llm_response(LLM_4o, full_prompt, "planner", config=current_config)
    
    # Try to parse the response as structured data
    try:
        # Try to extract JSON from the response
        import json
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            plan_data = json.loads(json_match.group())
            if 'tasks' in plan_data:
                plan = Plan(**plan_data)
            else:
                # Create a basic plan structure
                plan = Plan(tasks=[
                    {"id": "outline_agent", "description": f"Research {user_query} and generate outline"},
                    {"id": "artist_agent", "description": "Create layout instructions"},
                    {"id": "slide_agent", "description": "Generate presentation slides"}
                ])
        else:
            # Fallback plan
            plan = Plan(tasks=[
                {"id": "outline_agent", "description": f"Research {user_query} and generate outline"},
                {"id": "artist_agent", "description": "Create layout instructions"},
                {"id": "slide_agent", "description": "Generate presentation slides"}
            ])
    except Exception as e:
        logger.warning(f"Failed to parse plan from streamed response: {e}")
        plan = Plan(tasks=[
            {"id": "outline_agent", "description": f"Research {user_query} and generate outline"},
            {"id": "artist_agent", "description": "Create layout instructions"},
            {"id": "slide_agent", "description": "Generate presentation slides"}
        ])
    
    # Convert plan to string for message content - avoid completion keywords
    plan_content = f"Created workflow plan with {len(plan.tasks)} tasks: {[task.description for task in plan.tasks]}"
    
    # Complete planner span
    if planner_span:
        planner_span.end(output={
            "plan": plan.dict(),
            "num_tasks": len(plan.tasks),
            "task_agents": [task.id if hasattr(task, 'id') else task.get('id', 'unknown') for task in plan.tasks]
        })
    
    # Update the state with the generated plan
    return Command(
        update={
            "messages": [AIMessage(content=plan_content, name="planner")],
            "plan": plan.dict()
        },
        goto="supervisor"
   )

def outline_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    import json  # Import json at the beginning of the function
    
    logger.info("Outline agent: Starting outline generation")
    logger.debug(f"Input state for outline_agent: {state['messages']}")
    
    # Create Langfuse span for outline agent
    trace = get_current_trace()
    outline_span = None
    if trace:
        user_query = get_research_topic(state["messages"])
        outline_span = trace.span(
            name="outline_agent",
            input={
                "user_query": user_query,
                "outline_attempts": state.get("outline_attempts", 0)
            },
            metadata={"agent": "outline_agent"}
        ) 
    
    # Ensure the outline agent gets the original user request, not just the planner message
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    planner_messages = [msg for msg in state["messages"] if hasattr(msg, 'name') and msg.name == 'planner']
    
    # Include both user request and planner context for the outline agent
    outline_agent_messages = []
    if user_messages:
        # Add the original user request
        outline_agent_messages.extend(user_messages)
    if planner_messages:
        # Add the planner's plan for context
        outline_agent_messages.extend(planner_messages)
    
    # If we don't have user messages, fall back to all messages
    if not outline_agent_messages:
        outline_agent_messages = state["messages"]
    
    agent_input = {"messages": outline_agent_messages}
    
    # Log the input messages for debugging
    logger.info(f"Outline agent input messages ({len(outline_agent_messages)} total):")
    for i, msg in enumerate(outline_agent_messages):
        logger.info(f"Input message {i}: type={type(msg)}, content='{str(msg.content)[:100]}...'")
    
    current_config = RunnableConfig()
    current_config.update(config) 
    current_config["callbacks"] = []

    result = stream_react_agent_response(outline_agent, agent_input, "outline_agent", config=current_config)
    
    with open("outline_agent_result.txt", "w", encoding="utf-8") as f:
        f.write(str(result))
    
    logger.info("Outline agent: React agent invocation completed.")
    logger.info(f"Raw result from outline_agent: {str(result)[:1000]}...")
    
    # Log the complete result structure for debugging
    logger.info(f"Result type: {type(result)}")
    if isinstance(result, dict):
        logger.info(f"Result keys: {list(result.keys())}")
        if 'messages' in result:
            logger.info(f"Number of messages: {len(result['messages'])}")
            for i, msg in enumerate(result['messages']):
                logger.info(f"Message {i}: type={type(msg)}, name={getattr(msg, 'name', 'no_name')}, content_length={len(str(msg.content)) if hasattr(msg, 'content') else 0}")
                if hasattr(msg, 'content'):
                    logger.info(f"Message {i} content preview: {str(msg.content)[:200]}...")
        elif 'generate_structured_response' in result:
            logger.info(f"Found structured response: {str(result['generate_structured_response'])[:1000]}...")
        else:
            logger.info(f"Result content (first 500 chars): {str(result)[:500]}...")
    else:
        logger.info(f"Result content: {str(result)[:500]}...")

    extracted_outline_content = "" # Renamed for clarity
    images = []
    found_info = []
    # Default message if no outline is extracted but agent runs.
    agent_output_message_content = "Outline agent completed its turn. No specific outline content was extracted."
    
    # Handle different result formats from the outline agent
    returned_messages = []
    
    if isinstance(result, dict):
        if 'messages' in result:
            returned_messages = result['messages']
        elif 'agent' in result and isinstance(result['agent'], dict) and 'messages' in result['agent']:
            # Handle nested agent result format
            returned_messages = result['agent']['messages']
            logger.info(f"Found messages in nested agent format: {len(returned_messages)} messages")
        elif 'generate_structured_response' in result:
            # Handle structured response format (this was the previous case)
            pass
    
    # Check if result has structured response instead of messages
    if not returned_messages and 'generate_structured_response' in result:
        logger.info("Outline agent returned structured response instead of messages")
        structured_response = result['generate_structured_response']
        logger.info(f"Structured response content: {str(structured_response)[:500]}...")
        
        # Try to extract outline from structured response
        if hasattr(structured_response, 'outline') or (isinstance(structured_response, dict) and 'outline' in structured_response):
            outline_data = structured_response.outline if hasattr(structured_response, 'outline') else structured_response['outline']
            logger.info(f"Found outline data with {len(outline_data)} slides")
            
            # Convert structured outline to text format
            outline_text_parts = []
            for slide in outline_data:
                if hasattr(slide, 'slide_number'):
                    slide_num = slide.slide_number
                    slide_title = slide.slide_title if hasattr(slide, 'slide_title') else 'Untitled'
                    slide_content = slide.slide_body if hasattr(slide, 'slide_body') else ''
                    slide_script = slide.slide_script if hasattr(slide, 'slide_script') else ''
                elif isinstance(slide, dict):
                    slide_num = slide.get('slide_number', 0)
                    slide_title = slide.get('slide_title', 'Untitled')
                    slide_content = slide.get('slide_body', '')
                    slide_script = slide.get('slide_script', '')
                else:
                    continue
                
                outline_text_parts.append(f"Slide {slide_num}: {slide_title}")
                if slide_content:
                    outline_text_parts.append(f"Content: {slide_content}")
                if slide_script:
                    outline_text_parts.append(f"Script: {slide_script}")
                outline_text_parts.append("")  # Add spacing between slides
            
            extracted_outline_content = "\n".join(outline_text_parts)
            agent_output_message_content = f"Generated structured outline with {len(outline_data)} slides"
            logger.info(f"Successfully extracted structured outline with {len(outline_data)} slides")

    if returned_messages or extracted_outline_content:
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
            last_ai_message = final_ai_messages[-1]
            
            # Check if the message has structured content (Outline schema)
            if hasattr(last_ai_message, 'content'):
                content = last_ai_message.content
                
                # Try to parse as structured Outline response
                try:
                    if isinstance(content, dict) and 'outline' in content:
                        # Direct structured response
                        outline_data = content['outline']
                    elif isinstance(content, str):
                        # Try to parse JSON string
                        parsed_content = json.loads(content)
                        if 'outline' in parsed_content:
                            outline_data = parsed_content['outline']
                        else:
                            # Fallback to text content
                            extracted_outline_content = content.strip()
                            agent_output_message_content = extracted_outline_content
                            logger.info(f"Extracted outline content from agent's last AIMessage (text format). Length: {len(extracted_outline_content)}")
                            logger.info(f"Outline content preview: {extracted_outline_content[:300]}...")
                            outline_data = None
                    else:
                        outline_data = None
                        
                    if outline_data:
                        # Convert structured outline to text format
                        outline_text_parts = []
                        for slide in outline_data:
                            slide_num = slide.get('slide_number', 0)
                            slide_content = slide.get('slide_content', '')
                            slide_script = slide.get('slide_script', '')
                            
                            outline_text_parts.append(f"Slide {slide_num}: {slide_content}")
                            if slide_script:
                                outline_text_parts.append(f"Script: {slide_script}")
                        
                        extracted_outline_content = "\n\n".join(outline_text_parts)
                        agent_output_message_content = f"Generated structured outline with {len(outline_data)} slides"
                        logger.info(f"Extracted structured outline with {len(outline_data)} slides.")
                        
                except (json.JSONDecodeError, KeyError, TypeError) as e:
                    logger.warning(f"Failed to parse structured outline, falling back to text: {e}")
                    # Fallback to text content
                    if isinstance(content, str) and content.strip():
                        extracted_outline_content = content.strip()
                        agent_output_message_content = extracted_outline_content
                        logger.info("Extracted outline content from agent's last AIMessage (fallback text).")

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
    
    # Complete outline agent span
    if outline_span:
        outline_span.end(output={
            "is_generated": is_generated,
            "outline_length": len(extracted_outline_content),
            "images_found": len(images),
            "research_info_found": len(found_info),
            "outline_preview": extracted_outline_content[:200]
        })
    
    return Command(
        update={
            "messages": [AIMessage(content=agent_output_message_content, name="outline_agent")],
            "is_outline_generated": is_generated, # Set the boolean flag
            "is_outline_approved": False, # Always require approval after outline generation
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
    
    # Create Langfuse span for artist agent
    trace = get_current_trace()
    artist_span = None
    if trace:
        artist_span = trace.span(
            name="artist_agent",
            input={
                "outline_length": len(state.get("outline_content", "")),
                "has_outline": bool(state.get("outline_content"))
            },
            metadata={"agent": "artist_agent"}
        )
    
    outline_content = state.get("outline_content", "")
    prompt_system = prompt_system_layout.format(outline_content=outline_content)
    
    current_config = RunnableConfig()
    current_config.update(config)
    current_config["callbacks"] = []
    
    # Stream the artist response
    instructions_content = stream_llm_response(LLM_4o, prompt_system, "artist_agent", config=current_config)
    
    logger.info(f"Artist agent: Generated layout instructions (length: {len(instructions_content)})")
    
    # Complete artist agent span
    if artist_span:
        artist_span.end(output={
            "layout_instructions_length": len(instructions_content),
            "layout_preview": instructions_content[:200]
        })
    
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
    
    # Create Langfuse span for slide agent
    trace = get_current_trace()
    slide_span = None
    if trace:
        slide_span = trace.span(
            name="slide_agent",
            input={
                "outline_length": len(state.get("outline_content", "")),
                "layout_instructions_length": len(state.get("layout_instructions", "")),
                "images_available": len(state.get("images", [])),
                "research_info_available": len(state.get("found_information", []))
            },
            metadata={"agent": "slide_agent"}
        )

    # Retrieve the outline content from state (preferred) or fallback to messages
    actual_outline_str = state.get("outline_content", "")
    
    if not actual_outline_str.strip():
        # Fallback: search in messages if state doesn't have outline_content
        logger.warning("Slide agent: outline_content not found in state, searching messages...")
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'name') and msg.name == "outline_agent" and isinstance(msg, AIMessage):
                    if isinstance(msg.content, str) and msg.content.strip():
                        actual_outline_str = msg.content
                        logger.info(f"Retrieved outline content from outline_agent's last message (length: {len(actual_outline_str)}).")
                        break
    
    if not actual_outline_str.strip():
        logger.warning("Slide agent: Outline content not found in state or message history. Proceeding might result in poor slides.")
        actual_outline_str = "Outline not available. Please ensure the outline agent ran successfully and provided content."
    else:
        logger.info(f"Slide agent: Using outline content from state (length: {len(actual_outline_str)}).")

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

GENERAL INSTRUCTIONS FOR SLIDE GENERATION:
{slide_gen_instructions}

CRITICAL CONSTRAINTS:
- You MUST generate slides ONE AT A TIME, in sequence (slide 1, then slide 2, then slide 3, etc.)
- You MUST NOT call multiple generate_slide tools in parallel
- After each generate_slide call, wait for the result before proceeding to the next slide
- Call generate_slide ONLY ONCE per reasoning cycle
- Think step by step: analyze outline → decide on slide 1 content → call generate_slide for slide 1 → wait for result → then proceed to slide 2

Strategy:
1. Analyze the outline and identify how many slides to create
2. For slide 1 ONLY: determine its content, select relevant images/research, then call generate_slide
3. After slide 1 is complete, then work on slide 2
4. Continue this pattern until all slides are generated
5. Each generate_slide call should include: slide number, specific content, relevant images, and design instructions

REMEMBER: ONE SLIDE AT A TIME. NO PARALLEL TOOL CALLS."""
    
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
    current_config["callbacks"] = []

    # Stream the slide agent response
    result = stream_react_agent_response(slide_agent, agent_invoke_state, "slide_agent", config=current_config)
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
    
    # Complete slide agent span
    if slide_span:
        slide_span.end(output={
            "slides_generated": len(generated_slides_info),
            "slide_numbers": [s.get("slide_number", 0) for s in generated_slides_info],
            "final_response_preview": agent_final_response_content[:200]
        })
    
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