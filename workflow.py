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
from datetime import datetime
from langchain_core.tools import tool
from langchain_core.runnables import RunnableConfig
from langgraph.types import Command
from langgraph.checkpoint.memory import InMemorySaver
from langgraph.graph import MessagesState, END, START
from langchain_core.messages import BaseMessage, ToolMessage, HumanMessage, AIMessage
from langchain_core.messages import ToolMessage

from prompt import prompt_system_outline, prompt_system_slide, planning_instructions, prompt_system_layout
from utils.utils import get_research_topic
from utils.schemas import SearchQueryList, Plan, Outline, Slide_Content, StructuredOutline, SlideStructure, LayoutInstructions

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
    # New structured data fields
    structured_outline: Optional[dict]  # Will store StructuredOutline.dict()
    original_user_request: Optional[str]  # Store the original user request
    # Legacy fields (keep for backward compatibility)
    outline_content: str
    layout_instructions: str
    outline_attempts: int
    plan: dict

members = ["planner", "outline_agent", "slide_agent", "artist_agent"]
options = members + ["FINISH"]

class Router(TypedDict):
    next: Literal[*options]

outline_agent = create_react_agent(
    model=LLM,  # Use GPT-4o for better ReAct agent compatibility and reliability
    # remove image search from tools for now
    # tools=[image_search, crawl_url, web_search],
    tools=[crawl_url, web_search],
    prompt=prompt_system_outline,
)

# Structured outline agent using with_structured_output
structured_outline_agent = LLM_4o.with_structured_output(StructuredOutline)

# Structured artist agent using with_structured_output  
structured_artist_agent = LLM_4o.with_structured_output(LayoutInstructions)

slide_agent = create_react_agent(
    model=LLM_4o,
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
    Intelligent supervisor agent that analyzes workflow state and decides which agent should run next.
    Uses LLM reasoning instead of hardcoded routing logic.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command indicating which node to go to next
    """
    logger.info("Supervisor agent: Starting intelligent workflow supervision")
    
    # Create Langfuse span for supervisor decision
    trace = get_current_trace()
    supervisor_span = None
    if trace:
        supervisor_span = trace.span(
            name="supervisor_agent",
            input={
                "is_outline_generated": state.get("is_outline_generated", False),
                "is_outline_approved": state.get("is_outline_approved", False),
                "has_layout": bool(state.get("layout_instructions")),
                "has_slides": bool(state.get("slides")),
                "outline_attempts": state.get("outline_attempts", 0),
                "message_count": len(state.get("messages", []))
            },
            metadata={"agent": "supervisor"}
        )
    
    # Analyze current state
    outline_attempts = state.get("outline_attempts", 0)
    is_outline_generated_flag = state.get("is_outline_generated", False)
    is_outline_approved = state.get("is_outline_approved", False)
    current_slides = state.get("slides", [])
    structured_outline = state.get("structured_outline")
    has_structured_outline = bool(structured_outline)
    layout_instructions = state.get("layout_instructions", "")
    current_plan = state.get("plan", {})
    last_message_obj = state["messages"][-1] if state["messages"] else None
    last_message_content = last_message_obj.content if last_message_obj else ""
    last_message_sender = getattr(last_message_obj, 'name', None) or (last_message_obj.type if last_message_obj else "")

    # Check if structured outline has layout instructions
    has_layout_in_structured = False
    if structured_outline and 'slides' in structured_outline:
        slides_data = structured_outline['slides']
        has_layout_in_structured = any(
            slide.get('layout_instructions') for slide in slides_data
        )
    
    # Get recent messages for context
    recent_messages = state["messages"][-5:] if len(state["messages"]) > 5 else state["messages"]
    messages_context = "\n".join([
        f"{msg.type if hasattr(msg, 'type') else 'message'} ({getattr(msg, 'name', 'unknown')}): {str(msg.content)[:200]}..."
        for msg in recent_messages
    ])
    
    # Detect user content in messages
    user_content_detection = _detect_user_provided_content(state["messages"])
    
    # Check for critical safety conditions first (before LLM reasoning)
    MAX_OUTLINE_ATTEMPTS = 3
    if not is_outline_generated_flag and outline_attempts >= MAX_OUTLINE_ATTEMPTS:
        logger.warning(f"Supervisor: Max outline attempts ({MAX_OUTLINE_ATTEMPTS}) reached. Routing to FINISH.")
        complete_supervisor_span(supervisor_span, "FINISH", f"max_outline_attempts_reached_{MAX_OUTLINE_ATTEMPTS}")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Check for explicit completion signals
    completion_phrases = ["presentation completed", "slide generation completed", "workflow completed", "all slides generated"]
    has_completion_signal = any(phrase in last_message_content.lower() for phrase in completion_phrases)
    
    if has_completion_signal and last_message_sender == "slide_agent" and current_slides:
        logger.info("Supervisor: Detected explicit completion signal from slide_agent. Routing to FINISH.")
        complete_supervisor_span(supervisor_span, "FINISH", "slide_agent_completion_signal")
        return Command(goto="FINISH", update={"next": "FINISH", "outline_attempts": outline_attempts})

    # Create supervisor agent prompt
    supervisor_prompt = f"""You are an intelligent workflow supervisor for an AI slide generation system. Your job is to analyze the current workflow state and decide which agent should run next.

AVAILABLE AGENTS:
- planner: Creates initial workflow plan based on user request
- outline_agent: Researches topic and generates presentation outline with numbered slides
- artist_agent: Creates layout and design instructions for slides based on outline
- slide_agent: Generates actual HTML slides using outline, layout, and research data
- FINISH: Complete the workflow

CURRENT WORKFLOW STATE:
- Plan exists: {bool(current_plan)}
- Outline generated: {is_outline_generated_flag}
- Structured outline exists: {has_structured_outline}
- Outline approved: {is_outline_approved}
- Layout instructions exist: {bool(layout_instructions) or has_layout_in_structured}
- Structured layout exists: {has_layout_in_structured}
- Slides generated: {len(current_slides)} slides
- Outline attempts: {outline_attempts}
- User provided outline: {user_content_detection.get('has_outline', False)}
- User provided layout: {user_content_detection.get('has_layout', False)}

RECENT CONVERSATION CONTEXT:
{messages_context}

LAST MESSAGE: {last_message_content[:300]}...
FROM: {last_message_sender}

WORKFLOW LOGIC RULES:
1. If no plan exists and no agents have run yet → planner
2. If no outline exists and user hasn't provided one → outline_agent
3. If outline exists but not approved → FINISH and show outline to user for approval
4. If outline approved but no layout (neither text nor structured) → artist_agent  
5. If outline approved and layout exists but no slides → slide_agent
6. If outline, layout, and slides all exist → FINISH
7. Handle user approval/rejection responses appropriately
8. If user provides outline/layout content, skip corresponding agents
9. If previous agent had errors, consider alternative approaches or FINISH
10. When outline is generated, always require user approval before proceeding
11. Structured outline and layout are preferred over text-based versions

CRITICAL WORKFLOW FLOW:
The complete flow is: outline_agent → (approval) → artist_agent → slide_agent → FINISH
Do NOT skip steps! After approval, continue to artist_agent, then slide_agent.
Approval means "continue the workflow", NOT "finish the workflow".

APPROVAL HANDLING:
- If user says "approve", "yes", "proceed" → approve outline and continue to artist_agent
- If user says "reject", "no", "redo" → regenerate outline
- If response unclear → ask for clarification

ERROR HANDLING:
- If agent previously errored → avoid infinite loops, consider FINISH
- If too many attempts → FINISH with current state

YOUR TASK:
Analyze the current state and conversation context. Decide which agent should run next or if workflow should finish.

Respond with JSON format:
{{
    "next_agent": "agent_name_or_FINISH",
    "reasoning": "Brief explanation of decision",
    "state_updates": {{
        "key": "value if any state needs updating"
    }}
}}

Available next_agent options: planner, outline_agent, artist_agent, slide_agent, FINISH"""

    # Use supervisor LLM to make decision
    current_config = RunnableConfig()
    current_config.update(config)
    current_config["callbacks"] = []
    
    try:
        # Don't stream supervisor reasoning - use invoke to get decision internally
        response = LLM_4o.invoke(supervisor_prompt, config=current_config)
        response_content = response.content if hasattr(response, 'content') else str(response)
        
        # Parse the JSON response
        import json
        import re
        
        # Try to extract JSON from response
        json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
        if json_match:
            try:
                decision = json.loads(json_match.group())
                next_agent = decision.get("next_agent", "FINISH")
                reasoning = decision.get("reasoning", "No reasoning provided")
                state_updates = decision.get("state_updates", {})
                
                logger.info(f"Supervisor decision: {next_agent} - Reasoning: {reasoning}")
                
                # Validate the decision
                valid_agents = ["planner", "outline_agent", "artist_agent", "slide_agent", "FINISH"]
                if next_agent not in valid_agents:
                    logger.warning(f"Invalid agent decision: {next_agent}, defaulting to FINISH")
                    next_agent = "FINISH"
                
                # Handle special cases based on supervisor decision
                update_dict = {"next": next_agent, "outline_attempts": outline_attempts}
                
                # Apply any state updates from supervisor
                if state_updates:
                    update_dict.update(state_updates)
                
                # Handle approval scenarios - ONLY if last message is from user
                if isinstance(last_message_obj, HumanMessage):
                    approval_keywords = ["approve", "yes", "proceed", "continue", "looks good", "ok", "accept"]
                    rejection_keywords = ["reject", "no", "redo", "regenerate", "change", "modify"]
                    
                    user_response_lower = last_message_content.lower().strip()
                    is_approval = any(keyword in user_response_lower for keyword in approval_keywords)
                    is_rejection = any(keyword in user_response_lower for keyword in rejection_keywords)
                    
                    # Handle approval when outline exists but not yet approved
                    if (is_outline_generated_flag and not is_outline_approved and is_approval):
                        # Override next_agent to continue workflow instead of finishing
                        next_agent = "artist_agent"
                        update_dict.update({
                            "is_outline_approved": True,
                            "messages": [AIMessage(content="Outline approved! Proceeding with layout design...")]
                        })
                        logger.info("Supervisor: User approved the outline, proceeding to artist_agent for layout design")
                    # Handle rejection when outline exists but not approved
                    elif (is_outline_generated_flag and not is_outline_approved and is_rejection):
                        update_dict.update({
                "is_outline_generated": False,
                            "is_outline_approved": False,
                            "structured_outline": None,  # Clear structured outline
                            "outline_content": "",
                            "messages": [AIMessage(content="Regenerating outline based on feedback...")]
                        })
                        logger.info("Supervisor: User rejected the outline, will regenerate")
                    # Handle case where outline was just approved (state already updated) but workflow needs to continue
                    elif (is_outline_generated_flag and is_outline_approved and is_approval and 
                          not layout_instructions and not has_layout_in_structured and not current_slides):
                        # Override LLM decision if it decided to FINISH prematurely
                        if next_agent == "FINISH":
                            next_agent = "artist_agent"
                            update_dict.update({
                                "messages": [AIMessage(content="Outline approved! Proceeding with layout design...")]
                            })
                            logger.info("Supervisor: Outline was approved, overriding FINISH decision to continue with artist_agent")
                        # If LLM already decided correctly, don't override
                        elif next_agent == "artist_agent":
                            logger.info("Supervisor: LLM correctly decided to proceed to artist_agent after approval")
                
                # Handle new topic detection
                if next_agent == "planner" and is_outline_generated_flag:
                    # Reset for new topic
                    update_dict.update({
                        "is_outline_generated": False,
                        "is_outline_approved": False,
                        "structured_outline": None,  # Clear structured outline
                "outline_content": "",
                "layout_instructions": "",
                "slides": [],
                "images": [],
                "found_information": [],
                "outline_attempts": 0,
                        "plan": {}
                    })
                
                # Show outline for approval when needed
                if next_agent == "FINISH" and is_outline_generated_flag and not is_outline_approved and "messages" not in update_dict:
                    # Show outline for approval - prefer structured outline if available
                    structured_outline = state.get("structured_outline")
                    outline_content = state.get("outline_content", "")
                    
                    display_content = ""
                    if structured_outline and 'slides' in structured_outline:
                        # Format structured outline for display
                        title = structured_outline.get('presentation_title', 'Presentation')
                        slides = structured_outline['slides']
                        display_parts = [f"# {title}\n"]
                        
                        for slide in slides:
                            slide_num = slide.get('slide_number', 0)
                            slide_title = slide.get('title', 'Untitled')
                            slide_content = slide.get('content', '')
                            
                            display_parts.append(f"\n## Slide {slide_num}: {slide_title}")
                            if slide_content:
                                # Truncate content if too long
                                content_preview = slide_content[:300] + "..." if len(slide_content) > 300 else slide_content
                                display_parts.append(f"{content_preview}\n")
                        
                        display_content = "\n".join(display_parts)
                        logger.info(f"Supervisor showing structured outline with {len(slides)} slides for approval")
                    elif outline_content and outline_content.strip():
                        display_content = outline_content.strip()
                        logger.info(f"Supervisor showing text outline for approval. Content length: {len(display_content)}")
                    
                    if display_content:
                        # Limit length for better display
                        if len(display_content) > 2000:
                            display_content = display_content[:2000] + "\n\n... (content truncated)"
                        approval_message = f"{display_content}\n\nPlease respond with:\n• 'approve' or 'yes' to proceed with layout design\n• 'reject' or 'no' to regenerate the outline"
                    else:
                        logger.warning("No outline content found in state for approval display")
                        approval_message = f"I've generated an outline for your presentation. Please review it above and respond with:\n• 'approve' or 'yes' to proceed with layout design\n• 'reject' or 'no' to regenerate the outline"
                    update_dict["messages"] = [AIMessage(content=approval_message, name="supervisor")]
                
                # Complete supervisor span
                complete_supervisor_span(supervisor_span, next_agent, reasoning, {
                    "supervisor_decision": next_agent,
                    "reasoning": reasoning,
                    "state_updates_applied": bool(state_updates)
                })
                
                # Route to decision
                if next_agent == "FINISH":
                    return Command(goto="FINISH", update=update_dict)
                else:
                    return Command(goto=next_agent, update=update_dict)
                    
            except json.JSONDecodeError as e:
                logger.error(f"Failed to parse supervisor JSON response: {e}")
                
        # Fallback if parsing fails
        logger.warning("Supervisor LLM response parsing failed, using fallback logic")
        
    except Exception as e:
        logger.error(f"Error in supervisor LLM call: {e}")
        
    # Fallback to basic routing logic if LLM fails
    logger.info("Using fallback routing logic due to LLM error")
    
    # Basic fallback logic
    if not current_plan and not any(hasattr(msg, 'name') and msg.name == "planner" for msg in state["messages"]):
        complete_supervisor_span(supervisor_span, "planner", "fallback_no_plan")
        return Command(goto="planner", update={"next": "planner", "outline_attempts": outline_attempts})
    elif not is_outline_generated_flag:
        complete_supervisor_span(supervisor_span, "outline_agent", "fallback_no_outline")
        return Command(goto="outline_agent", update={"next": "outline_agent", "outline_attempts": outline_attempts + 1})
    elif is_outline_generated_flag and not is_outline_approved:
        # Show outline for approval in fallback case
        structured_outline = state.get("structured_outline")
        outline_content = state.get("outline_content", "")
        
        display_content = ""
        if structured_outline and 'slides' in structured_outline:
            # Format structured outline for display
            title = structured_outline.get('presentation_title', 'Presentation')
            slides = structured_outline['slides']
            display_parts = [f"# {title}\n"]
            
            for slide in slides:
                slide_num = slide.get('slide_number', 0)
                slide_title = slide.get('title', 'Untitled')
                slide_content = slide.get('content', '')
                
                display_parts.append(f"\n## Slide {slide_num}: {slide_title}")
                if slide_content:
                    content_preview = slide_content[:300] + "..." if len(slide_content) > 300 else slide_content
                    display_parts.append(f"{content_preview}\n")
            
            display_content = "\n".join(display_parts)
        elif outline_content and outline_content.strip():
            display_content = outline_content.strip()
        
        if display_content:
            if len(display_content) > 2000:
                display_content = display_content[:2000] + "\n\n... (content truncated)"
            approval_message = f"{display_content}\n\nPlease respond with:\n• 'approve' or 'yes' to proceed with layout design\n• 'reject' or 'no' to regenerate the outline"
        else:
            approval_message = f"I've generated an outline for your presentation. Please review it above and respond with:\n• 'approve' or 'yes' to proceed with layout design\n• 'reject' or 'no' to regenerate the outline"
        
        complete_supervisor_span(supervisor_span, "FINISH", "fallback_waiting_approval")
        return Command(goto="FINISH", update={
            "next": "FINISH", 
            "outline_attempts": outline_attempts,
            "messages": [AIMessage(content=approval_message, name="supervisor")]
        })
    elif is_outline_generated_flag and is_outline_approved and not (layout_instructions or has_layout_in_structured):
        complete_supervisor_span(supervisor_span, "artist_agent", "fallback_need_layout")
        return Command(goto="artist_agent", update={"next": "artist_agent", "outline_attempts": outline_attempts})
    elif is_outline_generated_flag and is_outline_approved and (layout_instructions or has_layout_in_structured) and not current_slides:
        complete_supervisor_span(supervisor_span, "slide_agent", "fallback_need_slides")
        return Command(goto="slide_agent", update={"next": "slide_agent", "outline_attempts": outline_attempts})
    else:
        complete_supervisor_span(supervisor_span, "FINISH", "fallback_complete")
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
        original_request = state.get("original_user_request")
        user_query = get_research_topic(state["messages"], original_request)
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
    
    original_request = state.get("original_user_request")
    user_query = get_research_topic(state["messages"], original_request)
    formatted_prompt = planning_instructions.format(user_query=user_query)
    full_prompt = f"{prompt_system_planner}\n\n{formatted_prompt}"
    
    current_config = RunnableConfig()
    current_config.update(config)
    current_config["callbacks"] = []
    
    # Don't stream planner JSON - use invoke to get plan internally
    response = LLM_4o.invoke(full_prompt, config=current_config)
    response_content = response.content if hasattr(response, 'content') else str(response)
    
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
    
    logger.info("Outline agent: Starting structured outline generation")
    logger.debug(f"Input state for outline_agent: {state['messages']}") 
    
    # Create Langfuse span for outline agent
    trace = get_current_trace()
    outline_span = None
    if trace:
        user_query = get_research_topic(state["messages"])
        outline_span = trace.span(
            name="outline_agent_structured",
            input={
                "user_query": user_query,
                "outline_attempts": state.get("outline_attempts", 0)
            },
            metadata={"agent": "outline_agent"}
        ) 
    
    # Get user query for outline generation - prioritize original request if available
    original_request = state.get("original_user_request")
    user_query = get_research_topic(state["messages"], original_request)
    logger.info(f"DEBUG: Original request from state: '{original_request}'")
    logger.info(f"DEBUG: Final extracted user_query: '{user_query}'")
    
    # Get plan context to understand specific requirements
    plan_context = ""
    plan_data = state.get("plan", {})
    if plan_data and "tasks" in plan_data:
        for task in plan_data["tasks"]:
            if task.get("id") == "outline_agent":
                plan_context = f"Plan requirement: {task.get('description', '')}"
                logger.info(f"Found plan context for outline_agent: {plan_context}")
                break
    
    # If we still have supervisor text despite original_request, fall back to a clean message
    if "🎯 Supervisor:" in user_query or "supervisor" in user_query.lower():
        logger.warning(f"User query still contains supervisor text: '{user_query}', falling back to generic topic")
        user_query = "Create a comprehensive presentation"
    
    # Create focused research input for React agent
    # Include plan context in the research message
    research_directive = f"Create a detailed presentation outline about: {user_query}"
    if plan_context:
        research_directive += f"\n\nSpecific requirements from plan: {plan_context}"
    if original_request:
        research_directive += f"\n\nOriginal user request: {original_request}"
        
    research_message = HumanMessage(content=research_directive)
    
    # Include original user messages for additional context if available
    user_messages = [msg for msg in state["messages"] if isinstance(msg, HumanMessage)]
    planner_messages = [msg for msg in state["messages"] if hasattr(msg, 'name') and msg.name == 'planner']
    
    research_messages = [research_message]  # Start with clear directive
    
    # Add original user messages for context (but research_message takes priority)
    if user_messages:
        research_messages.extend(user_messages[:2])  # Limit to first 2 for context
    
    # Add planner context if available
    if planner_messages:
        research_messages.extend(planner_messages[:1])  # Limit to 1 for context
    
    agent_input = {"messages": research_messages}
    
    # Log the input messages for debugging
    logger.info(f"Outline agent research phase - input messages ({len(research_messages)} total):")
    logger.info(f"Extracted user_query: '{user_query}'")
    for i, msg in enumerate(research_messages):
        logger.info(f"Input message {i}: type={type(msg)}, content='{str(msg.content)[:100]}...'")
    
    current_config = RunnableConfig()
    current_config.update(config) 
    current_config["callbacks"] = []

    # Run research phase using React agent
    logger.info(f"Running research phase with React agent for topic: {user_query}")
    research_result = stream_react_agent_response(outline_agent, agent_input, "outline_agent", config=current_config)
    
    with open("outline_agent_result.txt", "w", encoding="utf-8") as f:
        f.write(str(research_result))
    
    logger.info("Outline agent: React agent research phase completed.")
    logger.info(f"Raw research result from outline_agent: {str(research_result)[:1000]}...")

    # Extract research data from the React agent result
    images = []
    found_info = []
    research_summary = ""
    
    # Extract tool results from research phase
    returned_messages = []
    if isinstance(research_result, dict):
        if 'messages' in research_result:
            returned_messages = research_result['messages']
        elif 'agent' in research_result and isinstance(research_result['agent'], dict) and 'messages' in research_result['agent']:
            returned_messages = research_result['agent']['messages']
    
    logger.info(f"Processing {len(returned_messages)} research messages for data extraction...")
    
    # Extract images and research information from tool calls
    for msg in returned_messages:
        if isinstance(msg, ToolMessage):
            logger.debug(f"Processing tool message: {msg.name} - {str(msg.content)[:100]}...")
            try:
                content = json.loads(msg.content)
                if msg.name == "image_search":
                    logger.info(f"Found image_search tool message. Content keys: {content.keys()}")
                    results_data = content.get("results", content) 
                    logger.info(f"Image results_data type: {type(results_data)}, length: {len(results_data) if isinstance(results_data, list) else 'not a list'}")
                    if isinstance(results_data, list):
                        for i, item in enumerate(results_data[:3]):  # Log first 3 items for debug
                            logger.info(f"Image item {i}: {type(item)}, keys: {item.keys() if isinstance(item, dict) else 'not a dict'}")
                        # Extract image data properly
                        extracted_images = []
                        for item in results_data:
                            if isinstance(item, dict):
                                # Extract the image URL from the item
                                img_url = item.get("img_src", item.get("url", item.get("image_url", "")))
                                if img_url:
                                    extracted_images.append({
                                        "url": img_url,
                                        "title": item.get("title", item.get("alt", "Image from search")),
                                        "content": item.get("content", item.get("description", "")),
                                        "alt": item.get("alt", item.get("title", "Search result image")),
                                        "source": "image_search"
                                    })
                            elif isinstance(item, str) and (item.startswith('http') and any(ext in item.lower() for ext in ['.jpg', '.jpeg', '.png', '.gif', '.webp'])):
                                extracted_images.append({
                                    "url": item, 
                                    "title": "Image from search",
                                    "alt": "Search result image",
                                    "source": "image_search"
                                })
                        images.extend(extracted_images)
                        logger.info(f"Successfully extracted {len(extracted_images)} images from image_search")
                    else:
                        logger.warning(f"Image search results_data is not a list: {type(results_data)}")
                        # Try to extract individual image if it's a single result
                        if isinstance(results_data, dict):
                            img_url = results_data.get("img_src", results_data.get("url", results_data.get("image_url", "")))
                            if img_url:
                                images.append({
                                    "url": img_url,
                                    "title": results_data.get("title", "Image from search"),
                                    "alt": results_data.get("alt", "Search result image"),
                                    "source": "image_search"
                                })
                                logger.info(f"Extracted single image from image_search: {img_url}")
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
                if msg.name == "image_search": 
                    # Try to extract URLs from raw content as fallback
                    import re
                    url_pattern = r'https?://[^\s]+\.(jpg|jpeg|png|gif|webp)'
                    urls = re.findall(url_pattern, str(msg.content), re.IGNORECASE)
                    for url in urls[:3]:  # Limit to 3
                        images.append({
                            "url": url, 
                            "title": "Image from search (JSON parse failed)",
                            "alt": "Search result image",
                            "source": "image_search_fallback"
                        })
                    logger.info(f"Fallback extracted {len(urls)} image URLs from raw content")
                else: 
                    found_info.append({"source": msg.name, "content": str(msg.content), "error": "JSONDecodeError"})
        elif isinstance(msg, AIMessage) and hasattr(msg, 'content') and msg.content:
            # Collect research insights from AI messages
            if research_summary:
                research_summary += "\n\n"
            research_summary += msg.content
    
    logger.info(f"Research phase complete: {len(images)} images, {len(found_info)} research items")
    
    # Now generate structured outline using research data
    logger.info("Starting structured outline generation phase...")
    
    # Check if there's a requested slide count from modification request
    requested_slide_count = state.get('requested_slide_count', None)
    slide_count_instruction = ""
    if requested_slide_count:
        slide_count_instruction = f"\n\nIMPORTANT: Generate exactly {requested_slide_count} slides to maintain consistency with the previous presentation structure."
        logger.info(f"Using requested slide count: {requested_slide_count}")
    
    outline_generation_prompt = f"""Based on the user request and the research data gathered, create a comprehensive presentation outline.

USER REQUEST: {user_query}

RESEARCH SUMMARY:
{research_summary[:2000] if research_summary else "No research summary available"}

RESEARCH INFORMATION FOUND:
{str(found_info[:3])[:1000] if found_info else "No research data found"}

IMAGES AVAILABLE:
{str(images[:3])[:500] if images else "No images found"}{slide_count_instruction}

Create a structured presentation outline with:
- A clear presentation title
- Number of slides as requested by the user with meaningful titles and detailed content (else use 4-7 slides)
- Always let the refenences at footnote of the slide
- Each slide should have substantial content that can be expanded into a full slide
- Use the research data to make the content accurate and informative
- Focus on creating educational, well-structured content

The outline should be comprehensive enough to create a professional presentation."""
    
    try:
        logger.info(f"Generating structured outline with prompt: {outline_generation_prompt}")
        # Generate structured outline using with_structured_output
        structured_outline = structured_outline_agent.invoke(outline_generation_prompt, config=current_config)
        logger.info(f"Successfully generated structured outline with {len(structured_outline.slides)} slides")
        
        # Convert to dict for storage in state
        structured_outline_dict = structured_outline.dict()
        
        # Create text representation for compatibility
        outline_text_parts = [f"# {structured_outline.presentation_title}\n"]
        for slide in structured_outline.slides:
            outline_text_parts.append(f"\n## Slide {slide.slide_number}: {slide.title}")
            outline_text_parts.append(f"{slide.content}\n")
        
        extracted_outline_content = "\n".join(outline_text_parts)
        agent_output_message_content = f"Generated structured outline: '{structured_outline.presentation_title}' with {len(structured_outline.slides)} slides"
        
        is_generated = True
        logger.info(f"✅ SUCCESS: Generated structured outline with {len(structured_outline.slides)} slides")
        
    except Exception as e:
        logger.error(f"Failed to generate structured outline: {e}")
        # Fallback to basic outline
        structured_outline_dict = None
        extracted_outline_content = f"""Presentation Outline for: {user_query}

Slide 1: Introduction
- Topic overview and objectives
- Key context and background

Slide 2: Main Concepts  
- Core principles and definitions
- Essential background information

Slide 3: Key Points and Analysis
- Detailed explanations
- Important findings and insights

Slide 4: Applications and Examples
- Real-world applications
- Case studies and examples

Slide 5: Conclusion and Summary
- Key takeaways
- Future considerations and next steps"""
        
        agent_output_message_content = f"Generated basic outline (structured generation failed): {e}"
        is_generated = True
    
    # No need to extract images - we'll pass full outline content to slide agent

    logger.info(f"Outline agent: Completed. is_outline_generated: {is_generated}. Images: {len(images)}. Info: {len(found_info)}.")
    logger.debug(f"Outline content for history message (first 100 chars): '{agent_output_message_content[:100]}'")
    
    # Complete outline agent span
    if outline_span:
        outline_span.end(output={
            "is_generated": is_generated,
            "outline_length": len(extracted_outline_content),
            "structured_outline_available": structured_outline_dict is not None,
            "images_found": len(images),
            "research_info_found": len(found_info),
            "outline_preview": extracted_outline_content[:200]
        })
    
    return Command(
        update={
            "messages": [AIMessage(content=agent_output_message_content, name="outline_agent")],
            "is_outline_generated": is_generated,
            "is_outline_approved": False,  # Always require approval after outline generation
            "structured_outline": structured_outline_dict,  # Store structured data
            "outline_content": extracted_outline_content,  # Keep for compatibility
            "images": images,
            "found_information": found_info,
            "outline_attempts": state.get("outline_attempts", 0) 
        },
        goto="supervisor"
    )
    
def artist_agent_node(state: AgentState) -> Command[Literal["supervisor"]]:
    """
    Agent that creates layout instructions for each slide using structured output.
    
    Args:
        state: The current state of the workflow
        
    Returns:
        Command to update the state and proceed to supervisor
    """
    logger.info("Artist agent: Starting structured layout generation")
    
    # Create Langfuse span for artist agent
    trace = get_current_trace()
    artist_span = None
    if trace:
        artist_span = trace.span(
            name="artist_agent_structured",
            input={
                "has_structured_outline": bool(state.get("structured_outline")),
                "outline_length": len(state.get("outline_content", ""))
            },
            metadata={"agent": "artist_agent"}
        )
    
    # Get plan context and presentation topic for intelligent color decision
    plan_context = ""
    presentation_topic = ""
    plan_data = state.get("plan", {})
    if plan_data and "tasks" in plan_data:
        for task in plan_data["tasks"]:
            if task.get("id") == "artist_agent":
                plan_context = task.get('description', '')
                logger.info(f"Found plan context for artist_agent: {plan_context}")
                break
    
    # Get presentation topic from original request or outline
    original_request = state.get("original_user_request", "")
    if original_request:
        presentation_topic = original_request
    elif outline_content:
        # Extract topic from outline title if available
        lines = outline_content.split('\n')
        for line in lines:
            if line.strip() and (line.startswith('#') or 'presentation' in line.lower()):
                presentation_topic = line.strip().replace('#', '').strip()
                break
        if not presentation_topic:
            presentation_topic = original_request or "General presentation"
    
    # Pass user's original request to let LLM analyze design requirements
    user_design_context = ""
    if original_request:
        user_design_context = f"User's original request: '{original_request}'"
    
    logger.info(f"Presentation topic: {presentation_topic}")
    logger.info(f"User design context: {user_design_context}")
    
    # Get structured outline or fallback to text outline
    structured_outline = state.get("structured_outline")
    outline_content = state.get("outline_content", "")
    
    if structured_outline:
        logger.info("Using structured outline for layout generation")
        # Use structured outline data
        outline_data = StructuredOutline(**structured_outline)
        slides_for_layout = outline_data.slides
        
        # Create prompt using structured data
        slides_summary = "\n".join([
            f"Slide {slide.slide_number}: {slide.title}\nContent: {slide.content[:200]}..."
            for slide in slides_for_layout
        ])
        
        layout_prompt = f"""You are a professional presentation designer. Create layout instructions for each slide in this presentation.

PRESENTATION: {outline_data.presentation_title}
TOPIC: {presentation_topic}

SLIDES TO DESIGN:
{slides_summary}

USER REQUIREMENTS:
{user_design_context if user_design_context else "No specific user requirements mentioned"}
{f"Plan context: {plan_context}" if plan_context else "No specific plan requirements"}

INSTRUCTIONS FOR INTELLIGENT DESIGN AND COLOR SELECTION:
Analyze the user's request above and make intelligent design decisions:
- If user mentions specific colors/backgrounds (like galaxy, space, wine, etc.), incorporate those elements thoughtfully and appropriately
- If user specifies design styles (minimalist, modern, etc.), follow those preferences creatively
- If no specific design mentioned, intelligently choose colors and style based on the presentation topic:
  * Technology/AI/Tech topics: Modern blues, teals, or clean grays
  * Business/Finance: Professional blues, grays, or subtle greens
  * Health/Medical: Clean whites, soft blues, or medical greens
  * Education/Academic: Warm blues, oranges, or scholarly colors
  * Creative/Art: Vibrant but balanced color schemes
  * Environmental: Natural greens, earth tones
  * Food/Hospitality: Warm oranges, reds, or appetizing colors
  * Historical/Ancient civilizations: Rich heritage colors - Byzantine (imperial purple, gold, deep red), Roman (imperial red, marble white, gold), Egyptian (sand, gold, deep blue), Medieval (deep burgundy, forest green, gold), Renaissance (rich burgundy, deep blue, gold accents)
  * Cultural/Regional topics: Colors that reflect the cultural heritage and traditions of the region
  * Default: Professional blue and gray color scheme

For each slide, provide specific layout and design instructions that include:
- Visual hierarchy and positioning
- Intelligent color scheme selection based on user requirements and topic
- Typography recommendations that match the overall design theme
- Image/graphic placement recommendations
- Content organization and spacing
- Design style (modern, professional, creative, cosmic, minimalist, etc.)

IMPORTANT: Be creative and choose colors/designs that enhance the presentation topic and user requirements. Make thoughtful design decisions that create engaging presentations."""
        
    else:
        logger.warning("No structured outline found, using text outline fallback")
        # Fallback to text-based outline
        base_layout_prompt = prompt_system_layout.format(outline_content=outline_content)
        
        # Add intelligent design and color selection to the base prompt
        requirements_text = f"""

TOPIC: {presentation_topic}

USER REQUIREMENTS:
{user_design_context if user_design_context else "No specific user requirements mentioned"}
{f"Plan context: {plan_context}" if plan_context else "No specific plan requirements"}

INSTRUCTIONS FOR INTELLIGENT DESIGN AND COLOR SELECTION:
Analyze the user's request above and make intelligent design decisions:
- If user mentions specific colors/backgrounds (like galaxy, space, wine, etc.), incorporate those elements thoughtfully and appropriately
- If user specifies design styles (minimalist, modern, etc.), follow those preferences creatively
- If no specific design mentioned, intelligently choose colors and style based on the presentation topic:
  * Technology/AI/Tech topics: Modern blues, teals, or clean grays
  * Business/Finance: Professional blues, grays, or subtle greens
  * Health/Medical: Clean whites, soft blues, or medical greens
  * Education/Academic: Warm blues, oranges, or scholarly colors
  * Creative/Art: Vibrant but balanced color schemes
  * Environmental: Natural greens, earth tones
  * Food/Hospitality: Warm oranges, reds, or appetizing colors
  * Historical/Ancient civilizations: Rich heritage colors - Byzantine (imperial purple, gold, deep red), Roman (imperial red, marble white, gold), Egyptian (sand, gold, deep blue), Medieval (deep burgundy, forest green, gold), Renaissance (rich burgundy, deep blue, gold accents)
  * Cultural/Regional topics: Colors that reflect the cultural heritage and traditions of the region
  * Default: Professional blue and gray color scheme

IMPORTANT: Be creative and choose colors/designs that enhance the presentation topic and user requirements. Make thoughtful design decisions that create engaging presentations."""
        
        layout_prompt = base_layout_prompt + requirements_text
        
        # Parse outline to create slides structure for layout
        slides_for_layout = []
        lines = outline_content.split('\n')
        current_slide_num = 1
        
        for line in lines:
            line = line.strip()
            if line and ('slide' in line.lower() or line.startswith('#')):
                title = line.replace('#', '').replace('Slide', '').replace(str(current_slide_num), '').replace(':', '').strip()
                slides_for_layout.append(SlideStructure(
                    slide_number=current_slide_num,
                    title=title,
                    content=f"Content for {title}"
                ))
                current_slide_num += 1
        
        if not slides_for_layout:
            # Create basic slides if parsing fails
            slides_for_layout = [
                SlideStructure(slide_number=1, title="Introduction", content="Introduction content"),
                SlideStructure(slide_number=2, title="Main Points", content="Main content"),
                SlideStructure(slide_number=3, title="Conclusion", content="Conclusion content")
            ]
    
    current_config = RunnableConfig()
    current_config.update(config)
    current_config["callbacks"] = []
    
    try:
        # Generate structured layout using with_structured_output
        logger.info(f"Generating layouts for {len(slides_for_layout)} slides...")
        
        # Create the input with slide data that needs layout
        layout_input_data = LayoutInstructions(slide_layouts=slides_for_layout)
        
        # Use structured agent to generate layout instructions
        structured_layout = structured_artist_agent.invoke(layout_prompt, config=current_config)
        
        logger.info(f"Successfully generated structured layout for {len(structured_layout.slide_layouts)} slides")
        
        # Update the structured outline with layout instructions
        updated_slides = []
        layout_text_parts = []
        
        for layout_slide in structured_layout.slide_layouts:
            # Update slide with layout instructions
            updated_slide = SlideStructure(
                slide_number=layout_slide.slide_number,
                title=layout_slide.title,
                content=layout_slide.content,
                layout_instructions=layout_slide.layout_instructions
            )
            updated_slides.append(updated_slide)
            
            # Create text representation
            layout_text_parts.append(f"Slide {layout_slide.slide_number} Layout:")
            if layout_slide.layout_instructions:
                layout_text_parts.append(layout_slide.layout_instructions)
            layout_text_parts.append("")
        
        # Update structured outline with layout instructions
        if structured_outline:
            updated_outline_dict = structured_outline.copy()
            updated_outline_dict['slides'] = [slide.dict() for slide in updated_slides]
        else:
            # Create new structured outline if it didn't exist
            updated_outline_dict = StructuredOutline(
                presentation_title="Generated Presentation",
                slides=updated_slides
            ).dict()
        
        layout_instructions_text = "\n".join(layout_text_parts)
        agent_message = f"Generated layout instructions for {len(updated_slides)} slides"
        
        logger.info(f"✅ SUCCESS: Generated structured layout for {len(updated_slides)} slides")
        
    except Exception as e:
        logger.error(f"Failed to generate structured layout: {e}")
        # Fallback to simple layout generation
        layout_instructions_text = stream_llm_response(LLM_4o, layout_prompt, "artist_agent", config=current_config)
        updated_outline_dict = structured_outline  # Keep original
        agent_message = f"Generated layout instructions (structured generation failed): {e}"
    
    # Complete artist agent span
    if artist_span:
        artist_span.end(output={
            "layout_generated": True,
            "slides_with_layout": len(updated_slides) if 'updated_slides' in locals() else 0,
            "structured_layout_success": 'updated_slides' in locals()
        })
    
    return Command(
        update={
            "messages": [AIMessage(content=agent_message, name="artist_agent")],
            "structured_outline": updated_outline_dict,  # Updated with layout instructions
            "layout_instructions": layout_instructions_text  # Keep for compatibility
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
    
    # Get plan context for slide generation requirements
    plan_context = ""
    plan_data = state.get("plan", {})
    if plan_data and "tasks" in plan_data:
        for task in plan_data["tasks"]:
            if task.get("id") == "slide_agent":
                plan_context = task.get('description', '')
                logger.info(f"Found plan context for slide_agent: {plan_context}")
                break
    
    # Pass user's original request to let slide generation tools analyze design requirements
    original_request = state.get("original_user_request", "")
    user_design_context = ""
    if original_request:
        user_design_context = f"User's original request: '{original_request}'"
    
    logger.info(f"Slide agent user design context: {user_design_context}")

    # Get structured outline or fallback to text outline
    structured_outline = state.get("structured_outline")
    outline_content = state.get("outline_content", "")
    
    slide_data_list = []
    presentation_title = "Generated Presentation"
    
    if structured_outline and 'slides' in structured_outline:
        logger.info("Slide agent: Using structured outline data")
        # Use structured outline data
        presentation_title = structured_outline.get('presentation_title', 'Generated Presentation')
        slides_data = structured_outline['slides']
        
        for slide_data in slides_data:
            slide_info = {
                'slide_number': slide_data.get('slide_number', 1),
                'title': slide_data.get('title', 'Untitled'),
                'content': slide_data.get('content', ''),
                'layout_instructions': slide_data.get('layout_instructions', '')
            }
            slide_data_list.append(slide_info)
        
        logger.info(f"Slide agent: Using structured outline with {len(slide_data_list)} slides")
        
        # Create text representation for tools that still need it
        actual_outline_str = f"# {presentation_title}\n\n"
        for slide in slide_data_list:
            actual_outline_str += f"## Slide {slide['slide_number']}: {slide['title']}\n"
            actual_outline_str += f"{slide['content']}\n"
            if slide['layout_instructions']:
                actual_outline_str += f"Layout: {slide['layout_instructions']}\n"
            actual_outline_str += "\n"
            
    else:
        logger.warning("Slide agent: No structured outline found, using text outline fallback")
        # Fallback to text-based outline
        actual_outline_str = outline_content
    
    if not actual_outline_str.strip():
        # Search in messages if state doesn't have outline_content
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
    
    # Parse text outline to create slide data
    lines = actual_outline_str.split('\n')
    current_slide_num = 1
    current_title = ""
    current_content = []
    
    for line in lines:
        line = line.strip()
        if line and ('slide' in line.lower() or line.startswith('#')):
            # Save previous slide if exists
            if current_title:
                slide_data_list.append({
                    'slide_number': current_slide_num - 1,
                    'title': current_title,
                    'content': '\n'.join(current_content),
                    'layout_instructions': ''
                })
            
            # Start new slide
            current_title = line.replace('#', '').replace('Slide', '').replace(str(current_slide_num), '').replace(':', '').strip()
            current_content = []
            current_slide_num += 1
        elif line:
            current_content.append(line)
    
    # Add last slide
    if current_title:
        slide_data_list.append({
            'slide_number': current_slide_num - 1,
            'title': current_title,
            'content': '\n'.join(current_content),
            'layout_instructions': ''
        })
    
    # If no slides found, create basic ones
    if not slide_data_list:
        slide_data_list = [
            {'slide_number': 1, 'title': 'Introduction', 'content': 'Introduction content', 'layout_instructions': ''},
            {'slide_number': 2, 'title': 'Main Content', 'content': 'Main presentation content', 'layout_instructions': ''},
            {'slide_number': 3, 'title': 'Conclusion', 'content': 'Conclusion and summary', 'layout_instructions': ''}
        ]
    
    logger.info(f"Slide agent: Parsed text outline into {len(slide_data_list)} slides")

    images = state.get("images", [])
    found_info = state.get("found_information", [])
    layout_instructions = state.get("layout_instructions", "")
    
    # Simplified: No need for complex image extraction since we pass full outline to AI
    logger.info(f"Slide agent: Found {len(images)} images in state")
    logger.info(f"Slide agent: Found {len(found_info)} research items in state")
    logger.info(f"Slide agent: Will pass full outline content to AI for image suggestions")
    
    instruction_file_path = os.path.join(os.getcwd(), "rules", "instruction.txt")
    slide_gen_instructions = ""
    try:
        with open(instruction_file_path, "r") as f:
            slide_gen_instructions = f.read()
    except FileNotFoundError:
        logger.warning(f"Slide generation instruction file not found at {instruction_file_path}. Using default.")
        slide_gen_instructions = "Create informative and visually appealing slides."
        
    # Generate slides directly using structured data
    logger.info(f"Slide agent: Generating {len(slide_data_list)} slides using structured data")
    generated_slides_info = []
    
    # Generate slides one by one using structured data
    for slide_info in slide_data_list:
        try:
            # Add delay between slides to avoid rate limiting
            if slide_info['slide_number'] > 1:  # Don't delay the first slide
                import time
                time.sleep(3)  # 3 second delay between slides
            
            # Prepare slide generation data using structured information
            slide_title = slide_info['title']
            slide_content = slide_info['content']
            slide_layout = slide_info['layout_instructions']
            slide_number = slide_info['slide_number']
            
            # Combine content and layout for comprehensive instructions
            combined_instructions = f"Create slide {slide_number}: {slide_title}\n\nContent: {slide_content}"
            if slide_layout:
                combined_instructions += f"\n\nLayout Instructions: {slide_layout}"
            if user_design_context:
                combined_instructions += f"\n\n{user_design_context}"
            if found_info:
                combined_instructions += f"\n\nUse relevant research: {str(found_info[:2])}"
            
            # Include the full outline content so AI can see suggested images and context
            if actual_outline_str and len(actual_outline_str.strip()) > 50:
                combined_instructions += f"\n\nFull presentation outline with image suggestions:\n{actual_outline_str}"
            
            # Prepare image URLs for this slide (keep existing logic for backward compatibility)
            slide_images = []
            if images:
                # Take first 2-3 most relevant images for the slide
                for img in images[:3]:
                    if isinstance(img, dict):
                        img_url = img.get('url', '')
                        if img_url:
                            slide_images.append({
                                "url": img_url,
                                "title": img.get('title', ''),
                                "alt": img.get('content', img.get('title', 'Presentation image'))
                            })
            
            images_json = json.dumps(slide_images) if slide_images else "[]"
            logger.info(f"Slide {slide_number}: Using full outline content with image suggestions")
            
            slide_data = {
                "slide_number": slide_number,
                "instructions": combined_instructions,
                "images_urls": images_json,
                "style": slide_layout if slide_layout else "modern, professional presentation style with clean layout",
                "content": f"{slide_title}: {slide_content}"
            }
            
            logger.info(f"Generating slide {slide_number}: {slide_title}")
            logger.info(f"=== Calling generate_slide tool ===")
            logger.info(f"Tool parameters:")
            logger.info(f"  - slide_number: {slide_number}")
            logger.info(f"  - style: {slide_data['style']}")
            logger.info(f"  - content: {slide_data['content']}")
            logger.info(f"  - images_urls: {images_json}")
            logger.info(f"  - instructions length: {len(combined_instructions)} characters")
            logger.info(f"  - instructions preview: {combined_instructions[:200]}...")
            
            # Generate the slide using the tool
            slide_result = generate_slide.invoke(slide_data)
            
            # Try to read the generated HTML file
            html_content = ""
            try:
                slide_file_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
                if os.path.exists(slide_file_path):
                    with open(slide_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                else:
                    html_content = slide_result
            except Exception as e:
                logger.error(f"Error reading slide file: {str(e)}")
                html_content = slide_result
            
            # Create slide info object
            slide_info_result = {
                "slide_number": slide_number,
                "content": html_content,
                "timestamp": datetime.now().isoformat(),
                "title": slide_title,
                "structured_data": slide_info  # Keep original structured data
            }
            
            generated_slides_info.append(slide_info_result)
            logger.info(f"✅ Successfully generated slide {slide_number}: {slide_title}")
            
        except Exception as e:
            logger.error(f"Error generating slide {slide_info['slide_number']}: {str(e)}")
            error_slide = {
                "slide_number": slide_info['slide_number'],
                "content": f"Error generating slide {slide_info['slide_number']}: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True,
                "title": slide_info['title']
            }
            generated_slides_info.append(error_slide)
    
    agent_final_response_content = f"Generated {len(generated_slides_info)} slides using structured outline data"
    logger.info(f"Slide agent node: {agent_final_response_content}")
    
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