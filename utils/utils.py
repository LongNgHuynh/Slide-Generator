from typing import List
from langchain_core.messages import AnyMessage, AIMessage, HumanMessage
from datetime import datetime

# Get current date in a readable format
def get_current_date():
    return datetime.now().strftime("%B %d, %Y")


def get_research_topic(messages: List[AnyMessage], original_request: str = None) -> str:
    """
    Get the research topic from the messages.
    Prioritizes the original_request parameter if provided, then searches for genuine user messages.
    """
    # First priority: use original_request if provided and valid
    if original_request and original_request.strip() and "No topic specified" not in original_request:
        return original_request.strip()
    
    # Second priority: try to get the original user request (first HumanMessage without agent name or supervisor symbols)
    for message in messages:
        if isinstance(message, HumanMessage):
            # Skip agent messages that might be HumanMessage format but from agents
            if (not hasattr(message, 'name') or not message.name) and "🎯" not in message.content:
                # Also check if content doesn't look like supervisor message
                content = message.content.strip()
                if not content.startswith("🎯 Supervisor:") and "supervisor" not in content.lower()[:50]:
                    return content
    
    # If no clean user message found, look for any HumanMessage that doesn't contain supervisor text
    for message in messages:
        if isinstance(message, HumanMessage):
            content = message.content.strip()
            if not content.startswith("🎯 Supervisor:") and "supervisor" not in content.lower()[:50]:
                return content
    
    # If still no clean message, look for any HumanMessage
        for message in messages:
            if isinstance(message, HumanMessage):
            return message.content
    
    # Fallback: if no HumanMessage found, use the last message
    if messages:
        return messages[-1].content
    
    # Final fallback
    return "No topic specified" 