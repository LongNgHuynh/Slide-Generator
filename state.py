from typing import TypedDict, Optional, Dict, List, Any, Annotated
from langgraph.graph import add_messages

class State(TypedDict):
    """Graph state containing research topic, generated content, and human feedback."""
    research_topic: str
    recommend_outline: Annotated[list[str], add_messages]
    human_feedback: Annotated[list[str], add_messages]
    