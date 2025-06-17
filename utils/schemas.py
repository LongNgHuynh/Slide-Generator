from pydantic import BaseModel, Field
from typing import List

class SearchQueryList(BaseModel):
    query: List[str] = Field(
        description="A list of search queries to be used for web research."
    )
    rationale: str = Field(
        description="A brief explanation of why these queries are relevant to the research topic."
    )
    
class Task(BaseModel):
    id: str = Field(description="Unique identifier for the task.")
    description: str = Field(description="A concise description of what this task aims to achieve.")


class Plan(BaseModel):
    tasks: List[Task] = Field(description="A list of tasks to be executed.")