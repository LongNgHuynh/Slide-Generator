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
    
    
class Slide_Content(BaseModel):
    slide_number: int = Field(description="The number of the slide.")
    slide_title: str = Field(description="The title of the slide.")
    slide_body: str = Field(description="All other content of the slide.")
    slide_script: str = Field(description="The script of the slide for user to read.")
    
class Outline(BaseModel):
    outline: List[Slide_Content] = Field(description="A list of slide content to be used to generate the outline.")