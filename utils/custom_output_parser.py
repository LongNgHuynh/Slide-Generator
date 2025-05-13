from langchain.agents import AgentOutputParser
from typing import Union, Dict, Any
from langchain.schema import AgentAction, AgentFinish
import re
import json
import logging

logger = logging.getLogger(__name__)

class CustomOutputParser(AgentOutputParser):
    
    def parse(self, llm_output: str) -> Union[AgentAction, AgentFinish]:
        # Check if agent should finish
        if "Final Answer:" in llm_output:
            return AgentFinish(
                # Return values is generally always a dictionary with a single `output` key
                # It is not recommended to try anything else at the moment :)
                return_values={"output": llm_output.split("Final Answer:")[-1].strip()},
                log=llm_output,
            )
        
        # Check for "Action: N/A" which means the model wants to provide an answer directly
        if "Action: N/A" in llm_output or "Action: n/a" in llm_output:
            # Extract content after "Action: N/A" or from the thought
            thought_match = re.search(r"Thought:(.*?)(?:Action:|$)", llm_output, re.DOTALL)
            content = thought_match.group(1).strip() if thought_match else llm_output
            return AgentFinish(
                return_values={"output": content},
                log=llm_output,
            )
            
        # Parse out the action and action input
        regex = r"Action\s*\d*\s*:(.*?)\nAction\s*\d*\s*Input\s*\d*\s*:[\s]*(.*)"
        match = re.search(regex, llm_output, re.DOTALL)
        if not match:
            raise ValueError(f"Could not parse LLM output: `{llm_output}`")
        action = match.group(1).strip()
        action_input = match.group(2).strip()
        
        # Special handling for generate_presentation_outline tool
        if action == "generate_presentation_outline":
            try:
                # Try to parse as JSON if it looks like JSON
                if action_input.strip().startswith('{') and action_input.strip().endswith('}'):
                    try:
                        json_input = json.loads(action_input)
                        return AgentAction(
                            tool=action,
                            tool_input={
                                "topic": json_input.get("topic", ""),
                                "instructions": json_input.get("instructions", "Create a detailed presentation outline")
                            },
                            log=llm_output
                        )
                    except json.JSONDecodeError:
                        pass
                
                # If not JSON, try to extract topic from text
                topic_match = re.search(r'topic=[\'"]?([^\'"]+)[\'"]?', action_input)
                instructions_match = re.search(r'instructions=[\'"]?([^\'"]+)[\'"]?', action_input)
                
                topic = topic_match.group(1) if topic_match else action_input
                instructions = instructions_match.group(1) if instructions_match else "Create a detailed presentation outline"
                
                return AgentAction(
                    tool=action,
                    tool_input={
                        "topic": topic,
                        "instructions": instructions
                    },
                    log=llm_output
                )
            except Exception as e:
                logger.error(f"Error parsing generate_presentation_outline input: {e}")
                # Fall back to normal parsing
        
        # Special handling for generate_presentation tool
        if action == "generate_presentation":
            try:
                # Sometimes the action input might be wrapped in extra quotes or have trailing newlines
                cleaned_input = action_input.strip()
                if cleaned_input.startswith('"') and cleaned_input.endswith('"'):
                    cleaned_input = cleaned_input[1:-1]  # Remove surrounding quotes
                
                # Try to parse as JSON
                try:
                    json_input = json.loads(cleaned_input)
                    # Make sure slide_number is an integer
                    if "slide_number" in json_input:
                        if isinstance(json_input["slide_number"], str):
                            json_input["slide_number"] = int(json_input["slide_number"])
                        # Return proper dict for the presentation tool
                        return AgentAction(
                            tool=action,
                            tool_input={
                                "slide_number": json_input.get("slide_number"),
                                "title": json_input.get("title", ""),
                                "content": [json_input.get("content", "")] if isinstance(json_input.get("content", ""), str) else json_input.get("content", []),
                                "layout": json_input.get("layout", "default"),
                                "style": json_input.get("style", "bg-white text-black")
                            },
                            log=llm_output
                        )
                except (json.JSONDecodeError, ValueError) as e:
                    logger.error(f"Error parsing JSON for presentation tool: {e}")
                    # Fall back to regex parsing if JSON parsing fails
                    slide_number_match = re.search(r'"slide_number":\s*(\d+)', cleaned_input)
                    title_match = re.search(r'"title":\s*"([^"]+)"', cleaned_input)
                    content_match = re.search(r'"content":\s*"([^"]+)"', cleaned_input)
                    
                    if slide_number_match and title_match and content_match:
                        return AgentAction(
                            tool=action,
                            tool_input={
                                "slide_number": int(slide_number_match.group(1)),
                                "title": title_match.group(1),
                                "content": [content_match.group(1)],
                                "layout": "default",
                                "style": "bg-white text-black"
                            },
                            log=llm_output
                        )
            except Exception as e:
                logger.error(f"Error in presentation tool parsing: {e}")
                # Continue with normal parsing as fallback
                
        # Return the action and action input for other tools
        return AgentAction(tool=action, tool_input=action_input, log=llm_output)