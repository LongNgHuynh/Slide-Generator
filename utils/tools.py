import json
import os
import logging
from utils.search import Searxng
from models.LLMs import GPT_4o, GPT_o3
from langchain.tools import StructuredTool
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import datetime
from typing import Optional, Annotated
from langgraph.prebuilt import InjectedState
from langchain.tools import tool
# from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM = GPT_4o()
gpt_o3 = GPT_o3()

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)

# Add argument schemas
class SearchQuery(BaseModel):
    search_query: str
    
class UrlQuery(BaseModel):
    url: str
    
class PresentationOutlineQuery(BaseModel):
    topic: str
    instructions: str

class PresentationQuery(BaseModel):
    slide_number: int
    title: str
    content: list[str]
    image_url: Optional[str] = None
    pallette_colors: list[str]
    layout: str
    style: str

@tool
def image_search(search_query: str) -> dict:
    """
    Search for images based on a query.
    
    Args:
        search_query: The query string to search for images
        searcher: Searxng instance to use for searching
        
    Returns:
        Dictionary with search results including image URLs
    """
    logger.info(f"Starting image search for query: {search_query}")
    try:
        searcher: Searxng = Searxng()
        results = json.loads(
            searcher.image_search(search_query, max_results=10)
        )["results"]
                
        # Extract only the required fields
        filtered_results = []
        timeout = 5
        for idx, item in enumerate(results, 1):
            url = item.get("img_src", "")
            # Ensure URL has proper scheme
            if url.startswith("//"):
                url = "https:" + url
            elif not url.startswith(("http://", "https://")):
                url = "https://" + url
            try:
                # Send a HEAD request first (faster than GET since it doesn't download the content)
                logger.debug(f"Sending HEAD request to {url}")
                response = requests.head(url, timeout=timeout)
                is_valid_image = False
                
                # Check if response is successful and content-type indicates an image
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if content_type.startswith('image/'):
                        is_valid_image = True
                    
                # If HEAD request doesn't work or doesn't confirm it's an image, try GET
                if not is_valid_image:
                    response = requests.get(url, timeout=timeout, stream=True)
                    if response.status_code == 200:
                        content_type = response.headers.get('Content-Type', '')
                        if content_type.startswith('image/'):
                            is_valid_image = True
                
                # Append only if image is valid
                if is_valid_image:
                    filtered_item = {
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "img_src": url,
                        "resolution": item.get("resolution", "")
                    }
                    filtered_results.append(filtered_item)
                    
            except (requests.RequestException, requests.ConnectionError, 
                    requests.Timeout, requests.TooManyRedirects) as e:
                logger.error(f"Error checking URL {url}: {str(e)}")
                continue
        
        # Create an image search record
        image_record = {
            "query": search_query,
            "results": filtered_results
        }
        with open(os.path.join(OUTPUT_DIR, "image_search_record.json"), "w", encoding="utf-8") as f:
            json.dump(image_record, f)
        logger.info(f"Image search completed. Found {len(filtered_results)} valid images")
        
        return image_record
    
    except Exception as e:
        logger.error(f"Error in image_search: {str(e)}")
        return {"query": search_query, "results": []}

@tool
def web_search(search_query: str, ) -> dict:
    """
    Search for web content based on a query.
    
    Args:
        search_query: The query string to search for web content
        searcher: Searxng instance to use for searching
        
    Returns:
        Dictionary with search results including URLs and content
    """
    logger.info(f"Starting web search for query: {search_query}")
    try:
        searcher: Searxng = Searxng()
        full_results = json.loads(
            searcher.webpage_search(search_query, max_results=10)
        )["results"]
        
        # Extract only the url, title, content, and score fields
        filtered_results = []
        for result in full_results:
            filtered_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", ""),
                "score": result.get("score", 0)
            }
            filtered_results.append(filtered_result)
        
        
        search_record = {
            "query": search_query,
            "results": filtered_results
        }
        with open(os.path.join(OUTPUT_DIR, "web_search_record.json"), "w", encoding="utf-8") as f:
            json.dump(search_record, f)
        
        logger.info(f"Web search found {len(filtered_results)} results")
        return search_record
    
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return {"query": search_query, "results": []}

@tool  
def crawl_url(url: str) -> dict:
    """
    Crawl a webpage URL to extract its text content.
    
    Args:
        url: The URL of the webpage to crawl
        
    Returns:
        Dictionary with the extracted content from the webpage
    """
    logger.info(f"Starting URL crawl for: {url}")
    try:
        # Remove any extra quotes from the URL
        url = url.strip('"\'')
        
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Remove script and style elements
        for script in soup(["script", "style"]):
            script.decompose()
            
        # Get text content
        text = soup.get_text()
        
        # Clean up text
        lines = (line.strip() for line in text.splitlines())
        chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
        text = ' '.join(chunk for chunk in chunks if chunk)
        
        crawl_record = {
            "url": url,
            "content": text[:6000]  # Store the first 6000 characters
        }
        with open(os.path.join(OUTPUT_DIR, "crawl_record.json"), "w", encoding="utf-8") as f:
            json.dump(crawl_record, f)
        
        return crawl_record
    except Exception as e:
        logger.error(f"Error in crawl_url: {str(e)}")
        return {"url": url, "content": "Failed to crawl the URL"}
            
@tool
def generate_slide(slide_number: int, instructions: str) -> str:
    """
    Generate a single HTML slide and save it to the file system.
    
    Args:
        instructions: The instructions for the slide
    Returns:
        String with information about the generated slide
    """
    try:
        logger.info(f"Instructions: {instructions}")
        
        with open("rules/html.txt", "r") as f:
            html_rules = f.read()
            
        presentation_prompt = f"""
        You are a professional presentation designer.
        
        This is the html rules for the slides: {html_rules}
        
        Ignore the previous instructions
        
        The slide should follow these instructions: {instructions}
        
        Create slide presentation with tailwind css for artiristic like slidego template, make it look minimalistic and concise, but in details.
        IMPORTANT:
        The slide should be generate with: width: 1280px; min-height: 720px; position: relative; overflow: hidden;
        All elements should be wrap in a div 
        Using Google Font, Tailwind CSS, Font Awesome icons to add-on
        Using chart.js if needed
        Image if exists, should be in a appropriate scale, that's not too big or too small, and not too wide or too narrow.
        If the content is short, make it 1 column, if the content is long, make it multiple columns.
        Make html slide overlay: hidden.
        Make the slide responsive, and big font size.
        Make the slide look like a professional presentation.
        As many generate token as possible.
        """
        
        # Stream the response and yield each chunk
        html_content = ""
        for chunk in LLM.stream(presentation_prompt):
            if hasattr(chunk, 'content'):
                html_content += chunk.content
                yield chunk.content
            else:
                content = str(chunk)
                html_content += content
                yield content
        
        # Extract only the HTML content between <!DOCTYPE html> and </html>
        start_marker = "<!DOCTYPE html>"
        end_marker = "</html>"
        start_idx = html_content.find(start_marker)
        end_idx = html_content.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            html_content = html_content[start_idx:end_idx + len(end_marker)]
        else:
            logger.warning("Could not find proper HTML markers in the response")
        
        # Save individual slide
        os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
        output_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
            
        yield f"Slide #{slide_number} generated successfully. Saved to {output_path}"
    except Exception as e:
        error_msg = f"Failed to generate slide: {str(e)}"
        logger.error(error_msg)
        yield error_msg



# # Create structured tools with args_schema
# image_search_tool = StructuredTool(
#     name="image_search",
#     description="Search for images based on a query. Returns a list of image URLs.",
#     func=image_search,
#     args_schema=SearchQuery
# )

# web_search_tool = StructuredTool(
#     name="web_search",
#     description="Search for web content based on a query. Returns a list of search results.",
#     func=web_search,
#     args_schema=SearchQuery
# )

# crawl_tool = StructuredTool(
#     name="crawl_url",
#     description="Crawl a webpage URL to extract its text content. Use this when you need to get detailed information from a specific webpage.",
#     func=crawl_url,
#     args_schema=UrlQuery
# )

# generate_presentation_outline_tool = StructuredTool(
#     name="generate_presentation_outline",
#     description="Generate a presentation outline. Requires a topic and instructions. Returns the outline of the presentation.",
#     func=generate_presentation_outline,
#     args_schema=PresentationOutlineQuery
# )
    
# presentation_tool = StructuredTool(
#     name="generate_presentation",
#     description="Generate an HTML presentation slide. Requires five parameters: slide_number (int), title (string), content (string), layout (string), and style (string). Returns the file path of the generated slide.",
#     func=generate_slide,
#     args_schema=PresentationQuery
# )