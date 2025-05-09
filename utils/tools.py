import json
import os
import logging
from utils.search import Searxng
from models.LLMs import GPT_4o
from langchain.tools import StructuredTool
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import datetime
# from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM = GPT_4o()

# Create output directories if they don't exist
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)

# Add argument schemas
class SearchQuery(BaseModel):
    search_query: str
    
class UrlQuery(BaseModel):
    url: str
    
class PresentationQuery(BaseModel):
    slide_number: int
    title: str
    content: str


def image_search(search_query: str, searcher: Searxng = Searxng()) -> list[dict]:
    logger.info(f"Starting image search for query: {search_query}")
    try:
        logger.info("Fetching search results from Searxng")
        results = json.loads(
            searcher.image_search(search_query, max_results=10)
        )["results"]
        logger.info(f"Found {len(results)} initial results")
        
        # Extract only the required fields
        filtered_results = []
        timeout = 5
        for idx, item in enumerate(results, 1):
            url = item.get("img_src", "")
            logger.info(f"Processing result {idx}/{len(results)}: {url}")
            try:
                # Send a HEAD request first (faster than GET since it doesn't download the content)
                logger.debug(f"Sending HEAD request to {url}")
                response = requests.head(url, timeout=timeout)
                is_valid_image = False
                
                # Check if response is successful and content-type indicates an image
                if response.status_code == 200:
                    content_type = response.headers.get('Content-Type', '')
                    if content_type.startswith('image/'):
                        logger.info(f"Valid image confirmed via HEAD request: {url}")
                        is_valid_image = True
                    
                # If HEAD request doesn't work or doesn't confirm it's an image, try GET
                if not is_valid_image:
                    logger.debug(f"Sending GET request to {url}")
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
                
        # Save results to JSON file
        output_file = os.path.join(OUTPUT_DIR, "images.json")
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(filtered_results, f, indent=4, ensure_ascii=False)
        logger.info(f"Saved {len(filtered_results)} results to {output_file}")
        
        logger.info(f"Image search completed. Found {len(filtered_results)} valid images")
        return filtered_results
    
    except Exception as e:
        logger.error(f"Error in image_search: {str(e)}")
        return []
    
def web_search(search_query: str, searcher: Searxng = Searxng()) -> list[str]:
    logger.info(f"Starting web search for query: {search_query}")
    try:
        content = json.loads(
            searcher.webpage_search(search_query, max_results=10)
        )["results"]
        
        
        
        return content
    
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return []
    
def crawl_url(url: str) -> str:
    logger.info(f"Starting URL crawl for: {url}")
    try:
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
        
        return text[:6000]
    except Exception as e:
        logger.error(f"Error in crawl_url: {str(e)}")
        return "Failed to crawl the URL"
    
def generate_presentation(slide_number: int, title: str, content: str) -> str:
    """
    Generate a single HTML slide and save it to the file system.
    
    Args:
        slide_number: The slide number (integer)
        title: The slide title
        content: The content for the slide
        
    Returns:
        String with information about the generated slide
    """
    try:
        logger.info(f"Generating slide #{slide_number}: {title}")
        
        # Read HTML slide template
        try:
            with open("rules/html.txt", "r") as f:
                html_rules = f.read()
        except FileNotFoundError:
            # Fallback template if file not found
            html_rules = """
            <!DOCTYPE html>
            <html>
            <head>
                <meta charset="UTF-8">
                <title>Slide</title>
                <style>
                    body { font-family: Arial, sans-serif; margin: 0; padding: 0; }
                    .slide { width: 100%; height: 100vh; padding: 2em; box-sizing: border-box; }
                    h1 { color: #333; }
                </style>
            </head>
            <body>
                <div class="slide">
                    <h1>{title}</h1>
                    <div class="content">{content}</div>
                </div>
            </body>
            </html>
            """
            
        presentation_prompt = f"""
        Create a single HTML slide about {title}. 
        Use this content for the slide: {content}
        
        The slide should be in the following format:
        {html_rules}
        
        Return only the complete HTML code for this single slide.
        The HTML must be complete and ready to use, including all necessary styles and scripts.
        """
        
        # Get the response and extract the content
        response = LLM.invoke(presentation_prompt)
        html_content = response.content if hasattr(response, 'content') else str(response)
        
        # Save individual slide
        os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
        output_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Update the slide collection
        update_slide_collection(slide_number, title, html_content)
            
        return f"Slide #{slide_number} '{title}' generated successfully. Saved to {output_path}"
    except Exception as e:
        logger.error(f"Failed to generate slide: {str(e)}")
        return f"Failed to generate slide: {str(e)}"


def update_slide_collection(slide_number: int, title: str, html_content: str) -> None:
    """
    Update the collection of slides by adding or updating a slide.
    
    Args:
        slide_number: The slide number
        title: The slide title
        html_content: The HTML content of the slide
    """
    collection_file = os.path.join(GENERATED_SLIDES_DIR, "slides_collection.json")
    
    # Create or load the collection
    if os.path.exists(collection_file):
        try:
            with open(collection_file, "r", encoding="utf-8") as f:
                collection = json.load(f)
        except (json.JSONDecodeError, IOError):
            # Create new collection if file is corrupted
            collection = {"slides": [], "last_updated": ""}
    else:
        collection = {"slides": [], "last_updated": ""}
    
    # Check if this slide already exists
    slide_exists = False
    for i, slide in enumerate(collection["slides"]):
        if slide.get("slide_number") == slide_number:
            # Update existing slide
            collection["slides"][i] = {
                "slide_number": slide_number,
                "title": title,
                "html_content": html_content,
                "file_path": f"slide_{slide_number:03d}.html"
            }
            slide_exists = True
            break
    
    # Add new slide if it doesn't exist
    if not slide_exists:
        collection["slides"].append({
            "slide_number": slide_number,
            "title": title,
            "html_content": html_content,
            "file_path": f"slide_{slide_number:03d}.html"
        })
    
    # Sort slides by number
    collection["slides"].sort(key=lambda x: x.get("slide_number", 0))
    
    # Update timestamp
    collection["last_updated"] = datetime.datetime.now().isoformat()
    
    # Save the updated collection
    with open(collection_file, "w", encoding="utf-8") as f:
        json.dump(collection, f, indent=2, ensure_ascii=False)
    
    # Generate a combined HTML file with all slides
    generate_combined_presentation(collection["slides"])


def generate_combined_presentation(slides: list) -> None:
    """
    Generate a combined HTML presentation with all slides.
    
    Args:
        slides: List of slide objects
    """
    if not slides:
        return
        
    combined_html = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Complete Presentation</title>
    <style>
        body { 
            font-family: Arial, sans-serif;
            margin: 0;
            padding: 0;
            background-color: #f5f5f5;
        }
        .slide-container {
            width: 100%;
            max-width: 900px;
            margin: 20px auto;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 4px 12px rgba(0,0,0,0.1);
            page-break-after: always;
        }
        .slide-header {
            background: #4285f4;
            color: white;
            padding: 10px 20px;
            display: flex;
            justify-content: space-between;
            align-items: center;
        }
        .slide-number {
            font-size: 14px;
            background: rgba(255,255,255,0.2);
            padding: 3px 8px;
            border-radius: 12px;
        }
        .slide-content {
            padding: 20px;
            min-height: 400px;
        }
        h1 { margin-top: 0; }
        @media print {
            .slide-container {
                page-break-after: always;
                box-shadow: none;
                margin: 0;
                width: 100%;
            }
        }
    </style>
</head>
<body>
    <h1 style="text-align: center; padding: 20px;">Complete Presentation</h1>
"""

    # Add each slide
    for slide in slides:
        slide_number = slide.get("slide_number", 0)
        title = slide.get("title", "Untitled")
        
        combined_html += f"""
    <div class="slide-container">
        <div class="slide-header">
            <h2>{title}</h2>
            <span class="slide-number">Slide {slide_number}</span>
        </div>
        <div class="slide-content">
            {extract_slide_content(slide.get("html_content", ""))}
        </div>
    </div>
"""

    # Close HTML tags
    combined_html += """
    <script>
        // Print button functionality could be added here
    </script>
</body>
</html>
"""

    # Save combined presentation
    combined_file_path = os.path.join(GENERATED_SLIDES_DIR, "complete_presentation.html")
    with open(combined_file_path, "w", encoding="utf-8") as f:
        f.write(combined_html)
    
    logger.info(f"Generated combined presentation with {len(slides)} slides at {combined_file_path}")


def extract_slide_content(html_content: str) -> str:
    """
    Extract the main content from an HTML slide, removing unnecessary elements.
    
    Args:
        html_content: Full HTML content of a slide
        
    Returns:
        Cleaned HTML content
    """
    try:
        # Parse HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find the main content
        # This will depend on the structure of your slides, adjust as needed
        content_div = soup.find('div', class_='content') or soup.find('div', class_='slide-content')
        
        if content_div:
            return str(content_div)
        
        # Fallback to body content if no specific content div is found
        body = soup.find('body')
        if body:
            return ''.join(str(content) for content in body.contents)
            
        # Last resort
        return html_content
        
    except Exception as e:
        logger.error(f"Error extracting slide content: {e}")
        return html_content

# Create structured tools with args_schema
image_search_tool = StructuredTool(
    name="image_search",
    description="Search for images based on a query. Returns a list of image URLs.",
    func=image_search,
    args_schema=SearchQuery
)

web_search_tool = StructuredTool(
    name="web_search",
    description="Search for web content based on a query. Returns a list of search results.",
    func=web_search,
    args_schema=SearchQuery
)

crawl_tool = StructuredTool(
    name="crawl_url",
    description="Crawl a webpage URL to extract its text content. Use this when you need to get detailed information from a specific webpage.",
    func=crawl_url,
    args_schema=UrlQuery
)

presentation_tool = StructuredTool(
    name="generate_presentation",
    description="Generate an HTML presentation from the given title and content for only 1 slide. Returns the file path of the generated presentation.",
    func=generate_presentation,
    args_schema=PresentationQuery
)