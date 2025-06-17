import json
import os
import logging
from utils.search import Searxng
from models.LLMs import GPT_4o, GPT_o3, Gemini, Claude_3_7_Sonnet
from pydantic import BaseModel
import requests
from bs4 import BeautifulSoup
import datetime
from typing import Optional, Annotated
# from langgraph.prebuilt import InjectedState
from langchain.tools import tool
# from typing import List, Dict

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM_4o = GPT_4o()
LLM_o3 = GPT_o3()
LLM_claude = Claude_3_7_Sonnet()
LLM = Gemini()

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
    Search for images based on a query and download them to a timestamped folder.
    
    Args:
        search_query: The query string to search for images
        
    Returns:
        Dictionary with search results including image URLs and local paths
    """
    logger.info(f"Starting image search for query: {search_query}")
    try:
        # Create timestamp for folder name
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        images_root = os.path.join(os.getcwd(), "images")
        download_folder = os.path.join(images_root, f"folder_{timestamp}")
        
        # Create directories if they don't exist
        os.makedirs(download_folder, exist_ok=True)
        
        # Create list.txt file
        list_file_path = os.path.join(download_folder, "image_lists.txt")
        
        searcher: Searxng = Searxng()
        results = json.loads(
            searcher.image_search(search_query, max_results=10)
        )["results"]
                
        # Extract only the required fields and download images
        filtered_results = []
        downloaded_images = []
        timeout = 5
        
        for idx, item in enumerate(results, 1):
            url = item.get("img_src", "")
            # Ensure URL has proper scheme
            if url.startswith("//"):
                url = "https:" + url
            elif not url.startswith(("http://", "https://")):
                url = "https://" + url
                
            try:
                # Download the image directly
                response = requests.get(url, timeout=timeout, stream=True)
                if response.status_code == 200:
                    # Get file extension from content type or URL
                    content_type = response.headers.get('Content-Type', '')
                    ext = content_type.split('/')[-1] if content_type else url.split('.')[-1]
                    if not ext or len(ext) > 4:  # If no valid extension found, default to jpg
                        ext = 'jpg'
                    
                    # Create image filename
                    image_filename = f"image_{idx}.{ext}"
                    image_path = os.path.join(download_folder, image_filename)
                    
                    # Download and save the image
                    with open(image_path, 'wb') as f:
                        for chunk in response.iter_content(chunk_size=8192):
                            if chunk:
                                f.write(chunk)
                    
                    # Add to downloaded images list
                    downloaded_images.append({
                        "filename": image_filename,
                        "url": url,
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "resolution": item.get("resolution", "")
                    })
                    
                    # Create relative path from generated_slides directory to the image
                    rel_path = os.path.join(GENERATED_SLIDES_DIR, image_filename)
                    
                    filtered_item = {
                        "title": item.get("title", ""),
                        "content": item.get("content", ""),
                        "img_src": url,
                        "relative_path": rel_path,
                        "resolution": item.get("resolution", "")
                    }
                    filtered_results.append(filtered_item)
                    
            except (requests.RequestException, requests.ConnectionError, 
                    requests.Timeout, requests.TooManyRedirects) as e:
                logger.error(f"Error checking/downloading URL {url}: {str(e)}")
                continue
        
        # Save list of successfully downloaded images
        with open(list_file_path, "w", encoding="utf-8") as f:
            for img in downloaded_images:
                f.write(f"Filename: {img['filename']}\n")
                f.write(f"URL: {img['url']}\n")
                f.write(f"Title: {img['title']}\n")
                f.write(f"Content: {img['content']}\n")
                f.write(f"Resolution: {img['resolution']}\n")
                f.write("-" * 50 + "\n")
        
        # Create an image search record
        image_record = {
            "query": search_query,
            "timestamp": timestamp,
            "download_folder": download_folder,
            "results": filtered_results
        }
        
        with open(os.path.join(OUTPUT_DIR, "image_search_record.json"), "w", encoding="utf-8") as f:
            json.dump(image_record, f)
            
        logger.info(f"Image search completed. Downloaded {len(filtered_results)} images to {download_folder}")
        
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
def generate_slide(slide_number: int, instructions: str, images_urls: str, style: str, content: str) -> str:
    """Generate a single HTML slide and save it to the file system.
    The 'instructions' argument should contain ALL text, data, and specific guidance for this slide's content, 
    including any relevant research information or outline points.
    
    Args:
        slide_number: The slide number
        instructions: Detailed instructions for the slide content
        images_urls: JSON string or text containing image URLs and/or local paths with relative paths
        style: The style for the slide
        content: The main content for the slide
    """
    try:
        logger.info(f"generate_slide tool called for slide #{slide_number}")
        logger.debug(f"generate_slide image_urls: {images_urls}")
        logger.debug(f"generate_slide instructions (first 150 chars): {instructions[:150]}...")
        
        rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
        html_rules = ""
        try:
            with open(rules_html_path, "r", encoding="utf-8") as f:
                html_rules = f.read()
        except FileNotFoundError:
            logger.warning(f"HTML rules file not found at {rules_html_path}. Using placeholder.")
            html_rules = "<!-- HTML rules not found -->"
            
        presentation_prompt = f"""
You are a professional presentation designer specializing in Material Design principles.

This is example of a slide: {html_rules}
You should ingore all previous instructions and examples.

Available Images to incorporate if relevant (use your judgment based on instructions):
{images_urls}

IMPORTANT IMAGE USAGE NOTES:
- Images may include both web URLs (http/https) and local relative paths.
- For local images, use the 'relative_path' when available, as these are downloaded and more reliable.
- Always include fallback handling (e.g., alt text) for images that might fail to load.
- Prefer local relative paths over web URLs for reliability.

Design parameters for this slide:
Style: {style}
Content: {content}

Core content and detailed instructions for THIS SPECIFIC SLIDE (text, layout, etc.):
{instructions}

== DESIGN & LAYOUT GUIDELINES ==
You must create a complete, visually rich HTML slide using Tailwind CSS. All content must be wrapped in a responsive `<div>` (or `<section>`) that splits content into multiple columns or rows as needed.

DO NOT use black (#000000) or white (#FFFFFF) as background. Instead:
- Use soft, rich, or vibrant **solid colors** that harmonize with the topic and style
- Suggested palettes: `#fef6e4`, `#e0f2f1`, `#ede7f6`, `#e3f2fd`, `#f3e5f5`, `#fff3e0`, `#e8f5e9`
- Ensure **high contrast** for readability (background vs text)

== SLIDE CONTAINER ==
- Use a fixed wrapper: `width: 1280px; min-height: 720px; position: relative; overflow: hidden;`
- Inside, create a structured, responsive layout with grid or flex
- Scale appropriately: 
  - If content is short, use **1 centered column**
  - If content is long, split into **2+ columns** or sections based on logical grouping

== COMPONENT STYLING ==
- Always use:
  - Google Fonts (Roboto preferred)
  - Tailwind CSS
  - Font Awesome icons (if relevant)
  - Chart.js for data visualization (if data is included)
- Add icons where visually appropriate
- Text:
  - Use large, bold font for emphasis or headings
  - Use Tailwind utility classes to highlight important sections
- Images:
  - Must be wrapped in `<section>` with consistent padding/margins
  - Properly sized: not oversized or undersized; maintain visual harmony

== SLIDE PRESENTATION ==
- Make sure the final HTML:
  - Is complete and standalone
  - Contains clean structure, responsive design
  - Uses multi-column layout for dense content
  - Never uses hero image or background image
  - Has solid-color background only
  - Looks like a high-quality, professional slide

== OUTPUT FORMAT ==
Only generate full HTML code for the slide:
html<!DOCTYPE html>
<html lang="en">
<head>
    <!-- Material Design fonts and icons -->
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <link href="https://fonts.googleapis.com/icon?family=Material+Icons" rel="stylesheet">
    <!-- Tailwind CSS -->
    <script src="https://cdn.tailwindcss.com"></script>
    <!-- Chart.js -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
</head>
<body>
    <div class="slide-container" style="width: 1280px; min-height: 720px; position: relative; overflow: hidden;">
        <!-- Material Design structured content here -->
    </div>
</body>
</html>
"""

        
        response = LLM_claude.invoke(presentation_prompt) 
        html_content = response.content if hasattr(response, 'content') else str(response)
        
        start_marker = "<!DOCTYPE html>"
        end_marker = "</html>"
        start_idx = html_content.find(start_marker)
        end_idx = html_content.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            html_content = html_content[start_idx:end_idx + len(end_marker)]
        else:
            logger.warning("Could not find HTML markers in generate_slide LLM response.")
        
        os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
        output_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        return f"Slide #{slide_number} generated. Saved to {output_path}"
    except Exception as e:
        logger.error(f"Failed to generate slide #{slide_number}: {str(e)}", exc_info=True)
        return f"Failed to generate slide #{slide_number}: {str(e)}"
    
if __name__ == "__main__":
    rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
    with open(rules_html_path, "r", encoding="utf-8") as f:
        html_rules = f.read()
    print(html_rules)