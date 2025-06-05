import json
import os
import logging
from utils.search import Searxng
from models.LLMs import GPT_4o, GPT_o3
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
        
        # Create a timestamp for this search
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Create an image search record
        image_record = {
            "timestamp": timestamp,
            "query": search_query,
            "results": filtered_results
        }
        with open(os.path.join(OUTPUT_DIR, "image_search_record.json"), "w", encoding="utf-8") as f:
            json.dump(image_record, f)
        logger.info(f"Image search completed. Found {len(filtered_results)} valid images")
        
        return image_record
    
    except Exception as e:
        logger.error(f"Error in image_search: {str(e)}")
        return {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "query": search_query, "results": []}

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
        
        
        # Create a search record with timestamp and query
        search_record = {
            "query": search_query,
            "results": filtered_results
        }
        with open(os.path.join(OUTPUT_DIR, "web_search_record.json"), "w", encoding="utf-8") as f:
            json.dump(search_record, f)
        
        logger.info(f"Web search record: {search_record}")
        return search_record
    
    except Exception as e:
        logger.error(f"Error in web_search: {str(e)}")
        return {"timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "query": search_query, "results": []}

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
        
        # Create a crawl record with timestamp and URL
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

Base HTML structure/rules to consider: 
{html_rules}

Image URL(s) to incorporate if relevant (use your judgment based on instructions):
{images_urls}

Design parameters for this slide:
Style: {style}
Content: {content}

Core content and detailed instructions for THIS SPECIFIC SLIDE (this includes all text, data, and layout guidance):
{instructions}

Your task is to generate the complete HTML for this single slide following Material Design principles.

## Technical Specifications:
- **Dimensions**: width: 1280px; min-height: 720px; position: relative; overflow: hidden;
- **Material Design Framework**: Implement Google's Material Design 3 principles
- **Multi-column Layout**: Intelligently distribute content across columns based on content length and type
- **Image Handling**: All images must be wrapped in dedicated div containers with proper Material Design styling

## Material Design Requirements:
- **Typography**: Use Material Design typography scale (Roboto font family)
- **Color System**: Implement Material Design color tokens and elevation system
- **Spacing**: Follow 8dp grid system for consistent spacing
- **Shadows & Elevation**: Use appropriate Material Design shadow levels (elevation 0-24)
- **Rounded Corners**: Apply Material Design border radius (4dp, 8dp, 12dp, 16dp)
- **Component Styling**: Use Material Design component patterns (cards, buttons, chips, etc.)

## Layout Structure:
- **Multi-column Logic**:
  - Short content (< 200 words): Single column with generous spacing
  - Medium content (200-500 words): Two-column layout
  - Long content (> 500 words): Three-column layout or structured sections
  - Mixed content: Asymmetrical columns based on content hierarchy

## Image Handling Requirements:
- **Mandatory Container**: ALL images must be wrapped in `<div class="material-image-container">`
- **Responsive Scaling with Height Control**: Images must scale responsively using Tailwind classes:
  - `w-full h-auto` for full-width responsive images
  - `max-h-96` or `max-h-80` to prevent excessive height (adjust based on content)
  - `object-cover` or `object-contain` for proper aspect ratio handling
  - `max-w-full` to prevent overflow
- **Height Constraints by Content Type**:
  - Small images: `max-h-48` (192px)
  - Medium images: `max-h-64` (256px) 
  - Large images: `max-h-80` (320px)
  - Hero-style images: `max-h-96` (384px)
- **Material Design Styling**: Apply Material Design elevation and corners:
  - `shadow-md rounded-lg` for standard elevation
  - `shadow-lg rounded-xl` for emphasized images
- **Responsive Breakpoints**: Use Tailwind responsive prefixes for different screen sizes:
  - `sm:max-h-32 md:max-h-48 lg:max-h-64` for adaptive height sizing
  - `sm:w-1/2 md:w-1/3 lg:w-1/4` for adaptive width sizing
- **Example Structure**:
```html
<div class="material-image-container shadow-md rounded-lg overflow-hidden max-w-md mx-auto">
    <img src="image-url" alt="description" class="w-full h-auto max-h-64 object-cover">
</div>
Content Organization:

Header Section: Material Design app bar styling with title and optional subtitle
Main Content: Organized in Material Design cards or sections
Image Containers:

Wrap ALL images in <div class="material-image-container"> with Material Design styling
Apply appropriate elevation and rounded corners using Tailwind classes
Include proper aspect ratio containers for responsive scaling
Images must be responsive and scale properly across all screen sizes
Use Tailwind responsive image classes: w-full h-auto object-cover or object-contain
No hero background images - images should be content elements only


Footer/Action Area: Material Design button styling if actions are needed

Styling Requirements:

CSS Framework: Use Tailwind CSS exclusively for all styling (no custom CSS)
Typography:

Headline: text-4xl md:text-5xl lg:text-6xl font-normal
Subheading: text-xl md:text-2xl font-medium
Body: text-base md:text-lg leading-relaxed
Caption: text-sm text-gray-600


Color Palette: Use Material Design color system

Primary: Blue palette (blue-500, blue-600, etc.)
Secondary: Teal palette
Surface: Gray-50, White
On-surface: Gray-900, Gray-700


Spacing: Use consistent spacing scale (space-4, space-6, space-8, space-12, space-16)
Cards: Use shadow-md, shadow-lg with rounded-lg or rounded-xl

Interactive Elements:

Hover States: Implement Material Design hover effects
Focus States: Proper focus indicators for accessibility
Transitions: Smooth transitions using duration-200 or duration-300

Background Styling:

Main Background: Solid Material Design surface color (never hero images)
Section Backgrounds: Use Material Design surface variants and elevation
Gradient Accents: Subtle Material Design inspired gradients only as accents

Responsive Design:

Breakpoints: sm:, md:, lg:, xl: for different screen sizes
Typography: Responsive text sizing using Tailwind responsive prefixes
Layout: Adaptive column layouts that stack on smaller screens
Images: All images must use Material Design responsive scaling with controlled height using max-h-* classes and proper aspect ratios

Charts and Data Visualization:

Chart.js Integration: Use Chart.js for all data visualizations and charts
Chart Styling: Apply Material Design color palette to Chart.js charts
Responsive Charts: Ensure charts are responsive and properly sized within their containers
Chart Types: Support bar, line, pie, doughnut, and other Chart.js chart types as needed
Icons: Integrate Material Design Icons (Google Fonts Icons) where appropriate
Charts: Use Chart.js exclusively for all data visualizations with Material Design color palette
Accessibility: Proper ARIA labels, semantic HTML, and keyboard navigation support
Motion: Subtle animations following Material Design motion principles

Code Structure:
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
        <!-- Use Chart.js for any data visualizations -->
    </div>
</body>
</html>
Generate only the complete HTML code for the slide, ensuring it follows all Material Design principles and multi-column layout requirements specified above.
"""
        
        response = LLM.invoke(presentation_prompt) 
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