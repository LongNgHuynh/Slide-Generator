import json
import os
import logging
from utils.search import Searxng
from models.LLMs import GPT_4o, GPT_o3, Gemini, Claude_3_7_Sonnet, Gemini_2_5_Flash
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
# Import schemas for structured output
try:
    from utils.schemas import ColorPalette, CoverSlideResponse
except ImportError:
    # Fallback if import fails
    logger.warning("Could not import ColorPalette and CoverSlideResponse schemas")
    pass



OUTPUT_DIR = os.path.join(os.getcwd(), "semi_output")
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")
LLM_4o = GPT_4o()
LLM_o3 = GPT_o3()
LLM_claude = Claude_3_7_Sonnet()
LLM = Gemini()
LLM_2_5_Flash = Gemini_2_5_Flash()

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
    logger.info(f"=== image_search tool called ===")
    logger.info(f"Search Query: {search_query}")
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
                        # "relative_path": rel_path,
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
    logger.info(f"=== web_search tool called ===")
    logger.info(f"Search Query: {search_query}")
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
    Crawl a specific URL to extract content and metadata.
    
    Args:
        url: The URL to crawl
        
    Returns:
        Dictionary with crawled content and metadata
    """
    logger.info(f"=== crawl_url tool called ===")
    logger.info(f"URL to crawl: {url}")
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
            
def extract_color_palette_from_html(html_content: str) -> dict:
    """Extract color palette from generated HTML using AI analysis"""
    try:
        # Use AI to extract colors from the HTML
        extract_prompt = f"""Analyze this HTML and extract the main color palette being used.

HTML CONTENT:
{html_content[:2000]}...

Extract the 5 main colors used in this slide:
1. Primary color (main headings, important elements)  
2. Secondary color (subheadings, secondary elements)
3. Accent color (highlights, interactive elements)
4. Background color (main slide background)
5. Text color (main text content)

Look for colors in:
- CSS background-color properties
- CSS color properties  
- Inline styles
- Tailwind classes

Respond with JSON format:
{{
    "primary": "#colorcode",
    "secondary": "#colorcode",
    "accent": "#colorcode", 
    "background": "#colorcode",
    "text": "#colorcode"
}}

Only return the JSON, no explanations."""

        response = LLM_2_5_Flash.invoke(extract_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            palette = json.loads(json_match.group())
            logger.info(f"Successfully extracted color palette from HTML")
            return palette
        
        return None
        
    except Exception as e:
        logger.warning(f"Error extracting color palette from HTML: {e}")
        return None

def parse_color_requirements_from_instructions(instructions: str) -> dict:
    """Parse design and color requirements from slide instructions using AI analysis"""
    
    # Only analyze if the instructions contain a user request
    if "User's original request:" not in instructions:
        return None  # No user request to analyze
    
    try:
        # Use AI to analyze the user's request for design requirements
        analysis_prompt = f"""Analyze this user request for specific design and color requirements:

{instructions}

If the user has specified specific colors, backgrounds, or design styles in their request, provide a color palette.

Examples of specific requirements:
- "galaxy background" or "cosmic" → Use space/galaxy colors
- "wine color" or "wine background" → Use wine/burgundy colors  
- "blue theme" → Use blue colors
- "minimalist" → Use clean, simple colors
- "modern" → Use contemporary color schemes
- etc.

If you detect specific design requirements, respond with JSON in this format:
{{
    "detected": true,
    "style": "galaxy/wine/blue/minimalist/etc",
    "palette": {{
        "primary": "#colorcode",
        "secondary": "#colorcode", 
        "accent": "#colorcode",
        "background": "#colorcode",
        "text": "#colorcode"
    }}
}}

If no specific design requirements are mentioned, respond with:
{{"detected": false}}

Only detect EXPLICIT mentions of colors/designs, not general topic-based inference."""

        # Use a lightweight LLM for quick analysis
        response = LLM_2_5_Flash.invoke(analysis_prompt)
        response_text = response.content if hasattr(response, 'content') else str(response)
        
        # Parse JSON response
        import json
        import re
        json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
        if json_match:
            analysis = json.loads(json_match.group())
            
            if analysis.get("detected") and "palette" in analysis:
                logger.info(f"AI detected design requirement: {analysis.get('style', 'unknown')}")
                return analysis["palette"]
        
        return None
        
    except Exception as e:
        logger.warning(f"Error in AI color analysis: {e}")
        return None
            
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
        # Enhanced logging for all parameters
        logger.info(f"=== generate_slide tool called ===")
        logger.info(f"Slide Number: {slide_number}")
        logger.info(f"Style: {style}")
        logger.info(f"Content: {content}")
        logger.info(f"Images URLs (raw): {images_urls}")
        logger.info(f"Instructions length: {len(instructions)} characters")
        logger.info(f"Instructions preview: {instructions[:200]}...")
        
        # Log full instructions if not too long
        if len(instructions) < 1000:
            logger.info(f"Full instructions: {instructions}")
        else:
            logger.info(f"Instructions (first 500 chars): {instructions[:500]}...")
            logger.info(f"Instructions (last 200 chars): ...{instructions[-200:]}")
        
        # Parse and log image data
        slide_images = []
        try:
            if images_urls and images_urls != "[]":
                parsed_images = json.loads(images_urls)
                if isinstance(parsed_images, list):
                    slide_images = parsed_images
                    logger.info(f"Successfully parsed {len(slide_images)} images:")
                    for i, img in enumerate(slide_images):
                        logger.info(f"  Image {i+1}: URL={img.get('url', 'N/A')[:50]}..., Title={img.get('title', 'N/A')[:30]}...")
                else:
                    logger.warning(f"Parsed images is not a list: {type(parsed_images)}")
        except json.JSONDecodeError as e:
            logger.warning(f"Could not parse images_urls JSON: {e}")
            logger.warning(f"Raw images_urls: {images_urls}")
        
        logger.debug(f"generate_slide image_urls: {images_urls}")
        logger.debug(f"generate_slide instructions (first 150 chars): {instructions[:150]}...")
        
        # Load existing color palette first to ensure consistency
        color_palette = None
        palette_path = os.path.join(GENERATED_SLIDES_DIR, "color_palette.json")
        
        # Check if palette already exists (from previous slides)
        if os.path.exists(palette_path):
            # Implement retry logic for file loading
            max_retries = 3
            retry_count = 0
            while retry_count < max_retries:
                try:
                    # Add small delay to avoid file locking issues
                    import time
                    time.sleep(0.1 * retry_count)  # Progressive delay
                    
                    with open(palette_path, "r", encoding="utf-8") as f:
                        color_palette = json.load(f)
                    logger.info(f"Using existing color palette from cover slide with {len(color_palette)} colors")
                    break  # Success, exit retry loop
                except (FileNotFoundError, json.JSONDecodeError) as e:
                    logger.warning(f"Attempt {retry_count + 1}: Could not load color palette: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to load color palette after {max_retries} attempts, using default colors")
                        # Use fallback color palette
                        color_palette = {
                            "primary": "#2563EB",
                            "secondary": "#64748B",
                            "accent": "#F59E0B",
                            "background": "#F8FAFC",
                            "text": "#1E293B"
                        }
                except Exception as e:
                    logger.warning(f"Attempt {retry_count + 1}: Unexpected error loading color palette: {str(e)}")
                    retry_count += 1
                    if retry_count >= max_retries:
                        logger.error(f"Failed to load color palette after {max_retries} attempts, using default colors")
                        # Use fallback color palette
                        color_palette = {
                            "primary": "#2563EB", 
                            "secondary": "#64748B",
                            "accent": "#F59E0B",
                            "background": "#F8FAFC",
                            "text": "#1E293B"
                        }
        else:
            # No existing palette - check for user requirements and create initial palette
            user_specified_palette = parse_color_requirements_from_instructions(instructions)
            
            if user_specified_palette:
                color_palette = user_specified_palette
                logger.info(f"Creating initial color palette from user requirements: {color_palette}")
                
                # Save the new palette to file for consistency across all slides
                try:
                    with open(palette_path, "w", encoding="utf-8") as f:
                        json.dump(color_palette, f, indent=2)
                    logger.info(f"Saved initial color palette to {palette_path}")
                except Exception as e:
                    logger.warning(f"Could not save initial color palette: {e}")
            else:
                # Create topic-aware color palette if no user requirements found
                topic_based_palette = create_topic_based_color_palette(instructions, content)
                if topic_based_palette:
                    color_palette = topic_based_palette
                    logger.info(f"Created topic-based color palette: {color_palette}")
                    
                    # Save the new palette to file for consistency across all slides
                    try:
                        with open(palette_path, "w", encoding="utf-8") as f:
                            json.dump(color_palette, f, indent=2)
                        logger.info(f"Saved topic-based color palette to {palette_path}")
                    except Exception as e:
                        logger.warning(f"Could not save topic-based color palette: {e}")
                else:
                    logger.info("No existing color palette and no user requirements found, will use default color guidelines")
        
        rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
        html_rules = ""
        try:
            with open(rules_html_path, "r", encoding="utf-8") as f:
                html_rules = f.read()
        except FileNotFoundError:
            logger.warning(f"HTML rules file not found at {rules_html_path}. Using placeholder.")
            html_rules = "<!-- HTML rules not found -->"
        
        # Create color palette section for prompt
        color_instructions = ""
        if color_palette:
            color_instructions = f"""
MANDATORY COLOR PALETTE (MUST USE THESE EXACT COLORS FOR ALL SLIDES):
- Primary Color: {color_palette.get('primary', '#2563EB')}
- Secondary Color: {color_palette.get('secondary', '#64748B')}
- Accent Color: {color_palette.get('accent', '#F59E0B')}
- Background Color: {color_palette.get('background', '#F8FAFC')}
- Text Color: {color_palette.get('text', '#1E293B')}

CRITICAL CONSISTENCY REQUIREMENTS:
- ALL slides in this presentation MUST use this exact color palette
- Use the background color ({color_palette.get('background', '#F8FAFC')}) as the main slide background
- Use primary color for main headings and important elements
- Use secondary color for subheadings and secondary elements
- Use accent color for highlights and interactive elements
- Ensure text color provides good contrast on the background

Add this comment at the top of your HTML to document color usage:
<!-- USING PRESENTATION COLORS: PRIMARY:{color_palette.get('primary', '#2563EB')} SECONDARY:{color_palette.get('secondary', '#64748B')} ACCENT:{color_palette.get('accent', '#F59E0B')} BACKGROUND:{color_palette.get('background', '#F8FAFC')} TEXT:{color_palette.get('text', '#1E293B')} -->
"""
        else:
            color_instructions = """
COLOR GUIDELINES (CREATE INITIAL PALETTE):
- Create a cohesive color palette based on user requirements and topic
- Use soft, professional colors that harmonize with the content
- Ensure high contrast for readability
- Avoid pure black (#000000) or white (#FFFFFF) backgrounds
- Select colors appropriate to the topic and user requests
- This palette will be saved and used for ALL slides in the presentation

IMPORTANT: Choose colors that work well together and create a consistent visual theme.
"""
            
        # Create image section for prompt
        image_instructions = ""
        if slide_images:
            image_instructions = f"""
IMPORTANT: THIS SLIDE HAS {len(slide_images)} IMAGES AVAILABLE - YOU MUST INCLUDE THEM!

Available Images (MUST USE AT LEAST 1-2 OF THESE):
"""
            for i, img in enumerate(slide_images):
                img_url = img.get('url', '')
                img_title = img.get('title', f'Image {i+1}')
                img_alt = img.get('alt', img_title)
                image_instructions += f"""
- Image {i+1}: {img_url}
  Title: {img_title}
  Alt text: {img_alt}
"""
            
            image_instructions += """
CRITICAL IMAGE REQUIREMENTS:
- You MUST include at least 1-2 of these images in the slide
- Use <img> tags with proper src, alt, and styling
- Make images responsive with Tailwind classes like 'w-full h-auto max-w-md'
- Position images strategically to complement the text content
- If image URLs are long, you can truncate them in logs but use full URLs in HTML
- Add proper error handling with alt text in case images fail to load
"""
        else:
            image_instructions = """
No specific images provided for this slide. Focus on creating engaging text-based content with icons and visual elements using Tailwind CSS and Material Icons.
"""
            
        presentation_prompt = f"""
You are a professional presentation designer specializing in Material Design principles.

This is example of a slide: {html_rules}
You should ignore all previous instructions and examples.

{color_instructions}

{image_instructions}

Design parameters for this slide:
Style: {style}
Content: {content}

Core content and detailed instructions for THIS SPECIFIC SLIDE (text, layout, etc.):
{instructions}

== DESIGN & LAYOUT GUIDELINES ==
You must create a complete, visually rich HTML slide using Tailwind CSS. All content must be wrapped in a responsive `<div>` (or `<section>`) that splits content into multiple columns or rows as needed.

{"USE THE MANDATORY COLOR PALETTE SPECIFIED ABOVE." if color_palette else "Choose appropriate colors based on content and ensure good contrast."}

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
  - {"MAINTAINS COLOR CONSISTENCY with the cover slide theme" if color_palette else "Uses professional, topic-appropriate colors"}

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
        
        # If no color palette exists yet, try to extract and save one from the generated HTML
        if not color_palette and slide_number == 1:
            try:
                extracted_palette = extract_color_palette_from_html(html_content)
                if extracted_palette:
                    with open(palette_path, "w", encoding="utf-8") as f:
                        json.dump(extracted_palette, f, indent=2)
                    logger.info(f"Extracted and saved color palette from slide {slide_number}: {extracted_palette}")
            except Exception as e:
                logger.warning(f"Could not extract color palette from generated HTML: {e}")
        
        logger.info(f"Successfully generated slide #{slide_number}. Saved to {output_path}")
        return html_content  # Return the HTML content instead of status message
    except Exception as e:
        logger.error(f"Failed to generate slide #{slide_number}: {str(e)}", exc_info=True)
        return f"Failed to generate slide #{slide_number}: {str(e)}"
    
@tool
def update_slide(slide_number: int, new_content: str, original_html: str, preserve_design: bool = True) -> str:
    """
    Update an existing slide with new content while preserving the design.
    
    Args:
        slide_number: The slide number to update
        new_content: The new content to replace the existing content
        original_html: The original HTML content of the slide
        preserve_design: Whether to preserve the original design (default: True)
        
    Returns:
        Updated HTML content for the slide
    """
    logger.info(f"=== update_slide tool called ===")
    logger.info(f"Slide Number: {slide_number}")
    logger.info(f"New Content: {new_content}")
    logger.info(f"Preserve Design: {preserve_design}")
    logger.info(f"Original HTML length: {len(original_html)} characters")
    logger.info(f"Original HTML preview: {original_html[:200]}...")
    
    try:
        if preserve_design:
            # Use AI to intelligently update the slide while preserving design
            rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
            html_rules = ""
            try:
                with open(rules_html_path, "r", encoding="utf-8") as f:
                    html_rules = f.read()
            except FileNotFoundError:
                html_rules = "<!-- HTML rules not found -->"
            
            update_prompt = f"""
You are a professional slide content editor. Update the existing slide with new content while maintaining the exact same design, layout, and visual styling.

ORIGINAL SLIDE HTML:
{original_html}

NEW CONTENT TO INCORPORATE:
{new_content}

DESIGN PRESERVATION RULES:
{html_rules}

CRITICAL REQUIREMENTS:
1. Maintain EXACT same color scheme, fonts, and visual hierarchy
2. Keep the same HTML structure and CSS classes
3. Preserve all images, charts, and visual elements
4. Only change the text content, not the design
5. Ensure content fits within existing layout constraints
6. Maintain 1280x720px slide dimensions
7. If new content is longer/shorter, adjust text sizing appropriately but keep layout structure

OUTPUT: Return only the complete updated HTML code with new content but preserved design.
"""
            
            response = LLM.invoke(update_prompt)
            updated_html = response.content if hasattr(response, 'content') else str(response)
            
            # Extract HTML content
            start_marker = "<!DOCTYPE html>"
            end_marker = "</html>"
            start_idx = updated_html.find(start_marker)
            end_idx = updated_html.find(end_marker)
            if start_idx != -1 and end_idx != -1:
                updated_html = updated_html[start_idx:end_idx + len(end_marker)]
            
        else:
            # Create a completely new slide with the new content
            return generate_slide.invoke({
                "slide_number": slide_number,
                "instructions": new_content,
                "images_urls": "[]",
                "style": "modern, professional presentation style",
                "content": new_content
            })
        
        # Save the updated slide
        output_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(updated_html)
            
        logger.info(f"Successfully updated slide #{slide_number}")
        return updated_html
        
    except Exception as e:
        logger.error(f"Failed to update slide #{slide_number}: {str(e)}", exc_info=True)
        return f"Failed to update slide #{slide_number}: {str(e)}"

@tool
def generate_cover_slide(title: str, subtitle: str, topic: str, author: str = "", style: str = "modern") -> dict:
    """Generate the cover slide (first slide) of a presentation and extract its color palette.
    This slide will establish the visual theme and color scheme for the entire presentation.
    
    Args:
        title: Main title of the presentation
        subtitle: Subtitle or description
        topic: The main topic/theme to determine appropriate colors
        author: Author or presenter name (optional)
        style: Style preference for the presentation
        
    Returns:
        Dictionary containing the HTML content and extracted color palette
    """
    try:
        # Enhanced logging for all parameters
        logger.info(f"=== generate_cover_slide tool called ===")
        logger.info(f"Title: {title}")
        logger.info(f"Subtitle: {subtitle}")
        logger.info(f"Topic: {topic}")
        logger.info(f"Author: {author}")
        logger.info(f"Style: {style}")
        
        logger.info(f"generate_cover_slide tool called for topic: {topic}")
        
        # Load HTML rules
        rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
        html_rules = ""
        try:
            with open(rules_html_path, "r", encoding="utf-8") as f:
                html_rules = f.read()
        except FileNotFoundError:
            logger.warning(f"HTML rules file not found at {rules_html_path}")
            html_rules = "<!-- HTML rules not found -->"
        
        cover_slide_prompt = f"""
You are a professional presentation designer creating the COVER SLIDE for a presentation.

PRESENTATION DETAILS:
Title: {title}
Subtitle: {subtitle}
Topic: {topic}
Author: {author}
Style: {style}

COLOR PALETTE SELECTION TASK:
You must choose an appropriate color palette that fits the topic "{topic}" and create a cohesive visual theme.

TOPIC-SPECIFIC COLOR GUIDELINES:
- Technology/AI topics: Deep blues (#1E40AF), tech cyans (#06B6D4), modern purples (#6366F1)
- Business/Corporate: Professional blues (#1E40AF), slate grays (#64748B), gold accents (#F59E0B)
- Nature/Environment: Forest greens (#059669), earth browns (#92400E), natural teals (#0D9488)
- Health/Medical: Clean blues (#0EA5E9), medical greens (#10B981), safe whites (#F0F9FF)
- Education: Warm oranges (#EA580C), academic blues (#1D4ED8), knowledge purples (#7C3AED)
- Creative/Art: Vibrant magentas (#C026D3), artistic oranges (#EA580C), creative purples (#A855F7)
- Science: Scientific blues (#1E40AF), lab greens (#059669), neutral grays (#6B7280)
- Byzantine Empire: Imperial purple (#7C2D92), deep red (#B91C1C), gold accents (#F59E0B), warm ivory background (#FEF7F0)
- Roman Empire: Imperial red (#B91C1C), stone gray (#78716C), gold accents (#F59E0B), marble white background (#FAFAF9)
- Egyptian: Deep blue (#1E40AF), sand gold (#CA8A04), bright gold accents (#F59E0B), sand background (#FFFBEB)
- Medieval: Deep burgundy (#7F1D1D), forest green (#14532D), gold accents (#F59E0B), parchment background (#F7F3F0)
- Renaissance: Rich burgundy (#7F1D1D), deep blue (#1E40AF), gold accents (#F59E0B), warm cream background (#FEF7F0)
- Ancient/Historical: Royal purple (#7C2D92), ancient bronze (#92400E), gold accents (#F59E0B), aged parchment background (#FEF7F0)
- Chinese History: Chinese red (#B91C1C), imperial gold (#CA8A04), bright gold accents (#F59E0B), silk white background (#FFFBEB)
- Japanese History: Traditional deep red (#7F1D1D), charcoal black (#1F2937), gold accents (#F59E0B), pure white background (#FEFEFE)

COLOR SELECTION RULES:
1. Choose exactly 5 colors: primary, secondary, accent, background, text
2. Ensure minimum 4.5:1 contrast ratio between text and background
3. Primary: Main brand/theme color (for titles)
4. Secondary: Supporting color (for subtitles)
5. Accent: Highlight color (for call-to-action elements)
6. Background: Main slide background (light/neutral)
7. Text: Main reading color (dark for light backgrounds)

DESIGN REQUIREMENTS:
1. Size: 1280x720px (standard presentation ratio)
2. Center-aligned design with clear visual hierarchy
3. Title: Large (48-72px), bold, using primary color
4. Subtitle: Medium (24-32px), using secondary color
5. Author: Small (16-20px), positioned at bottom
6. Include 2-3 subtle geometric elements using accent color
7. Use gradients or solid colors only (NO images)
8. Modern, professional appearance

LAYOUT STRUCTURE:
- Main content area: Centered vertically and horizontally
- Title: Most prominent, top of content area
- Subtitle: Below title with appropriate spacing
- Author: Bottom of slide, smaller font
- Decorative elements: Subtle lines, shapes, or patterns
- Background: Clean, using background color

TECHNICAL SPECIFICATIONS:
- HTML5 semantic structure
- Tailwind CSS for styling
- Google Fonts (Roboto family recommended)
- Responsive design principles
- Clean, accessible markup

CRITICAL COLOR PALETTE REQUIREMENT:
You MUST include this exact comment at the very beginning of your HTML (right after <!DOCTYPE html>):

   <!-- COLOR PALETTE: PRIMARY:#hexcode SECONDARY:#hexcode ACCENT:#hexcode BACKGROUND:#hexcode TEXT:#hexcode -->

Example:
<!-- COLOR PALETTE: PRIMARY:#1E40AF SECONDARY:#6366F1 ACCENT:#06B6D4 BACKGROUND:#F8FAFC TEXT:#1E293B -->

IMPORTANT:
- Use ONLY hex color codes (6 digits with #)
- Apply these exact colors consistently throughout the HTML
- The comment MUST be on its own line after <!DOCTYPE html>
- Make sure all colors work well together and fit the topic

HTML RULES REFERENCE:
{html_rules}

OUTPUT: Generate complete HTML code for the cover slide with the color palette comment at the top and consistent color usage throughout.
"""

        response = LLM.invoke(cover_slide_prompt)
        html_content = response.content if hasattr(response, 'content') else str(response)
        
        # Extract HTML content
        start_marker = "<!DOCTYPE html>"
        end_marker = "</html>"
        start_idx = html_content.find(start_marker)
        end_idx = html_content.find(end_marker)
        if start_idx != -1 and end_idx != -1:
            html_content = html_content[start_idx:end_idx + len(end_marker)]
        else:
            logger.warning("Could not find HTML markers in cover slide response")
        
        # Extract color palette from HTML using multiple methods
        color_palette = {}
        try:
            import re
            
            # Method 1: Look for color palette comment
            palette_match = re.search(r'<!-- COLOR PALETTE: (.+?) -->', html_content, re.IGNORECASE)
            if palette_match:
                palette_text = palette_match.group(1)
                # Parse color values
                color_matches = re.findall(r'(\w+):(#[A-Fa-f0-9]{6})', palette_text)
                for color_name, color_value in color_matches:
                    color_palette[color_name.lower()] = color_value
                logger.info(f"Extracted color palette from HTML comment: {color_palette}")
            
            # Method 2: If comment method failed, extract from CSS styles/classes
            if not color_palette or len(color_palette) < 3:
                logger.info("Trying to extract colors from CSS styles...")
                
                # Look for hex colors in the HTML
                hex_colors = re.findall(r'#[A-Fa-f0-9]{6}', html_content)
                unique_colors = list(set(hex_colors))  # Remove duplicates
                
                if len(unique_colors) >= 3:
                    # Assign colors based on common patterns
                    color_palette = {
                        "primary": unique_colors[0] if len(unique_colors) > 0 else "#2563EB",
                        "secondary": unique_colors[1] if len(unique_colors) > 1 else "#64748B",
                        "accent": unique_colors[2] if len(unique_colors) > 2 else "#F59E0B",
                        "background": unique_colors[3] if len(unique_colors) > 3 else "#F8FAFC",
                        "text": unique_colors[4] if len(unique_colors) > 4 else "#1E293B"
                    }
                    logger.info(f"Extracted color palette from CSS: {color_palette}")
                    logger.info(f"Found {len(unique_colors)} unique colors: {unique_colors}")
            
            # Method 3: Use AI to analyze and extract colors
            if not color_palette or len(color_palette) < 3:
                logger.info("Using AI to analyze and extract color palette from generated HTML...")
                analyze_prompt = f"""
Analyze this HTML content and extract the main color palette used. Look for background colors, text colors, border colors, etc.

HTML CONTENT (first 2000 chars):
{html_content[:2000]}

Return ONLY a JSON object in this exact format:
{{
    "primary": "#hexcode",
    "secondary": "#hexcode", 
    "accent": "#hexcode",
    "background": "#hexcode",
    "text": "#hexcode"
}}

IMPORTANT: 
- Only return the JSON object, no other text
- Use actual hex color codes found in the HTML
- If you can't find 5 colors, use appropriate defaults for missing ones
- Ensure colors are appropriate for the topic: {topic}
"""
                
                try:
                    analyze_response = LLM.invoke(analyze_prompt)
                    analyze_content = analyze_response.content if hasattr(analyze_response, 'content') else str(analyze_response)
                    
                    # Try to parse JSON from response
                    import json
                    json_match = re.search(r'\{[^}]+\}', analyze_content)
                    if json_match:
                        extracted_palette = json.loads(json_match.group())
                        if all(key in extracted_palette for key in ['primary', 'secondary', 'accent', 'background', 'text']):
                            color_palette = extracted_palette
                            logger.info(f"Successfully extracted color palette using AI analysis: {color_palette}")
                        else:
                            logger.warning("AI-extracted palette missing required keys")
                    else:
                        logger.warning("Could not find JSON in AI analysis response")
                except Exception as ai_error:
                    logger.warning(f"AI color analysis failed: {str(ai_error)}")
            
        except Exception as e:
            logger.error(f"Error extracting color palette: {str(e)}")
            # Ultimate fallback
            color_palette = {
                "primary": "#2563EB",
                "secondary": "#64748B",
                "accent": "#F59E0B", 
                "background": "#F8FAFC",
                "text": "#1E293B"
            }
        
        # Save the cover slide
        os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
        output_path = os.path.join(GENERATED_SLIDES_DIR, "slide_001.html")
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(html_content)
        
        # Save color palette for reference by other slides
        palette_path = os.path.join(GENERATED_SLIDES_DIR, "color_palette.json")
        try:
            # Ensure directory exists
            os.makedirs(GENERATED_SLIDES_DIR, exist_ok=True)
            
            # Ensure color palette has all required fields before saving
            complete_palette = {
                "primary": color_palette.get("primary", "#2563EB"),
                "secondary": color_palette.get("secondary", "#64748B"),
                "accent": color_palette.get("accent", "#F59E0B"),
                "background": color_palette.get("background", "#F8FAFC"),
                "text": color_palette.get("text", "#1E293B")
            }
            
            # Update color_palette with complete version
            color_palette.update(complete_palette)
            
            # Write with explicit flushing to ensure data is written
            import json  # Import json for saving
            with open(palette_path, "w", encoding="utf-8") as f:
                json.dump(complete_palette, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            logger.info(f"Structured color palette saved successfully to {palette_path}: {complete_palette}")
        except Exception as e:
            logger.error(f"Failed to save color palette: {str(e)}")
            # Continue without failing the entire function
        
        logger.info(f"Cover slide generated with AI-chosen colors: {color_palette}")
        
        # Return structured response with validated color palette
        try:
            # Validate and structure the color palette
            if 'ColorPalette' in globals():
                structured_palette = ColorPalette(**color_palette)
                structured_response = CoverSlideResponse(
                    html_content=html_content,
                    color_palette=structured_palette,
                    topic=topic
                )
                
                # Convert to dict for tool return
                result = {
                    "html_content": html_content,
                    "color_palette": structured_palette.dict(),
                    "slide_path": output_path,
                    "palette_path": palette_path,
                    "topic": topic,
                    "structured": True  # Flag indicating structured format
                }
            else:
                # Fallback if schemas not available
                result = {
                    "html_content": html_content,
                    "color_palette": color_palette,
                    "slide_path": output_path,
                    "palette_path": palette_path,
                    "topic": topic,
                    "structured": False
                }
            
            logger.info(f"Cover slide generated successfully with structured color palette: {color_palette}")
            return result
            
        except Exception as structure_error:
            logger.warning(f"Could not structure response, using fallback: {structure_error}")
        return {
            "html_content": html_content,
            "color_palette": color_palette,
            "slide_path": output_path,
            "palette_path": palette_path,
                "topic": topic,
                "structured": False
        }
        
    except Exception as e:
        logger.error(f"Failed to generate cover slide: {str(e)}", exc_info=True)
        
        # Create fallback color palette based on topic even in error case
        error_color_palette = {}
        try:
            topic_lower = topic.lower() if topic else ""
            if any(word in topic_lower for word in ['ai', 'tech', 'digital']):
                error_color_palette = {
                    "primary": "#1E40AF", "secondary": "#6366F1", "accent": "#06B6D4",
                    "background": "#F8FAFC", "text": "#1E293B"
                }
            elif any(word in topic_lower for word in ['business', 'corporate']):
                error_color_palette = {
                    "primary": "#1E40AF", "secondary": "#64748B", "accent": "#F59E0B",
                    "background": "#F8FAFC", "text": "#1E293B"
                }
            elif any(word in topic_lower for word in ['byzantine', 'constantinople', 'eastern roman']):
                error_color_palette = {
                    "primary": "#7C2D92", "secondary": "#B91C1C", "accent": "#F59E0B",
                    "background": "#FEF7F0", "text": "#7C2D12"
                }
            elif any(word in topic_lower for word in ['roman', 'rome', 'latin', 'caesar']):
                error_color_palette = {
                    "primary": "#B91C1C", "secondary": "#78716C", "accent": "#F59E0B",
                    "background": "#FAFAF9", "text": "#44403C"
                }
            elif any(word in topic_lower for word in ['ancient', 'history', 'historical', 'civilization', 'empire']):
                error_color_palette = {
                    "primary": "#7C2D92", "secondary": "#92400E", "accent": "#F59E0B",
                    "background": "#FEF7F0", "text": "#78350F"
                }
            else:
                error_color_palette = {
                    "primary": "#2563EB", "secondary": "#64748B", "accent": "#F59E0B",
                    "background": "#F8FAFC", "text": "#1E293B"
                }
        except:
            error_color_palette = {
                "primary": "#2563EB", "secondary": "#64748B", "accent": "#F59E0B",
                "background": "#F8FAFC", "text": "#1E293B"
            }
        
        return {
            "html_content": f"Failed to generate cover slide: {str(e)}",
            "color_palette": error_color_palette,
            "slide_path": "",
            "palette_path": "",
            "topic": topic,
            "structured": True,  # Even error responses have structured palette
            "error": str(e)
        }

def create_topic_based_color_palette(instructions: str, content: str) -> dict:
    """Create a color palette based on detected topic from instructions and content"""
    
    try:
        # Combine instructions and content for topic analysis
        combined_text = f"{instructions} {content}".lower()
        
        logger.info(f"Analyzing topic from combined text: {combined_text[:200]}...")
        
        # Historical/Cultural topics
        if any(word in combined_text for word in ['byzantine', 'constantinople', 'eastern roman']):
            return {
                "primary": "#7C2D92",      # Imperial purple
                "secondary": "#B91C1C",    # Deep red
                "accent": "#F59E0B",       # Gold accent
                "background": "#FEF7F0",   # Warm ivory
                "text": "#7C2D12"          # Dark brown
            }
        elif any(word in combined_text for word in ['roman', 'rome', 'latin', 'caesar']):
            return {
                "primary": "#B91C1C",      # Imperial red
                "secondary": "#78716C",    # Stone gray
                "accent": "#F59E0B",       # Gold accent
                "background": "#FAFAF9",   # Marble white
                "text": "#44403C"          # Dark stone
            }
        elif any(word in combined_text for word in ['egyptian', 'egypt', 'pharaoh', 'pyramid']):
            return {
                "primary": "#1E40AF",      # Deep blue (Nile)
                "secondary": "#CA8A04",    # Sand gold
                "accent": "#F59E0B",       # Bright gold
                "background": "#FFFBEB",   # Sand background
                "text": "#78350F"          # Dark bronze
            }
        elif any(word in combined_text for word in ['medieval', 'middle age', 'knight', 'feudal']):
            return {
                "primary": "#7F1D1D",      # Deep burgundy
                "secondary": "#14532D",    # Forest green
                "accent": "#F59E0B",       # Gold accent
                "background": "#F7F3F0",   # Parchment
                "text": "#451A03"          # Dark brown
            }
        elif any(word in combined_text for word in ['renaissance', 'florence', 'da vinci', 'michelangelo']):
            return {
                "primary": "#7F1D1D",      # Rich burgundy
                "secondary": "#1E40AF",    # Deep blue
                "accent": "#F59E0B",       # Gold accent
                "background": "#FEF7F0",   # Warm cream
                "text": "#7C2D12"          # Rich brown
            }
        elif any(word in combined_text for word in ['ancient', 'history', 'historical', 'civilization', 'empire']):
            return {
                "primary": "#7C2D92",      # Royal purple
                "secondary": "#92400E",    # Ancient bronze
                "accent": "#F59E0B",       # Gold accent
                "background": "#FEF7F0",   # Aged parchment
                "text": "#78350F"          # Deep brown
            }
        elif any(word in combined_text for word in ['chinese', 'china', 'dynasty', 'confucius']):
            return {
                "primary": "#B91C1C",      # Chinese red
                "secondary": "#CA8A04",    # Imperial gold
                "accent": "#F59E0B",       # Bright gold
                "background": "#FFFBEB",   # Silk white
                "text": "#7C2D12"          # Dark lacquer
            }
        elif any(word in combined_text for word in ['japanese', 'japan', 'samurai', 'edo', 'meiji']):
            return {
                "primary": "#7F1D1D",      # Deep red (traditional)
                "secondary": "#1F2937",    # Charcoal black
                "accent": "#F59E0B",       # Gold accent
                "background": "#FEFEFE",   # Pure white
                "text": "#111827"          # Black ink
            }
        
        # Modern topics
        elif any(word in combined_text for word in ['ai', 'artificial intelligence', 'machine learning', 'tech', 'digital']):
            return {
                "primary": "#1E40AF",      # Tech blue
                "secondary": "#6366F1",    # Modern purple-blue
                "accent": "#06B6D4",       # Cyan accent
                "background": "#F8FAFC",   # Light gray-blue
                "text": "#1E293B"          # Dark slate
            }
        elif any(word in combined_text for word in ['business', 'finance', 'corporate']):
            return {
                "primary": "#1E40AF",      # Professional blue
                "secondary": "#64748B",    # Slate gray
                "accent": "#F59E0B",       # Gold accent
                "background": "#F8FAFC",   # Off-white
                "text": "#1E293B"          # Dark text
            }
        elif any(word in combined_text for word in ['nature', 'environment', 'green', 'eco']):
            return {
                "primary": "#059669",      # Green
                "secondary": "#0D9488",    # Teal
                "accent": "#84CC16",       # Lime accent
                "background": "#F0FDF4",   # Light green
                "text": "#064E3B"          # Dark green
            }
        elif any(word in combined_text for word in ['health', 'medical', 'healthcare']):
            return {
                "primary": "#0EA5E9",      # Medical blue
                "secondary": "#06B6D4",    # Light blue
                "accent": "#10B981",       # Health green
                "background": "#F0F9FF",   # Light blue bg
                "text": "#0C4A6E"          # Dark blue text
            }
        
        # No specific topic detected
        return None
        
    except Exception as e:
        logger.warning(f"Error in topic-based color palette creation: {e}")
        return None

if __name__ == "__main__":
    rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
    with open(rules_html_path, "r", encoding="utf-8") as f:
        html_rules = f.read()
    print(html_rules)