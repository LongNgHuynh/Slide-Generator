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
        
        # Load existing color palette if cover slide was generated
        color_palette = None
        palette_path = os.path.join(GENERATED_SLIDES_DIR, "color_palette.json")
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
            logger.info("No existing color palette found, will use default color guidelines")
        
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
MANDATORY COLOR PALETTE (MUST USE THESE EXACT COLORS TO MATCH COVER SLIDE):
- Primary Color: {color_palette.get('primary', '#2563EB')}
- Secondary Color: {color_palette.get('secondary', '#64748B')}
- Accent Color: {color_palette.get('accent', '#F59E0B')}
- Background Color: {color_palette.get('background', '#F8FAFC')}
- Text Color: {color_palette.get('text', '#1E293B')}

These colors were chosen by AI for the cover slide based on the presentation topic.
CRITICAL: You MUST use this exact color palette to maintain visual consistency with the cover slide and overall presentation theme.

Add this comment at the top of your HTML to document color usage:
<!-- USING COVER SLIDE COLORS: PRIMARY:{color_palette.get('primary', '#2563EB')} SECONDARY:{color_palette.get('secondary', '#64748B')} ACCENT:{color_palette.get('accent', '#F59E0B')} BACKGROUND:{color_palette.get('background', '#F8FAFC')} TEXT:{color_palette.get('text', '#1E293B')} -->
"""
        else:
            color_instructions = """
COLOR GUIDELINES:
- Use soft, professional colors that harmonize with the content
- Ensure high contrast for readability
- Avoid pure black (#000000) or white (#FFFFFF) backgrounds
- Select colors appropriate to the topic (e.g., tech = blues, nature = greens, business = navy/gray)

Since no cover slide exists yet, choose appropriate colors for this topic and ensure consistency if generating multiple slides.
"""
            
        presentation_prompt = f"""
You are a professional presentation designer specializing in Material Design principles.

This is example of a slide: {html_rules}
You should ignore all previous instructions and examples.

{color_instructions}

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
        return f"Slide #{slide_number} generated. Saved to {output_path}"
    except Exception as e:
        logger.error(f"Failed to generate slide #{slide_number}: {str(e)}", exc_info=True)
        return f"Failed to generate slide #{slide_number}: {str(e)}"
    
@tool
def update_slide(slide_number: int, new_content: str, original_html: str, preserve_design: bool = True) -> str:
    """Update an existing slide with new content while optionally preserving the original design.
    
    Args:
        slide_number: The slide number to update
        new_content: The new text content for the slide
        original_html: The original HTML content of the slide
        preserve_design: Whether to preserve the original design and styling
    """
    try:
        logger.info(f"update_slide tool called for slide #{slide_number}")
        
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

COLOR SELECTION GUIDELINES:
- Analyze the topic and choose colors that reinforce the subject matter
- For technology topics: Consider blues, cyans, silvers, or modern tech colors
- For historical topics: Consider earth tones, ancient colors, warm browns
- For business topics: Consider professional blues, grays, accent golds
- For nature topics: Consider greens, browns, earth tones
- For medical topics: Consider clean blues, whites, soft greens
- For creative topics: Consider vibrant, artistic color combinations
- Use 3-5 colors maximum (primary, secondary, accent, background, text)
- Ensure high contrast for readability (minimum 4.5:1 ratio)
- Choose colors that work well together and create professional appearance

DESIGN REQUIREMENTS FOR COVER SLIDE:
1. Choose and use a cohesive color palette appropriate to the topic
2. Create a professional, visually striking cover slide
3. Include title, subtitle, author (if provided)
4. Size: 1280x720px (standard presentation ratio)
5. Use clean, modern typography with Google Fonts
6. Apply appropriate visual hierarchy
7. Include subtle geometric elements or patterns that complement the theme
8. NO background images - use solid colors and gradients only
9. Ensure high contrast for readability

LAYOUT STRUCTURE:
- Center-aligned design with clear hierarchy
- Title: Large, bold, using primary or accent color
- Subtitle: Medium size, using secondary color
- Author: Small, positioned at bottom, using text color
- Add subtle design elements (lines, shapes) using accent color
- Use background color as the main background

TECHNICAL SPECIFICATIONS:
- Use Tailwind CSS for styling
- Include Google Fonts (Roboto family)
- Material Design principles
- Responsive design within 1280x720 container
- Clean, semantic HTML structure

HTML TEMPLATE REFERENCE:
{html_rules}

CRITICAL INSTRUCTIONS:
1. First choose your color palette based on the topic "{topic}"
2. Use EXACTLY those colors throughout the slide
3. In your HTML, add a comment at the top specifying your chosen colors like this:
   <!-- COLOR PALETTE: PRIMARY:#hexcode SECONDARY:#hexcode ACCENT:#hexcode BACKGROUND:#hexcode TEXT:#hexcode -->
4. Apply these colors consistently throughout the slide design

OUTPUT: Generate complete HTML code for the cover slide with your chosen color palette clearly specified in the HTML comment.
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
        
        # Extract color palette from HTML comment
        color_palette = {}
        try:
            import re
            # Look for color palette comment
            palette_match = re.search(r'<!-- COLOR PALETTE: (.+?) -->', html_content)
            if palette_match:
                palette_text = palette_match.group(1)
                # Parse color values
                color_matches = re.findall(r'(\w+):(#[A-Fa-f0-9]{6})', palette_text)
                for color_name, color_value in color_matches:
                    color_palette[color_name.lower()] = color_value
                logger.info(f"Extracted color palette from AI: {color_palette}")
            else:
                logger.warning("Could not find color palette comment in HTML")
                # Fallback: extract colors from CSS classes or styles
                color_palette = {
                    "primary": "#2563EB",
                    "secondary": "#64748B", 
                    "accent": "#F59E0B",
                    "background": "#F8FAFC",
                    "text": "#1E293B"
                }
        except Exception as e:
            logger.error(f"Error extracting color palette: {str(e)}")
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
            
            # Write with explicit flushing to ensure data is written
            with open(palette_path, "w", encoding="utf-8") as f:
                json.dump(color_palette, f, indent=2, ensure_ascii=False)
                f.flush()  # Force write to disk
                os.fsync(f.fileno())  # Ensure OS writes to disk
            
            logger.info(f"Color palette saved successfully to {palette_path}")
        except Exception as e:
            logger.error(f"Failed to save color palette: {str(e)}")
            # Continue without failing the entire function
        
        logger.info(f"Cover slide generated with AI-chosen colors: {color_palette}")
        
        return {
            "html_content": html_content,
            "color_palette": color_palette,
            "slide_path": output_path,
            "palette_path": palette_path,
            "topic": topic
        }
        
    except Exception as e:
        logger.error(f"Failed to generate cover slide: {str(e)}", exc_info=True)
        return {
            "html_content": f"Failed to generate cover slide: {str(e)}",
            "color_palette": {},
            "error": str(e)
        }

if __name__ == "__main__":
    rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
    with open(rules_html_path, "r", encoding="utf-8") as f:
        html_rules = f.read()
    print(html_rules)