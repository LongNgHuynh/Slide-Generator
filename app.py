from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import threading
import uuid
from datetime import datetime
import os
import logging
import re
from workflow import app as workflow_app, AgentState, set_streaming_callback, create_workflow_trace, langfuse
from langchain_core.messages import HumanMessage, AIMessage
import asyncio
import queue
from models.LLMs import Claude_3_7_Sonnet, Gemini, Gemini_2_5_Flash, GPT_o3
from bs4 import BeautifulSoup

# Initialize all LLM models for fallback
LLM_CLAUDE = Claude_3_7_Sonnet()
LLM_GEMINI = Gemini()
LLM_GEMINI_FLASH = Gemini_2_5_Flash()
LLM_GPT_O3 = GPT_o3()

# Fallback order: Claude -> Gemini -> Gemini Flash -> GPT-o3
LLM_FALLBACKS = [
    ("Gemini", LLM_GEMINI),
    ("Gemini 2.5 Flash", LLM_GEMINI_FLASH),
    ("GPT-o3", LLM_GPT_O3)
]

LLM = LLM_CLAUDE  # Default LLM for backward compatibility

# Import constants from workflow
GENERATED_SLIDES_DIR = os.path.join(os.getcwd(), "generated_slides")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config['SECRET_KEY'] = 'your-secret-key-here'
socketio = SocketIO(app, cors_allowed_origins="*")

# Store active sessions
active_sessions = {}
slide_queues = {}

@app.route('/')
def index():
    return render_template('index.html')

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    active_sessions[request.sid] = {
        'session_id': session_id,
        'created_at': datetime.now(),
        'slides': []
    }
    slide_queues[session_id] = queue.Queue()
    emit('session_created', {'session_id': session_id})
    logger.info(f"Client connected: {request.sid} with session {session_id}")
    logger.info(f"Active sessions: {list(active_sessions.keys())}")

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_sessions:
        session_id = active_sessions[request.sid]['session_id']
        del active_sessions[request.sid]
        if session_id in slide_queues:
            del slide_queues[session_id]
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('ai_edit_slide')
def handle_ai_edit_slide(data):
    """Handle AI-assisted slide editing requests from frontend"""
    slide_number = data.get('slide_number')
    user_request = data.get('user_request', '')
    current_content = data.get('current_content', '')
    session_id = active_sessions.get(request.sid, {}).get('session_id')
    client_sid = request.sid
    
    logger.info(f"Received AI edit request for slide {slide_number}: '{user_request}'")
    
    if not session_id:
        emit('error', {'message': 'Session not found'})
        return
    
    if not slide_number or not user_request.strip() or not current_content.strip():
        logger.error(f"Invalid AI edit data: slide_number={slide_number}, user_request='{user_request}', content_length={len(current_content)}")
        emit('error', {'message': 'Invalid AI edit request data'})
        return
    
    logger.info(f"Processing AI edit request for slide {slide_number}")
    
    try:
        # Use the AI to edit the slide
        updated_html = ai_edit_slide_content(current_content, user_request, slide_number)
        
        # Save updated slide to file
        slide_file_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        with open(slide_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_html)
        logger.info(f"Saved AI-edited HTML to file: {slide_file_path}")
        
        # Update session slides data
        if request.sid in active_sessions:
            session_data = active_sessions[request.sid]
            session_slides = session_data.get('slides', [])
            for slide in session_slides:
                if slide.get('slide_number') == slide_number:
                    slide['content'] = updated_html
                    logger.info(f"Updated slide {slide_number} content in session")
                    break
        
        # Send updated slide back to client
        emit('ai_slide_edited', {
            'slide_number': slide_number,
            'content': updated_html,
            'timestamp': datetime.now().isoformat(),
            'user_request': user_request,
            'ai_model_used': getattr(updated_html, '_ai_model_used', 'Unknown')
        })
        
        logger.info(f"Successfully AI-edited slide {slide_number} and sent to client")
        
    except Exception as e:
        logger.error(f"Error AI-editing slide {slide_number}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        emit('error', {
            'message': f'Failed to AI-edit slide {slide_number}: {str(e)}'
        })

def ai_edit_slide_content(current_html, user_request, slide_number):
    """
    Use AI to edit slide content based on user request with fallback models
    """
    logger.info(f"AI editing slide {slide_number} with request: '{user_request}'")
    
    # Create Langfuse trace for AI slide editing
    ai_edit_trace = langfuse.trace(
        name="ai_slide_edit",
        input={
            "slide_number": slide_number,
            "user_request": user_request,
            "html_length": len(current_html)
        },
        metadata={
            "feature": "ai_slide_editing",
            "slide_number": slide_number
        }
    )
    
    try:
        # Try simplified edit with LLM fallbacks
        result = create_simplified_edit_with_fallbacks(current_html, user_request, slide_number, ai_edit_trace)
        
        # Complete trace with success
        ai_edit_trace.update(
            output={
                "success": True,
                "result_length": len(result) if result else 0
            }
        )
        
        return result
    except Exception as e:
        # Complete trace with error
        ai_edit_trace.update(
            output={
                "success": False,
                "error": str(e)
            }
        )
        raise

def optimize_html_for_ai(html_content):
    """
    Optimize HTML content to reduce token usage while preserving structure
    """
    try:
        
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Remove comments
        from bs4 import Comment
        for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
            comment.extract()
        
        # Minimize whitespace in style tags
        for style_tag in soup.find_all('style'):
            if style_tag.string:
                # Remove extra whitespace from CSS
                css_content = style_tag.string
                css_content = ' '.join(css_content.split())  # Normalize whitespace
                style_tag.string = css_content
        
        # Get the optimized HTML
        optimized = str(soup)
        
        # Additional cleanup
        # Remove excessive whitespace between tags
        optimized = re.sub(r'>\s+<', '><', optimized)
        
        logger.info(f"HTML optimized: {len(html_content)} -> {len(optimized)} chars")
        return optimized
        
    except Exception as e:
        logger.warning(f"Failed to optimize HTML: {e}")
        return html_content

def estimate_token_count(text):
    """
    Rough estimation of token count (approximately 4 chars per token)
    """
    return len(text) // 4

def create_simplified_edit_with_fallbacks(current_html, user_request, slide_number, trace=None):
    """
    Create a simplified edit with LLM fallbacks when token limits are exceeded
    """
    logger.info(f"Creating simplified edit for slide {slide_number} with LLM fallbacks")
    
    # Simple AI prompt with complete HTML
    simple_prompt = f"""Edit this slide based on the request.

Current slide HTML:
{current_html}

User request: {user_request}

IMPORTANT: Retain all the design and content, follow the request only. Return the complete modified HTML."""
    
    # Try each LLM in fallback order
    for model_name, llm_model in LLM_FALLBACKS:
        # Create span for each model attempt
        model_span = None
        if trace:
            model_span = trace.span(
                name=f"llm_attempt_{model_name.lower().replace(' ', '_')}",
                input={
                    "model": model_name,
                    "prompt_length": len(simple_prompt)
                },
                metadata={"model_name": model_name}
            )
        
        try:
            logger.info(f"Trying {model_name} for slide {slide_number}")
            
            response = llm_model.invoke(simple_prompt)
            updated_html = response.content if hasattr(response, 'content') else str(response)
            
            # Validate response
            if updated_html and len(updated_html.strip()) > 100:
                # Extract HTML if wrapped in explanation
                if "<!DOCTYPE html>" in updated_html or "<html" in updated_html:
                    # Extract HTML content
                    start_markers = ["<!DOCTYPE html>", "<html"]
                    end_markers = ["</html>"]
                    
                    start_idx = -1
                    end_idx = -1
                    
                    for marker in start_markers:
                        idx = updated_html.find(marker)
                        if idx != -1:
                            start_idx = idx
                            break
                    
                    for marker in end_markers:
                        idx = updated_html.rfind(marker)
                        if idx != -1:
                            end_idx = idx + len(marker)
                            break
                    
                    if start_idx != -1 and end_idx != -1 and end_idx > start_idx:
                        updated_html = updated_html[start_idx:end_idx]
                    
                    # Tag the HTML with the model used (for logging/debugging)
                    updated_html += f"\n<!-- Generated by {model_name} -->"
                    
                    # Complete successful model span
                    if model_span:
                        model_span.end(output={
                            "success": True,
                            "model_used": model_name,
                            "response_length": len(updated_html)
                        })
                    
                    logger.info(f"Successfully edited slide {slide_number} using {model_name}")
                    return updated_html
                else:
                    # No HTML structure, create basic slide with content
                    logger.warning(f"{model_name} returned content without HTML structure")
                    updated_html = create_basic_slide_html(slide_number, updated_html, user_request)
                    updated_html += f"\n<!-- Content by {model_name}, formatted by template -->"
                    
                    # Complete model span with template fallback
                    if model_span:
                        model_span.end(output={
                            "success": True,
                            "model_used": model_name,
                            "template_fallback": True,
                            "response_length": len(updated_html)
                        })
                    
                    return updated_html
            else:
                logger.warning(f"{model_name} returned insufficient content")
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{model_name} failed for slide {slide_number}: {error_msg}")
            
            # Complete failed model span
            if model_span:
                model_span.end(output={
                    "success": False,
                    "error": error_msg,
                    "error_type": "token_limit" if "maximum tokens" in error_msg.lower() or "token limit" in error_msg.lower() else "other"
                })
            
            # Check if it's a token limit error
            if "maximum tokens" in error_msg.lower() or "token limit" in error_msg.lower():
                logger.info(f"{model_name} hit token limit, trying next model")
                continue
            else:
                logger.warning(f"{model_name} failed with non-token error: {error_msg}")
                continue
    
    # All LLMs failed, create basic slide
    logger.error(f"All LLM models failed for slide {slide_number}, creating basic slide")
    return create_basic_slide_html(slide_number, f"AI processing failed. Request: {user_request}", user_request)

def create_simplified_edit(current_html, user_request, slide_number):
    """
    Legacy function - now redirects to fallback version
    """
    return create_simplified_edit_with_fallbacks(current_html, user_request, slide_number, None)

def create_basic_slide_html(slide_number, content, user_request):
    """
    Create a basic slide HTML when AI response doesn't contain proper HTML structure
    """
    basic_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slide {slide_number}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div class="slide-container" style="width: 1280px; min-height: 720px; position: relative; overflow: hidden; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);">
        <div class="h-full flex flex-col justify-center items-center p-16">
            <div class="bg-white rounded-lg shadow-2xl p-12 max-w-5xl w-full">
                <h1 class="text-4xl font-bold text-gray-800 mb-8 text-center">Slide {slide_number}</h1>
                <div class="text-lg text-gray-700 leading-relaxed">
                    <p class="mb-4"><strong>Modified based on:</strong> {user_request}</p>
                    <div class="border-t pt-4">{content}</div>
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return basic_html

@socketio.on('edit_text')
def handle_text_edit(data):
    """Handle individual text element editing requests from frontend"""
    slide_number = data.get('slide_number')
    text_id = data.get('text_id')
    new_text = data.get('new_text', '')
    original_text = data.get('original_text', '')
    is_rich_text = data.get('is_rich_text', False)
    session_id = active_sessions.get(request.sid, {}).get('session_id')
    client_sid = request.sid
    
    logger.info(f"Received edit_text request: slide={slide_number}, text_id={text_id}, is_rich_text={is_rich_text}")
    logger.info(f"Original text: '{original_text}'")
    logger.info(f"New text: '{new_text}'")
    
    if not session_id:
        emit('error', {'message': 'Session not found'})
        return
    
    if not slide_number or not new_text.strip() or text_id is None:
        logger.error(f"Invalid text edit data: slide_number={slide_number}, new_text='{new_text}', text_id={text_id}")
        emit('error', {'message': 'Invalid text edit data'})
        return
    
    logger.info(f"Handling text edit request for slide {slide_number}, text element {text_id}")
    
    # Process text edit directly (not in background thread to avoid context issues)
    try:
        # Read current slide HTML
        slide_file_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        current_html = ""
        
        logger.info(f"Looking for slide file: {slide_file_path}")
        
        if os.path.exists(slide_file_path):
            with open(slide_file_path, 'r', encoding='utf-8') as f:
                current_html = f.read()
            logger.info(f"Read slide from file, length: {len(current_html)}")
        else:
            # If file doesn't exist, get from session data
            logger.info(f"Slide file not found, checking session data")
            if request.sid in active_sessions:
                session_data = active_sessions[request.sid]
                session_slides = session_data.get('slides', [])
                logger.info(f"Found {len(session_slides)} slides in session")
                for slide in session_slides:
                    if slide.get('slide_number') == slide_number:
                        current_html = slide.get('content', '')
                        logger.info(f"Found slide in session, content length: {len(current_html)}")
                        break
        
        if not current_html:
            logger.error("Could not find slide content to edit")
            emit('error', {'message': 'Could not find slide content to edit'})
            return
        
        logger.info(f"Processing text update with is_rich_text={is_rich_text}")
        
        # Update the specific text element in the HTML
        updated_html = update_text_in_html(current_html, text_id, original_text, new_text, is_rich_text)
        
        logger.info(f"HTML updated, new length: {len(updated_html)}")
        
        # Save updated slide to file
        with open(slide_file_path, 'w', encoding='utf-8') as f:
            f.write(updated_html)
        logger.info(f"Saved updated HTML to file: {slide_file_path}")
        
        # Update session slides data
        session_found = False
        if request.sid in active_sessions:
            session_data = active_sessions[request.sid]
            session_slides = session_data.get('slides', [])
            logger.info(f"Found session with {len(session_slides)} slides")
            for slide in session_slides:
                if slide.get('slide_number') == slide_number:
                    slide['content'] = updated_html
                    logger.info(f"Updated slide content in session")
                    session_found = True
                    break
            
            if not session_found:
                logger.warning(f"Slide {slide_number} not found in session slides")
        else:
            logger.warning(f"Session {request.sid} not found in active_sessions")
        
        # Send updated slide back to client
        emit('text_updated', {
            'slide_number': slide_number,
            'content': updated_html,
            'timestamp': datetime.now().isoformat()
        })
        
        logger.info(f"Successfully updated text in slide {slide_number} and sent to client")
        
    except Exception as e:
        logger.error(f"Error updating text in slide {slide_number}: {str(e)}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        emit('error', {
            'message': f'Failed to update text in slide {slide_number}: {str(e)}'
        })

def regenerate_slide_html(slide_number, new_text_content, original_html):
    """
    Regenerate slide HTML with updated text content while preserving design and layout
    """
    from models.LLMs import Claude_3_7_Sonnet
    
    logger.info(f"Regenerating HTML for slide {slide_number} with new content")
    
    # Load HTML rules for consistency
    rules_html_path = os.path.join(os.getcwd(), "rules", "html.txt")
    html_rules = ""
    try:
        with open(rules_html_path, "r", encoding="utf-8") as f:
            html_rules = f.read()
    except FileNotFoundError:
        logger.warning(f"HTML rules file not found at {rules_html_path}")
        html_rules = "<!-- HTML rules not found -->"
    
    # Create prompt for updating slide content
    update_prompt = f"""
You are a professional presentation slide editor. Your task is to update an existing slide with new text content while preserving the original design, layout, and styling.

ORIGINAL SLIDE HTML:
{original_html}

NEW TEXT CONTENT TO INCORPORATE:
{new_text_content}

DESIGN RULES TO FOLLOW:
{html_rules}

INSTRUCTIONS:
1. Keep the same overall design, color scheme, and layout structure as the original slide
2. Replace the text content with the new provided content while maintaining visual hierarchy
3. Preserve all styling, fonts, colors, and positioning from the original
4. Keep the same HTML structure and CSS classes
5. Maintain the 1280x720px dimensions and responsive design
6. Ensure the new content fits well within the existing layout
7. If the new content is longer/shorter than original, adjust spacing appropriately but keep the design consistent
8. Preserve any images, charts, or visual elements from the original slide
9. Use the same font families, sizes, and styling as the original

CRITICAL: Only return the complete, updated HTML code. Do not include any explanations or additional text.

The output should be a fully functional HTML slide that looks consistent with the original design but contains the updated text content.
"""

    try:
        llm_claude = Claude_3_7_Sonnet()
        response = llm_claude.invoke(update_prompt)
        updated_html = response.content if hasattr(response, 'content') else str(response)
        
        # Extract HTML content if it's wrapped in explanation text
        start_marker = "<!DOCTYPE html>"
        end_marker = "</html>"
        start_idx = updated_html.find(start_marker)
        end_idx = updated_html.find(end_marker)
        
        if start_idx != -1 and end_idx != -1:
            updated_html = updated_html[start_idx:end_idx + len(end_marker)]
        else:
            logger.warning("Could not find HTML markers in regenerated slide content")
            # Fallback: try to find just the HTML content
            if "<html" in updated_html.lower():
                # Extract everything from first <html tag to last </html>
                html_start = updated_html.lower().find("<html")
                html_end = updated_html.lower().rfind("</html>") + 7
                if html_start != -1 and html_end != -1:
                    updated_html = updated_html[html_start:html_end]
        
        logger.info(f"Successfully regenerated slide {slide_number} HTML")
        return updated_html
        
    except Exception as e:
        logger.error(f"Error regenerating slide HTML: {str(e)}")
        # Fallback: create a simple updated slide
        return create_fallback_slide_html(slide_number, new_text_content)

def create_fallback_slide_html(slide_number, text_content):
    """
    Create a simple fallback slide HTML when regeneration fails
    """
    fallback_html = f"""<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Slide {slide_number}</title>
    <link href="https://fonts.googleapis.com/css2?family=Roboto:wght@300;400;500;700&display=swap" rel="stylesheet">
    <script src="https://cdn.tailwindcss.com"></script>
</head>
<body>
    <div class="slide-container" style="width: 1280px; min-height: 720px; position: relative; overflow: hidden; background: #f8f9fa;">
        <div class="h-full flex flex-col justify-center items-center p-16">
            <div class="bg-white rounded-lg shadow-lg p-12 max-w-4xl w-full">
                <h1 class="text-4xl font-bold text-gray-800 mb-8 text-center">Slide {slide_number}</h1>
                <div class="text-lg text-gray-700 leading-relaxed whitespace-pre-wrap">
                    {text_content}
                </div>
            </div>
        </div>
    </div>
</body>
</html>"""
    return fallback_html

def update_text_in_html(html_content, text_id, original_text, new_text, is_rich_text=False):
    """
    Update a specific text element in HTML content based on text_id and original text
    Supports both plain text and rich text (HTML) updates
    """
    
    try:
        # Parse the HTML
        soup = BeautifulSoup(html_content, 'html.parser')
        
        # Find all text elements
        text_elements = soup.find_all(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div', 'li', 'td', 'th'])
        
        # Filter to get meaningful text elements (same logic as frontend)
        meaningful_elements = []
        for element in text_elements:
            text = element.get_text(strip=True)
            if len(text) >= 3:
                # Skip elements that contain other text elements
                has_text_children = element.find(['h1', 'h2', 'h3', 'h4', 'h5', 'h6', 'p', 'span', 'div', 'li', 'td', 'th'])
                if not has_text_children:
                    meaningful_elements.append(element)
        
        # Update the specific element by text_id
        if text_id < len(meaningful_elements):
            target_element = meaningful_elements[text_id]
            current_text = target_element.get_text(strip=True)
            
            # Verify this is the right element by checking if original text matches
            if current_text == original_text.strip():
                if is_rich_text:
                    # Handle rich text (HTML) content
                    target_element.clear()
                    # Parse the new HTML content and insert it
                    new_soup = BeautifulSoup(new_text, 'html.parser')
                    for content in new_soup.contents:
                        if hasattr(content, 'name'):
                            # It's a tag
                            target_element.append(content)
                        else:
                            # It's text
                            target_element.append(str(content))
                    
                    logger.info(f"Updated rich text element {text_id}: '{original_text}' -> rich HTML content")
                else:
                    # Handle plain text content
                    if target_element.string:
                        # Simple case: element has only text
                        target_element.string.replace_with(new_text)
                    else:
                        # Complex case: element has mixed content, replace text nodes
                        target_element.clear()
                        target_element.append(new_text)
                    
                    logger.info(f"Updated text element {text_id}: '{original_text}' -> '{new_text}'")
            else:
                # Fallback: try to find element by text content
                for i, element in enumerate(meaningful_elements):
                    if element.get_text(strip=True) == original_text.strip():
                        if is_rich_text:
                            element.clear()
                            new_soup = BeautifulSoup(new_text, 'html.parser')
                            for content in new_soup.contents:
                                if hasattr(content, 'name'):
                                    element.append(content)
                                else:
                                    element.append(str(content))
                        else:
                            if element.string:
                                element.string.replace_with(new_text)
                            else:
                                element.clear()
                                element.append(new_text)
                        
                        logger.info(f"Updated element by content match: '{original_text}' -> new content")
                        break
                else:
                    logger.warning(f"Could not find text element to update: '{original_text}'")
        else:
            logger.warning(f"Text element {text_id} not found in HTML")
        
        return str(soup)
        
    except Exception as e:
        logger.error(f"Error updating text in HTML: {str(e)}")
        # Fallback: simple text replacement
        return html_content.replace(original_text, new_text, 1)

@socketio.on('send_message')
def handle_message(data):
    message = data.get('message', '')
    session_id = active_sessions.get(request.sid, {}).get('session_id')
    client_sid = request.sid  # Capture the session ID before starting thread
    
    if not session_id:
        emit('error', {'message': 'Session not found'})
        return
    
    # Emit user message to chat
    emit('chat_message', {
        'type': 'user',
        'message': message,
        'timestamp': datetime.now().isoformat()
    })
    
    # Start workflow in background thread
    def run_workflow():
        try:
            # Check if this is an approval/rejection response
            message_lower = message.lower().strip()
            approval_keywords = ["approve", "yes", "proceed", "continue", "looks good", "ok", "accept"]
            rejection_keywords = ["reject", "no", "redo", "regenerate", "change", "modify"]
            
            is_approval_response = any(keyword in message_lower for keyword in approval_keywords)
            is_rejection_response = any(keyword in message_lower for keyword in rejection_keywords)
            
            # Get existing workflow state from session if this is an approval/rejection
            existing_state = None
            if (is_approval_response or is_rejection_response):
                # Look for session by sid since that's how we store it
                for sid, session_data in active_sessions.items():
                    if session_data.get('session_id') == session_id:
                        existing_state = session_data.get('workflow_state')
                        logger.info(f"Found existing workflow state for session {session_id}")
                        break
            
            if existing_state:
                # Continue existing workflow with approval/rejection
                logger.info(f"Continuing existing workflow with {'approval' if is_approval_response else 'rejection'}")
                logger.info(f"Existing state keys: {list(existing_state.keys())}")
                logger.info(f"Outline generated: {existing_state.get('is_outline_generated')}, approved: {existing_state.get('is_outline_approved')}")
                initial_state = existing_state.copy()
                # Add the new human message to the existing state
                initial_state["messages"].append(HumanMessage(content=message))
            else:
                # Create new initial state for workflow
                logger.info("Starting new workflow - no existing state found")
                if is_approval_response or is_rejection_response:
                    logger.warning("This looks like an approval/rejection but no existing state was found!")
                
                # Create Langfuse trace for new workflow
                trace = create_workflow_trace(session_id, message)
                logger.info(f"Created Langfuse trace for session {session_id}")
                
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "is_outline_generated": False,
                    "is_outline_approved": False,
                    "images": [],
                    "found_information": [],
                    "slides": [],
                    "summary": [],
                    "outline_attempts": 0
                }
            
            # Stream workflow updates
            def stream_callback(update_type, data):
                if update_type == 'slide_generated':
                    # Emit slide immediately when generated
                    socketio.emit('slide_generated', data, room=client_sid)
                elif update_type == 'agent_update':
                    # Emit agent status updates
                    socketio.emit('agent_status', data, room=client_sid)
                elif update_type == 'workflow_complete':
                    # Emit completion status
                    socketio.emit('workflow_complete', data, room=client_sid)
                elif update_type == 'agent_stream_start':
                    # Emit streaming start
                    socketio.emit('agent_stream_start', data, room=client_sid)
                elif update_type == 'agent_stream_token':
                    # Emit individual tokens
                    socketio.emit('agent_stream_token', data, room=client_sid)
                elif update_type == 'agent_stream_end':
                    # Emit streaming end
                    socketio.emit('agent_stream_end', data, room=client_sid)
                elif update_type == 'agent_tool_call':
                    # Emit tool call notifications
                    socketio.emit('agent_tool_call', data, room=client_sid)
                elif update_type == 'supervisor_message':
                    # Emit supervisor messages (like approval requests)
                    socketio.emit('chat_message', {
                        'type': 'assistant',
                        'message': data.get('message', ''),
                        'timestamp': datetime.now().isoformat()
                    }, room=client_sid)
            
            # Set the streaming callback for the workflow
            set_streaming_callback(stream_callback)
            
            # Run the modified workflow with streaming
            result = run_workflow_with_streaming(initial_state, stream_callback, session_id)
            
            # Save final workflow state to session
            for sid, session_data in active_sessions.items():
                if session_data.get('session_id') == session_id:
                    session_data['workflow_state'] = result
                    logger.info(f"Saved final workflow state to session {session_id}")
                    break
            
            # Complete Langfuse workflow trace
            from workflow import get_current_trace
            workflow_trace = get_current_trace()
            if workflow_trace:
                workflow_trace.update(
                    output={
                        "workflow_completed": True,
                        "slides_generated": len(result.get('slides', [])),
                        "outline_generated": result.get('is_outline_generated', False),
                        "outline_approved": result.get('is_outline_approved', False)
                    }
                )
                logger.info(f"Completed Langfuse workflow trace for session {session_id}")
            
            # Send final completion message only if workflow actually completed
            if result.get('slides') and len(result.get('slides', [])) > 0:
                socketio.emit('chat_message', {
                    'type': 'assistant',
                    'message': 'Presentation generation completed!',
                    'timestamp': datetime.now().isoformat()
                }, room=client_sid)
            
        except Exception as e:
            logger.error(f"Error in workflow: {str(e)}")
            socketio.emit('error', {
                'message': f'Error generating presentation: {str(e)}'
            }, room=client_sid)
    
    # Start workflow in background thread
    thread = threading.Thread(target=run_workflow)
    thread.daemon = True
    thread.start()
    
    # Send acknowledgment
    emit('chat_message', {
        'type': 'assistant',
        'message': 'Starting presentation generation...',
        'timestamp': datetime.now().isoformat()
    })

def run_workflow_with_streaming(initial_state, callback, session_id):
    """Modified workflow runner that streams updates"""
    from workflow import supervisor_node, planner_node, outline_agent_node, artist_agent_node, slide_agent_node, get_current_trace
    
    # Create workflow execution span
    workflow_trace = get_current_trace()
    execution_span = None
    if workflow_trace:
        execution_span = workflow_trace.span(
            name="workflow_execution",
            input={
                "session_id": session_id,
                "initial_state_keys": list(initial_state.keys())
            },
            metadata={"session_id": session_id}
        )
    
    state = initial_state
    current_node = "supervisor"
    max_iterations = 50
    iteration = 0
    
    # Check if we're continuing from an existing workflow (approval/rejection)
    if state.get('is_outline_generated') and not state.get('is_outline_approved'):
        logger.info("Continuing workflow from outline approval state")
    
    while current_node != "FINISH" and iteration < max_iterations:
        iteration += 1
        logger.info(f"Iteration {iteration}: Running node {current_node}")
        
        try:
            # Create span for each node execution
            node_span = None
            if execution_span:
                node_span = execution_span.span(
                    name=f"node_execution_{current_node}",
                    input={
                        "node": current_node,
                        "iteration": iteration,
                        "state_summary": {
                            "outline_generated": state.get('is_outline_generated', False),
                            "outline_approved": state.get('is_outline_approved', False),
                            "has_layout": bool(state.get('layout_instructions')),
                            "slides_count": len(state.get('slides', []))
                        }
                    },
                    metadata={"node": current_node, "iteration": iteration}
                )
            
            if current_node == "supervisor":
                command = supervisor_node(state)
                next_node = command.goto if hasattr(command, 'goto') else command.get('next', 'FINISH')
                
                # Update state with supervisor decision
                if hasattr(command, 'update'):
                    state.update(command.update)
                    
                    # Check if supervisor added a message (like approval request)
                    if 'messages' in command.update:
                        new_messages = command.update['messages']
                        for msg in new_messages:
                            if hasattr(msg, 'name') and msg.name == 'supervisor':
                                # This is a supervisor message, emit it
                                callback('supervisor_message', {'message': msg.content})
                
                callback('agent_update', {
                    'agent': 'supervisor',
                    'status': f'Routing to {next_node}',
                    'iteration': iteration
                })
                
                current_node = next_node
                
            elif current_node == "planner":
                command = planner_node(state)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                callback('agent_update', {
                    'agent': 'planner',
                    'status': 'Planning completed - workflow plan created',
                    'plan_created': bool(state.get('plan'))
                })
                
                current_node = "supervisor"
                
            elif current_node == "outline_agent":
                command = outline_agent_node(state)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                # Save state to session after outline generation (for approval continuation)
                for sid, session_data in active_sessions.items():
                    if session_data.get('session_id') == session_id:
                        session_data['workflow_state'] = state.copy()
                        logger.info(f"Saved workflow state after outline generation to session {session_id}")
                        break
                
                callback('agent_update', {
                    'agent': 'outline_agent',
                    'status': 'Outline generation completed',
                    'outline_generated': state.get('is_outline_generated', False)
                })
                
                current_node = "supervisor"
                
            elif current_node == "artist_agent":
                command = artist_agent_node(state)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                callback('agent_update', {
                    'agent': 'artist_agent',
                    'status': 'Layout design completed',
                    'layout_created': bool(state.get('layout_instructions'))
                })
                
                current_node = "supervisor"
                
            elif current_node == "slide_agent":
                # Modified slide agent to stream slides
                command = slide_agent_node_with_streaming(state, callback, session_id)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                callback('agent_update', {
                    'agent': 'slide_agent',
                    'status': f'Generated {len(state.get("slides", []))} slides'
                })
                
                current_node = "supervisor"
            
            else:
                logger.warning(f"Unknown node: {current_node}")
                if node_span:
                    node_span.end(output={"error": f"Unknown node: {current_node}"})
                break
            
            # Complete node span with success
            if node_span:
                node_span.end(output={
                    "success": True,
                    "next_node": current_node,
                    "state_updated": True
                })
                
        except Exception as e:
            logger.error(f"Error in node {current_node}: {str(e)}")
            callback('error', {'message': str(e)})
            
            # Complete node span with error
            if node_span:
                node_span.end(output={
                    "success": False,
                    "error": str(e)
                })
            
            break
    
    callback('workflow_complete', {
        'total_slides': len(state.get('slides', [])),
        'iterations': iteration
    })
    
    # Complete workflow execution span
    if execution_span:
        execution_span.end(output={
            "final_node": current_node,
            "total_iterations": iteration,
            "slides_generated": len(state.get('slides', [])),
            "workflow_completed": current_node == "FINISH"
        })
    
    return state

def slide_agent_node_with_streaming(state, callback, session_id):
    """Modified slide agent that streams slides as they're generated"""
    from workflow import slide_agent_node
    from utils.tools import generate_slide
    import json
    import re
    
    logger.info("Slide agent with streaming: Starting slide generation")
    
    # Get the outline content from state (preferred) or messages (fallback)
    actual_outline_str = state.get("outline_content", "")
    
    # Fallback: check messages if outline_content is not in state
    if not actual_outline_str:
        if state.get("messages"):
            for msg in reversed(state["messages"]):
                if hasattr(msg, 'name') and msg.name == "outline_agent":
                    actual_outline_str = msg.content
                    break
    
    if not actual_outline_str:
        actual_outline_str = "Outline not available."
    
    # Debug: log the actual outline content
    logger.info(f"Actual outline content: {actual_outline_str[:500]}...")  # First 500 chars
    
    images = state.get("images", [])
    found_info = state.get("found_information", [])
    
    # Read slide generation instructions
    instruction_file_path = os.path.join(os.getcwd(), "rules", "instruction.txt")
    slide_gen_instructions = ""
    try:
        with open(instruction_file_path, "r") as f:
            slide_gen_instructions = f.read()
    except FileNotFoundError:
        slide_gen_instructions = "Create informative and visually appealing slides."
    
    # Parse outline to identify slides with multiple strategies
    slide_lines = []
    
    # Strategy 1: Look for explicit slide markers
    for line in actual_outline_str.split('\n'):
        line_clean = line.strip()
        if line_clean and re.search(r'slide\s+\d+|^\d+\.|^-\s*\w+|^[*•]\s*\w+|^\d+\)|^[a-zA-Z]\)', line_clean, re.IGNORECASE):
            slide_lines.append(line_clean)
    
    # Strategy 2: If no explicit slides found, split by sections/paragraphs
    if not slide_lines:
        sections = [s.strip() for s in actual_outline_str.split('\n') if s.strip() and len(s.strip()) > 10]
        slide_lines = sections
    
    # Strategy 3: If still no content, split by sentences for basic slides
    if not slide_lines:
        sentences = [s.strip() for s in actual_outline_str.split('.') if s.strip() and len(s.strip()) > 20]
        slide_lines = sentences[:10]  # Limit to 10
    
    # Debug logging to see what slides were found
    logger.info(f"Found {len(slide_lines)} slide lines from outline: {slide_lines[:3]}...")  # Show first 3
    
    if not slide_lines:
        # Emergency fallback: create basic slides
        slide_lines = [
            "Introduction and Overview",
            "Main Topic Discussion", 
            "Key Points and Details",
            "Examples and Applications",
            "Conclusion and Summary"
        ]
        logger.info(f"Using emergency fallback slides: {len(slide_lines)} slides")
    
    # Let slide_agent decide the appropriate number of slides based on content
    # Filter slide_lines to only include meaningful slide content (not bullet points)
    meaningful_slides = []
    for line in slide_lines:
        line_lower = line.lower().strip()
        # Only include lines that look like actual slide titles/content
        if (re.search(r'slide\s+\d+', line_lower) or 
            re.search(r'^##\s+slide', line_lower) or
            (len(line.split()) > 3 and not line_lower.startswith('-') and not line_lower.startswith('•'))):
            meaningful_slides.append(line)
    
    # If we found meaningful slide markers, use them; otherwise use original strategy
    if meaningful_slides:
        slide_lines = meaningful_slides
    
    # Cap at reasonable number for good user experience
    max_slides = 10  # Reasonable maximum for most presentations
    if len(slide_lines) > max_slides:
        slide_lines = slide_lines[:max_slides]
        logger.info(f"Capped slides at {max_slides} for better user experience")
    logger.info(f"Final slide count: {len(slide_lines)} slides to generate")
    
    generated_slides_info = []
    
    # Generate slides one by one and stream them
    for i, slide_content in enumerate(slide_lines, 1):
        try:
            # Add delay between slides to avoid rate limiting
            if i > 1:  # Don't delay the first slide
                import time
                time.sleep(3)  # 3 second delay between slides
            
            # Prepare slide generation data
            slide_data = {
                "slide_number": i,
                "instructions": f"Create slide {i} with content: {slide_content}. Use relevant research: {found_info[:2]}",
                "images_urls": json.dumps([img for img in images[:2]]) if images else "[]",
                "style": "modern, professional presentation style with clean layout",
                "content": slide_content
            }
            
            # Generate the slide
            slide_result = generate_slide.invoke(slide_data)
            
            # Try to read the generated HTML file
            html_content = ""
            try:
                slide_file_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{i:03d}.html")
                if os.path.exists(slide_file_path):
                    with open(slide_file_path, 'r', encoding='utf-8') as f:
                        html_content = f.read()
                else:
                    html_content = slide_result
            except Exception as e:
                logger.error(f"Error reading slide file: {str(e)}")
                html_content = slide_result
            
            # Parse the result and stream it
            slide_info = {
                "slide_number": i,
                "content": html_content,
                "timestamp": datetime.now().isoformat()
            }
            
            generated_slides_info.append(slide_info)
            
            # Update session slides data immediately
            if session_id in active_sessions:
                if 'slides' not in active_sessions[session_id]:
                    active_sessions[session_id]['slides'] = []
                active_sessions[session_id]['slides'].append(slide_info)
                logger.info(f"Added slide {i} to session {session_id}")
            
            # Stream the slide immediately
            callback('slide_generated', slide_info)
            
            logger.info(f"Streamed slide {i}")
            
        except Exception as e:
            logger.error(f"Error generating slide {i}: {str(e)}")
            error_slide = {
                "slide_number": i,
                "content": f"Error generating slide {i}: {str(e)}",
                "timestamp": datetime.now().isoformat(),
                "error": True
            }
            generated_slides_info.append(error_slide)
            callback('slide_generated', error_slide)
    
    # Return command to update state - preserve existing messages
    
    return type('Command', (), {
        'update': {
            "messages": state.get("messages", []) + [AIMessage(
                content=f"Generated {len(generated_slides_info)} slides",
                name='slide_agent'
            )],
            "slides": generated_slides_info
        },
        'goto': "supervisor"
    })()

if __name__ == '__main__':
    # Disable auto-reloader to avoid Windows socket issues
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, use_reloader=False)
