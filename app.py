from flask import Flask, render_template, request, jsonify, send_file
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
from utils.pdf_export import (
    export_single_slide_to_pdf, 
    export_all_slides_in_directory, 
    export_slides_to_pdf,
    get_all_slides_in_directory
)

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

@app.route('/export_pdf/<int:slide_number>')
def export_single_slide_pdf(slide_number):
    """Export a single slide to PDF"""
    try:
        logger.info(f"PDF export request for slide {slide_number}")
        
        # Find the slide file
        slide_file = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
        
        if not os.path.exists(slide_file):
            return jsonify({
                "success": False,
                "error": f"Slide {slide_number} not found"
            }), 404
        
        # Create output path in exports directory
        exports_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"slide_{slide_number:03d}_{timestamp}.pdf"
        output_path = os.path.join(exports_dir, output_filename)
        
        # Export to PDF
        result = export_single_slide_to_pdf(slide_file, output_path)
        
        if result["success"]:
            logger.info(f"Successfully exported slide {slide_number} to PDF")
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype='application/pdf'
            )
        else:
            logger.error(f"Failed to export slide {slide_number}: {result['error']}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error exporting slide {slide_number} to PDF: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/export_pdf/all')
def export_all_slides_pdf():
    """Export all slides to a combined PDF"""
    try:
        logger.info("PDF export request for all slides")
        
        # Get all slides in the directory
        slide_files = get_all_slides_in_directory(GENERATED_SLIDES_DIR)
        
        if not slide_files:
            return jsonify({
                "success": False,
                "error": "No slides found to export"
            }), 404
        
        logger.info(f"Found {len(slide_files)} slides to export")
        
        # Create output path in exports directory
        exports_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"presentation_{timestamp}.pdf"
        output_path = os.path.join(exports_dir, output_filename)
        
        # Export all slides to combined PDF
        result = export_all_slides_in_directory(GENERATED_SLIDES_DIR, output_path)
        
        if result["success"]:
            logger.info(f"Successfully exported {len(slide_files)} slides to combined PDF")
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype='application/pdf'
            )
        else:
            logger.error(f"Failed to export slides: {result['error']}")
            return jsonify(result), 500
            
    except Exception as e:
        logger.error(f"Error exporting all slides to PDF: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/export_pdf/individual')
def export_individual_slides_pdf():
    """Export all slides to individual PDF files and return as ZIP"""
    try:
        logger.info("PDF export request for individual slides")
        
        # Get all slides in the directory
        slide_files = get_all_slides_in_directory(GENERATED_SLIDES_DIR)
        
        if not slide_files:
            return jsonify({
                "success": False,
                "error": "No slides found to export"
            }), 404
        
        logger.info(f"Found {len(slide_files)} slides to export individually")
        
        # Create temporary directory for individual PDFs
        import tempfile
        import zipfile
        
        with tempfile.TemporaryDirectory() as temp_dir:
            pdf_files = []
            
            # Export each slide individually
            for i, slide_file in enumerate(slide_files, 1):
                slide_name = os.path.splitext(os.path.basename(slide_file))[0]
                pdf_filename = f"{slide_name}.pdf"
                pdf_path = os.path.join(temp_dir, pdf_filename)
                
                result = export_single_slide_to_pdf(slide_file, pdf_path)
                if result["success"]:
                    pdf_files.append((pdf_path, pdf_filename))
                    logger.info(f"Exported slide {i} successfully")
                else:
                    logger.warning(f"Failed to export slide {i}: {result['error']}")
            
            if not pdf_files:
                return jsonify({
                    "success": False,
                    "error": "No slides could be exported"
                }), 500
            
            # Create ZIP file with all PDFs
            exports_dir = os.path.join(os.getcwd(), "exports")
            os.makedirs(exports_dir, exist_ok=True)
            
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            zip_filename = f"slides_individual_{timestamp}.zip"
            zip_path = os.path.join(exports_dir, zip_filename)
            
            with zipfile.ZipFile(zip_path, 'w', zipfile.ZIP_DEFLATED) as zip_file:
                for pdf_path, pdf_filename in pdf_files:
                    zip_file.write(pdf_path, pdf_filename)
            
            logger.info(f"Successfully created ZIP with {len(pdf_files)} individual PDFs")
            
            return send_file(
                zip_path,
                as_attachment=True,
                download_name=zip_filename,
                mimetype='application/zip'
            )
            
    except Exception as e:
        logger.error(f"Error exporting individual slides to PDF: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/export_selected_slides_pdf', methods=['POST'])
def export_selected_slides_pdf():
    """Export selected slides to a combined PDF"""
    try:
        logger.info("Export selected slides to PDF request received")
        
        data = request.get_json()
        slide_numbers = data.get('slide_numbers', [])
        
        if not slide_numbers:
            return jsonify({
                "success": False,
                "error": "No slide numbers provided"
            }), 400
        
        # Get selected slide files
        slide_files = []
        for slide_num in slide_numbers:
            slide_file = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_num:03d}.html")
            if os.path.exists(slide_file):
                slide_files.append(slide_file)
        
        if not slide_files:
            return jsonify({
                "success": False,
                "error": "None of the selected slides were found"
            }), 404
        
        # Create output path in exports directory
        exports_dir = os.path.join(os.getcwd(), "exports")
        os.makedirs(exports_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        slides_str = "_".join(map(str, sorted(slide_numbers)))
        output_filename = f"selected_slides_{slides_str}_{timestamp}.pdf"
        output_path = os.path.join(exports_dir, output_filename)
        
        # Export using our utility
        result = export_slides_to_pdf(slide_files, output_path, method="auto", combine=True)
        
        if result["success"]:
            return send_file(
                output_path,
                as_attachment=True,
                download_name=output_filename,
                mimetype='application/pdf'
            )
        else:
            return jsonify({
                "success": False,
                "error": result["error"]
            }), 500
            
    except Exception as e:
        logger.error(f"Error in export_selected_slides_pdf: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@app.route('/api/slides/available')
def get_available_slides():
    """Get list of available slides for export"""
    try:
        slide_files = get_all_slides_in_directory(GENERATED_SLIDES_DIR)
        
        slides_info = []
        for slide_file in slide_files:
            filename = os.path.basename(slide_file)
            # Extract slide number from filename (e.g., slide_001.html -> 1)
            slide_number = int(filename.split('_')[1].split('.')[0])
            
            slides_info.append({
                "slide_number": slide_number,
                "filename": filename,
                "path": slide_file,
                "exists": os.path.exists(slide_file)
            })
        
        return jsonify({
            "success": True,
            "slides": slides_info,
            "total_slides": len(slides_info)
        })
        
    except Exception as e:
        logger.error(f"Error getting available slides: {str(e)}")
        return jsonify({
            "success": False,
            "error": str(e)
        }), 500

@socketio.on('connect')
def handle_connect():
    session_id = str(uuid.uuid4())
    active_sessions[request.sid] = {
        'session_id': session_id,
        'created_at': datetime.now(),
        'slides': []
    }
    slide_queues[session_id] = queue.Queue()
    
    # Clear previous session data to ensure fresh start
    # This prevents color palette from previous presentations being reused
    try:
        palette_path = os.path.join(GENERATED_SLIDES_DIR, "color_palette.json")
        if os.path.exists(palette_path):
            os.remove(palette_path)
            logger.info(f"Cleared previous color palette for new session: {session_id}")
    except Exception as e:
        logger.warning(f"Could not clear previous color palette: {e}")
    
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
        
        if request.sid not in active_sessions:
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
                            target_element.append(content)
                        else:
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
            # Get existing workflow state from session 
            existing_state = None
            conversation_history = []
            
            # Look for session by sid to get both state and conversation history
            for sid, session_data in active_sessions.items():
                if session_data.get('session_id') == session_id:
                    existing_state = session_data.get('workflow_state')
                    # Build conversation history from workflow state messages if available
                    if existing_state and 'messages' in existing_state:
                        for msg in existing_state['messages']:
                            if hasattr(msg, 'content'):
                                conversation_history.append({
                                    'type': msg.__class__.__name__, 
                                    'content': msg.content[:200] + "..." if len(msg.content) > 200 else msg.content,
                                    'name': getattr(msg, 'name', 'unknown')
                                })
                    break
            
            # Use LLM to analyze the conversation context and determine workflow strategy
            workflow_decision = analyze_conversation_context(message, existing_state, conversation_history)
            
            logger.info(f"LLM workflow decision: {workflow_decision}")
            
            if workflow_decision['strategy'] == 'continue_existing':
                # Continue existing workflow
                logger.info(f"Continuing existing workflow: {workflow_decision['reasoning']}")
                
                # Debug: Log existing state
                if existing_state:
                    logger.info(f"DEBUG: Existing state keys: {existing_state.keys()}")
                    logger.info(f"DEBUG: is_outline_generated: {existing_state.get('is_outline_generated')}")
                    logger.info(f"DEBUG: is_outline_approved: {existing_state.get('is_outline_approved')}")
                    logger.info(f"DEBUG: structured_outline exists: {bool(existing_state.get('structured_outline'))}")
                    logger.info(f"DEBUG: plan exists: {bool(existing_state.get('plan'))}")
                else:
                    logger.warning("DEBUG: existing_state is None!")
                
                initial_state = existing_state.copy()
                initial_state["messages"].append(HumanMessage(content=message))
                
                # Apply any LLM-suggested state modifications
                if workflow_decision.get('state_modifications'):
                    logger.info(f"DEBUG: Applying state modifications: {workflow_decision['state_modifications']}")
                    initial_state.update(workflow_decision['state_modifications'])
                    logger.info(f"Applied LLM state modifications: {workflow_decision['state_modifications'].keys()}")
                    
                # Debug: Log final initial state
                logger.info(f"DEBUG: Final initial_state - is_outline_generated: {initial_state.get('is_outline_generated')}")
                logger.info(f"DEBUG: Final initial_state - is_outline_approved: {initial_state.get('is_outline_approved')}")
                logger.info(f"DEBUG: Final initial_state - plan exists: {bool(initial_state.get('plan'))}")
                    
            elif workflow_decision['strategy'] == 'modify_existing':
                # Modify existing presentation
                logger.info(f"Modifying existing presentation: {workflow_decision['reasoning']}")
                initial_state = existing_state.copy()
                
                # LLM determines what to preserve and what to regenerate
                preservation_strategy = workflow_decision.get('preservation_strategy', {})
                logger.info(f"LLM preservation strategy: {preservation_strategy}")
                
                # Apply preservation strategy
                if preservation_strategy.get('preserve_slide_count') and existing_state.get('structured_outline'):
                    existing_outline = existing_state.get('structured_outline')
                    if existing_outline and 'slides' in existing_outline:
                        slide_count = len(existing_outline['slides'])
                        initial_state['requested_slide_count'] = slide_count
                        logger.info(f"LLM preserved slide count: {slide_count}")
                
                # Reset components as determined by LLM
                reset_components = workflow_decision.get('reset_components', [])
                for component in reset_components:
                    if component == 'outline':
                        initial_state.update({
                            "is_outline_generated": False,
                            "is_outline_approved": False,
                            "structured_outline": None,
                            "outline_content": ""
                        })
                    elif component == 'slides':
                        initial_state["slides"] = []
                    elif component == 'layout':
                        initial_state["layout_instructions"] = ""
                
                initial_state.update({
                    "original_user_request": f"{message} (modification of existing presentation)",
                    "outline_attempts": 0
                })
                initial_state["messages"].append(HumanMessage(content=message))
                logger.info(f"Reset components as determined by LLM: {reset_components}")
                
            else:
                # Create new workflow
                logger.info(f"Starting new workflow: {workflow_decision['reasoning']}")
                
                # Create Langfuse trace for new workflow
                trace = create_workflow_trace(session_id, message)
                logger.info(f"Created Langfuse trace for session {session_id}")
                
                initial_state = {
                    "messages": [HumanMessage(content=message)],
                    "is_outline_generated": False,
                    "is_outline_approved": False,
                    "structured_outline": None,
                    "original_user_request": message,
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
                    # Skip agent status updates to reduce chat noise
                    pass
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
                    message_content = data.get('message', '')
                    # Only show outline approval requests, hide all other supervisor messages
                    if ('approve' in message_content and 'reject' in message_content):
                        message_type = 'supervisor'  # This will trigger the approval interface
                        socketio.emit('chat_message', {
                            'type': message_type,
                            'message': message_content,
                            'timestamp': datetime.now().isoformat()
                        }, room=client_sid)
                    # Skip all other supervisor messages (reasoning, routing, etc.)
            
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
                        "outline_approved": result.get('is_outline_approved', False),
                        "llm_decision": workflow_decision
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
                
                current_node = next_node
                
            elif current_node == "planner":
                command = planner_node(state)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                # Skip planner status messages
                # callback('agent_update', {
                #     'agent': 'planner',
                #     'status': 'Planning completed - workflow plan created',
                #     'plan_created': bool(state.get('plan'))
                # })
                
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
                
                # Skip outline agent status messages
                # callback('agent_update', {
                #     'agent': 'outline_agent',
                #     'status': 'Outline generation completed',
                #     'outline_generated': state.get('is_outline_generated', False)
                # })
                
                current_node = "supervisor"
                
            elif current_node == "artist_agent":
                command = artist_agent_node(state)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                # Skip artist agent status messages
                # callback('agent_update', {
                #     'agent': 'artist_agent',
                #     'status': 'Layout design completed',
                #     'layout_created': bool(state.get('layout_instructions'))
                # })
                
                current_node = "supervisor"
                
            elif current_node == "slide_agent":
                # Modified slide agent to stream slides
                command = slide_agent_node_with_streaming(state, callback, session_id)
                
                # Update state
                if hasattr(command, 'update'):
                    state.update(command.update)
                
                # Skip slide agent status messages
                # callback('agent_update', {
                #     'agent': 'slide_agent',
                #     'status': f'Generated {len(state.get("slides", []))} slides'
                # })
                
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
    
    # Pass user's original request to let slide generation tools analyze design requirements
    user_design_context = ""
    original_request = state.get("original_user_request", "")
    if original_request:
        user_design_context = f"User's original request: '{original_request}'"
    
    logger.info(f"Streaming slide agent user design context: {user_design_context}")
    
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
            instructions_text = f"Create slide {i} with content: {slide_content}"
            if user_design_context:
                instructions_text += f". {user_design_context}. Analyze the user's request and apply appropriate design elements."
            instructions_text += f". Use relevant research: {found_info[:2]}"
            
            # Include the full outline content so AI can see suggested images and context
            if actual_outline_str and len(actual_outline_str.strip()) > 50:
                instructions_text += f"\n\nFull presentation outline with image suggestions:\n{actual_outline_str}"
            
            slide_data = {
                "slide_number": i,
                "instructions": instructions_text,
                "images_urls": json.dumps([img for img in images[:2]]) if images else "[]",
                "style": "modern, professional presentation style with clean layout",
                "content": slide_content
            }
            
            logger.info(f"=== Calling generate_slide tool (streaming) ===")
            logger.info(f"Tool parameters:")
            logger.info(f"  - slide_number: {i}")
            logger.info(f"  - style: {slide_data['style']}")
            logger.info(f"  - content: {slide_data['content']}")
            logger.info(f"  - images_urls: {slide_data['images_urls']}")
            logger.info(f"  - instructions length: {len(instructions_text)} characters")
            logger.info(f"  - instructions preview: {instructions_text[:200]}...")
            
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

def analyze_conversation_context(current_message: str, existing_state: dict, conversation_history: list) -> dict:
    """
    Use LLM to analyze conversation context and determine workflow strategy with fallback models
    """
    
    # Build context summary
    context_summary = "No previous context"
    if existing_state:
        context_parts = []
        if existing_state.get('is_outline_generated'):
            outline_info = "Outline generated"
            if existing_state.get('structured_outline', {}).get('slides'):
                slide_count = len(existing_state['structured_outline']['slides'])
                outline_info += f" ({slide_count} slides)"
            context_parts.append(outline_info)
        
        if existing_state.get('is_outline_approved'):
            context_parts.append("Outline approved")
        
        if existing_state.get('layout_instructions'):
            context_parts.append("Layout created")
            
        if existing_state.get('slides'):
            slides_count = len(existing_state['slides'])
            context_parts.append(f"Slides generated ({slides_count} slides)")
        
        context_summary = ", ".join(context_parts) if context_parts else "Session started"
    
    # Build conversation history summary (limit to avoid token issues)
    history_summary = "No conversation history"
    if conversation_history:
        history_parts = []
        for msg in conversation_history[-3:]:  # Reduced to last 3 messages to save tokens
            msg_type = msg['type'].replace('Message', '').lower()
            name = msg.get('name', 'unknown')
            content = msg['content'][:50] + "..." if len(msg['content']) > 50 else msg['content']  # Reduced content length
            history_parts.append(f"{msg_type}({name}): {content}")
        history_summary = "\n".join(history_parts)
    
    # Create concise analysis prompt for LLM
    analysis_prompt = f"""Analyze conversation context and determine workflow strategy.

USER MESSAGE: "{current_message}"
STATE: {context_summary}
HISTORY: {history_summary}

STRATEGIES:
- "new_workflow": Start new presentation
- "continue_existing": Continue current workflow (approvals, rejections)
- "modify_existing": Modify existing presentation

RESPOND WITH JSON:
{{
    "strategy": "new_workflow|continue_existing|modify_existing",
    "reasoning": "Brief explanation",
    "preservation_strategy": {{"preserve_slide_count": boolean}},
    "reset_components": ["outline", "slides", "layout"],
    "state_modifications": {{"key": "value"}}
}}

IMPORTANT FIELD NAMES FOR state_modifications:
- To approve outline: "is_outline_approved": true
- To mark outline generated: "is_outline_generated": true  
- To reset outline: "is_outline_generated": false, "is_outline_approved": false

DETECT:
- Approval: yes, approve, ok, good, proceed (any language) → set "is_outline_approved": true
- Rejection: no, reject, redo, change (any language) → reset outline generation
- Modification: edit, modify, change, sửa lại, thay đổi"""

    # Define fallback models in order of preference
    analysis_fallbacks = [
        ("Claude", LLM_CLAUDE),
        ("Gemini", LLM_GEMINI),
        ("Gemini Flash", LLM_GEMINI_FLASH),
        ("GPT-o3", LLM_GPT_O3)
    ]
    
    # Try each LLM in fallback order
    for model_name, llm_model in analysis_fallbacks:
        try:
            logger.info(f"Trying {model_name} for conversation analysis")
            
            response = llm_model.invoke(analysis_prompt)
            response_content = response.content if hasattr(response, 'content') else str(response)
            
            # Parse JSON response
            import json
            import re
            
            # Extract JSON from response
            json_match = re.search(r'\{.*\}', response_content, re.DOTALL)
            if json_match:
                decision = json.loads(json_match.group())
                
                # Validate and provide defaults
                valid_strategies = ["new_workflow", "continue_existing", "modify_existing"]
                if decision.get("strategy") not in valid_strategies:
                    decision["strategy"] = "new_workflow"
                
                # Ensure required fields exist
                decision.setdefault("reasoning", "No reasoning provided")
                decision.setdefault("preservation_strategy", {})
                decision.setdefault("reset_components", [])
                decision.setdefault("state_modifications", {})
                
                logger.info(f"✅ {model_name} conversation analysis successful: {decision['strategy']} - {decision['reasoning']}")
                return decision
            else:
                logger.warning(f"{model_name} could not parse JSON from response")
                
        except Exception as e:
            error_msg = str(e)
            logger.warning(f"{model_name} failed for conversation analysis: {error_msg}")
            
            # Check if it's a token limit error
            if "maximum tokens" in error_msg.lower() or "token limit" in error_msg.lower() or "131072" in error_msg:
                logger.info(f"{model_name} hit token limit, trying next model")
                continue
            else:
                logger.warning(f"{model_name} failed with non-token error: {error_msg}")
                continue
    
    # All LLMs failed, use enhanced fallback logic
    logger.warning("All LLM models failed for conversation analysis, using enhanced fallback logic")
    
    # Enhanced fallback logic with better approval detection
    current_message_lower = current_message.lower().strip()
    
    # Check for approval keywords in multiple languages
    approval_keywords = [
        "approve", "yes", "ok", "good", "proceed", "continue", "accept",
        "đồng ý", "chấp nhận", "tiếp tục", "được", "ok", "yes"
    ]
    rejection_keywords = [
        "reject", "no", "redo", "regenerate", "change", "modify", "không",
        "từ chối", "làm lại", "thay đổi", "sửa lại"
    ]
    
    is_approval = any(keyword in current_message_lower for keyword in approval_keywords)
    is_rejection = any(keyword in current_message_lower for keyword in rejection_keywords)
    
    if not existing_state:
        return {
            "strategy": "new_workflow",
            "reasoning": "No existing state - starting new workflow",
            "preservation_strategy": {},
            "reset_components": [],
            "state_modifications": {}
        }
    elif existing_state.get('is_outline_generated') and not existing_state.get('is_outline_approved'):
        if is_approval:
            return {
                "strategy": "continue_existing",
                "reasoning": "User approved outline - continuing to next step",
                "preservation_strategy": {},
                "reset_components": [],
                "state_modifications": {"is_outline_approved": True}
            }
        elif is_rejection:
            return {
                "strategy": "continue_existing",
                "reasoning": "User rejected outline - regenerating",
                "preservation_strategy": {},
                "reset_components": ["outline"],
                "state_modifications": {"is_outline_generated": False, "is_outline_approved": False}
            }
        else:
            return {
                "strategy": "continue_existing", 
                "reasoning": "Outline awaiting approval - continuing existing workflow",
                "preservation_strategy": {},
                "reset_components": [],
                "state_modifications": {}
            }
    else:
        # Default to modification if there's existing state
        return {
            "strategy": "modify_existing",
            "reasoning": "Existing state found - treating as modification request",
            "preservation_strategy": {"preserve_slide_count": True},
            "reset_components": ["outline", "slides"],
            "state_modifications": {}
        }

if __name__ == '__main__':
    # Disable auto-reloader to avoid Windows socket issues
    socketio.run(app, debug=False, host='0.0.0.0', port=5000, use_reloader=False)
