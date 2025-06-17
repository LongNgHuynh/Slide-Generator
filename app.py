from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import json
import threading
import uuid
from datetime import datetime
import os
import logging
from workflow import app as workflow_app, AgentState
from langchain_core.messages import HumanMessage
import asyncio
import queue

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

@socketio.on('disconnect')
def handle_disconnect():
    if request.sid in active_sessions:
        session_id = active_sessions[request.sid]['session_id']
        del active_sessions[request.sid]
        if session_id in slide_queues:
            del slide_queues[session_id]
    logger.info(f"Client disconnected: {request.sid}")

@socketio.on('edit_text')
def handle_text_edit(data):
    """Handle individual text element editing requests from frontend"""
    slide_number = data.get('slide_number')
    text_id = data.get('text_id')
    new_text = data.get('new_text', '')
    original_text = data.get('original_text', '')
    session_id = active_sessions.get(request.sid, {}).get('session_id')
    client_sid = request.sid
    
    if not session_id:
        emit('error', {'message': 'Session not found'})
        return
    
    if not slide_number or not new_text.strip() or text_id is None:
        emit('error', {'message': 'Invalid text edit data'})
        return
    
    logger.info(f"Handling text edit request for slide {slide_number}, text element {text_id}")
    
    # Process text edit in background thread
    def process_text_edit():
        try:
            # Read current slide HTML
            slide_file_path = os.path.join(GENERATED_SLIDES_DIR, f"slide_{slide_number:03d}.html")
            current_html = ""
            
            if os.path.exists(slide_file_path):
                with open(slide_file_path, 'r', encoding='utf-8') as f:
                    current_html = f.read()
            else:
                # If file doesn't exist, get from session data
                if session_id in active_sessions:
                    session_slides = active_sessions[session_id].get('slides', [])
                    for slide in session_slides:
                        if slide.get('slide_number') == slide_number:
                            current_html = slide.get('content', '')
                            break
            
            if not current_html:
                raise Exception("Could not find slide content to edit")
            
            # Update the specific text element in the HTML
            updated_html = update_text_in_html(current_html, text_id, original_text, new_text)
            
            # Save updated slide to file
            with open(slide_file_path, 'w', encoding='utf-8') as f:
                f.write(updated_html)
            
            # Update session slides data
            if session_id in active_sessions:
                session_slides = active_sessions[session_id].get('slides', [])
                for slide in session_slides:
                    if slide.get('slide_number') == slide_number:
                        slide['content'] = updated_html
                        break
            
            # Send updated slide back to client
            socketio.emit('text_updated', {
                'slide_number': slide_number,
                'content': updated_html,
                'timestamp': datetime.now().isoformat()
            }, room=client_sid)
            
            logger.info(f"Successfully updated text in slide {slide_number}")
            
        except Exception as e:
            logger.error(f"Error updating text in slide {slide_number}: {str(e)}")
            socketio.emit('error', {
                'message': f'Failed to update text in slide {slide_number}: {str(e)}'
            }, room=client_sid)
    
    # Start processing in background
    thread = threading.Thread(target=process_text_edit)
    thread.daemon = True
    thread.start()
    
    # Send acknowledgment
    emit('chat_message', {
        'type': 'system',
        'message': f'Processing text edit for slide {slide_number}...',
        'timestamp': datetime.now().isoformat()
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

def update_text_in_html(html_content, text_id, original_text, new_text):
    """
    Update a specific text element in HTML content based on text_id and original text
    """
    from bs4 import BeautifulSoup
    import re
    
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
                # Replace the text content while preserving HTML structure
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
                        if element.string:
                            element.string.replace_with(new_text)
                        else:
                            element.clear()
                            element.append(new_text)
                        logger.info(f"Updated text element by content match: '{original_text}' -> '{new_text}'")
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
            # Create initial state for workflow
            initial_state = {
                "messages": [HumanMessage(content=message)],
                "is_outline_generated": False,
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
            
            # Run the modified workflow with streaming
            result = run_workflow_with_streaming(initial_state, stream_callback, session_id)
            
            # Send final completion message
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
    from workflow import supervisor_node, planner_node, outline_agent_node, artist_agent_node, slide_agent_node
    
    state = initial_state
    current_node = "supervisor"
    max_iterations = 50
    iteration = 0
    
    while current_node != "FINISH" and iteration < max_iterations:
        iteration += 1
        logger.info(f"Iteration {iteration}: Running node {current_node}")
        
        try:
            if current_node == "supervisor":
                command = supervisor_node(state)
                next_node = command.goto if hasattr(command, 'goto') else command.get('next', 'FINISH')
                
                # Update state with supervisor decision
                if hasattr(command, 'update'):
                    state.update(command.update)
                
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
                break
                
        except Exception as e:
            logger.error(f"Error in node {current_node}: {str(e)}")
            callback('error', {'message': str(e)})
            break
    
    callback('workflow_complete', {
        'total_slides': len(state.get('slides', [])),
        'iterations': iteration
    })
    
    return state

def slide_agent_node_with_streaming(state, callback, session_id):
    """Modified slide agent that streams slides as they're generated"""
    from workflow import slide_agent_node
    from utils.tools import generate_slide
    import json
    import re
    
    logger.info("Slide agent with streaming: Starting slide generation")
    
    # Get the outline content
    actual_outline_str = ""
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
    
    # Ensure we have at least the requested number of slides for the user request
    # Check if user requested a specific number
    user_request = ""
    if state.get("messages"):
        for msg in state["messages"]:
            if hasattr(msg, 'type') and msg.type == 'human':
                user_request = msg.content.lower()
                break
    
    logger.info(f"User request for slide count analysis: {user_request}")
    
    # Extract number of slides requested with multiple patterns
    import re as regex
    requested_slides = None
    
    # Pattern 1: "5 slides", "make 3 slides", "create 7 slides"
    slide_count_match = regex.search(r'(\d+)\s*slides?', user_request)
    if slide_count_match:
        requested_slides = int(slide_count_match.group(1))
    
    # Pattern 2: "presentation with 5", "5-slide presentation"
    if not requested_slides:
        alt_match = regex.search(r'(\d+)[-\s]*slide', user_request)
        if alt_match:
            requested_slides = int(alt_match.group(1))
    
    # Pattern 3: "about 5", "around 3"
    if not requested_slides:
        about_match = regex.search(r'(?:about|around|roughly)\s*(\d+)', user_request)
        if about_match:
            requested_slides = int(about_match.group(1))
    
    # Default: if no specific number requested, use a reasonable default
    if not requested_slides:
        requested_slides = max(5, len(slide_lines))  # At least 5 slides, or more if outline has more
    
    logger.info(f"Requested slides: {requested_slides}, Current slide_lines: {len(slide_lines)}")
    
    # Ensure we generate the requested number of slides
    if len(slide_lines) < requested_slides:
        # Expand content to reach requested number
        base_content = slide_lines if slide_lines else [
            "Introduction and Overview",
            "Background Information", 
            "Main Topic Analysis",
            "Key Points Discussion",
            "Examples and Case Studies",
            "Benefits and Advantages",
            "Challenges and Solutions",
            "Future Implications", 
            "Recommendations",
            "Conclusion and Summary"
        ]
        
        # Duplicate and expand content
        original_count = len(slide_lines)
        content_index = 0
        while len(slide_lines) < requested_slides:
            if content_index < len(base_content):
                new_content = base_content[content_index % len(base_content)]
                slide_lines.append(f"{new_content}")
                content_index += 1
            else:
                # If we run out of base content, create generic slides
                slide_num = len(slide_lines) + 1
                slide_lines.append(f"Additional Topic Discussion - Part {slide_num - original_count}")
    
    # Limit to requested number (in case we had too many)
    slide_lines = slide_lines[:requested_slides]
    logger.info(f"Final slide count: {len(slide_lines)} slides to generate")
    
    generated_slides_info = []
    
    # Generate slides one by one and stream them
    for i, slide_content in enumerate(slide_lines, 1):
        try:
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
    from langchain_core.messages import AIMessage
    
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
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
