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
    from workflow import supervisor_node, outline_agent_node, slide_agent_node
    
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
    
    # Parse outline to identify slides
    slide_lines = []
    for line in actual_outline_str.split('\n'):
        if re.search(r'slide\s+\d+|^\d+\.|^-\s*\w+', line.strip(), re.IGNORECASE):
            slide_lines.append(line.strip())
    
    if not slide_lines:
        # Fallback: create slides based on sections
        sections = [s.strip() for s in actual_outline_str.split('\n') if s.strip()]
        slide_lines = sections[:10]  # Limit to 10 slides
    
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
    
    # Return command to update state
    return type('Command', (), {
        'update': {
            "messages": [type('HumanMessage', (), {
                'content': f"Generated {len(generated_slides_info)} slides",
                'name': 'slide_agent'
            })()],
            "slides": generated_slides_info
        },
        'goto': "supervisor"
    })()

if __name__ == '__main__':
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
