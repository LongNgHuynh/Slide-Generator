# 👨‍💻 Developer Guide - AI Slide Generator

## 📖 Tổng quan

Hướng dẫn này dành cho developers muốn đóng góp, mở rộng, hoặc hiểu sâu về codebase của AI Slide Generator.

## 🏗️ Cấu trúc project

```
Slide-Generator/
├── app.py                 # Flask server chính
├── workflow.py            # LangGraph workflow engine
├── pyproject.toml         # Project dependencies
├── models/
│   └── LLMs.py           # AI model configurations
├── utils/
│   ├── tools.py          # LangChain tools
│   ├── pdf_export.py     # PDF export functionality
│   ├── pptx_export.py    # PPTX export functionality
│   ├── schemas.py        # Pydantic data models
│   └── search.py         # Search utilities
├── templates/
│   └── index.html        # Frontend interface
├── static/
│   └── style.css         # Styling
├── rules/
│   ├── html.txt          # HTML generation rules
│   └── instruction.txt   # Slide generation instructions
├── generated_slides/     # Generated HTML slides
├── exports/              # Exported files
└── docs/                 # Documentation
```

## 🚀 Development Setup

### 1. Environment Setup

```bash
# Clone repository
git clone https://github.com/LongNgHuynh/Slide-Generator.git
cd Slide-Generator

# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows
venv\Scripts\activate
# macOS/Linux
source venv/bin/activate

# Install dependencies
pip install uv
uv sync

# Install Playwright browsers
playwright install chromium

# Install pre-commit hooks (optional)
pre-commit install
```

### 2. Environment Variables

Tạo file `.env`:

```env
# Development settings
FLASK_ENV=development
DEBUG=True

# AI Model APIs
AZURE_OPENAI_API_KEY=your_key
AZURE_OPENAI_ENDPOINT=your_endpoint
GOOGLE_API_KEY=your_key
AWS_ACCESS_KEY_ID=your_key
AWS_SECRET_ACCESS_KEY=your_secret
AWS_REGION_NAME=us-east-1

# Optional services
CONVERTAPI_API_KEY=your_key
LANGFUSE_SECRET_KEY=your_key
LANGFUSE_PUBLIC_KEY=your_key

# Search settings
SEARXNG_URL=https://searx.be
```

### 3. IDE Configuration

#### VS Code Settings (`.vscode/settings.json`)
```json
{
  "python.defaultInterpreter": "./venv/bin/python",
  "python.linting.enabled": true,
  "python.linting.pylintEnabled": true,
  "python.formatting.provider": "black",
  "python.sortImports.path": "isort",
  "editor.formatOnSave": true,
  "files.exclude": {
    "**/__pycache__": true,
    "**/*.pyc": true,
    ".venv/": true
  }
}
```

## 🔧 Core Components

### 1. Flask Application (`app.py`)

#### Structure
```python
# Main components
app = Flask(__name__)
socketio = SocketIO(app, cors_allowed_origins="*")

# Session management
active_sessions = {}
slide_queues = {}

# Socket.IO event handlers
@socketio.on('send_message')
def handle_message(data):
    # Process user messages
    pass

@socketio.on('edit_text')
def handle_text_edit(data):
    # Handle text editing
    pass
```

#### Adding New Endpoints
```python
@app.route('/api/new-feature')
def new_feature():
    """Add new REST endpoint"""
    try:
        # Implementation
        return jsonify({"success": True, "data": result})
    except Exception as e:
        return jsonify({"success": False, "error": str(e)}), 500

@socketio.on('new_event')
def handle_new_event(data):
    """Add new Socket.IO event"""
    try:
        # Process event
        emit('response_event', response_data)
    except Exception as e:
        emit('error', {'message': str(e)})
```

### 2. Workflow Engine (`workflow.py`)

#### LangGraph State Management
```python
class AgentState(TypedDict):
    """Extend state for new features"""
    messages: Annotated[List[BaseMessage], add_messages]
    # Core fields
    is_outline_generated: bool
    is_outline_approved: bool
    # Add new fields
    new_feature_enabled: bool
    custom_data: Optional[dict]

def new_agent_node(state: AgentState) -> Command:
    """Create new agent"""
    # Agent logic
    return Command(
        update={"new_field": value},
        goto="next_agent"
    )

# Add to graph
graph.add_node("new_agent", new_agent_node)
graph.add_edge("supervisor", "new_agent")
```

#### Agent Development Pattern
```python
def create_agent_template(state: AgentState) -> Command:
    """Template for new agents"""
    
    # 1. Create Langfuse span
    trace = get_current_trace()
    agent_span = trace.span(
        name="new_agent",
        input={"state_summary": "relevant_state"},
        metadata={"agent": "new_agent"}
    )
    
    try:
        # 2. Process state
        result = process_agent_logic(state)
        
        # 3. Update state
        updated_state = {
            "messages": [AIMessage(content="Result", name="new_agent")],
            "new_field": result
        }
        
        # 4. Complete span
        agent_span.end(output={"success": True, "result": result})
        
        return Command(update=updated_state, goto="supervisor")
        
    except Exception as e:
        agent_span.end(output={"success": False, "error": str(e)})
        raise
```

### 3. Tools Development (`utils/tools.py`)

#### Creating New Tools
```python
@tool
def new_research_tool(query: str, options: Optional[dict] = None) -> dict:
    """
    Template for new research tool
    
    Args:
        query: Research query
        options: Additional options
        
    Returns:
        Dictionary with results
    """
    logger.info(f"=== new_research_tool called ===")
    logger.info(f"Query: {query}")
    
    try:
        # Tool implementation
        results = perform_research(query, options)
        
        # Save results for debugging
        record = {
            "query": query,
            "results": results,
            "timestamp": datetime.now().isoformat()
        }
        
        with open(os.path.join(OUTPUT_DIR, "new_tool_record.json"), "w") as f:
            json.dump(record, f, indent=2)
        
        logger.info(f"Found {len(results)} results")
        return record
        
    except Exception as e:
        logger.error(f"Error in new_research_tool: {str(e)}")
        return {"query": query, "results": [], "error": str(e)}

# Register tool with agents
research_agent = create_react_agent(
    model=LLM,
    tools=[web_search, image_search, new_research_tool],  # Add new tool
    prompt=agent_prompt
)
```

#### Tool Testing
```python
def test_new_tool():
    """Test new tool functionality"""
    result = new_research_tool.invoke({"query": "test query"})
    assert result["query"] == "test query"
    assert "results" in result
    print(f"Tool test passed: {result}")

if __name__ == "__main__":
    test_new_tool()
```

### 4. Model Integration (`models/LLMs.py`)

#### Adding New AI Models
```python
class NewAIModel(BaseModel):
    """Template for new AI model integration"""
    
    def __init__(self, **kwargs):
        super().__init__(
            api_key=os.getenv("NEW_AI_API_KEY"),
            base_url=os.getenv("NEW_AI_BASE_URL"),
            model="new-model-name",
            **kwargs
        )
    
    def invoke(self, prompt: str, config=None) -> str:
        """Override invoke method if needed"""
        try:
            response = super().invoke(prompt, config)
            return response
        except Exception as e:
            logger.error(f"NewAIModel error: {e}")
            raise

# Add to fallback list
LLM_FALLBACKS.append(("New AI Model", NewAIModel()))
```

#### Model Testing
```python
def test_model_integration():
    """Test new model integration"""
    model = NewAIModel()
    response = model.invoke("Hello, how are you?")
    assert len(response) > 0
    print(f"Model test passed: {response[:100]}...")

if __name__ == "__main__":
    test_model_integration()
```

## 🎨 Frontend Development

### 1. HTML Structure (`templates/index.html`)

#### Adding New UI Components
```html
<!-- New feature panel -->
<div class="new-feature-panel">
    <div class="panel-header">
        <span>🆕 New Feature</span>
    </div>
    <div class="feature-content" id="newFeatureContent">
        <!-- Feature content -->
    </div>
    <div class="feature-controls">
        <button id="newFeatureBtn" onclick="handleNewFeature()">
            Activate Feature
        </button>
    </div>
</div>
```

#### JavaScript Event Handling
```javascript
// Socket.IO events
socket.on('new_feature_update', (data) => {
    updateNewFeature(data);
});

// UI interactions
function handleNewFeature() {
    const data = collectNewFeatureData();
    socket.emit('new_feature_request', data);
}

function updateNewFeature(data) {
    const content = document.getElementById('newFeatureContent');
    content.innerHTML = renderNewFeatureData(data);
}
```

### 2. Styling (`static/style.css`)

#### CSS Organization
```css
/* New feature styles */
.new-feature-panel {
    /* Base styles */
    display: flex;
    flex-direction: column;
    
    /* Feature-specific styles */
    background: var(--bg-secondary);
    border-radius: 8px;
    padding: 1rem;
}

.new-feature-panel .panel-header {
    /* Header styles */
    font-weight: bold;
    color: var(--text-primary);
}

/* Responsive design */
@media (max-width: 768px) {
    .new-feature-panel {
        padding: 0.5rem;
    }
}
```

## 🧪 Testing

### 1. Unit Tests

#### Test Structure
```python
# tests/test_tools.py
import pytest
from utils.tools import web_search, image_search

def test_web_search():
    """Test web search functionality"""
    result = web_search.invoke("AI technology")
    assert result["success"] == True
    assert len(result["results"]) > 0

def test_image_search():
    """Test image search functionality"""
    result = image_search.invoke("machine learning")
    assert "results" in result
    assert isinstance(result["results"], list)

# tests/test_workflow.py
import pytest
from workflow import supervisor_node, AgentState

def test_supervisor_routing():
    """Test supervisor routing logic"""
    state = AgentState(
        messages=[],
        is_outline_generated=False,
        is_outline_approved=False
    )
    
    command = supervisor_node(state)
    assert command.goto in ["planner", "outline_agent", "FINISH"]
```

#### Running Tests
```bash
# Install test dependencies
pip install pytest pytest-asyncio pytest-cov

# Run all tests
pytest

# Run with coverage
pytest --cov=utils --cov=models --cov-report=html

# Run specific test file
pytest tests/test_tools.py -v

# Run with debugging
pytest tests/test_workflow.py -v -s
```

### 2. Integration Tests

```python
# tests/test_integration.py
import pytest
from app import app, socketio
from flask_socketio import SocketIOTestClient

class TestSlideGeneration:
    """Integration tests for slide generation"""
    
    def setup_method(self):
        """Setup test client"""
        self.client = app.test_client()
        self.socketio_client = SocketIOTestClient(app, socketio)
    
    def test_complete_workflow(self):
        """Test complete slide generation workflow"""
        # Connect to Socket.IO
        received = self.socketio_client.get_received()
        
        # Send message to generate slides
        self.socketio_client.emit('send_message', {
            'message': 'Create presentation about testing'
        })
        
        # Check for responses
        received = self.socketio_client.get_received()
        assert len(received) > 0
        
        # Verify slide generation
        # Additional assertions...
    
    def test_export_functionality(self):
        """Test PDF/PPTX export"""
        response = self.client.get('/api/slides/available')
        assert response.status_code == 200
        
        data = response.get_json()
        if data["total_slides"] > 0:
            response = self.client.get('/export_pdf/all')
            assert response.status_code == 200
            assert response.content_type == 'application/pdf'
```

### 3. Performance Tests

```python
# tests/test_performance.py
import time
import pytest
from utils.tools import web_search

def test_search_performance():
    """Test search tool performance"""
    start_time = time.time()
    result = web_search.invoke("performance testing")
    end_time = time.time()
    
    assert end_time - start_time < 10.0  # Should complete within 10 seconds
    assert len(result["results"]) > 0

@pytest.mark.performance
def test_concurrent_requests():
    """Test concurrent request handling"""
    import concurrent.futures
    
    def make_request():
        return web_search.invoke("concurrent test")
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=3) as executor:
        futures = [executor.submit(make_request) for _ in range(3)]
        results = [future.result() for future in futures]
    
    assert all(result["success"] for result in results)
```

## 🔍 Debugging

### 1. Logging Configuration

```python
# Enhanced logging setup
import logging
from datetime import datetime

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'debug_{datetime.now().strftime("%Y%m%d")}.log'),
        logging.StreamHandler()
    ]
)

# Module-specific loggers
logger = logging.getLogger(__name__)
workflow_logger = logging.getLogger('workflow')
tools_logger = logging.getLogger('tools')

# Usage in code
logger.debug(f"Processing request: {data}")
workflow_logger.info(f"Agent {agent_name} starting")
tools_logger.error(f"Tool failed: {error}", exc_info=True)
```

### 2. Debug Utilities

```python
# utils/debug.py
import json
import traceback
from datetime import datetime

def debug_state(state: AgentState, stage: str):
    """Debug workflow state"""
    debug_info = {
        "stage": stage,
        "timestamp": datetime.now().isoformat(),
        "state_keys": list(state.keys()),
        "message_count": len(state.get("messages", [])),
        "outline_generated": state.get("is_outline_generated", False),
        "outline_approved": state.get("is_outline_approved", False),
        "slides_count": len(state.get("slides", []))
    }
    
    with open(f"debug_state_{stage}.json", "w") as f:
        json.dump(debug_info, f, indent=2)
    
    print(f"🐛 Debug: {stage} - {debug_info}")

def debug_exception(e: Exception, context: str = ""):
    """Debug exception with full context"""
    error_info = {
        "error": str(e),
        "type": type(e).__name__,
        "context": context,
        "traceback": traceback.format_exc(),
        "timestamp": datetime.now().isoformat()
    }
    
    with open("debug_errors.json", "a") as f:
        json.dump(error_info, f, indent=2)
        f.write("\n")
    
    print(f"🚨 Error: {context} - {str(e)}")
```

### 3. Development Tools

```python
# Development server with hot reload
if __name__ == '__main__':
    import os
    if os.getenv('FLASK_ENV') == 'development':
        socketio.run(
            app,
            debug=True,
            host='0.0.0.0',
            port=5000,
            use_reloader=True,
            reloader_options={'extra_files': ['rules/html.txt', 'rules/instruction.txt']}
        )
    else:
        socketio.run(app, debug=False, host='0.0.0.0', port=5000)
```

## 📦 Deployment

### 1. Docker Setup

```dockerfile
# Dockerfile
FROM python:3.11-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \
    chromium \
    chromium-driver \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml uv.lock ./
RUN pip install uv && uv sync --frozen

# Copy application code
COPY . .

# Install Playwright browsers
RUN playwright install chromium

EXPOSE 5000

CMD ["python", "app.py"]
```

```yaml
# docker-compose.yml
version: '3.8'

services:
  slide-generator:
    build: .
    ports:
      - "5000:5000"
    environment:
      - FLASK_ENV=production
      - AZURE_OPENAI_API_KEY=${AZURE_OPENAI_API_KEY}
      - GOOGLE_API_KEY=${GOOGLE_API_KEY}
      - AWS_ACCESS_KEY_ID=${AWS_ACCESS_KEY_ID}
      - AWS_SECRET_ACCESS_KEY=${AWS_SECRET_ACCESS_KEY}
    volumes:
      - ./exports:/app/exports
      - ./generated_slides:/app/generated_slides
    restart: unless-stopped
```

### 2. Production Configuration

```python
# config.py
import os

class Config:
    SECRET_KEY = os.getenv('SECRET_KEY', 'dev-secret-key')
    DEBUG = False
    TESTING = False

class DevelopmentConfig(Config):
    DEBUG = True
    
class ProductionConfig(Config):
    DEBUG = False
    # Production-specific settings
    
class TestingConfig(Config):
    TESTING = True

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'testing': TestingConfig,
    'default': DevelopmentConfig
}
```

## 🔧 Best Practices

### 1. Code Style

```python
# Follow PEP 8 and type hints
from typing import Dict, List, Optional, Union

def process_slides(
    slides: List[Dict[str, str]], 
    options: Optional[Dict[str, Union[str, int]]] = None
) -> Dict[str, bool]:
    """
    Process slides with proper type hints
    
    Args:
        slides: List of slide dictionaries
        options: Optional processing options
        
    Returns:
        Processing results
    """
    if options is None:
        options = {}
    
    # Implementation
    return {"success": True, "processed": len(slides)}
```

### 2. Error Handling

```python
class SlideGenerationError(Exception):
    """Custom exception for slide generation errors"""
    pass

def safe_ai_call(prompt: str, model, retries: int = 3) -> str:
    """AI call with proper error handling"""
    for attempt in range(retries):
        try:
            response = model.invoke(prompt)
            return response.content
        except Exception as e:
            logger.warning(f"AI call attempt {attempt + 1} failed: {e}")
            if attempt == retries - 1:
                raise SlideGenerationError(f"AI call failed after {retries} attempts: {e}")
            time.sleep(2 ** attempt)  # Exponential backoff
```

### 3. Performance Optimization

```python
import asyncio
from functools import lru_cache

@lru_cache(maxsize=100)
def cached_research(query: str) -> dict:
    """Cache research results"""
    return perform_expensive_research(query)

async def parallel_processing(tasks: List[str]) -> List[dict]:
    """Process tasks in parallel"""
    async def process_task(task: str) -> dict:
        return await async_process(task)
    
    results = await asyncio.gather(*[process_task(task) for task in tasks])
    return results
```

## 🤝 Contributing

### 1. Pull Request Process

1. **Fork** repository
2. **Create feature branch**: `git checkout -b feature/amazing-feature`
3. **Write tests** cho new functionality
4. **Update documentation** if needed
5. **Run tests**: `pytest`
6. **Check code style**: `black . && isort .`
7. **Commit changes**: `git commit -m 'Add amazing feature'`
8. **Push to branch**: `git push origin feature/amazing-feature`
9. **Create Pull Request**

### 2. Code Review Checklist

- [ ] Code follows project style guidelines
- [ ] Tests are included and passing
- [ ] Documentation is updated
- [ ] No breaking changes (or properly documented)
- [ ] Error handling is proper
- [ ] Performance impact is considered
- [ ] Security implications are reviewed

### 3. Release Process

```bash
# 1. Update version in pyproject.toml
# 2. Create changelog
# 3. Tag release
git tag -a v1.2.0 -m "Release version 1.2.0"
git push origin v1.2.0

# 4. Build and test
uv build
pytest

# 5. Deploy to production
```

---

**Happy coding! 🚀** 