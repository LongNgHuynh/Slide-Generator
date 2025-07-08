# Presentation Slide Generator

An AI-powered application that automatically generates professional presentation slides using advanced language models and AI technologies.

## Features

- Generate complete presentations with just a text prompt
- AI-powered content generation using multiple LLM providers
- Advanced workflow orchestration with LangGraph
- Support for multiple AI providers (OpenAI, Anthropic, Google)
- Comprehensive logging and monitoring with Langfuse
- AWS integration capabilities

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/slide-generator.git
   cd slide-generator
   ```

2. Install uv (recommended package installer):
   ```
   # On Windows (PowerShell)
   (Invoke-WebRequest -Uri https://astral.sh/uv/install.ps1 -UseBasicParsing).Content | pwsh -Command -
   
   # On macOS/Linux
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

3. Create and activate a virtual environment using uv:
   ```
   uv venv --python=3.11
   # On Windows
   .venv\Scripts\activate
   # On macOS/Linux
   source .venv/bin/activate
   ```

4. Install dependencies using uv:
   ```
   uv pip install -e .
   ```

5. Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ANTHROPIC_API_KEY=your_anthropic_api_key
   GOOGLE_API_KEY=your_google_api_key
   LANGCHAIN_API_KEY=your_langsmith_api_key
   LANGCHAIN_PROJECT=slide-generator
   ```

## Development

To start the LangGraph development server and view your workflows in LangSmith:

1. Make sure you have set up your LangSmith API key in the `.env` file
2. Run the development server:
   ```
   langgraph dev
   ```

3. You can now view and debug your workflows in the LangSmith interface

## Project Structure

  - `workflows/` - LangGraph workflow definitions
  - `models/` - LLM model configurations and integrations
  - `utils/` - Utility functions and tools
- `generated_slides/` - Directory where generated slides are stored

## Dependencies

The project uses several key dependencies:
- LangChain and LangGraph for workflow orchestration
- Multiple LLM providers (OpenAI, Anthropic, Google)
- Langfuse for monitoring and logging
- BeautifulSoup4 for web scraping

## Requirements

- Python 3.11 or higher
- API keys for your chosen LLM providers
- Internet connection for web scraping and API access

## Running the Web Application

To use the real-time web interface for generating and viewing slides:

1. **Install dependencies** (if you haven't already):

2. **Set up your environment variables** in a `.env` file (see above for required keys).

3. **Start the Flask web app**:
   ```bash
   python app.py
   ```

4. **Open your browser** and go to [http://localhost:5000](http://localhost:5000)

- Use the chat panel to request a presentation (e.g., "Create a presentation about AI in education").
- Slides will appear in real-time on the right as they are generated.

