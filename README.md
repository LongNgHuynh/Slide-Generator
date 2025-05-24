# Presentation Slide Generator

An AI-powered web application that automatically generates professional presentation slides based on your topic.

## Features

- Generate complete presentations with just a text prompt
- AI researches your topic, finds images, and creates slides
- View slides in the browser
- Responsive web interface
- Automatic slide summaries with visual enhancement suggestions

## Setup Instructions

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/slide-generator.git
   cd slide-generator
   ```

2. Create and activate a virtual environment:
   ```
   python -m venv venv
   # On Windows
   venv\Scripts\activate
   # On macOS/Linux
   source venv/bin/activate
   ```

3. Install dependencies:
   ```
   pip install -r requirements.txt
   ```

4. Set up environment variables:
   Create a `.env` file in the project root with your API keys:
   ```
   OPENAI_API_KEY=your_openai_api_key
   ```

5. Make sure you have the required directories:
   ```
   mkdir -p generated_slides
   ```

## Running the Application

1. Start the Flask server:
   ```
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://localhost:5000
   ```

3. Enter a topic for your presentation and click "Generate Presentation"

## Project Structure

- `app.py` - Flask web application
- `new_workflow.py` - Core presentation generation logic using LangGraph
- `templates/` - HTML templates for the web interface
- `generated_slides/` - Directory where generated slides are stored
- `utils/` - Utility functions and tools
- `models/` - LLM model configurations

## Usage

1. Enter a topic or specific instructions for your presentation
2. Wait for the AI to generate your slides (this may take a few minutes)
3. View and navigate through the generated slides in the browser
4. Open individual slides in a new tab for full-screen viewing

## Requirements

- Python 3.9+
- OpenAI API key
- Internet connection (for web search and image retrieval)

## License

MIT License

