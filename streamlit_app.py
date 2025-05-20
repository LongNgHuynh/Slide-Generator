import streamlit as st
import logging
import json
import os
import pprint
from langchain_core.messages import HumanMessage
from new_workflow import app, config, logger, AgentState

# Configure page
st.set_page_config(page_title="Slide Generator", layout="wide")

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "results" not in st.session_state:
    st.session_state.results = None
if "is_processing" not in st.session_state:
    st.session_state.is_processing = False

# App header
st.title("AI Presentation Slide Generator")
st.markdown("Generate professional presentation slides using AI")

# Main UI
with st.container():
    col1, col2 = st.columns([2, 1])
    
    with col1:
        user_input = st.text_area("Enter your presentation topic and instructions:", height=150)
        
        # Process button
        if st.button("Generate Presentation", type="primary", disabled=st.session_state.is_processing):
            if user_input:
                with st.spinner("Generating slides..."):
                    st.session_state.is_processing = True
                    
                    # Create initial state
                    initial_state = {
                        "messages": [HumanMessage(content=user_input)],
                        "outline": "",
                        "images": [],
                        "found_information": [],
                        "input": user_input,
                        "slides": [],
                        "summary": []
                    }
                    
                    try:
                        # Execute workflow
                        result = app.invoke(initial_state, config=config)
                        st.session_state.results = result
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        st.session_state.messages.append({"role": "system", "content": "Presentation generated successfully!"})
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
                        st.session_state.messages.append({"role": "user", "content": user_input})
                        st.session_state.messages.append({"role": "system", "content": f"Error: {str(e)}"})
                        logger.error(f"Error in workflow execution: {str(e)}", exc_info=True)
                    
                    st.session_state.is_processing = False
                    st.rerun()
    
    # Display conversation history
    with col2:
        st.subheader("Conversation")
        for msg in st.session_state.messages:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

# Display results if available
if st.session_state.results:
    st.header("Generated Presentation")
    
    # Display summary if available
    if st.session_state.results.get("summary"):
        st.subheader("Presentation Summary")
        st.write(st.session_state.results["summary"])
    
    # Create tabs for different sections
    tab1, tab2, tab3, tab4 = st.tabs(["Slides", "Outline", "Images", "Debug Info"])
    
    with tab1:
        slides = st.session_state.results.get("slides", [])
        if slides:
            for slide in slides:
                with st.expander(f"Slide {slide.get('slide_number', 'Unknown')}"):
                    st.write(slide.get("content", "No content available"))
            
            # Display slides from files
            slide_files = sorted([f for f in os.listdir("generated_slides") if f.endswith(".html")])
            if slide_files:
                st.subheader("Preview Slides")
                selected_slide = st.selectbox("Select a slide to view", slide_files)
                if selected_slide:
                    slide_path = os.path.join("generated_slides", selected_slide)
                    with open(slide_path, "r", encoding="utf-8") as f:
                        html_content = f.read()
                    st.components.v1.html(html_content, height=600)
        else:
            st.info("No slides have been generated yet.")
    
    with tab2:
        outline = st.session_state.results.get("outline", "")
        if outline:
            st.write(outline)
        else:
            st.info("No outline has been generated yet.")
    
    with tab3:
        images = st.session_state.results.get("images", [])
        if images:
            cols = st.columns(3)
            for i, img in enumerate(images):
                if "url" in img:
                    cols[i % 3].image(img["url"], width=200)
        else:
            st.info("No images have been found yet.")
    
    with tab4:
        st.json(st.session_state.results)

# Footer
st.markdown("---")
st.markdown("Powered by LangGraph and GPT-4o") 