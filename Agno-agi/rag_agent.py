import streamlit as st
from agno.agent import Agent
from data_ingestion import data_ingestion
from agent_with_streamlit import get_agent

import time
import os
import tempfile

# UI Config
st.set_page_config(page_title="RAG Agent", layout="centered")
st.title("Agentic Rag for Agno Agent")
st.markdown("Parse and embed via LlamaParse and Qdrant, then generate response via Agno Agent.")

# API Key Inputs
with st.sidebar:
    llama_cloud_api_key = st.text_input("LlamaCloud API Key", type="password")
    qdrant_api_key = st.text_input("Qdrant API Key", type="password")

uploaded_file = st.file_uploader("Upload PDF Document", type=["pdf"])

# Document Ingestion
if uploaded_file and llama_cloud_api_key and qdrant_api_key:
    if st.button("Starting Document Ingestion"):    
        with tempfile.NamedTemporaryFile(delete=False, suffix='.pdf') as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_file_path = tmp_file.name

        # time counter
        start_time = time.time()

        # initialise and store knowledge base into session state
        with st.spinner("Ingesting document..."):
            knowledge_base = data_ingestion(file=tmp_file_path, llama_api_key=llama_cloud_api_key, qdrant_api_key=qdrant_api_key)
            st.session_state['knowledge_base'] = knowledge_base
            
            # Clear chat history when a new document is uploaded
            if "messages" in st.session_state:
                st.session_state.messages = []

            # time counter
            end_time = time.time()
            elapsed_time = end_time - start_time

            # Calculate minutes and seconds
            minutes, seconds = divmod(elapsed_time, 60)
            
            # Format the success message dynamically
            if minutes > 0:
                time_str = f"{int(minutes)}m {seconds:.1f}s"
            else:
                time_str = f"{seconds:.2f} seconds"

            st.success(f"Document Ingestion Time: {time_str} seconds")
                
        # remove temporary file from the disk
        os.remove(tmp_file_path)

# --- Agent Interaction (Updated to Chatbot UI) ---

if "knowledge_base" in st.session_state:
    st.divider()
    st.subheader("Chat with Agent")

    agent = get_agent(st.session_state['knowledge_base'])

    # Initialize chat history in session state
    if "messages" not in st.session_state or not st.session_state.messages:
        st.session_state.messages = [
            {"role": "assistant", "content": "Hello! I have reviewed your document. What would you like to know?"}
        ]

    # Display existing chat messages
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # React to user input
    if prompt := st.chat_input("Ask a question about your uploaded document..."):
        # 1. Display user message in the UI
        with st.chat_message("user"):
            st.markdown(prompt)
        
        # 2. Add user message to chat history
        st.session_state.messages.append({"role": "user", "content": prompt})

        # 3. Generate and display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Use stream=True to get an iterator
                    response_stream = agent.run(prompt, stream=True)
                    
                    # Create a generator to yield string chunks for Streamlit's write_stream
                    def stream_text():
                        for chunk in response_stream:
                            if hasattr(chunk, 'content') and chunk.content:
                                yield chunk.content
                            elif isinstance(chunk, str):
                                yield chunk
                    
                    # st.write_stream streams the text and returns the complete final string
                    full_response = st.write_stream(stream_text())
                    
                    # 4. Add assistant response to chat history
                    st.session_state.messages.append({
                        "role": "assistant", 
                        "content": full_response
                    })
                except Exception as e:
                    st.error(f"An error occurred while querying the agent: {e}")