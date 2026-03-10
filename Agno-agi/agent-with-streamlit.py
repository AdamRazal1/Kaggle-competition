import streamlit as st
import requests
import dotenv
import os
import re
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools import tool
from agno.skills import Skills, LocalSkills
from agno.db.sqlite import SqliteDb
from agno.tools.crawl4ai import Crawl4aiTools

# Load environment variables
dotenv.load_dotenv()

# Setup API Keys
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
pdf_api_key = os.getenv('PDF_ENDPOINT_API_KEY') 

# Generate PDF tool
@tool
def generate_pdf_from_html(html: str, filename: str) -> str:
    """
    Converts a raw HTML string into a PDF document.
    Use this tool whenever you are asked to generate, create, or save a PDF, report etc.
    
    Args:
        html (str): The raw HTML code to be converted into a PDF.
        filename (str): The name of the PDF file to be generated.
        
    Returns:
        str: A URL link to the generated PDF file, or an error message.
    """
    url = "https://api.pdfendpoint.com/v1/convert"

    payload = {
        "orientation": "vertical",
        "page_size": "A4",
        "margin_top": "2cm",
        "margin_bottom": "2cm",
        "margin_left": "2cm",
        "margin_right": "2cm",
        "filename": filename,
        "html": html,
    }

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {pdf_api_key}"
    }

    response = requests.request("POST", url, json=payload, headers=headers)

    return response.text

# Cache the agent initialization so it doesn't reload on every UI interaction
@st.cache_resource
def get_agent():
    db = SqliteDb('agno.db')

    deepseek_model = DeepSeek(id='deepseek-reasoner', api_key=deepseek_api_key)
    
    return Agent(
        model=deepseek_model,
        instructions=[
            "when generating pdf, always return the pdf's link to the user", 
            "do note that the pdf generated via html-to-pdf tool will have default configuration like this: orientation: vertical, page_size: A4, margin_top: 2cm, margin_bottom: 2cm, margin_left: 2cm, margin_right: 2cm"
        ],
        db=db,
        enable_agentic_memory=True,
        update_memory_on_run=True,
        add_memories_to_context=True,
        tools=[generate_pdf_from_html, Crawl4aiTools()],
        skills=Skills(loaders=[
            LocalSkills("/Users/user/Documents/Agno-agi/html-to-pdf"), 
            LocalSkills("/Users/user/Documents/Agno-agi/professional-writing-guideline")
        ]),
        markdown=True,
        debug_mode=True
    )

# Initialize the Agent
agent = get_agent()

# --- Streamlit User Interface ---

st.set_page_config(page_title="PDF Generator Agent", page_icon="📄")
st.title("📄 DeepSeek Document & PDF Agent")
st.markdown("Ask the agent to write a document and save it as a PDF.")

# Initialize chat history in session state
if "messages" not in st.session_state:
    st.session_state.messages = []
    st.session_state.messages.append({
        "role": "assistant", 
        "content": "Hello! I can help you draft documents like cover letters and generate them as PDFs. What would you like to create today?",
        "links": [] # Keep track of generated links
    })

# Display existing chat messages
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # If there are links saved in history, display them as buttons
        if message.get("links"):
            for link in message["links"]:
                st.link_button("View/Download PDF", link)

# React to user input
if prompt := st.chat_input("E.g., Generate a cover letter for a fresh CS grad..."):
    # 1. Display user message in the UI
    st.chat_message("user").markdown(prompt)
    
    # 2. Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 3. Generate and display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking and generating..."):
            
            # Use stream=True to get an iterator
            response_stream = agent.run(prompt, stream=True)
            
            # Create a generator to yield string chunks for Streamlit's write_stream
            def stream_text():
                for chunk in response_stream:
                    # Agno/Phidata yields RunResponse chunks, we want the .content attribute
                    if hasattr(chunk, 'content') and chunk.content:
                        yield chunk.content
                    elif isinstance(chunk, str):
                        yield chunk
            
            # st.write_stream streams the text and returns the complete final string
            full_response = st.write_stream(stream_text())
            
            # Extract URLs using regular expressions
            # This looks for standard http/https links
            urls = re.findall(r'(https?://[^\s)\]"\']+)', full_response)
            
            # Deduplicate URLs in case the agent mentions the same link twice
            unique_urls = list(dict.fromkeys(urls))
            
            # Render a link button for each URL found
            for url in unique_urls:
                st.link_button("View/Download PDF", url)
            
    # 4. Add assistant response and any extracted links to chat history
    st.session_state.messages.append({
        "role": "assistant", 
        "content": full_response,
        "links": unique_urls
    })