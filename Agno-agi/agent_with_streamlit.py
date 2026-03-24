import streamlit as st
import requests
import dotenv
import os
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.tools import tool
from agno.skills import Skills, LocalSkills
from agno.db.sqlite import SqliteDb

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
def get_agent(_knowledge_base=None):
    db = SqliteDb('agno.db')

    deepseek_model = DeepSeek(id='deepseek-reasoner', api_key=deepseek_api_key)
    
    return Agent(
        model=deepseek_model,
        instructions=[
            "always search through knowledge base for relevant information.",
            "when generating pdf, always return the pdf's link to the user", 
            "do note that the pdf generated via html-to-pdf tool will have default configuration like this: orientation: vertical, page_size: A4, margin_top: 2cm, margin_bottom: 2cm, margin_left: 2cm, margin_right: 2cm",
        ],
        db=db,
        enable_agentic_memory=True,
        update_memory_on_run=True,
        add_memories_to_context=True,
        knowledge=_knowledge_base,
        search_knowledge=True,
        tools=[generate_pdf_from_html],
        skills=Skills(loaders=[
            LocalSkills("/Users/user/Documents/GitHub/Kaggle-competition/Agno-agi/html-to-pdf"), 
            LocalSkills("/Users/user/Documents/GitHub/Kaggle-competition/Agno-agi/professional-writing-guideline")
        ]),
        markdown=True,
        debug_mode=True
    )