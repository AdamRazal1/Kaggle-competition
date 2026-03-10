import os
import requests
import dotenv
from pydantic import BaseModel
from agno.agent import Agent
from agno.models.deepseek import DeepSeek
from agno.models.anthropic import Claude
from agno.tools import tool
from agno.skills import Skills, LocalSkills
from agno.db.sqlite import SqliteDb
from agno.tools.tavily import TavilyTools
# Load environment variables
dotenv.load_dotenv()

# Setup API Keys
deepseek_api_key = os.getenv('DEEPSEEK_API_KEY')
anthropic_api_key = os.getenv('ANTHROPIC_API_KEY')
pdf_api_key = os.getenv('PDF_ENDPOINT_API_KEY')
tavily_api_key = os.getenv('TAVILY_API_KEY')

# generate pdf tools
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

# initializing model
deepseek_model = DeepSeek(id='deepseek-reasoner', api_key=deepseek_api_key)
claude_model = Claude(id='claude-opus-4-6',api_key=anthropic_api_key)

# initializing db

db = SqliteDb('agno.db')

# initializing web_crawler
web_crawler = TavilyTools(api_key=tavily_api_key)

# creating agent
agent = Agent(
    model=deepseek_model,
    instructions=["when generating pdf, always return the pdf's link to the user", "do note that the pdf generated via html-to-pdf tool will have default configuration like this: orientation: vertical, page_size: A4, margin_top: 2cm, margin_bottom: 2cm, margin_left: 2cm, margin_right: 2cm"],
    db=db,
    update_memory_on_run=True,
    enable_agentic_memory=True,
    add_memories_to_context=True,
    tools=[generate_pdf_from_html, web_crawler],
    skills=Skills(
        loaders=[
            LocalSkills("/Users/user/Documents/Agno-agi/html-to-pdf"), 
            LocalSkills("/Users/user/Documents/Agno-agi/professional-writing-guideline")
        ]
    ),
    markdown=True,
    debug_mode=True
)

if __name__ == "__main__":
    agent.print_response(
        "Create me 4-5 pages pdf example of academic research about penguin.",
        stream=True
    )

    db.clear_memories()