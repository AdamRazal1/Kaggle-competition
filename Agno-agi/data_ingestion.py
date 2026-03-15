from llama_cloud import AsyncLlamaCloud, LlamaCloud

import os
import httpx
import re
from dotenv import load_dotenv
load_dotenv()

LLAMA_CLOUD_API_KEY = os.getenv('LLAMA_CLOUD_API_KEY')

client = LlamaCloud(api_key=LLAMA_CLOUD_API_KEY)

# Upload and parse a document
file_obj = client.files.create(file="./content/input/test.pdf", purpose="parse")

result = client.parsing.parse(
    file_id=file_obj.id,
    tier="cost_effective",
    version="latest",

    # Options specific to the input file type, e.g. html, spreadsheet, presentation, etc.
    input_options={},

    # Control the output structure and markdown styling
    output_options={
        "markdown": {
            "tables": {
                "output_tables_as_markdown": False,
            },
        },
        # Saving images for later retrieval
        "images_to_save": ["screenshot", 'embedded', 'layout'],
    },

    # Options for controlling how we process the document
    processing_options={
        "ignore": {
            "ignore_diagonal_text": True,
        },
        "ocr_parameters": {
            "languages": ["fr"]
        }
    },

    # Parsed content to include in the returned response
    expand=["markdown", "items", "metadata"],
)

print(result.markdown.pages[0])
