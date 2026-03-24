import os
import tempfile
import logging
from llama_cloud import LlamaCloud
from agno.knowledge.knowledge import Knowledge
from agno.knowledge.reranker.cohere import CohereReranker
from agno.vectordb.qdrant import Qdrant, SearchType
from agno.knowledge.embedder.fastembed import FastEmbedEmbedder
from agno.knowledge.reader.markdown_reader import MarkdownReader
from agno.knowledge.chunking.markdown import MarkdownChunking

# --- Configure the Logger ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    datefmt='%H:%M:%S' # Keeps the timestamp concise
)
logger = logging.getLogger(__name__)

def data_ingestion(file: str, llama_api_key: str, qdrant_api_key: str) -> Knowledge:
    logger.info(f"Starting data ingestion for file: {file}")

    # 1. Parse Document with LlamaCloud
    logger.info("Initializing LlamaCloud client and uploading file...")
    client = LlamaCloud(api_key=llama_api_key)
    file_obj = client.files.create(file=file, purpose="parse")
    
    logger.info("Parsing document with LlamaCloud (this may take a moment)...")
    result = client.parsing.parse(
        file_id=file_obj.id,
        tier="cost_effective",
        version="latest",
        input_options={},
        output_options={
            "markdown": {
                "tables": {"output_tables_as_markdown": False},
            },
            "images_to_save": ["screenshot", 'embedded', 'layout'],
        },
        processing_options={
            "ignore": {"ignore_diagonal_text": True},
            "ocr_parameters": {"languages": ["fr"]}
        },
        expand=["markdown_full", "metadata"],
    )

    text_content = result.markdown_full
    text_metadata = result.metadata.model_dump()
    logger.info("Successfully parsed document and extracted metadata.")

    # 2. Initialize Vector DB & Knowledge Base
    logger.info("Connecting to Qdrant Vector DB...")
    vector_db = Qdrant(
        api_key=qdrant_api_key,
        url='https://d24e0cea-2938-4359-977c-fc39ab6375b8.sa-east-1-0.aws.cloud.qdrant.io',
        collection='Coomii_v3',
        embedder=FastEmbedEmbedder(),
        search_type=SearchType.vector,
        reranker=CohereReranker(model="rerank-v3.5"),
    )

    knowledge_base = Knowledge(
        name="Qdrant Vector Db",
        description="This is a knowledge base that uses a Qdrant Vector DB",
        vector_db=vector_db,
    )

    # 3. Handle Chunking via Temporary File
    logger.info("Creating temporary markdown file for Agno chunking...")
    with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False, encoding="utf-8") as temp_file:
        temp_file.write(text_content)
        temp_file_path = temp_file.name

    try:
        
        logger.info("Starting document chunking and insertion into DB...")
        knowledge_base.insert(
        path=temp_file_path,
        metadata=text_metadata,
        reader=MarkdownReader(
            name='Markdown Chunker',
            chunking_strategy= MarkdownChunking(chunk_size=512, overlap=64),
            ),
        )

    except Exception as e:
        # If something breaks here, it logs the error clearly before failing
        logger.error(f"An error occurred during chunking or DB insertion: {e}")
        raise

    finally:
        # 4. Clean Up
        logger.info("Cleaning up temporary system files...")
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)
            logger.info("Cleanup complete. Pipeline finished.")

    return knowledge_base