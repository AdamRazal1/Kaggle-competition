from qdrant_client import QdrantClient
from qdrant_client.http import models

# Initialize your client (update with your actual URL/API key if using Qdrant Cloud)
client = QdrantClient(url="https://d24e0cea-2938-4359-977c-fc39ab6375b8.sa-east-1-0.aws.cloud.qdrant.io", api_key='eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJhY2Nlc3MiOiJtIn0.QqnjgYrlOOnXF69FeiTojrL4SpaWqHAQ3NR9GfKKLYU') 

# Create the keyword index
client.create_payload_index(
    collection_name="Coomii_v3",
    field_name="content_hash",
    field_schema=models.PayloadSchemaType.KEYWORD,
)

print("Index created successfully!")