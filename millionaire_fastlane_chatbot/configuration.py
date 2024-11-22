from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
key = os.getenv("PINECONE_API_KEY")
pc = Pinecone(api_key=key)
# Create a serverless index
index_name = "millionairefastlanechatbot"

pc.create_index(
    name=index_name,
    dimension=768, # Replace with your model dimensions
    metric="cosine", # Replace with your model metric
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    ) 
)
load_dotenv()