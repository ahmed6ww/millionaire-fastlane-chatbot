from pinecone import Pinecone, ServerlessSpec
import os
from dotenv import load_dotenv
from langchain_groq import ChatGroq

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
groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(
    groq_api_key=groq_api_key,
    model_name="mixtral-8x7b-32768",
    temperature=0.4
)