import os
import streamlit as st
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

from millionaire_fastlane_chatbot.utils import read_pdf_files, chunk_data, create_embeddings, create_vector_store, initialize_llm, initialize_qa_chain, chatbot

# Load environment variables

# Pinecone.init(
#     api_key=os.getenv("PINECONE_API_KEY"),
#     environment=os.getenv("PINECONE_ENVIRONMENT"),
# )
# key = os.getenv("PINECONE_API_KEY")
# pc = Pinecone(api_key=key)
# # Create a serverless index
# index_name = "millionairefastlanechatbot"

# pc.create_index(
#     name=index_name,
#     dimension=768, # Replace with your model dimensions
#     metric="cosine", # Replace with your model metric
#     spec=ServerlessSpec(
#         cloud="aws",
#         region="us-east-1"
#     ) 
# )
# Function to load PDF files


# Streamlit App
st.title("Chat with the book 'The Millionaire Fastlane'")
st.write("This is a Retrieval-Augmented Generation (RAG) app. You can ask anything about the book 'The Millionaire Fastlane'. The app will provide responses based on the contents of the book.")

# Load and process documents
documents = chunk_data(read_pdf_files("document/"))

# Create embeddings and vector store
embeddings = create_embeddings()
index = create_vector_store(documents, embeddings)

# Initialize LLM and QA chain
llm = initialize_llm()
chain = initialize_qa_chain(llm)

# Initialize chat history
# Initialize chat history
if "messages" not in st.session_state:
    st.session_state.messages = []

# Display chat messages from history on app rerun
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Accept user input
if prompt := st.chat_input("What is up?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})
    # Display user message in chat message container
    with st.chat_message("user"):
        st.markdown(prompt)

    # RAG response generation
    with st.chat_message("assistant"):
            
            response = chatbot(prompt,index,chain)
           
            st.markdown(response)
            
    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})



# query = "who is the author of the book"


# answer = chatbot(query, index, chain)
# print(answer)