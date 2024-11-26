import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import SentenceTransformerEmbeddings
from langchain_community.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq


@st.cache_data
def read_pdf_files(pdf_directory):
    pdf_loader = PyPDFDirectoryLoader(pdf_directory)
    documents = pdf_loader.load()
    return documents

# Function to split documents into chunks
@st.cache_data
def chunk_data(_docs, chunk_size=1000, chunk_overlap=130):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(_docs)
    return docs

# Function to create embeddings
@st.cache_resource
def create_embeddings(_model_name="all-mpnet-base-v2"):
    embeddings = SentenceTransformerEmbeddings(model_name=_model_name)
    return embeddings

# Function to create vector store
@st.cache_resource
def create_vector_store(_documents, _embeddings, index_name="millionairefastlanechatbot"):
    index = Pinecone.from_documents(_documents, _embeddings, index_name=index_name)
    return index

# Function to initialize LLM
@st.cache_resource
def initialize_llm():
    groq_api_key = os.environ['GROQ_API_KEY']
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="mixtral-8x7b-32768",
        temperature=0.4,
    )
    return llm

# Function to retrieve query
def retrieve_query(query, index,k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

# Function to initialize QA chain
@st.cache_resource
def initialize_qa_chain(_llm):
    chain = load_qa_chain(_llm, chain_type="stuff")
    return chain

# Function to handle chatbot queries
def chatbot(query, index, chain):
    matching_results = retrieve_query(query, index)
    response = chain.run(input_documents=matching_results, question=query)
    return response
def get_index(name,embedding):
    index = Pinecone.from_existing_index(index_name=name, embedding=embedding)
    return index