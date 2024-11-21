import os
import streamlit as st
from dotenv import load_dotenv
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq

# Load environment variables
load_dotenv()

# Function to load PDF files
@st.cache_data
def read_pdf_files(pdf_directory):
    pdf_loader = PyPDFDirectoryLoader(pdf_directory)
    documents = pdf_loader.load()
    return documents

# Function to split documents into chunks
@st.cache_data
def chunk_data(_docs, chunk_size=800, chunk_overlap=50):
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
        model_name="llama-3.1-70b-versatile",
        temperature=0.7,
    )
    return llm

# Function to retrieve query
def retrieve_query(query, index, k=2):
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



