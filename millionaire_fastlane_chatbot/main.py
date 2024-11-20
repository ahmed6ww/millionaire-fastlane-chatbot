import langchain
import pinecone
from langchain.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import SentenceTransformerEmbeddings
from langchain.vectorstores import Pinecone
from langchain_groq import ChatGroq
import os
from dotenv import load_dotenv
load_dotenv()

def read_pdf_files(pdf_directory):
    pdf_loader = PyPDFDirectoryLoader(pdf_directory)
    documents = pdf_loader.load()
    return documents
    
doc = read_pdf_files("document/")
len(doc)

def chunk_data(docs,chunk_size=800, chunk_overlap=50):
   text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
   docs = text_splitter.split_documents(docs)
   return docs

documents = chunk_data(docs=doc)
len(documents)

embeddings = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")

vectors = embeddings.embed_query("how ")
len(vectors)


index_name = "millionairefastlanechatbot"
from langchain.vectorstores import Pinecone
index = Pinecone.from_documents( documents, embeddings,index_name=index_name)

def retrieve_query(query,k=2):
    matching_results = index.similarity_search(query, k=k)
    return matching_results

from langchain.chains.question_answering import load_qa_chain
from langchain_groq import ChatGroq


groq_api_key = os.environ['GROQ_API_KEY']
llm = ChatGroq(
            groq_api_key=groq_api_key,
            model_name="llama3-groq-70b-8192-tool-use-preview",
            temperature=0.5,
    )
chain = load_qa_chain(llm,chain_type="stuff")

def chatbot(query):
    matching_results = retrieve_query(query)
    print(matching_results)
    response = chain.run(input_documents=matching_results,question=query)
    return response
query = "what is fastlane roadmap.and what is the difference between fastlane roadmap and slowlane roadmap. Give strategies for fastlane roadmap.Give any story from the book"
answer = chatbot(query)
print(answer)



import streamlit as st
from streamlit import chat_message, chat_input, title, markdown, spinner
# Streamlit App
st.title("Simple RAG-Integrated Chat")

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
        with st.spinner("Generating response..."):
            response = chatbot(prompt)
            st.markdown(response)

    # Add assistant response to chat history
    st.session_state.messages.append({"role": "assistant", "content": response})