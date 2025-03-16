
# Taken from https://github.com/alejandro-ao/langchain-ask-pdf/tree/main

import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  # Replace OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.vectorstores import Qdrant  # Replace FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_groq import ChatGroq  # Replace OpenAI with ChatGroq
from qdrant_client import QdrantClient


def main():
    # load_dotenv()
    loaded = load_dotenv()
    print(f"dotenv loaded: {loaded}")
    groq_api_key = os.getenv("GROQ_API_KEY")  
    qdrant_api_key = os.getenv("QDRANT_API_KEY")  
    qdrant_url = "https://e5d1fcdb-954e-4c6e-8f44-921000e540e4.europe-west3-0.gcp.cloud.qdrant.io"
    collection_name = "test_collection"

    st.set_page_config(page_title="Ask your PDF")
    st.header("Ask your PDF 💬")
    
    # upload file
    pdf = st.file_uploader("Upload your PDF", type="pdf")
    
    # extract the text
    if pdf is not None:
      pdf_reader = PdfReader(pdf)
      text = ""
      for page in pdf_reader.pages:
        text += page.extract_text()
        
      # split into chunks
      text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
      )
      chunks = text_splitter.split_text(text)
      
      # create embeddings
      # embeddings = OpenAIEmbeddings()
      embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2
      
      # create vector database
      # knowledge_base = FAISS.from_texts(chunks, embeddings)

      # Initialize Qdrant client
      client = QdrantClient(
          url=qdrant_url,
          api_key=qdrant_api_key,
      )      

      # Create Qdrant vector store
      # First, check if collection exists, if not it will be created
      knowledge_base = Qdrant.from_texts(
          texts=chunks,
          embedding=embeddings,
          url=qdrant_url,
          collection_name=collection_name,
          api_key=qdrant_api_key,
          prefer_grpc=True,  # Usually faster
      )      
      
      # show user input
      user_question = st.text_input("Ask a question about your PDF:")
      if user_question:
        docs = knowledge_base.similarity_search(user_question)
        
        # llm = OpenAI()
        llm = ChatGroq(
            model="llama-3.1-8b-instant",  # Groq model, adjust as needed
            groq_api_key=groq_api_key, # st.secrets.get("GROQ_API_KEY") or os.getenv("GROQ_API_KEY")
        )        
        chain = load_qa_chain(llm, chain_type="stuff")
        with get_openai_callback() as cb:
          response = chain.run(input_documents=docs, question=user_question)
          print(cb)
           
        st.write(response)
    

    # Add a feature to use existing collection without reuploading
    st.sidebar.header("Use Existing Collection")
    use_existing = st.sidebar.checkbox("Query existing collection without uploading PDF")
    
    if use_existing and not pdf:
        user_question = st.text_input("Ask a question about your existing collection:")
        if user_question:
            # Connect to existing Qdrant collection
            embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
            knowledge_base = Qdrant(
                client=QdrantClient(url=qdrant_url, api_key=qdrant_api_key),
                collection_name=collection_name,
                embedding_function=embeddings.embed_query
            )
            
            docs = knowledge_base.similarity_search(user_question)
            
            llm = ChatGroq(
                model="llama-3.1-8b-instant",
                groq_api_key=groq_api_key,
            )        
            chain = load_qa_chain(llm, chain_type="stuff")
            with get_openai_callback() as cb:
                response = chain.run(input_documents=docs, question=user_question)
                print(cb)
                
            st.write(response)


if __name__ == '__main__':
    main()