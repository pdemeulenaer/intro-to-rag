
# Taken from https://github.com/alejandro-ao/langchain-ask-pdf/tree/main

import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.embeddings import HuggingFaceEmbeddings  # Replace OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
from langchain.callbacks import get_openai_callback
from langchain_groq import ChatGroq  # Replace OpenAI with ChatGroq


def main():
    # load_dotenv()
    loaded = load_dotenv()
    print(f"dotenv loaded: {loaded}")
    groq_api_key = os.getenv("GROQ_API_KEY")    
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
    #   embeddings = OpenAIEmbeddings()
      embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
      knowledge_base = FAISS.from_texts(chunks, embeddings)
      
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
    

if __name__ == '__main__':
    main()