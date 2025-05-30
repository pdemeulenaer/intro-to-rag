
import os
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template
from utils import (
    get_qdrant_vectorstore, 
    get_conversation_chain, 
    display_database_info
)



def handle_userinput(user_question):
    """
    Handle user input and generate response using the conversation chain.
    """
    if st.session_state.conversation is None:
        st.error("Please connect to the database first!")
        return
        
    try:
        response = st.session_state.conversation({'question': user_question})
        st.session_state.chat_history = response['chat_history']

        # Group messages into pairs: (user, bot)
        pairs = [
            (st.session_state.chat_history[i], st.session_state.chat_history[i + 1])
            for i in range(0, len(st.session_state.chat_history) - 1, 2)
        ]

        # Reverse the pairs and display user first, then bot
        for user_msg, bot_msg in reversed(pairs):
            st.write(user_template.replace("{{MSG}}", user_msg.content), unsafe_allow_html=True)
            st.write(bot_template.replace("{{MSG}}", bot_msg.content), unsafe_allow_html=True)
                
    except Exception as e:
        st.error(f"Error processing your question: {str(e)}")


def main():
    load_dotenv()

    st.set_page_config(
        page_title="RAG Chat with PDF Knowledge Base",
        page_icon="🤖",
        layout="wide"
    )
    st.write(css, unsafe_allow_html=True)

    # Initialize session state
    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None
    if "vectorstore_connected" not in st.session_state:
        st.session_state.vectorstore_connected = False

    # Main header
    st.header("🤖 RAG Chat with PDF Knowledge Base")
    st.markdown("Ask questions about the documents in your knowledge base!")

    # Sidebar for database connection
    with st.sidebar:
        st.subheader("📚 Knowledge Base")
        
        # Display database info
        display_database_info()
        
        st.markdown("---")
        
        # Connect to database button
        if st.button("🔌 Connect to Knowledge Base", type="primary"):
            with st.spinner("Connecting to Qdrant database..."):
                try:
                    vectorstore = get_qdrant_vectorstore()
                    st.session_state.conversation = get_conversation_chain(vectorstore)
                    st.session_state.vectorstore_connected = True
                    st.success("✅ Successfully connected to knowledge base!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Connection failed: {str(e)}")
                    st.session_state.vectorstore_connected = False
        
        # Reset conversation button
        if st.session_state.vectorstore_connected:
            if st.button("🔄 Reset Conversation"):
                st.session_state.chat_history = None
                if st.session_state.conversation:
                    st.session_state.conversation.memory.clear()
                st.success("Conversation reset!")
                st.rerun()

    # Main chat interface
    if st.session_state.vectorstore_connected:
        st.success("🟢 Knowledge base connected - Ready to answer questions!")
    else:
        st.warning("🟡 Please connect to the knowledge base first using the sidebar")

    # Chat input
    user_question = st.text_input(
        "💬 Ask a question about your documents:",
        placeholder="What would you like to know?",
        disabled=not st.session_state.vectorstore_connected
    )
    
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)

    # Display chat history if exists
    if st.session_state.chat_history:
        st.markdown("---")
        st.subheader("💬 Conversation History")


if __name__ == '__main__':
    main()


# # import streamlit as st
# # from dotenv import load_dotenv
# # from PyPDF2 import PdfReader
# # from langchain.text_splitter import CharacterTextSplitter
# # from langchain.embeddings import OpenAIEmbeddings, HuggingFaceInstructEmbeddings
# # from langchain.embeddings import HuggingFaceEmbeddings  # Replace OpenAIEmbeddings
# # from langchain.vectorstores import FAISS
# # # from langchain.chat_models import ChatOpenAI
# # # from langchain_huggingface import ChatHuggingFace, HuggingFacePipeline
# # from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
# # from langchain_groq import ChatGroq
# # from langchain.memory import ConversationBufferMemory
# # from langchain.chains import ConversationalRetrievalChain
# # from htmlTemplates import css, bot_template, user_template
# # from langchain_community.llms import HuggingFaceHub
# # from dotenv import load_dotenv
# # import os

# import os
# import streamlit as st
# from dotenv import load_dotenv
# from htmlTemplates import css, bot_template, user_template

# from utils import *


# def handle_userinput(user_question):
#     response = st.session_state.conversation({'question': user_question})
#     st.session_state.chat_history = response['chat_history']

#     for i, message in enumerate(st.session_state.chat_history):
#         if i % 2 == 0:
#             st.write(user_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)
#         else:
#             st.write(bot_template.replace(
#                 "{{MSG}}", message.content), unsafe_allow_html=True)


# def main():
#     load_dotenv()

#     st.set_page_config(page_title="Chat with multiple PDFs",
#                        page_icon=":books:")
#     st.write(css, unsafe_allow_html=True)

#     if "conversation" not in st.session_state:
#         st.session_state.conversation = None
#     if "chat_history" not in st.session_state:
#         st.session_state.chat_history = None

#     st.header("Chat with multiple PDFs :books:")
#     user_question = st.text_input("Ask a question about your documents:")
#     # if user_question:
#     #     handle_userinput(user_question)
#     if user_question and st.session_state.conversation:
#         handle_userinput(user_question)        

#     with st.sidebar:
#         st.subheader("Your documents")
#         pdf_docs = st.file_uploader(
#             "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
#         if st.button("Process"):
#             with st.spinner("Processing"):
#                 # get pdf text
#                 raw_text = get_pdf_text(pdf_docs)

#                 # get the text chunks
#                 # text_chunks = get_text_chunks_naive(raw_text) 
#                 text_chunks = get_text_chunks_recursive(raw_text)

#                 # create vector store
#                 vectorstore = get_vectorstore(text_chunks)

#                 # create conversation chain
#                 st.session_state.conversation = get_conversation_chain(
#                     vectorstore)


# if __name__ == '__main__':
#     main()