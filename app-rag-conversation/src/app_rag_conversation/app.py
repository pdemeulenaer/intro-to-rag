import os
import streamlit as st
from dotenv import load_dotenv
from htmlTemplates import css, bot_template, user_template

from utils import *


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
    st.set_page_config(page_title="Chat with multiple PDFs",
                       page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    # if user_question:
    #     handle_userinput(user_question)
    if user_question and st.session_state.conversation:
        handle_userinput(user_question)        

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True)
        if st.button("Process"):
            with st.spinner("Processing"):
                # get pdf text
                raw_text = get_pdf_text(pdf_docs)

                # get the text chunks
                # text_chunks = get_text_chunks_naive(raw_text) 
                text_chunks = get_text_chunks_recursive(raw_text)

                # create vector store
                vectorstore = get_vectorstore(text_chunks)

                # create conversation chain
                st.session_state.conversation = get_conversation_chain(
                    vectorstore)


if __name__ == '__main__':
    main()