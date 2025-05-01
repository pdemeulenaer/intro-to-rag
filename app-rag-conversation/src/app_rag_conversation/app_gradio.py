import os
import gradio as gr
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# PDF processing functions
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(
        separator="\n", chunk_size=1000, chunk_overlap=200, length_function=len
    )
    return text_splitter.split_text(text)

def get_vectorstore(text_chunks):
    if 'HUGGINGFACE_API_TOKEN' not in os.environ:
        raise ValueError("HUGGINGFACE_API_TOKEN not set")
    embeddings = HuggingFaceInferenceAPIEmbeddings(
        api_key=os.environ['HUGGINGFACE_API_TOKEN'],
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    return FAISS.from_texts(texts=text_chunks, embedding=embeddings)

def get_conversation_chain(vectorstore):
    groq_api_key = os.getenv("GROQ_API_KEY")
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found")
    llm = ChatGroq(
        groq_api_key=groq_api_key,
        model_name="llama3-8b-8192",
        temperature=0.5
    )
    memory = ConversationBufferMemory(memory_key='chat_history', return_messages=True)
    return ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )

# Global conversation state
conversation = None

# Gradio event handlers
def process_pdfs_and_setup_chat(pdf_files):
    global conversation
    if not pdf_files:
        return "Please upload at least one PDF file."
    raw_text = get_pdf_text(pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)
    return "PDFs processed successfully! You can now chat."

def chat_with_pdfs(user_input):
    if conversation is None:
        return "Please upload and process PDFs first."
    response = conversation({'question': user_input})
    return response['answer']

# Gradio Interface
with gr.Blocks(title="Chat with PDFs") as demo:
    gr.Markdown("# Chat with Multiple PDFs")
    
    with gr.Row():
        with gr.Column():
            pdf_input = gr.File(label="Upload your PDFs", file_count="multiple", file_types=[".pdf"])
            process_button = gr.Button("Process PDFs")
            process_output = gr.Textbox(label="Processing Status")
        
        with gr.Column():
            chatbot = gr.Chatbot(label="Chat History")
            user_question = gr.Textbox(label="Ask a question about your documents", placeholder="Type here...")
            submit_button = gr.Button("Submit")
    
    # Event handlers
    process_button.click(
        fn=process_pdfs_and_setup_chat,
        inputs=pdf_input,
        outputs=process_output
    )
    
    def update_chat(user_input, history):
        if not history:
            history = []
        response = chat_with_pdfs(user_input)
        history.append((user_input, response))
        return history, ""
    
    submit_button.click(
        fn=update_chat,
        inputs=[user_question, chatbot],
        outputs=[chatbot, user_question]
    )

if __name__ == "__main__":
    demo.launch()