from taipy.gui import Gui, notify
import os
from PyPDF2 import PdfReader
from langchain.text_splitter import CharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Global variables for state management
status = "Ready to upload PDFs"
history = []  # [{"Question": "", "Answer": ""}]
question = ""
pdf_files = []  # Explicitly bound to file_selector
conversation = None

# PDF processing functions
def get_pdf_text(pdf_files):
    text = ""
    for pdf in pdf_files:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text

def get_text_chunks(text):
    text_splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
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

# Taipy event handlers
def on_file_change(state):
    global pdf_files
    pdf_files = state.pdf_files if state.pdf_files else []
    state.pdf_files = pdf_files  # Ensure state sync
    state.status = f"Selected {len(pdf_files)} PDF(s)" if pdf_files else "No PDFs selected"
    print(f"on_file_change: pdf_files = {pdf_files}, state.pdf_files = {state.pdf_files}")

def process_pdfs(state):
    global conversation, status
    print(f"process_pdfs: state.pdf_files = {state.pdf_files}")
    if not state.pdf_files:
        state.status = "Please upload PDFs first."
        notify(state, "error", "No PDFs uploaded!")
        return
    raw_text = get_pdf_text(state.pdf_files)
    text_chunks = get_text_chunks(raw_text)
    vectorstore = get_vectorstore(text_chunks)
    conversation = get_conversation_chain(vectorstore)
    state.status = "PDFs processed successfully!"
    notify(state, "success", "Ready to chat!")

def submit_question(state):
    global conversation, history, status
    if not conversation:
        state.status = "Process PDFs first."
        notify(state, "error", "No PDFs processed yet!")
        return
    q = state.question
    response = conversation({'question': q})
    state.history.append({"Question": q, "Answer": response['answer']})
    state.question = ""
    state.status = "Question answered."
    print(f"submit_question: history = {state.history}")

# Taipy GUI definition
page = """
# Chat with PDFs
<|layout|columns=1 1|
<|Upload PDFs|file_selector|multiple=True|label=Upload your PDFs|value={pdf_files}|accept=application/pdf|on_change=on_file_change|>
<|Process PDFs|button|on_action=process_pdfs|>
<|Processing Status|text|value={status}|>
<|Chat History|table|data={history}|height=300px|columns=Question;Answer|>
<|Ask a question|input|value={question}|on_action=submit_question|>
|>
"""

# Run the GUI with debugging
gui = Gui(page)
gui.run(title="Chat with PDFs", port=5000, debug=True)