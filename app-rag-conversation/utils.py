import os
from PyPDF2 import PdfReader
import pymupdf
from langchain.text_splitter import CharacterTextSplitter
from langchain.vectorstores import FAISS
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain



# def get_pdf_text(pdf_docs):
#     """
#     Extract text from a list of PDF documents.

#     Caveats:
#     * No structure preservation — things like headings, paragraphs, tables, or bullet lists are flattened.
#     * No cleaning of noisy OCR or page breaks.
#     """
#     text = ""
#     for pdf in pdf_docs:
#         pdf_reader = PdfReader(pdf)
#         for page in pdf_reader.pages:
#             text += page.extract_text()
#     return text

def get_pdf_text(pdf_docs):
    """
    Based on PyMuPDF for better text extraction then PyPDF2.
    Source: https://pymupdf.readthedocs.io/en/latest/the-basics.html

    See in there how to expand on other document types and images
    """
    text = ""
    for pdf in pdf_docs:
        with pymupdf.open(stream=pdf.read(), filetype="pdf") as doc:
            for page in doc:
                text += page.get_text()
    return text


def get_text_chunks_naive(text):
    """
    Split text into chunks for embedding.
    
    Caveats:
    * Naive chunk boundaries: chunks might cut off mid-sentence or mid-thought.
    * Treats all line breaks equally — which ignores paragraph or section logic.
    """
    text_splitter = CharacterTextSplitter(
        separator="\n",
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(text)
    return chunks


# def get_vectorstore(text_chunks):
#     # embeddings = OpenAIEmbeddings()
#     # embeddings = HuggingFaceInstructEmbeddings(model_name="hkunlp/instructor-xl") # said to be better than OpenAIEmbeddings
#     embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2") # 384 for all-MiniLM-L6-v2, 768 for all-mpnet-base-v2      
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore


# def get_vectorstore(text_chunks, model_name="sentence-transformers/all-MiniLM-L6-v2"):
#     """
#     Create a vector store using Hugging Face Inference API embeddings.
    
#     Args:
#         text_chunks (list): List of text chunks to embed
#         model_name (str): Name of the embedding model
    
#     Returns:
#         FAISS: A FAISS vector store with embedded text chunks
#     """
#     # Ensure Hugging Face API token is set
#     if 'HUGGINGFACE_API_TOKEN' not in os.environ:
#         raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")
    
#     # Use LangChain's built-in Hugging Face Inference API Embeddings
#     embeddings = HuggingFaceInferenceAPIEmbeddings(
#         api_key=os.environ['HUGGINGFACE_API_TOKEN'],
#         model_name=model_name
#     )
    
#     # Create and return FAISS vector store
#     vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
#     return vectorstore

def get_vectorstore(text_chunks):
    """
    Create a vector store using Hugging Face Inference API embeddings with fallback.
    
    Args:
        text_chunks (list): List of text chunks to embed
    
    Returns:
        FAISS: A FAISS vector store with embedded text chunks
    """
    # Ensure Hugging Face API token is set
    if 'HUGGINGFACE_API_TOKEN' not in os.environ:
        raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")
    
    # List of models to try in order
    models_to_try = [
        # "intfloat/e5-base-v2",
        "sentence-transformers/all-MiniLM-L6-v2"
    ]
    
    # Try each model until one works
    for model_name in models_to_try:
        try:
            # Use LangChain's built-in Hugging Face Inference API Embeddings
            embeddings = HuggingFaceInferenceAPIEmbeddings(
                api_key=os.environ['HUGGINGFACE_API_TOKEN'],
                model_name=model_name
            )
            
            # Create and return FAISS vector store
            vectorstore = FAISS.from_texts(texts=text_chunks, embedding=embeddings)
            print(f"Successfully used model: {model_name}")
            return vectorstore
        
        except Exception as e:
            print(f"Failed to use model {model_name}: {e}")
            continue
    
    # If all models fail
    raise ValueError("Could not create embeddings with any of the specified models")



# def get_conversation_chain(vectorstore):
#     llm = ChatOpenAI()
#     # llm = HuggingFaceHub(repo_id="google/flan-t5-xxl", model_kwargs={"temperature":0.5, "max_length":512})

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain
# def get_conversation_chain(vectorstore):
#     # Load the model locally
#     # model_id = "meta-llama/Llama-2-7b-chat-hf" # (requires Hugging Face approval for access)
#     # model_id = "mistralai/Mixtral-8x7B-Instruct-v0.1" # (openly accessible, instruction-tuned)
#     model_id = "HuggingFaceH4/zephyr-7b-beta" # (a strong, open-source chat model)

#     tokenizer = AutoTokenizer.from_pretrained(model_id)
#     model = AutoModelForCausalLM.from_pretrained(model_id)
    
#     # Create a pipeline for text generation
#     hf_pipeline = pipeline(
#         "text-generation",
#         model=model,
#         tokenizer=tokenizer,
#         max_length=512,
#         temperature=0.5
#     )
    
#     # Wrap it in HuggingFacePipeline
#     llm = HuggingFacePipeline(pipeline=hf_pipeline)
    
#     # Use ChatHuggingFace for chat compatibility
#     chat_llm = ChatHuggingFace(llm=llm)

#     memory = ConversationBufferMemory(
#         memory_key='chat_history', return_messages=True)
#     conversation_chain = ConversationalRetrievalChain.from_llm(
#         llm=chat_llm,
#         retriever=vectorstore.as_retriever(),
#         memory=memory
#     )
#     return conversation_chain
def get_conversation_chain(vectorstore):

    # Retrieve the API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    llm = ChatGroq(
        groq_api_key = groq_api_key,
        model_name="llama3-8b-8192" , #"mixtral-8x7b-32768" is deprecated,  # Example model; check Groq’s docs for available options
        temperature=0.5
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=vectorstore.as_retriever(),
        memory=memory
    )
    return conversation_chain
