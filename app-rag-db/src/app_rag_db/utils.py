import os
# from PyPDF2 import PdfReader
import streamlit as st
import pymupdf
from langchain.text_splitter import CharacterTextSplitter, RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
from langchain_community.vectorstores import Qdrant
from huggingface_hub import InferenceClient
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.embeddings import HuggingFaceInferenceAPIEmbeddings
from typing import List
from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings
# from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from langchain_groq import ChatGroq
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.schema import Document
from langchain.schema.retriever import BaseRetriever
from qdrant_client import QdrantClient
import requests




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
    Split text into chunks for embedding using LangChain CharacterTextSplitter
    
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

def get_text_chunks_recursive(text):
    """
    Split text into chunks for embedding using LangChain RecursiveCharacterTextSplitter
    
    Caveats:
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)


# Custom multimodal embedding wrapper
class HFCLIPTextEmbedding(Embeddings):
    def __init__(self, model_name: str, api_token: str):
        self.client = InferenceClient(model=model_name, token=api_token)
    
    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]


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
        "sentence-transformers/clip-ViT-B-32",  # Primary model HF API NOT WORKING AS OF NOW
        "laion/CLIP-ViT-B-32-laion2B-s34B-b79K", # Fallback model HF API NOT WORKING AS OF NOW
        "openai/clip-vit-base-patch32", # Another fallback HF API NOT WORKING AS OF NOW
        "Salesforce/blip-image-captioning-base", # Another fallback HF API NOT WORKING AS OF NOW
        "sentence-transformers/all-MiniLM-L6-v2" # embedding model for TEXT ONLY        
    ]
    
    # Try each model until one works
    for model_name in models_to_try:
        try:
            # # Use LangChain's built-in Hugging Face Inference API Embeddings
            # embeddings = HuggingFaceInferenceAPIEmbeddings(
            #     api_key=os.environ['HUGGINGFACE_API_TOKEN'],
            #     model_name=model_name
            # )

            # Use HuggingFaceHub directly
            embeddings = HFCLIPTextEmbedding(
                model_name=model_name,
                api_token=os.environ['HUGGINGFACE_API_TOKEN']
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
def get_conversation_chain(retriever): #(vectorstore):
    """
    Create conversation chain with Groq LLM and vectorstore.
    """

    # Retrieve the API key from environment variables
    groq_api_key = os.getenv("GROQ_API_KEY")
    
    if not groq_api_key:
        raise ValueError("GROQ_API_KEY not found in .env file")

    llm = ChatGroq(
        groq_api_key = groq_api_key,
        # model_name="llama3-8b-8192" , #"mixtral-8x7b-32768" is deprecated,  # Example model; check Groq’s docs for available options
        model_name="llama-3.3-70b-versatile", # longer context: 128K
        temperature=0.5
    )
    
    memory = ConversationBufferMemory(
        memory_key='chat_history', return_messages=True)
    
    # conversation_chain = ConversationalRetrievalChain.from_llm(
    #     llm=llm,
    #     retriever=vectorstore.as_retriever(
    #         search_kwargs={"k": 5}  # Return top 5 most relevant chunks
    #     ),
    #     memory=memory
    # )
    conversation_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever,
        memory=memory
    )

    return conversation_chain



# TEST THESE FUNCTIONS (not accepted yet)

class RemoteEmbeddingsAPI(Embeddings):
    def __init__(self, endpoint_url: str):
        self.endpoint_url = endpoint_url.rstrip("/") + "/embed"

    def embed_query(self, text: str) -> List[float]:
        response = requests.post(
            self.endpoint_url,
            json={"text": text},
            timeout=10
        )
        response.raise_for_status()
        return response.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(text) for text in texts]


from typing import Any, List
from langchain.schema.retriever import BaseRetriever
from langchain.schema import Document
from langchain.callbacks.manager import CallbackManagerForRetrieverRun

class QdrantRetriever(BaseRetriever):
    """Custom retriever for Qdrant that properly handles the text field."""
    
    def __init__(self, qdrant_client, collection_name, embeddings, k=5):
        super().__init__()
        # Use object.__setattr__ to bypass Pydantic validation
        object.__setattr__(self, 'qdrant_client', qdrant_client)
        object.__setattr__(self, 'collection_name', collection_name)
        object.__setattr__(self, 'embeddings', embeddings)
        object.__setattr__(self, 'k', k)
    
    def _get_relevant_documents(
        self, 
        query: str, 
        *, 
        run_manager: CallbackManagerForRetrieverRun = None
    ) -> List[Document]:
        # Embed the query
        query_vector = self.embeddings.embed_query(query)
        
        # Search in Qdrant
        search_results = self.qdrant_client.search(
            collection_name=self.collection_name,
            query_vector=query_vector,
            limit=self.k,
            with_payload=True
        )
        
        # Convert to LangChain Documents
        documents = []
        for result in search_results:
            # Extract text content from the payload
            text_content = result.payload.get("text", "")
            
            # Create metadata (exclude text content from metadata)
            metadata = {k: v for k, v in result.payload.items() if k != "text"}
            metadata["score"] = result.score
            
            # Create Document object
            doc = Document(
                page_content=text_content,
                metadata=metadata
            )
            documents.append(doc)
        
        return documents



def get_qdrant_vectorstore():
    """
    Connect to existing Qdrant vector database and return vectorstore.
    """
    # Load environment variables
    QDRANT_URL = os.getenv("QDRANT_URL")
    QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
    COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_collection")  # Default to test_collection
    # HUGGINGFACE_API_TOKEN = os.getenv("HUGGINGFACE_API_TOKEN")
    EMBEDDING_API_URL = os.environ.get("EMBEDDING_API_URL")
    
    if not all([QDRANT_URL, QDRANT_API_KEY, EMBEDDING_API_URL]): #HUGGINGFACE_API_TOKEN
        raise ValueError("Missing required environment variables: QDRANT_URL, QDRANT_API_KEY, EMBEDDING_API_URL")
    # if not EMBEDDING_API_URL:
    #     raise ValueError("Please set the EMBEDDING_API_URL environment variable")

    # # Initialize embedding model (same as used during ingestion)
    # embedding_model = HFCLIPTextEmbedding(
    #     model_name="sentence-transformers/all-MiniLM-L6-v2",
    #     api_token=HUGGINGFACE_API_TOKEN
    # )

    # Initialize embedding model using remote API
    embedding_model = RemoteEmbeddingsAPI(endpoint_url=EMBEDDING_API_URL)    
    
    # Initialize Qdrant client
    qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
    
    # Check if collection exists
    if not qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
        raise ValueError(f"Collection '{COLLECTION_NAME}' does not exist in Qdrant database")
    
    # # Create Qdrant vectorstore
    # vectorstore = Qdrant(
    #     client=qdrant_client,
    #     collection_name=COLLECTION_NAME,
    #     embeddings=embedding_model
    # )

    # Create custom retriever
    retriever = QdrantRetriever(
        qdrant_client=qdrant_client,
        collection_name=COLLECTION_NAME,
        embeddings=embedding_model,
        k=5
    )
    
    return retriever
    
    # return vectorstore



def display_database_info():
    """
    Display information about the connected Qdrant database.
    """
    try:
        QDRANT_URL = os.getenv("QDRANT_URL")
        QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
        COLLECTION_NAME = os.getenv("COLLECTION_NAME", "test_collection")
        
        if QDRANT_URL and QDRANT_API_KEY:
            qdrant_client = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY)
            
            if qdrant_client.collection_exists(collection_name=COLLECTION_NAME):
                collection_info = qdrant_client.get_collection(collection_name=COLLECTION_NAME)
                points_count = collection_info.points_count
                
                st.success(f"✅ Connected to Qdrant collection: **{COLLECTION_NAME}**")
                st.info(f"📊 Total documents indexed: **{points_count:,}** chunks")
                
                # Get sample of file names
                points, _ = qdrant_client.scroll(
                    collection_name=COLLECTION_NAME, 
                    limit=10,
                    with_payload=True
                )
                
                if points:
                    file_names = set()
                    for point in points:
                        if 'file_name' in point.payload:
                            file_names.add(point.payload['file_name'])
                    
                    if file_names:
                        st.info(f"📄 Sample files: {', '.join(list(file_names)[:5])}")
                        if len(file_names) > 5:
                            st.info(f"... and {len(file_names) - 5} more files")
            else:
                st.error(f"❌ Collection '{COLLECTION_NAME}' not found in database")
        else:
            st.warning("⚠️ Database connection not configured")
            
    except Exception as e:
        st.error(f"❌ Error connecting to database: {str(e)}")
