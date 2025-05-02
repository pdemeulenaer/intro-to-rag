import os
import hashlib
import uuid
import pymupdf
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Qdrant
from langchain.embeddings import HuggingFaceEmbeddings
from qdrant_client import QdrantClient
from qdrant_client.http.models import Filter, FieldCondition, MatchValue
from qdrant_client.http.models import PointStruct
from qdrant_client.http.models import VectorParams, Distance, PayloadSchemaType
from typing import List
import statistics
from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv

load_dotenv()

# === Config ===
QDRANT_URL = os.getenv("QDRANT_URL") 
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "test_collection"
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "folder")


class HFCLIPTextEmbedding(Embeddings):
    def __init__(self, model_name: str, api_token: str):
        self.client = InferenceClient(model=model_name, token=api_token)
    
    def embed_query(self, text: str) -> List[float]:
        return self.client.feature_extraction(text)
    
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self.embed_query(t) for t in texts]

# === Text Extraction ===
def get_pdf_text(filepath):
    """
    Extract text from a PDF file using PyMuPDF.
    Args:
        filepath (str): Path to the PDF file.
    Returns:
        str: Extracted text from the PDF.
    """
    text = ""
    with pymupdf.open(filepath) as doc:
        for page in doc:
            text += page.get_text()
    return text


# === Chunking ===
def get_text_chunks_recursive(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)


# === File Hashing ===
def get_file_hash(filepath):
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# === Main Ingestion ===
def ingest_folder_to_qdrant(folder_path, qdrant_url, qdrant_api_key, collection_name):
    # Set up embedding model and Qdrant vector store

    # Ensure Hugging Face API token is set
    if 'HUGGINGFACE_API_TOKEN' not in os.environ:
        raise ValueError("Please set the HUGGINGFACE_API_TOKEN environment variable")

    # embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # Use HuggingFaceHub directly
    embedding_model = HFCLIPTextEmbedding(
        model_name="sentence-transformers/all-MiniLM-L6-v2",
        api_token=os.environ['HUGGINGFACE_API_TOKEN']
    )  

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    # # Create collection if it doesn't exist
    # vectorstore = Qdrant.from_texts(
    #     texts=[],  # Empty init
    #     embedding=embedding_model,
    #     url=qdrant_url,
    #     collection_name=collection_name,
    #     api_key=qdrant_api_key,
    #     prefer_grpc=True,
    # )

    # Create collection only if it doesn't exist
    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            vectors_config=VectorParams(size=384, distance=Distance.COSINE)
        )    
    # qdrant_client.recreate_collection(
    #     collection_name=collection_name,
    #     vectors_config=VectorParams(size=384, distance=Distance.COSINE)
    # )

    # Create payload indexes
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="file_hash",
        field_schema=PayloadSchemaType.KEYWORD
    )

    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="file_name",
        field_schema=PayloadSchemaType.KEYWORD
    )

    vectorstore = Qdrant(
        client=qdrant_client,
        collection_name=collection_name,
        embeddings=embedding_model
    )    

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)
        file_hash = get_file_hash(filepath)

        # Check if hash already in Qdrant
        # existing = qdrant_client.scroll(
        #     collection_name=collection_name,
        #     filter={
        #         "must": [{"key": "file_hash", "match": {"value": file_hash}}]
        #     },
        #     limit=1
        # )
        existing = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter=Filter(
                must=[
                    FieldCondition(
                        key="file_hash",
                        match=MatchValue(value=file_hash)
                    )
                ]
            ),
            limit=1
        )
        # print(f"Scroll result for {filename} (hash: {file_hash}): {existing}")

        if existing[0]:
            print(f"✔ Skipping (already indexed): {filename}")
            continue

        print(f"→ Processing: {filename}")
        text = get_pdf_text(filepath)
        chunks = get_text_chunks_recursive(text)
        chunk_lengths = [len(chunk) for chunk in chunks]
        print(f"   - {len(chunks)} chunks extracted")
        print(f"→ Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Median: {int(statistics.median(chunk_lengths))}")

        # Add chunks with metadata
        # vectorstore.add_texts(
        #     texts=chunks,
        #     metadatas=[{
        #         "file_name": filename,
        #         "file_hash": file_hash
        #     }] * len(chunks)
        # )

        vectors = embedding_model.embed_documents(chunks)

        qdrant_client.upsert(
            collection_name=collection_name,
            points=[
                PointStruct(
                    id=str(uuid.uuid4()),
                    vector=vec,
                    payload={
                        "file_name": filename,
                        "file_hash": file_hash,
                        "text": chunk  # or rename to page_content if needed
                    }
                )
                for chunk, vec in zip(chunks, vectors)
            ]
        )

        print(f"✅ Indexed: {filename}")

        points, _ = qdrant_client.scroll(collection_name=collection_name, limit=3)
        for pt in points:
            print(f"Payload: {pt.payload}")


# === Run ===
if __name__ == "__main__":
    ingest_folder_to_qdrant(
        folder_path=PDF_FOLDER,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME
    )
