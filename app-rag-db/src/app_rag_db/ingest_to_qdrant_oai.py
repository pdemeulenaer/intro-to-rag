import os
import httpx
import hashlib
import uuid
import pymupdf
from datetime import datetime
from langchain.text_splitter import RecursiveCharacterTextSplitter
from qdrant_client import QdrantClient
from qdrant_client.http.models import PointStruct, VectorParams, Distance, PayloadSchemaType
# from qdrant_client.http.models import TextIndexParams, TextIndexType
from typing import List, Generator, Tuple, Dict
import statistics
# from huggingface_hub import InferenceClient
from langchain.embeddings.base import Embeddings
from dotenv import load_dotenv
from openai import OpenAI

import instructor
from pydantic import BaseModel, Field


from .utils import (
    load_config,
    RemoteEmbeddingsAPI,
)

load_dotenv()

# === Config ===
QDRANT_URL = os.getenv("QDRANT_URL")
QDRANT_API_KEY = os.getenv("QDRANT_API_KEY")
COLLECTION_NAME = "test_collection_oai_test"
PDF_FOLDER = os.path.join(os.path.dirname(__file__), "folder")
EMBEDDING_API_URL = os.getenv("EMBEDDING_API_URL")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

config = load_config()

# === OpenAI Embedding Class ===
client = OpenAI(api_key=OPENAI_API_KEY)
# Wrap OpenAI client with Instructor
# client = instructor.from_openai(OpenAI(api_key=OPENAI_API_KEY))
groq_client = instructor.from_openai(
    OpenAI(
        base_url="https://api.groq.com/openai/v1",
        api_key=os.getenv("GROQ_API_KEY")
    )
)


SYSTEM_PROMPT = """
You are a research assistant that extracts structured metadata from scientific documents.
Your task is to generate concise, factual metadata that is directly grounded in the document text.
Do not hallucinate information. If a field cannot be determined, leave it empty.
"""

USER_PROMPT = """
Extract the following metadata from the provided text:

- **Title**: The scientific title of the document (if present).
- **Authors**: The main author(s) or PhD candidate.
- **Keywords**: 5–10 scientific keywords that are explicitly present in the text,
  or strongly implied by domain-specific terminology. Avoid generic terms like
  'research', 'study', 'thesis'. Each keyword must be a single word or short phrase.

The keywords must come from the text (or be obvious synonyms), not invented.

Return only valid JSON following this schema:
{
  "title": string,
  "authors": [string],
  "keywords": [string]
}

Text to analyze:
----------------
{input_text}
"""


class AdditionalMetadata(BaseModel):
    """Structured metadata extracted from a scientific PDF."""
    title: str = Field(..., description="The title of the document")
    authors: list[str] = Field(default_factory=list, description="List of authors of the document, as a string")
    keywords: list[str] = Field(default_factory=list, description="List of keywords (empty if none)")

# def extract_metadata_with_llm(text: str, config) -> AdditionalMetadata:
#     """Use Groq LLM to extract structured metadata from the first page text."""
#     return groq_client.chat.completions.create(
#         model=config["groq"]["summarization_model"],  # e.g., "mixtral-8x7b-32768"
#         response_model=AdditionalMetadata,
#         messages=[
#             {
#                 "role": "system",
#                 "content": "You are an academic assistant. Extract structured metadata from academic documents."
#             },
#             {
#                 "role": "user",
#                 "content": f"Extract title, authors, and keywords from the following text:\n\n{text[:3000]}"
#             }
#         ],
#         temperature=0,
#         max_tokens=300
#     )

def extract_metadata_with_llm(title_text: str, keywords_text: str, config) -> AdditionalMetadata:
    """Use Groq LLM to extract metadata (title/authors from first page, keywords from first 3 pages)."""
    return groq_client.chat.completions.create(
        model=config["groq"]["summarization_model"],
        response_model=AdditionalMetadata,
        messages=[
            {
                "role": "system",
                "content": "You are an academic assistant. Extract structured metadata from academic documents."
            },
            {
                "role": "user",
                "content": f"""
                Extract the following fields as JSON:
                - Title (from this text):\n{title_text[:1500]}
                - Authors (from this text):\n{title_text[:1500]}
                - Keywords (from this broader text, if present):\n{keywords_text[:8000]}
                """
            }
        ],
        temperature=0,
        max_tokens=500
    )

# def extract_metadata_with_llm(doc_text: str):
#     api_key = os.getenv("GROQ_API_KEY")
#     if not api_key:
#         raise ValueError("GROQ_API_KEY is not set in environment variables.")

#     headers = {
#         "Authorization": f"Bearer {api_key}",
#         "Content-Type": "application/json"
#     }

#     payload = {
#         "model": config["groq"]["metadata_model"],
#         "messages": [
#             {"role": "system", "content": SYSTEM_PROMPT},
#             {"role": "user", "content": USER_PROMPT.format(input_text=doc_text[:8000])}
#         ],
#         "temperature": 0,
#         "max_tokens": 500
#     }

#     response = httpx.post("https://api.groq.com/openai/v1/chat/completions",
#                           headers=headers, 
#                           json=payload, 
#                           timeout=60)
#     response.raise_for_status()

#     raw_text = response.json()["choices"][0]["message"]["content"].strip()

#     try:
#         data = json.loads(raw_text)
#     except Exception:
#         print(f"⚠️ Failed to parse JSON:\n{raw_text}")
#         data = {"title": None, "authors": [], "keywords": []}

#     return AdditionalMetadata(**data)


class OpenAIEmbeddings(Embeddings):
    """A wrapper for OpenAI's embedding model."""

    def __init__(self, model_name: str = "text-embedding-3-small"):
        self.model_name = model_name
        self.client = client
        self.dimensions = 1536  # text-embedding-3-small has a dimension of 1536 by default

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """Embeds a list of documents."""
        response = self.client.embeddings.create(
            input=texts,
            model=self.model_name,
        )
        return [item.embedding for item in response.data]

    def embed_query(self, text: str) -> List[float]:
        """Embeds a single query string."""
        return self.embed_documents([text])[0]


# === File Hashing ===
def get_file_hash(filepath) -> str:
    with open(filepath, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()


# === Chunking ===
def get_text_chunks_recursive(text) -> List[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=10000,
        chunk_overlap=2000,
        separators=["\n\n", "\n", ".", "!", "?", ",", " ", ""]
    )
    return splitter.split_text(text)


# === Chunk Generator with Metadata ===
def extract_chunks_with_metadata(filepath: str) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
    """
    Extract chunks of text with page numbers and collect metadata.
    Falls back to LLM extraction if PyMuPDF metadata is incomplete.
    """    
    chunks_with_page = []
    first_page_text, first_pages_text = "", ""
    with pymupdf.open(filepath) as doc:
        metadata = doc.metadata or {}
        for page_number, page in enumerate(doc, start=1):
            text = page.get_text()
            if page_number == 1:
                first_page_text = text   
            if page_number <= 3:  # capture first 3 pages for keywords
                first_pages_text += "\n" + text                         
            if not text.strip():
                continue
            page_chunks = get_text_chunks_recursive(text)
            for chunk in page_chunks:
                chunks_with_page.append((chunk, page_number))

        # Extract year from creationDate
        creation_date = metadata.get("creationDate")
        year = None
        if creation_date:
            try:
                # Strip "D:" if present
                clean_date = creation_date.lstrip("D:")
                # Try parsing
                dt = datetime.strptime(clean_date[:14], "%Y%m%d%H%M%S")
                year = str(dt.year)
            except Exception:
                pass

        # Prefer PDF metadata, fallback to LLM
        title = metadata.get("title")
        authors = metadata.get("author")
        keywords = metadata.get("keywords")            

        # if not title or not authors or not keywords:
        #     try:
        #         llm_meta = extract_metadata_with_llm(first_page_text, config)
        #         title = title or llm_meta.title
        #         authors = authors or llm_meta.authors
        #         keywords = keywords or llm_meta.keywords
        #     except Exception as e:
        #         print(f"⚠️ Metadata extraction with Groq failed: {e}")

        # Fallback to LLM if missing
        if not title or not authors or not keywords:
            try:
                llm_meta = extract_metadata_with_llm(
                    title_text=first_page_text,
                    keywords_text=first_pages_text,
                    config=config
                )
                title = title or llm_meta.title
                authors = authors or llm_meta.authors
                keywords = keywords or llm_meta.keywords
            except Exception as e:
                print(f"⚠️ Metadata extraction with Groq failed: {e}")

    return chunks_with_page, {
        "file_title": title,
        "authors": authors,
        "keywords": keywords,
        "creation_date": creation_date,
        "year": year
    }    

# def extract_chunks_with_metadata(filepath: str) -> Tuple[List[Tuple[str, int]], Dict[str, str]]:
#     """
#     Extract chunks of text with page numbers and collect metadata.
#     Falls back to LLM extraction if PyMuPDF metadata is incomplete.
#     """
#     chunks_with_page = []
#     with pymupdf.open(filepath) as doc:
#         metadata = doc.metadata or {}
#         title = metadata.get("title")
#         authors = [metadata.get("author")] if metadata.get("author") else []
#         keywords = metadata.get("keywords").split(",") if metadata.get("keywords") else []
#         creation_date = metadata.get("creationDate")

#         # Extract year from creationDate
#         year = None
#         if creation_date:
#             try:
#                 # Strip "D:" if present
#                 clean_date = creation_date.lstrip("D:")
#                 # Try parsing
#                 dt = datetime.strptime(clean_date[:14], "%Y%m%d%H%M%S")
#                 year = str(dt.year)
#             except Exception:
#                 pass        

#         # Collect text for chunking
#         for page_number, page in enumerate(doc, start=1):
#             text = page.get_text()
#             if not text.strip():
#                 continue
#             page_chunks = get_text_chunks_recursive(text)
#             for chunk in page_chunks:
#                 chunks_with_page.append((chunk, page_number))

#         # Use the first few pages for LLM metadata extraction if needed
#         if not title or not authors or not keywords:
#             preview_text = "\n".join(
#                 [doc[i].get_text() for i in range(min(10, len(doc)))]
#             )
#             try:
#                 llm_meta: AdditionalMetadata = extract_metadata_with_llm(preview_text)
#                 if not title and llm_meta.title:
#                     title = llm_meta.title
#                 if not authors and llm_meta.authors:
#                     authors = llm_meta.authors
#                 if not keywords and llm_meta.keywords:
#                     keywords = llm_meta.keywords
#             except Exception as e:
#                 print(f"⚠️ Metadata extraction with Groq failed: {e}")

#     return chunks_with_page, {
#         "file_title": title,
#         "authors": authors,
#         "keywords": keywords,
#         "creation_date": creation_date,
#         "year": year
#     }


# === Summarization with Groq ===
def summarize_chunk(text: str, config) -> str:
    """
    Use Groq's Mixtral model to summarize a long chunk of text.
    """
    api_key = os.getenv("GROQ_API_KEY")
    if not api_key:
        raise ValueError("GROQ_API_KEY is not set in environment variables.")

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    payload = {
        "model": config["groq"]["summarization_model"],
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that summarizes academic documents."},
            {"role": "user", "content": f"Summarize the following chunk:\n\n{text}"}
        ],
        "temperature": config["groq"]["temperature"],
        "max_tokens": config["groq"]["max_tokens"]
    }

    try:
        response = httpx.post("https://api.groq.com/openai/v1/chat/completions", headers=headers, json=payload, timeout=60)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"].strip()
    except Exception as e:
        print(f"⚠️ Groq summarization failed: {e}")
        return text[:200] + "..."


# === Ingestion Function ===
def ingest_folder_to_qdrant(folder_path: str, qdrant_url: str, qdrant_api_key: str, collection_name: str, config):
    # Initialize embedding model using the new OpenAIEmbeddings class
    embedding_model = OpenAIEmbeddings()

    qdrant_client = QdrantClient(url=qdrant_url, api_key=qdrant_api_key)

    if not qdrant_client.collection_exists(collection_name=collection_name):
        qdrant_client.create_collection(
            collection_name=collection_name,
            # Update vector size to 1536 for text-embedding-3-small
            vectors_config=VectorParams(size=embedding_model.dimensions, distance=Distance.COSINE)
        )

    # Create metadata indexes
    for field in ["file_hash", "file_name", "file_title", "authors", "keywords", "creation_date", "page_number"]:
        qdrant_client.create_payload_index(
            collection_name=collection_name,
            field_name=field,
            field_schema=PayloadSchemaType.KEYWORD
        )

    # ADD THIS BLOCK to create the text index    
    qdrant_client.create_payload_index(
        collection_name=collection_name,
        field_name="text",
        field_schema=PayloadSchemaType.TEXT,
        # field_schema=TextIndexParams(
        #     type=TextIndexType.TEXT
        # )
    )     

    for filename in os.listdir(folder_path):
        if not filename.lower().endswith(".pdf"):
            continue

        filepath = os.path.join(folder_path, filename)
        file_hash = get_file_hash(filepath)

        # Skip already ingested files
        existing = qdrant_client.scroll(
            collection_name=collection_name,
            scroll_filter={"must": [{"key": "file_hash", "match": {"value": file_hash}}]},
            limit=1
        )
        if existing[0]:
            print(f"✔ Skipping (already indexed): {filename}")
            continue

        print(f"→ Processing: {filename}")
        chunks_with_meta, doc_metadata = extract_chunks_with_metadata(filepath)

        texts = [chunk for chunk, _ in chunks_with_meta]
        page_numbers = [page for _, page in chunks_with_meta]

        vectors = embedding_model.embed_documents(texts)

        chunk_lengths = [len(c) for c in texts]
        print(f"    - {len(texts)} chunks extracted")
        print(f"→ Min: {min(chunk_lengths)}, Max: {max(chunk_lengths)}, Median: {int(statistics.median(chunk_lengths))}")

        points = []
        for chunk, vec, page_num in zip(texts, vectors, page_numbers):
            points.append(PointStruct(
                id=str(uuid.uuid4()),
                vector=vec,
                payload={
                    "file_name": filename,
                    "file_hash": file_hash,
                    "file_title": doc_metadata.get("file_title"),
                    "authors": doc_metadata.get("authors"),
                    "keywords": doc_metadata.get("keywords"),
                    "creation_date": doc_metadata.get("creation_date"),
                    "year": doc_metadata.get("year"),
                    "page_number": str(page_num),
                    "text": chunk,
                    "summary": summarize_chunk(chunk, config)
                }
            ))

        qdrant_client.upsert(collection_name=collection_name, points=points)
        print(f"✅ Indexed: {filename}")

        # Optional: show sample payloads
        sample, _ = qdrant_client.scroll(collection_name=collection_name, limit=2)
        for pt in sample:
            print(f"Sample payload:\n{pt.payload}")

if __name__ == "__main__":
    ingest_folder_to_qdrant(
        folder_path=PDF_FOLDER,
        qdrant_url=QDRANT_URL,
        qdrant_api_key=QDRANT_API_KEY,
        collection_name=COLLECTION_NAME,
        config=config
    )