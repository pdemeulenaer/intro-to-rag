Introduction to RAG

This repo is massively inspired by 
* videos from https://www.youtube.com/@alejandro_ao
* https://github.com/alejandro-ao

Content:

**app-one-pdf**: Streamlit app to query one PDF and store the vector in Qdrant Cloud (local test mode yet)

**app-rag-conversation**: Dockerized Streamlit app to upload multiple PDFs and have a conversation with it (i.e. using memory)

**app-rag-db**: (WIP) Dockerized Streamlit app to (1) upload folder containing many PDFs into Qdrant VS database and (2) query the VS database

langchain_multimodal.ipynb: multi-modal RAG, i.e. extracting text, images from a PDF (using unstructured) and converting the documents into embeddings (using Chroma as vector store)

**Docker image**: to create & run it:

* docker build -t rag-app:0.0.1 .
* docker run -p 8501:8501 --env-file .env rag-app:0.0.1
* docker image tag rag-app:0.0.1 pdemeulenaer/rag-app:0.0.1
* docker login
* docker image push pdemeulenaer/rag-app:0.0.1

