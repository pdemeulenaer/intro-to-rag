Introduction to RAG

This repo is massively inspired by 
* videos from https://www.youtube.com/@alejandro_ao
* https://github.com/alejandro-ao

Content:

**app-one-pdf**: Streamlit app to query one PDF and store the vector in Qdrant Cloud

**app-rag-conversation**: Streamlit app to upload multiple PDFs and have a conversation with it (i.e. using memory)

langchain_multimodal.ipynb: multi-modal RAG, i.e. extracting text, images from a PDF (using unstructured) and converting the documents into embeddings (using Chroma as vector store)