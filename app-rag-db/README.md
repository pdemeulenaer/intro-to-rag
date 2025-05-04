# app-rag-db

**Objective**: (WIP) Dockerized Streamlit app to (1) upload folder containing many PDFs into Qdrant VS database and (2) query the VS database

Here the app is served using `make serve` 

The ingestion of the PDF folder into Qdrant is done using `make ingest`


**Docker image**: to create & run it:

* docker build -t rag-app:0.0.1 .
* docker run -p 8501:8501 --env-file .env rag-app:0.0.1
* docker image tag rag-app:0.0.1 pdemeulenaer/rag-app:0.0.1
* docker login
* docker image push pdemeulenaer/rag-app:0.0.1

