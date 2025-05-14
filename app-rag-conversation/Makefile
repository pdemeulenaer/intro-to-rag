.ONESHELL:

SHELL := $(shell which bash)

# Image name
IMAGE_NAME := rag-app

# Read version from version.txt
VERSION := $(shell cat version.txt)

DOCKER_FOLDER := pdemeulenaer

# 0. General local commands

env-file:
	cp .env.sample .env

# conda:
# 	conda env create --file environment.yml --yes
# 	$(CONDA_ACTIVATE) databricks-llm-fine-tuning-demo

install:
#	pip install -r requirements.txt
#	pip install -e .
	poetry install
	poetry lock

# pre-commit:
# 	pre-commit install

# setup: env-file conda pre-commit

black:
	black .

lint:
	mypy src
	pylint src

test:
	behave tests/features/

doc: 
	mkdocs build	

quality: black lint test

quality-ci: lint test

serve:
	# python -m src.databricks_llm_fine_tuning_demo.model.train	
	poetry run streamlit run src/app_rag_conversation/app.py	

.PHONY: build run

build:
	@echo "Building image version: $(VERSION)"
	@docker build -t $(IMAGE_NAME):$(VERSION) .
	@echo "Built image: $(IMAGE_NAME):$(VERSION)"

run:
	@echo "Running image version: $(VERSION)"
	@docker run -d -p 8501:8501 --env-file .env --name $(IMAGE_NAME) $(IMAGE_NAME):$(VERSION)
	@echo "Running image: $(IMAGE_NAME):$(VERSION)"
	@echo "Access the app at http://localhost:8501"

tag:
	@echo "Tag image version: $(VERSION)"
	@docker tag $(IMAGE_NAME):$(VERSION) $(DOCKER_FOLDER)/$(IMAGE_NAME):$(VERSION)
	@echo "Tagged image: $(DOCKER_FOLDER)/$(IMAGE_NAME):$(VERSION)"

push:
	@echo "Pushing image version: $(VERSION)"
	@docker push $(DOCKER_FOLDER)/$(IMAGE_NAME):$(VERSION)
	@echo "Pushed image: $(DOCKER_FOLDER)/$(IMAGE_NAME):$(VERSION)"		

