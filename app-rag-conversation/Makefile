.ONESHELL:

SHELL := $(shell which bash)

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

