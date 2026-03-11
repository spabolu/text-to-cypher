# cypher-slm

A Colab-first project for fine-tuning a tiny open model to generate Cypher queries.

The first milestone is intentionally narrow:
- model: `HuggingFaceTB/SmolLM2-360M-Instruct`
- task: natural language to Cypher
- workflow: public data + schema-grounded synthetic data + QLoRA SFT + execution-based evaluation
- entrypoint: [`main.ipynb`](./main.ipynb)

## What this repo contains
- `main.ipynb`: end-to-end tutorial notebook for Colab
- `cypher_slm/`: reusable Python package for data prep, prompts, training, evaluation, and reporting
- `tests/`: lightweight tests for normalization and evaluation helpers

## Local development
```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

## Colab usage
Open `main.ipynb` in Google Colab and run top-to-bottom. The notebook is designed to teach the workflow as it runs:
1. install dependencies
2. ingest and normalize data
3. generate schema-aware synthetic examples
4. evaluate the untuned base model
5. fine-tune with QLoRA
6. evaluate the tuned model
7. export summary artifacts

## Why SmolLM2-360M
The point of the project is to demonstrate that a very small model can become useful on a narrow structured-generation task when the data pipeline and evaluation discipline are sound.
