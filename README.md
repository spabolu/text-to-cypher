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
1. clone the repo and install it in editable mode
2. verify the CUDA runtime and package versions
3. ingest and normalize data
4. generate schema-aware synthetic examples
5. baseline the untuned model
6. populate the in-notebook Neo4j config cell and let the notebook introspect the live Aura schema before execution-based evaluation
7. fine-tune with QLoRA
8. evaluate the tuned model and export summary artifacts

The notebook keeps placeholder values for Neo4j credentials on purpose. Fill them directly in the dedicated config cell when you run it in Colab, and do not commit real secrets back to GitHub.

## Why SmolLM2-360M
The point of the project is to demonstrate that a very small model can become useful on a narrow structured-generation task when the data pipeline and evaluation discipline are sound.
