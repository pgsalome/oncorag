# Oncorag2 - Hybrid Clinical Feature Extraction with RAG + Rule-Based Reasoning

**Challenge submission for the Intersystems AI Contest:** [Hybrid approach for clinical data curation: combining RAG and rule-based methods](https://openexchange.intersystems.com/contest/40)

Oncorag2 is a hybrid system designed to extract and curate oncology-related clinical features by combining rule-based regular expressions with retrieval-augmented generation (RAG). It integrates structured and unstructured information, powered by **LangChain**, **LLMs**, and the **Intersystems IRIS Vector Store**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pgsalome/oncorag/blob/main/notebooks/oncorag2_demo.ipynb)

---

## 🧠 Overview

Oncorag2 is built to tackle a fundamental challenge in clinical informatics: **how to extract accurate, structured data from highly variable and verbose clinical documentation**.

This system works by combining two powerful strategies:

- 🧠 **Rule-based logic**: curated regular expressions enable rapid and deterministic extraction of key clinical attributes (e.g., stage, mutation status, treatment) across known document sections.
- 🔍 **Retrieval-Augmented Generation (RAG)**: to handle ambiguity, rare edge cases, or unstructured passages, we retrieve relevant patient-specific document chunks using **Intersystems IRIS Vector Store** and pass them to an LLM for reasoning.

This hybrid method achieves:
- 🧮 **Faster inference** with lower token usage due to targeted retrieval
- 🎯 **Higher accuracy** by grounding answers in indexed patient-specific contexts
- 💬 **Conversational access** to structured and free-text patient data (via chatbot interface)

Together, this enables scalable clinical curation with a rich and explainable interface — combining the rigor of symbolic logic with the flexibility of foundation models.

This project tackles clinical data heterogeneity through a **hybrid approach** that:
- 🧬 Generates entity-specific clinical feature templates using LLM agents
- 📑 Extracts features from real or synthetic clinical documents using regex-based rule matching
- 🤖 Combines structured features with context-aware retrieval-augmented generation (RAG)
- 🧠 Leverages **Intersystems IRIS Vector Store** to store and retrieve contextual embeddings efficiently
- 📉 Reduces token usage and improves accuracy by performing targeted retrieval from a curated context index
- 🩺 Enables natural-language **conversations with patient data** across structured and unstructured fields

Use cases include:
- Oncology cohort curation
- Clinical report digitization
- NLP research on multimodal hospital records
- LLM-based QA interfaces for medical data science teams

---

## ⚙️ Installation

> **Note:** This project uses a hybrid setup with both `pip` packages and Git-based dependencies. To ensure everything works smoothly:
>
> - The `smolagents` package is installed directly from GitHub using `pip install git+https://github.com/huggingface/smolagents.git`
> - Additional dependencies are installed via `requirements.txt`
> - The `setup_local.py` script handles virtual environment creation and installation automatically

---

### 1: Local Setup
```bash
python setup_local.py
```
This script will:
- Create a virtual environment
- Install dependencies
- Prompt you for `.env` setup if missing

Activate your virtual environment:
- macOS/Linux: `source venv/bin/activate`
- Windows: `./venv/Scripts/activate`

### 2: Docker Setup
```bash
python setup_docker.py
```
This builds and launches Docker containers for all services including:
- the backend logic for clinical feature extraction
- the Intersystems IRIS Vector Store (used to persist context chunks)
- a Jupyter notebook server (if included in your `docker-compose.yml`)

After the containers are running, you can:

1. Access the Jupyter server at `http://localhost:8888` or `http://localhost:8889` (depending on your port mapping).
2. Run any of the scripts from inside the `oncorag2-app` or `oncorag2-jupyter` containers using:
   ```bash
   docker exec -it oncorag2-app bash
   python scripts/run_feature_generation.py
   ```
3. View logs to monitor the IRIS database or other containers:
   ```bash
   docker logs oncorag2-iris-1
   ```
4. Stop all services when done:
   ```bash
   docker compose down
   ```


### 3: Manual Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔁 Script-Based Workflow

### 1️⃣ `scripts/run_feature_generation.py`
This launches an LLM-powered assistant that walks you through defining clinical features for your use case (e.g., NSCLC, breast cancer). It:
- Guides the user to select a clinical entity and areas of interest (e.g., biomarkers, staging)
- Uses a smolagent-powered code agent to iteratively generate 5 novel features per round
- Formats and validates each feature (name, prompt, category, regex, etc.)
- Saves the feature schema to a JSON config under `config/` (e.g., `nsclc_feature_config.json`)

### 2️⃣ `scripts/run_data_extraction.py`
This script uses the previously generated feature config to extract data from clinical documents. It:
- Accepts a directory of patient PDFs (or defaults to generating synthetic PDFs)
- Converts PDFs to markdown and anonymizes names
- Applies rule-based regex patterns for primary and fallback categories
- Extracts structured data and indexed context snippets
- Populates both a tabular CSV (`output/extracted_data.csv`) and a context-aware vector store (IRIS)

### 3️⃣ `scripts/run_chatbot.py`
This activates the final conversational interface for querying the extracted patient data. It:
- Connects to the IRIS Vector Store where patient contexts were embedded
- Loads the structured `extracted_data.csv` file for cross-reference
- Allows the user to ask natural language questions (e.g., "What is the cancer stage?", "Is there EGFR mutation?")
- Retrieves relevant chunks using similarity search and synthesizes answers using the LLM
- Can optionally print retrieved evidence for transparency (`--verbose`)

```bash
python scripts/run_chatbot.py --extracted-data output/extracted_data.csv --verbose
```

---

## 🧪 Python API Usage

```python
from oncorag2.feature_extraction import FeatureExtractionAgent

agent = FeatureExtractionAgent()
features = agent.generate_features(
    entity="lung cancer",
    areas_of_interest="biomarkers, staging"
)
agent.save_features("lung_cancer_features.json")
```

---

## 🧠 Try It in Colab

[📓 `oncorag2_demo.ipynb`](https://colab.research.google.com/github/pgsalome/oncorag/blob/main/notebooks/oncorag2_demo.ipynb)

Colab covers:
1. Cloning and setup
2. Feature generation
3. Data extraction
4. Chatbot usage (communicating directly with patient-level data)

---

## 🔐 Environment Variables
Create a `.env` file with:

```
OPENAI_API_KEY=your_openai_api_key
ANTHROPIC_API_KEY=...
GROQ_API_KEY=...
HUGGINGFACE_API_KEY=...
COHERE_API_KEY=...
LOG_LEVEL=INFO
LOG_FILE=oncorag.log
```

---

## 🪪 License
MIT License — see the [LICENSE](LICENSE) file.
