# Oncorag2 - Hybrid Clinical Feature Extraction with RAG + Rule-Based Reasoning

**Challenge submission for the Intersystems AI Contest:** [Hybrid approach for clinical data curation: combining RAG and rule-based methods](https://openexchange.intersystems.com/contest/40)

Oncorag2 is a hybrid system designed to extract and curate oncology-related clinical features by combining rule-based regular expressions with retrieval-augmented generation (RAG). It integrates structured and unstructured information, powered by **LangChain**, **LLMs**, and the **Intersystems IRIS Vector Store**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pgsalome/oncorag/blob/main/notebooks/oncorag2_demo.ipynb)

---

## 🧠 Overview

This project tackles clinical data heterogeneity through a **hybrid approach** that:
- 🧬 Generates entity-specific clinical feature templates using LLM agents
- 📑 Extracts data snippets from clinical documents using rule-based matching
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

### Option 1: Local Setup
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

### Option 2: Docker Setup
```bash
python setup_docker.py
```
This builds and launches Docker containers for all services.

### Option 3: Manual Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔁 Script-Based Workflow

### 1️⃣ `scripts/run_feature_generation.py`
Launch an LLM agent to design features interactively (saved to `config/`).

### 2️⃣ `scripts/run_data_extraction.py`
Apply rule-based extraction on patient PDFs or synthetic notes. If no data is provided, synthetic patients are generated.

### 3️⃣ `scripts/run_chatbot.py`
Launches a conversational agent backed by the IRIS vector DB to query structured + free-text context using RAG.

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

## 📚 Documentation
See [`docs/usage.md`](docs/usage.md) for advanced options and component details.

## 🪪 License
MIT License — see the [LICENSE](LICENSE) file.
