# Oncorag2 – Hybrid Clinical Feature Extraction with RAG + Rule-Based Reasoning

**Challenge submission for the InterSystems AI Contest:**  
🧠 [Hybrid approach for clinical data curation: combining RAG and rule-based methods](https://openexchange.intersystems.com/contest/40)

**Oncorag2** is a hybrid system designed to extract and curate oncology-related clinical features by combining **rule-based regular expressions** with **retrieval-augmented generation (RAG)**. It integrates structured and unstructured information, powered by **LangChain**, **LLMs**, and the **InterSystems IRIS Vector Store**.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

---

## 🧠 Overview

Oncorag2 addresses a central challenge in clinical informatics:  
**How to extract accurate, structured data from complex and variable clinical documentation.**

It employs a **hybrid methodology** that combines:

- 🧠 **Rule-Based Logic**: Curated regular expressions provide deterministic and rapid extraction of key attributes (e.g., staging, mutations, treatment) across known sections.
- 🔍 **Retrieval-Augmented Generation (RAG)**: For ambiguous or unstructured data, relevant document chunks are retrieved using **IRIS Vector Store** and processed with an LLM for context-aware reasoning.

### Benefits:
- 🧮 **Efficient Inference**: Targeted retrieval reduces token usage.
- 🎯 **Higher Accuracy**: Answers are grounded in patient-specific context.
- 💬 **Natural-Language Interface**: Supports conversational queries over both structured and unstructured data.

### Key Capabilities:
- 🧬 Generates LLM-defined clinical feature templates
- 📑 Applies regex for deterministic extraction from clinical texts
- 🤖 Combines rule-based and RAG outputs for robust coverage
- 🧠 Uses **IRIS Vector Store** for fast, relevant context retrieval
- 📉 Reduces hallucination risk by grounding responses
- 🩺 Enables rich, explainable **chat-based access** to patient data

### Use Cases:
- Oncology cohort curation  
- Digitization of clinical notes  
- NLP research on hospital records  
- LLM-based clinical QA pipelines  

---

## ⚙️ Installation

> **Note:** The project includes both `pip` and Git-based dependencies. Use the provided setup scripts for a smooth experience.

---

### 🔧 Local Setup
```bash
python setup_local.py
```
This script will:
- Create a virtual environment
- Install dependencies
- Prompt `.env` creation if missing

Activate the environment:
- macOS/Linux: `source venv/bin/activate`  
- Windows: `.\venv\Scripts\activate`

---

### 🐳 Docker Setup
```bash
python setup_docker.py
```
This builds and launches Docker containers for:
- Clinical feature extraction backend
- IRIS Vector Store
- (Optional) Jupyter Notebook Server

Once running:
1. Access the notebook server at `http://localhost:8888`.
2. Run scripts inside the container:
   ```bash
   docker exec -it oncorag2-app bash
   python scripts/run_feature_generation.py
   ```
3. Monitor logs:
   ```bash
   docker logs oncorag2-iris-1
   ```
4. Shut down services:
   ```bash
   docker compose down
   ```

---

### 🛠 Manual Setup
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

---

## 🔁 DEMO: Script-Based Workflow

### 1️⃣ `run_feature_generation.py`
- Guides definition of clinical features for your use case (e.g., NSCLC, breast cancer)
- Uses smolagent to generate 5 novel features per round
- Validates and stores feature schema (JSON in `config/`)

### 2️⃣ `run_data_extraction.py`
- Uses the saved config to extract features from clinical notes
- Converts and anonymizes PDFs
- Applies rule-based extraction with fallbacks
- Outputs both CSV and vector store for downstream use

### 3️⃣ `run_chatbot.py`
- Enables natural language querying of extracted data
- Integrates structured CSV + contextual search via IRIS
- Synthesizes grounded answers using the LLM
- Supports optional verbose output for retrieved context

```bash
python scripts/run_chatbot.py --extracted-data output/extracted_data.csv --verbose
```

---

## 🔐 Environment Variables

Create a `.env` file in the root directory with the following:

```bash
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
