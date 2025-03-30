# Oncorag2 - Oncology Report Analysis with Generative AI

A powerful tool for automatically generating and extracting clinical features from oncology reports using AI agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pgsalome/oncorag/blob/main/notebooks/oncorag2_demo.ipynb)

## Overview

Oncorag2 helps clinicians and medical researchers identify relevant data points in oncology reports. It uses AI to suggest specialized features tailored to different cancer types and clinical contexts, and can extract these features from clinical documentation.

Key capabilities:
- Generate entity-specific clinical features for different oncology types
- Avoid duplicates and ensure all features are properly formatted
- Save feature configurations as JSON for use in extraction pipelines
- Interactive mode for refining feature suggestions
- Support for multiple LLM providers (OpenAI, Anthropic, Groq, etc.)

## Installation

### Option 1: Local Installation

Run the setup script:

```bash
python setup_local.py
```

This will:
1. Create a virtual environment
2. Install required dependencies
3. Set up the environment variables (if no .env file exists)

After installation, activate the virtual environment:
- Windows: `./venv/Scripts/activate`
- Mac/Linux: `source venv/bin/activate`

### Option 2: Docker Installation

Run the Docker setup script:

```bash
python setup_docker.py
```

This will build and start Docker containers for the application.

### Option 3: Manual Installation

```bash
# Create a virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install -e .
```

## 🔧 Script-Based Workflow

Once the environment is set up, use these scripts in order:

### 1️⃣ `scripts/run_feature_generation.py`
Launches an interactive agent to help you define entity-specific clinical features. Features will be saved to the `config/` directory.

### 2️⃣ `scripts/run_data_extraction.py`
Extracts structured features from patient notes using the config file. If no data is provided, synthetic examples are generated.

```bash
python scripts/run_data_extraction.py \
  --features config/your_entity_feature_config.json \
  --data-dir data/sample_patient \
  --output output/extracted_data.csv
```

### 3️⃣ `scripts/run_chatbot.py`
Launches a retrieval-augmented chatbot that combines structured data and text-based evidence.

```bash
python scripts/run_chatbot.py --extracted-data output/extracted_data.csv
```

Use `--verbose` to see matched evidence in context.

---

## 🧪 Python Usage Example (Programmatic API)

```python
from oncorag2.feature_extraction import FeatureExtractionAgent

# Initialize the agent
agent = FeatureExtractionAgent()

# Generate entity-specific features
features = agent.generate_features(
    entity="lung cancer",
    areas_of_interest="biomarkers, staging, treatment response"
)

# Save to disk
agent.save_features("lung_cancer_features.json")
```

## 🧠 Try It on Google Colab

Click the badge above or open this notebook directly:

[📓 `oncorag2_demo.ipynb`](https://colab.research.google.com/github/pgsalome/oncorag/blob/main/notebooks/oncorag2_demo.ipynb)

This interactive notebook runs the full Oncorag2 pipeline:
1. Clone the repo & install dependencies
2. Run the feature generation agent
3. Extract features from real or synthetic notes
4. Launch a conversational clinical chatbot

## Environment Setup

The application uses environment variables for configuration. Create a `.env` file in the root directory with the following variables:

```
# LLM API Keys
OPENAI_API_KEY=your_openai_api_key_here
ANTHROPIC_API_KEY=your_anthropic_api_key_here
GROQ_API_KEY=your_groq_api_key_here
HUGGINGFACE_API_KEY=your_huggingface_api_key_here
COHERE_API_KEY=your_cohere_api_key_here

# Logging Configuration
LOG_LEVEL=INFO
LOG_FILE=oncorag.log
```

You only need to provide API keys for the platforms you intend to use.


## Documentation

For more detailed information, see the [documentation](docs/usage.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.