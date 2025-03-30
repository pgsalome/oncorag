# Oncorag2 - Oncology Report Analysis with Generative AI

A powerful tool for automatically generating and extracting clinical features from oncology reports using AI agents.

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)

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
- Windows: `.\venv\Scripts\activate`
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

## Quick Start

```python
from oncorag2.feature_extraction import FeatureExtractionAgent

# Initialize the agent with default settings (OpenAI, gpt-4o-mini)
agent = FeatureExtractionAgent()

# Or specify a different model and platform
agent = FeatureExtractionAgent(
    model_id="claude-3-opus-20240229",
    platform="anthropic"
)

# Generate features for a specific entity
features = agent.generate_features(
    entity="lung cancer",
    areas_of_interest="biomarkers, staging, treatment response"
)

# Save the features
agent.save_features("lung_cancer_features.json")
```

## Command Line Usage

You can run the agent from the command line:

```bash
# Interactive mode with default settings
oncorag2-generate

# With specific model and platform
oncorag2-generate --model claude-3-sonnet-20240229 --platform anthropic
```

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

## Supported LLM Platforms

Oncorag2 supports multiple LLM providers:

| Platform | Environment Variable | Default Models |
|----------|----------------------|----------------|
| OpenAI | OPENAI_API_KEY | gpt-4o-mini, gpt-4o, gpt-4 |
| Anthropic | ANTHROPIC_API_KEY | claude-3-opus-20240229, claude-3-sonnet-20240229 |
| Groq | GROQ_API_KEY | llama3-8b-8192, llama3-70b-8192 |
| HuggingFace | HUGGINGFACE_API_KEY | various models |
| Cohere | COHERE_API_KEY | command, command-light |

## Documentation

For more detailed information, see the [documentation](docs/usage.md).

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.