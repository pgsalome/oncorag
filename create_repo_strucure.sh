#!/bin/bash

# Create main directory structure
mkdir -p oncorag2/{agents,extractors,chatbot,utils}
mkdir -p {config,data,notebooks,output,scripts,tests}
mkdir -p .github/workflows
mkdir -p tests

# Create __init__.py files
touch oncorag2/__init__.py
touch oncorag2/agents/__init__.py
touch oncorag2/extractors/__init__.py
touch oncorag2/chatbot/__init__.py
touch oncorag2/utils/__init__.py
touch tests/__init__.py
touch config/__init__.py

# Create core agent files
touch oncorag2/agents/feature_agent.py
touch oncorag2/agents/tools.py
touch oncorag2/agents/prompts.py

# Create extractor files
touch oncorag2/extractors/document_processor.py
touch oncorag2/extractors/context_extractor.py
touch oncorag2/extractors/feature_extractor.py
touch oncorag2/extractors/pipeline.py

# Create chatbot files
touch oncorag2/chatbot/conversation.py
touch oncorag2/chatbot/query_engine.py

# Create utility files
touch oncorag2/utils/database.py
touch oncorag2/utils/pdf.py
touch oncorag2/utils/redaction.py
touch oncorag2/utils/config.py
touch oncorag2/utils/logging.py

# Create CLI file
touch oncorag2/cli.py



# Create configuration files
touch config/example_feature_config.json
touch .env.example
touch .gitignore
touch .dockerignore



# Create setup scripts
touch scripts/setup_local.py


# Create test files
touch tests/conftest.py
touch tests/test_agents.py
touch tests/test_extractors.py
touch tests/test_utils.py


# Create notebook examples
touch notebooks/example_workflow.ipynb

# Create placeholder files for data and output directories
touch data/.gitkeep
touch output/.gitkeep

echo "Repository structure created successfully!"