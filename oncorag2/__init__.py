"""
Oncorag2 - Oncology Report Analysis with Generative AI.

A powerful tool for automatically generating and extracting clinical
features from oncology reports using AI agents.
"""

import os
from dotenv import load_dotenv

# Set the environment variable to allow iris import to work with containerized IRIS
os.environ['IRISINSTALLDIR'] = os.environ.get('IRISINSTALLDIR', '/usr')

# Load environment variables from .env file
load_dotenv(override=True)

from oncorag2.agents import FeatureExtractionAgent
from oncorag2.extractors import ExtractionPipeline
from oncorag2.chatbot import PatientDataConversation

__version__ = "0.1.0"

__all__ = [
    'FeatureExtractionAgent',
    'ExtractionPipeline',
    'PatientDataConversation',
]