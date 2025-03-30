"""
Extractors module for the Oncorag2 package.

This module handles document processing, context extraction, and
feature extraction from oncology reports.
"""

from oncorag2.extractors.document_processor import DocumentProcessor
from oncorag2.extractors.context_extractor import ContextExtractor
from oncorag2.extractors.feature_extractor import FeatureExtractor
from oncorag2.extractors.pipeline import ExtractionPipeline

__all__ = [
    'DocumentProcessor',
    'ContextExtractor',
    'FeatureExtractor',
    'ExtractionPipeline',
]