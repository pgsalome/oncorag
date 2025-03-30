"""
Context extraction from oncology reports.

This module handles extracting relevant contexts from oncology reports using
feature-defined regex patterns and formatting them for vector storage.
"""

import re
import os
import pandas as pd
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

from langchain.docstore.document import Document

logger = logging.getLogger(__name__)


class ContextExtractor:
    """
    Extracts context snippets from medical documents using regex patterns.

    This class scans documents for relevant clinical information defined by
    feature regex patterns and extracts contextual snippets.
    """

    def __init__(self,
                 window_size: int = 200,
                 context_csv: Optional[str] = None,
                 raw_context_csv: Optional[str] = None):
        """
        Initialize the context extractor.

        Args:
            window_size: Context window size in characters
            context_csv: Path to CSV file for storing context snippets
            raw_context_csv: Path to CSV file for storing raw document text
        """
        self.window_size = window_size
        self.context_csv = context_csv
        self.raw_context_csv = raw_context_csv

        # Initialize CSV files if provided
        if context_csv and os.path.exists(context_csv):
            os.remove(context_csv)

        if raw_context_csv and os.path.exists(raw_context_csv):
            os.remove(raw_context_csv)

        logger.info(f"Context extractor initialized with window size: {window_size}")

    def extract_context_from_match(self, text: str, match: re.Match) -> str:
        """
        Extract context around a regex match with specified window size.

        Args:
            text: The source text
            match: Regex match object

        Returns:
            A string containing the context around the match
        """
        start = max(0, match.start() - self.window_size)
        end = min(len(text), match.end() + self.window_size)

        context = text[start:end]

        # Try to expand to complete sentences
        if start > 0:
            # Find the first sentence start before our window
            prev_text = text[max(0, start - 100):start]
            sentence_starts = list(re.finditer(r'[.!?]\s+[A-Z]', prev_text))
            if sentence_starts:
                last_start = sentence_starts[-1]
                start = max(0, start - 100) + last_start.end() - 1
                context = text[start:end]

        if end < len(text):
            # Find the first sentence end after our window
            next_text = text[end:min(len(text), end + 100)]
            sentence_end = re.search(r'[.!?]\s+', next_text)
            if sentence_end:
                end = end + sentence_end.start() + 1
                context = text[start:end]

        return context.strip()

    def extract_contexts_from_text(self,
                                   markdown_text: str,
                                   features: List[Dict],
                                   pdf_name: str,
                                   max_features: Optional[int] = None) -> Tuple[List[Document], List[Dict]]:
        """
        Extract contexts using regex patterns for features.

        Args:
            markdown_text: The markdown text to extract contexts from
            features: List of feature dictionaries with regex patterns
            pdf_name: Name of the source PDF file
            max_features: Maximum number of features to process (for testing)

        Returns:
            Tuple of (list of Document objects, list of context data dictionaries)
        """
        all_contexts = []
        all_context_data = []  # For CSV output

        feature_limit = max_features if max_features is not None else len(features)

        for feature in features[:feature_limit]:
            feature_name = feature["name"]
            feature_desc = feature["description"]

            if "regex_patterns" in feature:
                for pattern in feature["regex_patterns"]:
                    try:
                        for match in re.finditer(pattern, markdown_text, re.IGNORECASE):
                            context = self.extract_context_from_match(markdown_text, match)
                            matched_text = match.group(0)

                            if context:
                                # Create a document with metadata
                                doc = Document(
                                    page_content=context,
                                    metadata={
                                        "feature_name": feature_name,
                                        "feature_desc": feature_desc,
                                        "pattern": pattern,
                                        "matched_text": matched_text,
                                        "input_prompt": feature["input_prompt"],
                                        "expected_output_type": feature["expected_output_type"],
                                        "expected_range": feature["expected_range"],
                                        "pdf_source": pdf_name
                                    }
                                )
                                all_contexts.append(doc)

                                # Add context data for CSV output
                                all_context_data.append({
                                    "feature_name": feature_name,
                                    "feature_desc": feature_desc,
                                    "pattern": pattern,
                                    "matched_text": matched_text,
                                    "context": context,
                                    "pdf_source": pdf_name
                                })
                    except re.error as e:
                        logger.error(f"Error in regex pattern for feature {feature_name}: {e}")
                        continue

        logger.info(f"Extracted {len(all_contexts)} contexts from document {pdf_name}")
        return all_contexts, all_context_data

    def save_raw_text(self, markdown_text: str, pdf_name: str, patient_id: str, reference: str) -> bool:
        """
        Save raw markdown text to CSV file.

        Args:
            markdown_text: The markdown text to save
            pdf_name: Name of the source PDF
            patient_id: Patient identifier
            reference: Reference type (e.g., "summary", "notes")

        Returns:
            True if successful, False otherwise
        """
        if not self.raw_context_csv:
            return False

        try:
            raw_df = pd.DataFrame([{
                "patient_id": patient_id,
                "pdf_source": pdf_name,
                "reference": reference,
                "markdown_text": markdown_text
            }])

            file_exists = os.path.isfile(self.raw_context_csv)
            raw_df.to_csv(self.raw_context_csv, mode='a', header=not file_exists, index=False)
            logger.debug(f"Saved raw text for {pdf_name} to {self.raw_context_csv}")
            return True
        except Exception as e:
            logger.error(f"Error saving raw text to CSV: {str(e)}")
            return False

    def save_context_data(self, context_data: List[Dict]) -> bool:
        """
        Save context data to CSV file.

        Args:
            context_data: List of context data dictionaries

        Returns:
            True if successful, False otherwise
        """
        if not self.context_csv or not context_data:
            return False

        try:
            df = pd.DataFrame(context_data)
            file_exists = os.path.isfile(self.context_csv)
            df.to_csv(self.context_csv, mode='a', header=not file_exists, index=False)
            logger.debug(f"Saved {len(context_data)} context items to {self.context_csv}")
            return True
        except Exception as e:
            logger.error(f"Error saving context data to CSV: {str(e)}")
            return False