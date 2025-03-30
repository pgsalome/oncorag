"""
Conversational interface for querying patient data.

This module provides a user-friendly chat interface for querying the
extracted oncology patient data.
"""

import os
import logging
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from oncorag2.chatbot.query_engine import QueryEngine
from oncorag2.utils.database import setup_iris_vector_store

logger = logging.getLogger(__name__)


class PatientDataConversation:
    """
    Conversational interface for interacting with patient data.

    This class provides a natural language interface for querying
    patient data stored in the vector database.
    """

    def __init__(self,
                 collection_name: str = "patient_contexts",
                 extracted_data_path: Optional[str] = None,
                 connection_string: Optional[str] = None,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """
        Initialize the conversation interface.

        Args:
            collection_name: Name of the IRIS vector collection
            extracted_data_path: Path to the extracted data CSV
            connection_string: Connection string for IRIS
            model_name: LLM model name to use for responses
            temperature: Temperature for LLM sampling
        """
        self.collection_name = collection_name
        self.extracted_data_path = extracted_data_path

        # Connect to vector store
        self.vector_db, self.conn_string = setup_iris_vector_store(
            collection_name=collection_name,
            connection_string=connection_string,
            reset_collection=False  # Don't reset existing collection
        )

        # Initialize query engine
        self.query_engine = QueryEngine(
            vector_db=self.vector_db,
            model_name=model_name,
            temperature=temperature
        )

        # Load extracted data if provided
        self.extracted_data = None
        if extracted_data_path and os.path.exists(extracted_data_path):
            self.extracted_data = pd.read_csv(extracted_data_path)
            logger.info(f"Loaded extracted data with {len(self.extracted_data)} records")
        else:
            logger.warning(f"Warning: Could not find extracted data at {extracted_data_path}")

    def analyze_query(self, user_query: str) -> Optional[str]:
        """
        Determine if this is a query about specific patient data.

        Args:
            user_query: The user's query text

        Returns:
            Patient ID if detected, None otherwise
        """
        # First check if any patient ID from our data is mentioned
        if self.extracted_data is not None:
            for patient_id in self.extracted_data['patient_id'].unique():
                if str(patient_id) in user_query:
                    return str(patient_id)

        return None

    def process_query(self, user_query: str, patient_id: Optional[str] = None) -> Tuple[str, List[Tuple[Any, float]]]:
        """
        Process a user query and return a response.

        Args:
            user_query: The user's query text
            patient_id: Optional patient ID to filter results

        Returns:
            Tuple of (response, retrieved documents with scores)
        """
        # Detect patient ID if not provided
        if patient_id is None:
            patient_id = self.analyze_query(user_query)

        # Search the vector store
        results = self.query_engine.search_vector_store(user_query, k=3, patient_id=patient_id)

        if not results:
            return "I couldn't find any relevant information for your query.", []

        # Formulate an answer from the results
        answer = self.query_engine.formulate_answer(user_query, results)

        return answer, results

    def start_conversation(self, verbose: bool = False) -> None:
        """Start an interactive conversation where any question is treated as a vector search query."""
        print("=" * 60)
        print("Patient Data Conversation")
        print("Just ask any question about the patient data, and I'll search for relevant information.")
        print("Type 'exit' to quit.")
        print("=" * 60)

        while True:
            user_input = input("\nYou: ").strip()

            if user_input.lower() == 'exit':
                print("Goodbye!")
                break

            # Process the query
            answer, results = self.process_query(user_input)

            # Print only the answer
            print(f"\nAnswer: {answer}")

            # Show sources only if verbose is enabled
            if verbose:
                self._display_results(results)

    def _display_results(self, results: List[Tuple[Any, float]]) -> None:
        """
        Display retrieved results in a human-readable format.

        Args:
            results: List of (document, score) tuples from the vector search
        """
        if not results:
            return

        print("\nRetrieved information:")
        for i, (doc, score) in enumerate(results, 1):
            print(f"\n--- Source {i} " + "-" * 40)
            print(f"Relevance: {score:.4f}")

            # Show metadata for context
            if hasattr(doc, 'metadata') and doc.metadata:
                if 'patient_id' in doc.metadata:
                    print(f"Patient: {doc.metadata['patient_id']}")
                if 'feature_name' in doc.metadata:
                    print(f"Feature: {doc.metadata['feature_name']}")
                if 'pdf_source' in doc.metadata:
                    print(f"Source: {doc.metadata['pdf_source']}")

            # Show content
            print(f"\n{doc.page_content}")