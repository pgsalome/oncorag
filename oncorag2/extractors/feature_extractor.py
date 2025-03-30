"""
Feature extraction from context documents.

This module handles extracting structured feature data from context documents
using LLM-powered RAG techniques.
"""

import logging
import pandas as pd
from typing import Any, Dict, List, Optional, Union

from langchain.docstore.document import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extracts structured features from context documents using RAG techniques.

    This class uses vector search and LLMs to extract specific features from
    context documents based on feature definitions.
    """

    def __init__(self,
                 vector_db: Any,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """
        Initialize the feature extractor.

        Args:
            vector_db: Vector database for similarity search
            model_name: LLM model name to use for extraction
            temperature: Temperature for LLM sampling
        """
        self.vector_db = vector_db
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        logger.info(f"Feature extractor initialized with model {model_name}")

    def _create_prompt_template(self) -> ChatPromptTemplate:
        """
        Create the prompt template for RAG-based feature extraction.

        Returns:
            ChatPromptTemplate for feature extraction
        """
        template = """Answer the task based only on the following contexts:
        {context}

        Feature Description: {feature_desc}

        Task: {query}

        Expected Output Type: {expected_output_type}
        Expected Output Range: {expected_range}

        If the information is not available in the provided contexts, respond only with "Missing"
        """
        return ChatPromptTemplate.from_template(template)

    def retrieve_feature_data(self,
                              feature: Dict[str, Any],
                              patient_id: str) -> str:
        """
        Retrieve and process information for a specific feature.

        Implements fallback logic to check alternative document references
        when primary search returns "Missing".

        Args:
            feature: Feature dictionary with metadata
            patient_id: Patient ID to filter results

        Returns:
            The extracted information or "Missing"
        """
        # Construct a query based on the feature description and input prompt
        query = f"{feature['description']} {feature['input_prompt']}"

        # Get feature name and reference documents
        feature_name = feature["name"]
        primary_reference = feature.get("reference", "summary")
        fallback1 = feature.get("fallback_category_1", "")
        fallback2 = feature.get("fallback_category_2", "")

        # Define the prompt template for RAG
        prompt = self._create_prompt_template()

        # Try primary reference first
        result = self._search_reference(query, feature, patient_id, primary_reference, prompt)

        # If primary reference returns "Missing" and fallback1 is provided, try fallback1
        if result.strip().lower() == "missing" and fallback1:
            logger.info(
                f"Primary reference returned Missing for {feature_name}. Trying fallback_category_1: {fallback1}")
            result = self._search_reference(query, feature, patient_id, fallback1, prompt)

        # If fallback1 returns "Missing" and fallback2 is provided, try fallback2
        if result.strip().lower() == "missing" and fallback2:
            logger.info(f"Fallback1 returned Missing for {feature_name}. Trying fallback_category_2: {fallback2}")
            result = self._search_reference(query, feature, patient_id, fallback2, prompt)

        # If all attempts return "Missing", try one final search without reference filter
        if result.strip().lower() == "missing" and feature.get("search_everywhere", False):
            logger.info(
                f"All reference categories returned Missing for {feature_name}. Trying search across all documents.")
            result = self._search_all_documents(query, feature, patient_id, prompt)

        return result

    def _search_reference(self,
                          query: str,
                          feature: Dict[str, Any],
                          patient_id: str,
                          reference: str,
                          prompt: ChatPromptTemplate) -> str:
        """
        Search for information in a specific reference document.

        Args:
            query: Search query
            feature: Feature dictionary
            patient_id: Patient ID
            reference: Document reference to search in
            prompt: Prompt template

        Returns:
            The extracted information or "Missing"
        """
        feature_name = feature["name"]

        try:
            # Retrieve relevant documents with both feature_name and reference filters
            docs_with_score = self.vector_db.similarity_search_with_score(
                query,
                k=3,  # Get top 3 results
                filter={"patient_id": patient_id, "feature_name": feature_name, "reference": reference}
            )

            # If no documents found with feature_name filter, try just with reference filter
            if not docs_with_score:
                docs_with_score = self.vector_db.similarity_search_with_score(
                    query,
                    k=3,
                    filter={"patient_id": patient_id, "reference": reference}
                )

        except Exception as e:
            logger.error(f"Error querying vector store for feature {feature_name} in reference {reference}: {e}")
            docs_with_score = []

        # If no results, return "Missing"
        if not docs_with_score:
            return "Missing"

        # Combine contexts from retrieved documents
        contexts = []
        for doc, score in docs_with_score:
            contexts.append(f"Context (similarity score: {score:.4f}):\n{doc.page_content}")

        combined_context = "\n\n" + "\n\n".join(contexts)

        # Format the prompt
        formatted_prompt = prompt.format(
            query=feature["input_prompt"],
            context=combined_context,
            feature_desc=feature["description"],
            expected_output_type=feature["expected_output_type"],
            expected_range=feature["expected_range"]
        )

        # Get the response from the language model
        try:
            response = self.llm.invoke(formatted_prompt).content
            return response
        except Exception as e:
            logger.error(f"Error processing feature {feature_name} with reference {reference}: {e}")
            return "Error in processing"

    def _search_all_documents(self,
                              query: str,
                              feature: Dict[str, Any],
                              patient_id: str,
                              prompt: ChatPromptTemplate) -> str:
        """
        Search across all documents as a last resort.

        Args:
            query: Search query
            feature: Feature dictionary
            patient_id: Patient ID
            prompt: Prompt template

        Returns:
            The extracted information or "Missing"
        """
        feature_name = feature["name"]

        try:
            # Retrieve relevant documents with only patient_id filter
            docs_with_score = self.vector_db.similarity_search_with_score(
                query,
                k=5,  # Get top 5 results for broader search
                filter={"patient_id": patient_id}
            )
        except Exception as e:
            logger.error(f"Error querying all documents for feature {feature_name}: {e}")
            docs_with_score = []

        # If no results, return "Missing"
        if not docs_with_score:
            return "Missing"

        # Combine contexts from retrieved documents
        contexts = []
        for doc, score in docs_with_score:
            contexts.append(f"Context (similarity score: {score:.4f}):\n{doc.page_content}")

        combined_context = "\n\n" + "\n\n".join(contexts)

        # Format the prompt
        formatted_prompt = prompt.format(
            query=feature["input_prompt"],
            context=combined_context,
            feature_desc=feature["description"],
            expected_output_type=feature["expected_output_type"],
            expected_range=feature["expected_range"]
        )

        # Get the response from the language model
        try:
            response = self.llm.invoke(formatted_prompt).content
            return response
        except Exception as e:
            logger.error(f"Error processing feature {feature_name} across all documents: {e}")
            return "Error in processing"

    def process_patient_features(self,
                                 features: List[Dict[str, Any]],
                                 patient_id: str) -> Dict[str, Any]:
        """
        Process all features for a single patient.

        Args:
            features: List of feature dictionaries
            patient_id: Patient ID

        Returns:
            Dictionary of feature name to extracted value
        """
        results = {"patient_id": patient_id}

        for feature in features:
            feature_name = feature["name"]
            logger.info(f"Processing feature: {feature_name} for patient {patient_id}")

            # Retrieve feature data using vector similarity search
            result = self.retrieve_feature_data(feature, patient_id)
            results[feature_name] = result

        return results

    def save_to_csv(self,
                    extracted_data: Dict[str, Any],
                    output_csv: str = 'extracted_data.csv') -> bool:
        """
        Save extracted data to CSV.

        Args:
            extracted_data: Dictionary of extracted feature values
            output_csv: Path to save the CSV file

        Returns:
            True if successful, False otherwise
        """
        try:
            # Clean up the extracted data
            cleaned_data = self._clean_extracted_data(extracted_data)

            # Check if file exists to determine if header should be written
            file_exists = os.path.isfile(output_csv)

            df = pd.DataFrame([cleaned_data])
            df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
            logger.info(f"Data saved to {output_csv}")
            return True
        except Exception as e:
            logger.error(f"Error saving data to CSV: {str(e)}")
            return False

    def _clean_extracted_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Clean up the extracted data.

        Args:
            data: Dictionary of extracted data

        Returns:
            Cleaned dictionary
        """
        cleaned_data = {}
        for key, value in data.items():
            if isinstance(value, str):
                # If the value looks like a tuple ('label', 'value'), clean it
                if value.startswith("('") and value.endswith("')"):
                    value = value.split(",")[1].strip(" '")
                # Additional cleanup for any unwanted characters
                value = value.replace("('", "").replace("', '", "").replace("')", "").strip()
            cleaned_data[key] = value
        return cleaned_data