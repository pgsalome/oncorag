"""
Query engine for retrieving patient data.

This module handles vector search and answer generation using RAG.
"""

import logging
from typing import Any, Dict, List, Optional, Tuple

from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

logger = logging.getLogger(__name__)


class QueryEngine:
    """
    Query engine for retrieving and processing patient data.

    This class handles vector similarity search and answer formulation
    using RAG techniques.
    """

    def __init__(self,
                 vector_db: Any,
                 model_name: str = "gpt-3.5-turbo",
                 temperature: float = 0):
        """
        Initialize the query engine.

        Args:
            vector_db: Vector database for similarity search
            model_name: LLM model name to use for responses
            temperature: Temperature for LLM sampling
        """
        self.vector_db = vector_db
        self.llm = ChatOpenAI(model_name=model_name, temperature=temperature)
        logger.info(f"Query engine initialized with model {model_name}")

    def search_vector_store(self, query: str, k: int = 3, patient_id: Optional[str] = None) -> List[Tuple[Any, float]]:
        """
        Search the vector store for relevant documents.

        Args:
            query: The query string
            k: Number of results to return
            patient_id: Filter by patient ID

        Returns:
            List of (document, score) tuples
        """
        if not self.vector_db:
            logger.warning("Vector store not available.")
            return []

        try:
            # Set up filter if patient_id provided
            filter_dict = {"patient_id": patient_id} if patient_id else None

            # Execute search with optional filter
            if filter_dict:
                docs_with_score = self.vector_db.similarity_search_with_score(
                    query, k=k, filter=filter_dict
                )
            else:
                docs_with_score = self.vector_db.similarity_search_with_score(query, k=k)

            return docs_with_score

        except Exception as e:
            logger.error(f"Error querying vector store: {e}")
            return []

    def formulate_answer(self, query: str, results: List[Tuple[Any, float]]) -> str:
        """
        Formulate a conversational answer based on retrieved documents.

        Args:
            query: The user's query
            results: List of (document, score) tuples from the vector search

        Returns:
            A conversational answer based on the retrieved information
        """
        if not results:
            return "I couldn't find any relevant information to answer your question."

        # Prepare context from the retrieved documents
        contexts = []
        for doc, score in results:
            # Format the context with metadata
            context_entry = f"DOCUMENT (relevance: {score:.2f}):\n"

            if hasattr(doc, 'metadata') and doc.metadata:
                if 'patient_id' in doc.metadata:
                    context_entry += f"Patient ID: {doc.metadata['patient_id']}\n"
                if 'feature_name' in doc.metadata:
                    context_entry += f"Feature: {doc.metadata['feature_name']}\n"
                if 'pdf_source' in doc.metadata:
                    context_entry += f"Source: {doc.metadata['pdf_source']}\n"

            context_entry += f"Content: {doc.page_content}\n"
            contexts.append(context_entry)

        combined_context = "\n".join(contexts)

        # Create a prompt for the LLM
        prompt = ChatPromptTemplate.from_template(
            """
            Based on the following information, answer the user's question: "{query}"

            {context}

            Answer the question directly and conversationally using only the information provided.
            If the information doesn't contain an answer to the question, say so.
            Do not include phrases like "Based on the provided information" or "According to the documents".
            Just answer naturally as if you are a medical assistant.
            """
        )

        # Format the prompt
        formatted_prompt = prompt.format(
            query=query,
            context=combined_context
        )

        # Get answer from LLM
        try:
            response = self.llm.invoke(formatted_prompt).content
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return "I'm having trouble formulating an answer based on the information I found."