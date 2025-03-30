"""
Database utilities for IRIS vector store.

This module provides utility functions for setting up and managing
the IRIS vector store connection.
"""

import os
import logging
from typing import Any, Optional, Tuple

from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_iris import IRISVector

logger = logging.getLogger(__name__)


def setup_iris_vector_store(
        collection_name: str = "oncorag",
        connection_string: Optional[str] = None,
        reset_collection: bool = True
) -> Tuple[Any, str]:
    """
    Set up an IRIS vector store connection with option to reset the collection.

    Args:
        collection_name: Name of the vector collection
        connection_string: Connection string for IRIS
        reset_collection: If True, delete the collection if it exists

    Returns:
        Tuple of (IRISVector database instance, connection string)
    """
    # Default connection string if not provided
    if connection_string is None:
        username = '_SYSTEM'
        password = 'SYS'
        hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
        port = '1972'
        namespace = 'USER'
        connection_string = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

    # Initialize embeddings
    embeddings = OpenAIEmbeddings()

    if reset_collection:
        try:
            # Try to connect to existing store
            db = IRISVector(
                embedding_function=embeddings,
                collection_name=collection_name,
                connection_string=connection_string,
            )

            # If connection successful, delete the collection
            logger.info(f"Deleting existing vector collection: {collection_name}")
            db.delete_collection()
            logger.info(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            # If error occurs, collection might not exist
            logger.info(f"No existing collection to delete or error occurred: {e}")

    # Create a new vector store
    try:
        logger.info(f"Creating new vector store: {collection_name}")
        db = IRISVector.from_documents(
            embedding=embeddings,
            documents=[Document(page_content="Initialization document")],
            collection_name=collection_name,
            connection_string=connection_string,
        )

        # Try to remove the initialization document
        try:
            # This might not work depending on the IRISVector implementation
            db.delete(["Initialization document"])
        except:
            logger.info("Note: Could not remove initialization document, but collection was created.")
    except Exception as e:
        logger.error(f"Error creating vector store: {e}")
        logger.info("Attempting to connect to existing store instead...")

        # Connect to existing vector store
        db = IRISVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=connection_string,
        )

    return db, connection_string