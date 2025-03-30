import os
import pandas as pd
from langchain_openai import ChatOpenAI
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_iris import IRISVector
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# Set your OpenAI API key

# Set the environment variable to allow iris import to work with containerized IRIS
os.environ['IRISINSTALLDIR'] = '/usr'


class PatientDataConversation:
    def __init__(self, collection_name="patient_contexts", extracted_data_path="extracted_data10.csv"):
        """
        Initialize the conversation interface with the IRIS vector store connection.

        Args:
            collection_name (str): Name of the IRIS vector collection
            extracted_data_path (str): Path to the extracted data CSV
        """
        self.collection_name = collection_name
        self.extracted_data_path = extracted_data_path

        # Connect to vector store
        self.vector_db = self._connect_to_vector_store()

        # Load extracted data
        if os.path.exists(extracted_data_path):
            self.extracted_data = pd.read_csv(extracted_data_path)
            print(f"Loaded extracted data with {len(self.extracted_data)} records")
        else:
            self.extracted_data = None
            print(f"Warning: Could not find extracted data at {extracted_data_path}")

        # Initialize LLM
        self.llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    def _connect_to_vector_store(self):
        """Connect to the IRIS vector store."""
        try:
            # Connection string for IRIS
            username = '_SYSTEM'
            password = 'SYS'
            hostname = os.getenv('IRIS_HOSTNAME', 'localhost')
            port = '1972'
            namespace = 'USER'
            connection_string = f"iris://{username}:{password}@{hostname}:{port}/{namespace}"

            # Initialize embeddings
            embeddings = OpenAIEmbeddings()

            # Connect to existing vector store
            vector_db = IRISVector(
                embedding_function=embeddings,
                collection_name=self.collection_name,
                connection_string=connection_string,
            )

            # Test connection
            vector_db.get()
            print(f"Connected to existing IRIS vector store: {self.collection_name}")

            return vector_db

        except Exception as e:
            print(f"Error connecting to IRIS vector store: {e}")
            return None

    def search_vector_store(self, query, k=3, patient_id=None):
        """
        Search the vector store for relevant documents.

        Args:
            query (str): The query string
            k (int): Number of results to return
            patient_id (str, optional): Filter by patient ID

        Returns:
            list: List of (document, score) tuples
        """
        if not self.vector_db:
            print("Vector store not available.")
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
            print(f"Error querying vector store: {e}")
            return []

    def analyze_query(self, user_query):
        """
        Determine if this is a query about specific patient data.
        Returns patient_id if detected, None otherwise.
        """
        # First check if any patient ID from our data is mentioned
        if self.extracted_data is not None:
            for patient_id in self.extracted_data['patient_id'].unique():
                if str(patient_id) in user_query:
                    return str(patient_id)

        return None

    def formulate_answer(self, query, results):
        """
        Formulate a conversational answer based on retrieved documents.

        Args:
            query (str): The user's query
            results (list): List of (document, score) tuples from the vector search

        Returns:
            str: A conversational answer based on the retrieved information
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
        prompt = f"""
        Based on the following information, answer the user's question: "{query}"

        {combined_context}

        Answer the question directly and conversationally using only the information provided.
        If the information doesn't contain an answer to the question, say so.
        Do not include phrases like "Based on the provided information" or "According to the documents".
        Just answer naturally as if you are a medical assistant.
        """

        # Get answer from LLM
        try:
            response = self.llm.invoke(prompt).content
            return response
        except Exception as e:
            print(f"Error generating answer: {e}")
            return "I'm having trouble formulating an answer based on the information I found."

    def start_conversation(self):
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

            # Analyze if the query mentions a specific patient
            patient_id = self.analyze_query(user_input)

            # Search the vector store
            results = self.search_vector_store(user_input, k=3, patient_id=patient_id)

            if not results:
                print("\nI couldn't find any relevant information for your query.")
                continue

            # Formulate an answer from the results
            answer = self.formulate_answer(user_input, results)
            print(f"\nAnswer: {answer}")

            # Optional - can be commented out if you don't want to see the raw results
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


# Example usage
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Conversational Patient Data Query")
    parser.add_argument("--extracted_data", default="/tmp/pycharm_project_711/extracted_data10.csv", help="Path to extracted data CSV")
    parser.add_argument("--collection_name", default="patient_contexts", help="IRIS vector collection name")

    args = parser.parse_args()

    # Initialize and run
    conversation = PatientDataConversation(
        collection_name=args.collection_name,
        extracted_data_path=args.extracted_data
    )

    # Start conversation
    conversation.start_conversation()