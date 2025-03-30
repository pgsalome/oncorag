#!/usr/bin/env python3
"""
Run Oncorag2 Patient Chatbot for querying oncology records.

This script launches a conversational agent using LangChain + IRIS vector store
to retrieve patient-related data via RAG (Retrieval-Augmented Generation).
"""

import os
import sys
import argparse
from dotenv import load_dotenv
from oncorag2.chatbot.conversation import PatientDataConversation
from oncorag2.utils.logging import configure_logging

def main():
    parser = argparse.ArgumentParser(description="Launch Oncorag2 patient data chatbot")
    parser.add_argument(
        "-e", "--extracted-data",
        default="./output/extracted_data.csv",
        help="Path to extracted data CSV (default: ./output/extracted_data.csv)"
    )
    parser.add_argument(
        "-c", "--collection",
        default="patient_contexts",
        help="Name of IRIS vector collection (default: patient_contexts)"
    )
    parser.add_argument(
        "--model", default="gpt-3.5-turbo",
        help="OpenAI model name (default: gpt-3.5-turbo)"
    )
    parser.add_argument(
        "--temperature", type=float, default=0,
        help="Sampling temperature for LLM responses (default: 0)"
    )
    parser.add_argument(
        "--log-level", default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
        help="Logging level (default: INFO)"
    )
    parser.add_argument(
        "--connection-string", default=None,
        help="Optional: Override default IRIS connection string"
    )
    parser.add_argument("--verbose", action="store_true", help="Show retrieved document metadata")

    args = parser.parse_args()

    # Load .env and configure logs
    load_dotenv(override=True)
    configure_logging(log_level=args.log_level, console=True)

    # Print startup banner
    print("=" * 60)
    print(" 🧠 Oncorag2 Patient Data Chatbot ")
    print(" 💬 Ask questions about patients and get document-aware answers.")
    print("=" * 60)

    # Start chatbot
    try:
        conversation = PatientDataConversation(
            collection_name=args.collection,
            extracted_data_path=args.extracted_data,
            connection_string=args.connection_string,
            model_name=args.model,
            temperature=args.temperature
        )
        conversation.start_conversation(verbose=args.verbose)

        return 0
    except Exception as e:
        print(f"Fatal error starting chatbot: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())
