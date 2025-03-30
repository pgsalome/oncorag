"""
Command-line interface for Oncorag2.

This module provides a comprehensive CLI for the Oncorag2 package,
including feature generation, extraction, and querying functionality.
"""

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

from oncorag2.agents.feature_agent import FeatureExtractionAgent
from oncorag2.extractors.pipeline import ExtractionPipeline
from oncorag2.chatbot.conversation import PatientDataConversation
from oncorag2.utils.logging import configure_logging


def generate_features_cmd(args: argparse.Namespace) -> None:
    """Generate entity-specific features."""
    logger = logging.getLogger("oncorag2.cli")

    logger.info(f"Generating features for entity: {args.entity}")
    agent = FeatureExtractionAgent(
        model_id=args.model,
        platform=args.platform,
        interactive=args.interactive
    )

    features = agent.generate_features(
        entity=args.entity,
        areas_of_interest=args.areas_of_interest
    )

    if args.output:
        agent.save_features(args.output)
        logger.info(f"Generated features saved to {args.output}")

    print(f"\nGenerated {len(features)} features for {args.entity}")


def extract_data_cmd(args: argparse.Namespace) -> None:
    """Extract features from patient documents."""
    logger = logging.getLogger("oncorag2.cli")

    logger.info(f"Extracting data from: {args.data_dir}")
    pipeline = ExtractionPipeline(
        features_json=args.features,
        collection_name=args.collection_name
    )

    results = pipeline.process_directory(
        data_dir=args.data_dir,
        output_csv=args.output,
        context_csv=args.context_output,
        raw_context_csv=args.raw_output
    )

    print(f"\nExtracted data for {len(results)} patients to {args.output}")


def chat_cmd(args: argparse.Namespace) -> None:
    """Start interactive chat interface for patient data queries."""
    logger = logging.getLogger("oncorag2.cli")

    logger.info(f"Starting chat interface with data from: {args.extracted_data}")
    conversation = PatientDataConversation(
        collection_name=args.collection_name,
        extracted_data_path=args.extracted_data
    )

    conversation.start_conversation()


def setup_parser() -> argparse.ArgumentParser:
    """Set up the command-line argument parser."""
    parser = argparse.ArgumentParser(
        description="Oncorag2 - Oncology Report Analysis with Generative AI"
    )

    # Global arguments
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"],
                        help="Logging level")
    parser.add_argument("--log-file", help="Log file path")

    # Subcommands
    subparsers = parser.add_subparsers(title="commands", dest="command")

    # Generate features command
    gen_parser = subparsers.add_parser("generate", help="Generate entity-specific features")
    gen_parser.add_argument("--entity", "-e", required=True,
                            help="Oncology entity type (e.g., 'lung cancer')")
    gen_parser.add_argument("--areas-of-interest", "-a",
                            help="Specific areas of interest")
    gen_parser.add_argument("--model", "-m", default="gpt-4o-mini",
                            help="Model to use")
    gen_parser.add_argument("--platform", "-p", default="openai",
                            help="Platform (openai, anthropic, etc.)")
    gen_parser.add_argument("--interactive", "-i", action="store_true",
                            help="Interactive mode")
    gen_parser.add_argument("--output", "-o",
                            help="Output JSON file for features")
    gen_parser.set_defaults(func=generate_features_cmd)

    # Extract data command
    extract_parser = subparsers.add_parser("extract", help="Extract data from patient documents")
    extract_parser.add_argument("--data-dir", "-d", required=True,
                                help="Directory with patient data")
    extract_parser.add_argument("--features", "-f", required=True,
                                help="Features JSON file")
    extract_parser.add_argument("--output", "-o", default="extracted_data.csv",
                                help="Output CSV file")
    extract_parser.add_argument("--context-output", "-c", default="context_data.csv",
                                help="Context CSV file")
    extract_parser.add_argument("--raw-output", "-r", default="raw_context_data.csv",
                                help="Raw context CSV file")
    extract_parser.add_argument("--collection-name", default="patient_contexts",
                                help="Vector collection name")
    extract_parser.set_defaults(func=extract_data_cmd)

    # Chat command
    chat_parser = subparsers.add_parser("chat", help="Start chat interface for patient data queries")
    chat_parser.add_argument("--extracted-data", "-e", required=True,
                             help="Path to extracted data CSV")
    chat_parser.add_argument("--collection-name", default="patient_contexts",
                             help="IRIS vector collection name")
    chat_parser.set_defaults(func=chat_cmd)

    return parser


def main() -> None:
    """Main entry point for the CLI."""
    parser = setup_parser()
    args = parser.parse_args()

    # Configure logging
    configure_logging(
        log_level=args.log_level,
        log_file=args.log_file,
        console=True
    )

    # If no command specified, print help
    if not args.command:
        parser.print_help()
        return

    # Call the appropriate function
    args.func(args)


if __name__ == "__main__":
    main()