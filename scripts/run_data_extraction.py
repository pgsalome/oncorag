#!/usr/bin/env python3
"""
Run Oncorag2 data extractor on patient records.
If no data directory is provided or is empty, synthetic data is generated.
"""
import os
import sys
import argparse
from pathlib import Path
from dotenv import load_dotenv
from oncorag2.extractors.pipeline import ExtractionPipeline
from oncorag2.utils.logging import configure_logging

# Load environment variables
load_dotenv(override=True)

def generate_synthetic_data(output_dir=Path("data/sample_patient")):
    """Generate synthetic notes if no real data is provided."""
    output_dir.mkdir(parents=True, exist_ok=True)
    try:
        from oncorag2.utils.generate_synthetic_data import create_synthetic_notes
        print(f"Generating synthetic patient data in {output_dir}...")
        create_synthetic_notes(output_dir)
        return output_dir if output_dir.exists() else None
    except ImportError:
        print("Synthetic data generator not available.")
        return None

def is_data_dir_valid(data_dir: Path) -> bool:
    """Check if the data directory exists and contains at least one PDF."""
    return data_dir.exists() and any(data_dir.glob("*.pdf"))

def main():
    parser = argparse.ArgumentParser(description="Extract features from oncology reports")
    parser.add_argument("-d", "--data-dir", help="Directory containing patient data")
    parser.add_argument("-f", "--features", default="config/example_feature_config.json",
                        help="Path to feature config JSON (default: config/example_feature_config.json)")
    parser.add_argument("-o", "--output", default="./output/extracted_data.csv")
    parser.add_argument("-c", "--context-csv", default="./output/context_data.csv")
    parser.add_argument("-r", "--raw-csv", default="./output/raw_context_data.csv")
    parser.add_argument("--collection", default="patient_contexts")
    parser.add_argument("--reset", action="store_true")
    parser.add_argument("--log-level", default="INFO",
                        choices=["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"])
    args = parser.parse_args()

    configure_logging(log_level=args.log_level, console=True)

    # Step 1: Resolve or generate data directory
    data_dir = Path(args.data_dir) if args.data_dir else Path("data/sample_patient")

    if not is_data_dir_valid(data_dir):
        print("No valid data directory or empty directory found. Generating synthetic data...")
        generate_synthetic_data(output_dir=data_dir)
        if not data_dir:
            print("Failed to generate synthetic data. Exiting.")
            return 1

    # Step 2: Verify feature config file exists
    features_path = Path(args.features)
    if not features_path.exists():
        print(f"Feature configuration file not found at {features_path}.")
        return 1

    # Step 3: Ensure output directories exist
    for path in [args.output, args.context_csv, args.raw_csv]:
        Path(path).parent.mkdir(parents=True, exist_ok=True)

    # Step 4: Run extraction pipeline
    try:
        print(f"Initializing pipeline with features: {features_path}")
        pipeline = ExtractionPipeline(
            features_json=str(features_path),
            collection_name=args.collection,
            reset_collection=args.reset
        )

        print(f"Processing data from: {data_dir}")
        results = pipeline.process_directory(
            data_dir= os.path.dirname(data_dir),
            output_csv=args.output,
            context_csv=args.context_csv,
            raw_context_csv=args.raw_csv
        )

        print("\nExtraction complete!")
        print(f"Processed {len(results)} patients")
        print(f"Results saved to: {args.output}")
        print(f"Context snippets saved to: {args.context_csv}")
        print(f"Raw context data saved to: {args.raw_csv}")
        return 0

    except Exception as e:
        print(f"Error running extraction pipeline: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == "__main__":
    sys.exit(main())
