#!/usr/bin/env python
"""
Quick script to run the feature generator directly.
"""

import os
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from oncorag2.agents.feature_agent import FeatureExtractionAgent
from oncorag2.utils.logging import configure_logging


def main():
    # Set up logging
    configure_logging(log_level="INFO", console=True)

    # Initialize the agent
    print("Feature Extraction Assistant")
    print("----------------------------")

    model_id = os.environ.get("ONCORAG2_MODEL", "gpt-4o-mini")

    agent = FeatureExtractionAgent(
        model_id=model_id,
        interactive=True,  # Always interactive in this simple script
    )

    print(f"Loaded {len(agent.universal_features)} universal features")
    print("These features will be used as a reference and will be included in the final output")
    print()

    # Generate features (this will prompt the user for entity and areas of interest)
    features = agent.generate_features()

    # Ask for output filename
    entity_name = input("\nWhat oncology entity name should be used for the config file? ").strip().lower()
    if not entity_name:
        entity_name = "oncology_entity"

    # Convert spaces to underscores for filename
    filename = entity_name.replace(" ", "_")
    output_path = f"config/{filename}_feature_config.json"

    # Save the features
    agent.save_features(output_path)

    print(f"\nFeatures saved to {output_path}")
    print(f"Total features: {len(agent.get_combined_features().get('features', []))}")


if __name__ == "__main__":
    main()