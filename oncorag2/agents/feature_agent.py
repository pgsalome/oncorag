"""
Feature extraction agent for oncology reports.

This module provides a high-level API for generating clinical features
specific to oncology entities using large language models.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Set, Union

import openai
from smolagents import CodeAgent, LiteLLMModel

from oncorag2.agents.prompts import FEATURE_GENERATION_TASK
from oncorag2.agents.tools import (combine_all_features, format_features_for_display,
                                   generate_entity_features_batch,
                                   get_feature_names, UserInputTool)

logger = logging.getLogger(__name__)


class FeatureExtractionAgent:
    """
    Agent for generating entity-specific features for oncology reports.

    This agent uses LLMs to generate clinical features tailored to specific oncology
    entities, ensuring the features are formatted appropriately for extraction.
    """

    def __init__(
            self,
            model_id: str = "gpt-4o-mini",
            platform: str = "openai",
            api_key: Optional[str] = None,
            interactive: bool = False,
            universal_features_path: Optional[str] = None,
    ):
        """
        Initialize the feature extraction agent.

        Args:
            model_id: The model identifier to use (e.g., "gpt-4o-mini", "claude-3-opus-20240229")
            platform: The LLM platform to use (e.g., "openai", "anthropic")
            api_key: API key for the specified platform (defaults to environment variable)
            interactive: Whether to run in interactive mode
            universal_features_path: Path to JSON file with universal features
        """
        self.model_id = model_id
        self.platform = platform
        self.interactive = interactive

        # Set API key
        self._api_key = api_key or self._get_platform_api_key(platform)

        # Initialize variables to track features
        self.universal_features: List[Dict] = []
        self.entity_features: List[Dict] = []
        self.suggested_features: Set[str] = set()

        # Load universal features if provided
        if universal_features_path:
            self.load_universal_features(universal_features_path)
        else:
            # Try to load from default location
            config_path = Path("config/example_feature_config.json")
            if config_path.exists():
                self.load_universal_features(config_path)

        # Initialize LLM model and agent
        self._initialize_agent()

        logger.info(
            f"Feature extraction agent initialized with {platform}/{model_id} "
            f"and {len(self.universal_features)} universal features"
        )

    def _get_platform_api_key(self, platform: str) -> str:
        """Get the API key for the specified platform from environment variables."""
        env_var_map = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "groq": "GROQ_API_KEY",
            "huggingface": "HUGGINGFACE_API_KEY",
            "cohere": "COHERE_API_KEY",
        }

        env_var = env_var_map.get(platform.lower())
        if not env_var:
            raise ValueError(f"Unsupported platform: {platform}")

        api_key = os.environ.get(env_var)
        if not api_key:
            raise ValueError(f"API key not found for {platform}. Set {env_var} environment variable.")

        return api_key

    def _initialize_agent(self) -> None:
        """Initialize the LLM model and agent with appropriate tools."""
        # Define tools for the agent
        tools = [
            UserInputTool(),
            generate_entity_features_batch,
            get_feature_names,
            format_features_for_display,
            combine_all_features
        ]

        # Initialize the model
        model = LiteLLMModel(
            model_id=self.model_id,
            api_key=self._api_key
        )

        # Initialize the agent
        self.agent = CodeAgent(
            model=model,
            tools=tools,
            verbosity_level=1 if self.interactive else 0
        )

        # Set global variables for tools to access
        global UNIVERSAL_FEATURES, ENTITY_FEATURES, SUGGESTED_FEATURES
        UNIVERSAL_FEATURES = self.universal_features
        ENTITY_FEATURES = self.entity_features
        SUGGESTED_FEATURES = self.suggested_features

    def load_universal_features(self, config_path: Union[str, Path]) -> List[Dict]:
        """
        Load universal features from a JSON configuration file.

        Args:
            config_path: Path to the JSON file with universal features

        Returns:
            List of feature dictionaries
        """
        try:
            with open(config_path, "r") as f:
                data = json.load(f)
                self.universal_features = data.get("features", [])

                # Extract feature names to avoid duplicates
                for feature in self.universal_features:
                    full_name = feature.get("name", "").lower()
                    self.suggested_features.add(full_name)

                    # Also add base name (without number suffix)
                    name_parts = full_name.split("_")
                    if len(name_parts) > 0:
                        base_name = name_parts[0].lower()
                        self.suggested_features.add(base_name)

                logger.info(f"Loaded {len(self.universal_features)} universal features from {config_path}")
                return self.universal_features
        except (FileNotFoundError, json.JSONDecodeError) as e:
            logger.error(f"Error loading universal features: {str(e)}")
            return []

    def generate_features(
            self,
            entity: Optional[str] = None,
            areas_of_interest: Optional[str] = "",
            max_batches: int = 5
    ) -> List[Dict]:
        """
        Generate entity-specific features for the given oncology entity.

        Args:
            entity: The oncology entity (e.g., "lung cancer", "melanoma")
                   If None, will prompt the user in interactive mode
            areas_of_interest: Specific areas of interest for feature generation
                               If None, will prompt the user in interactive mode
            max_batches: Maximum number of feature batches to generate

        Returns:
            List of generated feature dictionaries
        """
        # Update global variables
        global UNIVERSAL_FEATURES, ENTITY_FEATURES, SUGGESTED_FEATURES
        UNIVERSAL_FEATURES = self.universal_features
        ENTITY_FEATURES = self.entity_features
        SUGGESTED_FEATURES = self.suggested_features

        # Determine if we need to ask for entity and areas_of_interest
        if self.interactive and entity is None:
            # Ask for entity if not provided
            user_message = "What oncology entity would you like to generate features for? (e.g., 'lung cancer', 'breast cancer', 'melanoma')"

            print("\n" + user_message)
            entity = input("> ").strip()

            if not entity:
                logger.warning("No entity provided. Using default 'cancer'")
                entity = "cancer"

            # Ask for areas of interest if not provided
            if areas_of_interest is None or areas_of_interest == "":
                user_message = f"Are there any specific areas of interest for {entity} you'd like to focus on? (press Enter to skip)"
                print("\n" + user_message)
                areas_of_interest = input("> ").strip()

        # Ensure we have an entity at this point
        if not entity:
            raise ValueError("Entity must be provided either as an argument or interactively")

        # Build the agent task
        if self.interactive:
            task = FEATURE_GENERATION_TASK.format(
                entity=entity,
                areas_of_interest=areas_of_interest or "general clinical features",
                max_batches=max_batches
            )
        else:
            # Use non-interactive task format
            task = FEATURE_GENERATION_NONINTERACTIVE_TASK.format(
                entity=entity,
                areas_of_interest=areas_of_interest or "general clinical features",
                max_batches=max_batches,
                num_features=5 * max_batches  # Default to 5 features per batch
            )

        # Run the agent to generate features
        logger.info(f"Generating features for {entity}")
        result = self.agent.run(task)

        # Process the result
        try:
            # If result is a string or the agent returned JSON as string
            if isinstance(result, str):
                if "```json" in result:
                    json_str = result.split("```json")[1].split("```")[0].strip()
                elif "```" in result:
                    json_str = result.split("```")[1].split("```")[0].strip()
                else:
                    json_str = result.strip()

                parsed = json.loads(json_str)
            else:
                # Result is already a dictionary
                parsed = result

            # Store the generated features in the instance
            self.entity_features = [f for f in ENTITY_FEATURES if isinstance(f, dict)]
            logger.info(f"Generated {len(self.entity_features)} entity-specific features")

            return self.entity_features

        except Exception as e:
            logger.error(f"Error processing feature generation result: {str(e)}")
            if isinstance(result, str):
                logger.debug(f"Raw result: {result[:500]}...")

            # Return whatever features we managed to generate
            self.entity_features = [f for f in ENTITY_FEATURES if isinstance(f, dict)]
            return self.entity_features

    def save_features(self, output_path: Union[str, Path]) -> bool:
        """
        Save generated features to a JSON file.

        Args:
            output_path: Path to save the JSON file

        Returns:
            True if successful, False otherwise
        """
        try:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # Combine universal and entity-specific features
            combined_features = combine_all_features()

            with open(output_path, "w") as f:
                json.dump(combined_features, f, indent=2)

            logger.info(f"Saved {len(combined_features.get('features', []))} features to {output_path}")
            return True

        except Exception as e:
            logger.error(f"Error saving features: {str(e)}")
            return False

    def get_combined_features(self) -> Dict[str, List[Dict]]:
        """
        Get a dictionary containing both universal and entity-specific features.

        Returns:
            Dictionary with a "features" key containing all features
        """
        return combine_all_features()