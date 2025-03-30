"""
Tool functions for the feature extraction agent.

These tools are used by the CodeAgent to generate, format, and manipulate
oncology-specific features.
"""

import re
from typing import Dict, List, Optional, Set, Union

from smolagents import tool

# Global variables to store feature data
# These are updated by the FeatureExtractionAgent
UNIVERSAL_FEATURES: List[Dict] = []
ENTITY_FEATURES: List[Dict] = []
SUGGESTED_FEATURES: Set[str] = set()


class UserInputTool:
    """Tool for getting user input during feature generation."""

    def __call__(self, question: str) -> str:
        """
        Get input from the user.

        Args:
            question: Question to ask the user

        Returns:
            User's response
        """
        return input(f"\n{question}\n> ")


@tool
def generate_entity_features_batch(
        entity: str,
        areas_of_interest: str = "",
        batch_number: int = 1,
        previously_suggested: Optional[List[str]] = None
) -> Dict:
    """
    Generate a batch of 5 entity-specific features for the given oncology entity.

    Args:
        entity: The oncology entity type (e.g., "lung cancer", "melanoma")
        areas_of_interest: Optional specific areas the user is interested in
        batch_number: Which batch of features to generate (1=first 5, 2=next 5, etc.)
        previously_suggested: List of previously suggested feature names to avoid duplicates

    Returns:
        A dictionary with a batch of entity-specific features and info about universal features
    """
    import openai
    import os
    import json

    # Access the global feature tracking variables
    global SUGGESTED_FEATURES, UNIVERSAL_FEATURES, ENTITY_FEATURES

    # Use OpenAI directly for generation
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # If previously_suggested list was provided, add to global set
    if previously_suggested:
        for feature in previously_suggested:
            SUGGESTED_FEATURES.add(feature.lower())

    # Calculate how many features we've already shown
    features_seen = (batch_number - 1) * 5

    # Universal feature names to avoid duplicates
    universal_feature_bases = set()
    for f in UNIVERSAL_FEATURES:
        name = f.get("name", "").lower()
        # Remove number suffix if present
        if "_" in name and name.split("_")[-1].isdigit():
            base_name = "_".join(name.split("_")[:-1])
        else:
            base_name = name
        universal_feature_bases.add(base_name)

    prompt = f"""
    Generate EXACTLY 5 NEW features specifically relevant to {entity}, focusing on {areas_of_interest if areas_of_interest else "general clinical features"}.

    This is batch #{batch_number} (features {features_seen + 1}-{features_seen + 5}).

    DO NOT DUPLICATE any of these features that have already been suggested or that exist in the universal features.

    ALREADY EXISTING UNIVERSAL FEATURES:
    {", ".join(sorted(universal_feature_bases))}

    PREVIOUSLY SUGGESTED FEATURES:
    {", ".join(sorted(SUGGESTED_FEATURES))}

    ALSO AVOID any features that are conceptually similar to those already suggested.

    Each feature must follow this structure:
    {{
      "name": "FEATURE_NAME", // Use underscores instead of spaces, like "egfr_mutation_status"
      "description": "Detailed explanation",
      "feature_group": "A relevant grouping category",
      "expected_output_type": "string|numeric|date|categorical",
      "unit": "N/A or appropriate unit",
      "input_prompt": "A direct question to the user",
      "expected_range": "Expected values",
      "reference": "summary",
      "fallback_category_1": "notes",
      "fallback_category_2": "diagnosis",
      "double_check": "no",
      "search_everywhere": true,
      "category": "Clinical category",
      "regex_patterns": [
        "(?i)Pattern1[:\\\\s]*(value1|value2)",
        "(?i)\\\\bPattern2\\\\b"
      ]
    }}

    Return ONLY the JSON with EXACTLY 5 entity-specific features in an array with the format:
    {{"entity_specific_features": [feature1, feature2, feature3, feature4, feature5]}}
    """

    try:
        # Call the OpenAI API
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are a clinical feature extraction specialist."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.7,
        )

        # Parse response content
        content = response.choices[0].message.content

        # Try to extract JSON
        try:
            # Find JSON content (might be wrapped in markdown code blocks)
            if "```json" in content:
                json_str = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                json_str = content.split("```")[1].split("```")[0].strip()
            else:
                json_str = content.strip()

            entity_data = json.loads(json_str)

            # Get the newly generated features
            new_features = entity_data.get("entity_specific_features", [])

            # Find the highest existing feature number
            highest_number = 0
            for feature in UNIVERSAL_FEATURES:
                name = feature.get("name", "")
                parts = name.split("_")
                if len(parts) > 0 and parts[-1].isdigit():
                    try:
                        number = int(parts[-1])
                        if number > highest_number:
                            highest_number = number
                    except ValueError:
                        pass

            # Process each new feature
            for feature in new_features:
                # Ensure name uses underscores instead of spaces
                feature_name = feature.get("name", "").replace(" ", "_").lower()

                # Add to suggested features set
                base_name = feature_name.split("_")[0] if "_" in feature_name else feature_name
                SUGGESTED_FEATURES.add(base_name)

                # Get the feature number if it exists
                if "_" in feature_name and feature_name.split("_")[-1].isdigit():
                    # Keep the existing number
                    pass
                else:
                    # Assign a new number based on highest existing feature number
                    highest_number += 1
                    feature["name"] = f"{feature_name}_{highest_number}"

                # Ensure all required fields are present
                required_fields = [
                    "description", "feature_group", "expected_output_type",
                    "unit", "input_prompt", "expected_range", "reference",
                    "fallback_category_1", "fallback_category_2", "double_check",
                    "search_everywhere", "category", "regex_patterns"
                ]

                for field in required_fields:
                    if field not in feature or feature[field] is None:
                        if field == "description":
                            continue  # Description should already be there
                        elif field == "feature_group":
                            feature[field] = "Biomarkers"
                        elif field == "expected_output_type":
                            feature[field] = "categorical"  # Default to categorical for biomarkers
                        elif field == "unit":
                            feature[field] = "N/A"
                        elif field == "input_prompt":
                            feature[field] = f"What is the {feature_name.replace('_', ' ')} value?"
                        elif field == "expected_range":
                            if "status" in feature_name.lower():
                                feature[field] = "Positive, Negative, Unknown"
                            elif "mutation" in feature_name.lower():
                                feature[field] = "Mutated, Not Mutated, Unknown"
                            elif "level" in feature_name.lower() or "expression" in feature_name.lower():
                                feature[field] = "0-100"
                            else:
                                feature[field] = "N/A"
                        elif field == "reference":
                            feature[field] = "summary"
                        elif field == "fallback_category_1":
                            feature[field] = "notes"
                        elif field == "fallback_category_2":
                            feature[field] = "diagnosis"
                        elif field == "double_check":
                            feature[field] = "no"
                        elif field == "search_everywhere":
                            feature[field] = True
                        elif field == "category":
                            feature[field] = "Clinical category"
                        elif field == "regex_patterns":
                            base_term = feature_name.replace("_", " ")
                            if feature.get("expected_output_type") == "categorical":
                                values = feature.get("expected_range", "").split(",")
                                values = [v.strip() for v in values]
                                value_pattern = "|".join(values)
                                feature[field] = [
                                    f"(?i){base_term}[:\\s]*({value_pattern})",
                                    f"(?i)\\b{base_term}\\b"
                                ]
                            else:
                                feature[field] = [
                                    f"(?i){base_term}[:\\s]*(.*?)",
                                    f"(?i)\\b{base_term}\\b"
                                ]

            # Add the features to the global list
            if not isinstance(ENTITY_FEATURES, list):
                ENTITY_FEATURES = []
            ENTITY_FEATURES.extend(new_features)

            # Return batch of features
            return {
                "batch_number": batch_number,
                "entity_specific_features": new_features,
                "entity_feature_count": len(ENTITY_FEATURES)
            }

        except json.JSONDecodeError:
            # Return error if can't parse
            return {
                "batch_number": batch_number,
                "entity_specific_features": [],
                "error": "Could not parse entity-specific features.",
                "entity_feature_count": len(ENTITY_FEATURES) if isinstance(ENTITY_FEATURES, list) else 0
            }

    except Exception as e:
        # Return error if API fails
        return {
            "batch_number": batch_number,
            "entity_specific_features": [],
            "error": f"Error generating features: {str(e)}",
            "entity_feature_count": len(ENTITY_FEATURES) if isinstance(ENTITY_FEATURES, list) else 0
        }


@tool
def get_feature_names(features_batch: Dict) -> List[str]:
    """
    Extract the names of features from a features batch.

    Args:
        features_batch: A batch of features returned by generate_entity_features_batch

    Returns:
        List of feature names
    """
    return [f.get("name") for f in features_batch.get("entity_specific_features", [])]


@tool
def format_features_for_display(features_batch: Dict) -> str:
    """
    Format features in a clean, readable format for display to the user.

    Args:
        features_batch: A batch of features returned by generate_entity_features_batch

    Returns:
        Formatted string with features in a clean, list format
    """
    features = features_batch.get("entity_specific_features", [])
    if not features:
        return "No features available."

    formatted_output = ""
    for i, feature in enumerate(features):
        # Get the base name without number suffix
        name_parts = feature.get("name", "").split("_")
        if len(name_parts) > 1 and name_parts[-1].isdigit():
            # Remove the number suffix but keep base parts
            display_name = " ".join(name_parts[:-1])
        else:
            display_name = feature.get("name", "").replace("_", " ")

        # Capitalize each word for display
        display_name = " ".join(part.title() for part in display_name.split())

        description = feature.get("description", "")
        expected_type = feature.get("expected_output_type", "")
        expected_range = feature.get("expected_range", "")

        formatted_output += f"* **{display_name}**\n"
        formatted_output += f"   * Description: {description}\n"

        # Format the type information based on whether it's categorical or numeric
        if expected_type.lower() == "categorical":
            formatted_output += f"   * Type: {expected_type.title()} ({expected_range})\n"
        elif expected_type.lower() == "numeric" and feature.get("unit", ""):
            formatted_output += f"   * Type: {expected_type.title()} ({feature.get('unit')}, {expected_range})\n"
        else:
            formatted_output += f"   * Type: {expected_type.title()}\n"

        if i < len(features) - 1:
            formatted_output += "\n"

    return formatted_output


@tool
def combine_all_features(entity_features_batches: Optional[List] = None) -> Dict[str, List[Dict]]:
    """
    Combine all entity-specific features with existing universal features.

    Args:
        entity_features_batches: Optional list of batches of entity-specific features
                                (not used in this implementation)

    Returns:
        Dictionary with all features (existing universal + new entity-specific)
    """
    # Use the global entity features and universal features
    global ENTITY_FEATURES, UNIVERSAL_FEATURES

    # Ensure the global entity features list exists
    if not isinstance(ENTITY_FEATURES, list):
        ENTITY_FEATURES = []

    # Start with a copy of the universal features
    combined_features = UNIVERSAL_FEATURES.copy()

    # Find highest existing feature number
    highest_number = 0
    for feature in combined_features:
        name = feature.get("name", "")
        parts = name.split("_")
        if len(parts) > 0 and parts[-1].isdigit():
            try:
                number = int(parts[-1])
                if number > highest_number:
                    highest_number = number
            except ValueError:
                pass

    # Process entity features to ensure proper formatting and all fields
    for feature in ENTITY_FEATURES:
        # Ensure name uses underscores, not spaces
        feature_name = feature.get("name", "").replace(" ", "_")

        # Check if name already has a number suffix
        if "_" in feature_name and feature_name.split("_")[-1].isdigit():
            base_name = "_".join(feature_name.split("_")[:-1])
        else:
            base_name = feature_name
            # Add number suffix
            highest_number += 1
            feature_name = f"{base_name}_{highest_number}"

        feature["name"] = feature_name

        # Make sure all required fields are present with valid values
        if "description" not in feature or not feature["description"]:
            feature["description"] = f"Description for {feature_name}"

        if "feature_group" not in feature or not feature["feature_group"]:
            feature["feature_group"] = "Biomarkers"

        if "expected_output_type" not in feature or not feature["expected_output_type"]:
            if "type" in feature:
                type_str = feature["type"]
                if "categorical" in type_str.lower():
                    feature["expected_output_type"] = "categorical"
                elif "numeric" in type_str.lower():
                    feature["expected_output_type"] = "numeric"
                else:
                    feature["expected_output_type"] = "string"
            else:
                feature["expected_output_type"] = "string"

        if "unit" not in feature or not feature["unit"]:
            feature["unit"] = "N/A"

        if "input_prompt" not in feature or not feature["input_prompt"]:
            feature["input_prompt"] = f"What is the {base_name.replace('_', ' ')} value?"

        if "expected_range" not in feature or not feature["expected_range"]:
            if "type" in feature:
                # Try to extract from type field
                match = re.search(r'\((.*?)\)', feature.get("type", ""))
                if match:
                    range_str = match.group(1)
                    if "," in range_str and feature["expected_output_type"] == "numeric":
                        # Numeric with unit and range
                        parts = range_str.split(",")
                        if len(parts) >= 2:
                            feature["unit"] = parts[0].strip()
                            feature["expected_range"] = parts[1].strip()
                    else:
                        feature["expected_range"] = range_str
                else:
                    # Default values based on name patterns
                    if "status" in feature_name.lower():
                        feature["expected_range"] = "Positive, Negative, Unknown"
                    elif "mutation" in feature_name.lower():
                        feature["expected_range"] = "Mutated, Not Mutated, Unknown"
                    elif "level" in feature_name.lower() or "expression" in feature_name.lower():
                        feature["expected_range"] = "0-100"
                    else:
                        feature["expected_range"] = "N/A"
            else:
                # Default values based on name patterns
                if "status" in feature_name.lower():
                    feature["expected_range"] = "Positive, Negative, Unknown"
                elif "mutation" in feature_name.lower():
                    feature["expected_range"] = "Mutated, Not Mutated, Unknown"
                elif "level" in feature_name.lower() or "expression" in feature_name.lower():
                    feature["expected_range"] = "0-100"
                else:
                    feature["expected_range"] = "N/A"

        # Add remaining fields if missing
        if "reference" not in feature or not feature["reference"]:
            feature["reference"] = "summary"

        if "fallback_category_1" not in feature or not feature["fallback_category_1"]:
            feature["fallback_category_1"] = "notes"

        if "fallback_category_2" not in feature or not feature["fallback_category_2"]:
            feature["fallback_category_2"] = "diagnosis"

        if "double_check" not in feature or not feature["double_check"]:
            feature["double_check"] = "no"

        if "search_everywhere" not in feature or feature["search_everywhere"] is None:
            feature["search_everywhere"] = True

        if "category" not in feature or not feature["category"]:
            feature["category"] = "Clinical category"

        if "regex_patterns" not in feature or not feature["regex_patterns"]:
            base_term = base_name.replace("_", " ")
            if feature["expected_output_type"] == "categorical":
                values = feature["expected_range"].split(",")
                values = [v.strip() for v in values]
                value_pattern = "|".join(values)
                feature["regex_patterns"] = [
                    f"(?i){base_term}[:\\s]*({value_pattern})",
                    f"(?i)\\b{base_term}\\b"
                ]
            else:
                feature["regex_patterns"] = [
                    f"(?i){base_term}[:\\s]*(.*?)",
                    f"(?i)\\b{base_term}\\b"
                ]

        # Remove any non-standard fields like "type"
        if "type" in feature:
            del feature["type"]

        # Add to combined features
        combined_features.append(feature)

    # Return combined features
    return {"features": combined_features}