import os
import json
import openai
from smolagents import (
    CodeAgent,
    LiteLLMModel,
    UserInputTool,
    tool
)


###############################################################################
# 0. LOAD UNIVERSAL FEATURES FROM CONFIG FILE
###############################################################################
def load_universal_features():
    """Load the universal features from the config file."""
    try:
        with open("config/example_feature_config.json", "r") as f:
            data = json.load(f)
            return data.get("features", [])
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading universal features: {e}")
        return []

def save_universal_features(features):
    """Save the updated universal features to the config file."""
    try:
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_path = os.path.join(config_dir, "example_feature_config.json")
        with open(config_path, "w") as f:
            json.dump({"features": features}, f, indent=2)
        print(f"Updated example_feature_config.json with {len(features)} features")
        return True
    except Exception as e:
        print(f"Error saving universal features: {e}")
        return False

# Modified function to save entity-specific features without updating the universal config
def save_entity_features(entity_name, features):
    """Save entity-specific features to a separate config file."""
    try:
        config_dir = "config"
        if not os.path.exists(config_dir):
            os.makedirs(config_dir)

        config_path = os.path.join(config_dir, f"{entity_name}_feature_config.json")

        # Check if all required fields are present
        for i, feature in enumerate(features):
            # Ensure all required fields are present
            if "name" not in feature:
                feature["name"] = f"unknown_feature_{i+1}"

            # Ensure name has underscores, not spaces
            feature["name"] = feature["name"].replace(" ", "_")

            # Add any missing required fields
            required_fields = [
                "description", "feature_group", "expected_output_type",
                "unit", "input_prompt", "expected_range", "reference",
                "fallback_category_1", "fallback_category_2", "double_check",
                "search_everywhere", "category", "regex_patterns"
            ]

            for field in required_fields:
                if field not in feature or feature[field] is None:
                    if field == "description":
                        feature[field] = f"Description for {feature['name']}"
                    elif field == "feature_group":
                        feature[field] = "Biomarkers"
                    elif field == "expected_output_type":
                        feature[field] = "string"
                    elif field == "unit":
                        feature[field] = "N/A"
                    elif field == "input_prompt":
                        feature[field] = f"What is the {feature['name'].replace('_', ' ')} value?"
                    elif field == "expected_range":
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
                        base_term = feature["name"].replace("_", " ")
                        feature[field] = [
                            f"(?i){base_term}[:\\s]*(.*?)",
                            f"(?i)\\b{base_term}\\b"
                        ]

        with open(config_path, "w") as f:
            json.dump({"features": features}, f, indent=2)
        print(f"Created {entity_name}_feature_config.json with {len(features)} features")
        return True
    except Exception as e:
        print(f"Error saving entity features: {e}")
        return False

# Load universal features at startup (quietly)
UNIVERSAL_FEATURES = load_universal_features()

# Extract all feature names (base names and full names) from universal features
UNIVERSAL_FEATURE_NAMES = set()
UNIVERSAL_FEATURE_FULL_NAMES = set()
for f in UNIVERSAL_FEATURES:
    full_name = f.get("name", "").lower()
    UNIVERSAL_FEATURE_FULL_NAMES.add(full_name)

    # Also add base name (without number suffix)
    name_parts = full_name.split("_")
    if len(name_parts) > 0:
        base_name = name_parts[0].lower()
        UNIVERSAL_FEATURE_NAMES.add(base_name)

# This will track features we've already suggested to avoid duplicates
SUGGESTED_FEATURES = set()
# This will store the entity-specific features we generate
ENTITY_FEATURES = []

###############################################################################
# 1. AGENT INSTRUCTIONS
###############################################################################
task = """
You are a clinical feature extraction specialist.

Have a conversational interaction with the user to:
1. Ask which oncology entity they're working with
2. Ask if they have any specific areas of interest for this entity
3. Generate 5 relevant entity-specific features at a time
4. After generating each set of 5 features, use the format_features_for_display tool to show them in a clean format

IMPORTANT GUIDELINES:
- NEVER suggest duplicate features that have already been shown or that are already in the universal features list
- NEVER suggest features that are conceptually similar to existing features
- Each feature should contribute unique value and measure a distinct clinical aspect
- Present features in small batches of 5 with clear names and descriptions USING ONLY the format_features_for_display tool
- When showing features, MAKE SURE to print the output from format_features_for_display in your response
- DO NOT show any extra information, logs, or technical details along with the features

ALWAYS FOLLOW THIS EXACT FORMAT FOR YOUR CODE BLOCKS:

Thoughts: I need to [action description]
Code:
```py
# Python code here
features_batch = generate_entity_features_batch(entity="entity_name", areas_of_interest="interest_area", batch_number=1)
formatted_features = format_features_for_display(features_batch)
print(formatted_features)  # This must be included to show the features
user_input_result = user_input(question="Would you like more features or are these sufficient?")
```

The agent MUST follow the above format with the exact structure - the smolagents parser requires this exact format.
ALWAYS include the print(formatted_features) line to ensure the features are displayed.

When finished, return the complete feature set as JSON.
"""

###############################################################################
# 2. TOOL THAT USES THE LLM TO GENERATE ENTITY-SPECIFIC FEATURES
###############################################################################
@tool
def generate_entity_features_batch(entity: str, areas_of_interest: str = "", batch_number: int = 1, previously_suggested: list = None) -> dict:
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
    # Use OpenAI directly for generation
    client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

    # Get all existing feature names to avoid duplicates
    global SUGGESTED_FEATURES, UNIVERSAL_FEATURES, UNIVERSAL_FEATURE_NAMES, UNIVERSAL_FEATURE_FULL_NAMES, ENTITY_FEATURES

    # If previously_suggested list was provided, add to global set
    if previously_suggested:
        for feature in previously_suggested:
            SUGGESTED_FEATURES.add(feature.lower())

    # Calculate how many features we've already shown
    features_seen = (batch_number - 1) * 5

    # Add all universal feature names to the suggested features set to avoid duplicates
    for f in UNIVERSAL_FEATURES:
        full_name = f.get("name", "").lower()
        # Extract base name (without number suffix)
        if "_" in full_name and full_name.split("_")[-1].isdigit():
            base_name = "_".join(full_name.split("_")[:-1]).lower()
        else:
            base_name = full_name.split("_")[0].lower() if "_" in full_name else full_name.lower()

        # Add both the full name and base name to suggested features
        SUGGESTED_FEATURES.add(base_name)

        # Also add common variations - replace underscores with spaces and vice versa
        SUGGESTED_FEATURES.add(base_name.replace("_", " "))
        SUGGESTED_FEATURES.add(base_name.replace(" ", "_"))

    # Create a complete list of all features to avoid (universal + previously suggested)
    all_features_to_avoid = UNIVERSAL_FEATURE_NAMES.union(SUGGESTED_FEATURES).union(UNIVERSAL_FEATURE_FULL_NAMES)

    # Extract all universal feature base names for clear display in the prompt
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
    
    This is batch #{batch_number} (features {features_seen+1}-{features_seen+5}).
    
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
            messages=[{"role": "system", "content": "You are a clinical feature extraction specialist."},
                      {"role": "user", "content": prompt}],
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

            # Create a collection for entity-specific features that we'll track separately
            entity_features = []

            # Add new features to our set of suggested features but don't modify UNIVERSAL_FEATURES
            for feature in new_features:
                # Ensure name uses underscores instead of spaces
                feature_name = feature.get("name", "").replace(" ", "_").lower()

                # Add to suggested features set
                base_name = feature_name.split("_")[0] if "_" in feature_name else feature_name
                SUGGESTED_FEATURES.add(base_name)
                UNIVERSAL_FEATURE_FULL_NAMES.add(feature_name)

                # Get the feature number if it exists
                if "_" in feature_name and feature_name.split("_")[-1].isdigit():
                    # Keep the existing number
                    pass
                else:
                    # Assign a new number based on highest existing feature number
                    highest_number += 1
                    feature["name"] = f"{feature_name}_{highest_number}"
                    feature_name = feature["name"].lower()
                    UNIVERSAL_FEATURE_FULL_NAMES.add(feature_name)

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

                # Add to entity features list
                entity_features.append(feature)

            # Add the features to the global list
            if not isinstance(ENTITY_FEATURES, list):
                ENTITY_FEATURES = []
            ENTITY_FEATURES.extend(entity_features)

            # Create a preview of the features to print to the console
            print(f"Generated {len(new_features)} new features for {entity} (total: {len(ENTITY_FEATURES)})")

            # Directly format the features for display to ensure they appear
            feature_display = ""
            for i, feature in enumerate(new_features):
                # Process feature name for display
                name = feature.get("name", "")
                parts = name.split("_")
                if parts[-1].isdigit():
                    display_name = " ".join(parts[:-1])
                else:
                    display_name = name.replace("_", " ")
                display_name = " ".join(word.title() for word in display_name.split())

                feature_display += f"* **{display_name}**\n"
                feature_display += f"   * Description: {feature.get('description', '')}\n"

                output_type = feature.get('expected_output_type', '').title()
                range_info = feature.get('expected_range', '')
                unit = feature.get('unit', '')

                if output_type.lower() == "categorical":
                    feature_display += f"   * Type: {output_type} ({range_info})\n"
                elif output_type.lower() == "numeric" and unit:
                    feature_display += f"   * Type: {output_type} ({unit}, {range_info})\n"
                else:
                    feature_display += f"   * Type: {output_type}\n"

                if i < len(new_features) - 1:
                    feature_display += "\n"

            # Print the features directly from this function
            print("\nGenerated features:\n")
            #print(feature_display)
            #print("\n")

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

###############################################################################
# 3. TOOL TO GET FEATURE NAMES FROM A BATCH
###############################################################################
@tool
def get_feature_names(features_batch: dict) -> list:
    """
    Extract the names of features from a features batch.

    Args:
        features_batch: A batch of features returned by generate_entity_features_batch

    Returns:
        List of feature names
    """
    return [f.get("name") for f in features_batch.get("entity_specific_features", [])]

@tool
def format_features_for_display(features_batch: dict) -> str:
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

    # Print the formatted output directly from the tool to ensure it appears in the log
    print("\n" + formatted_output + "\n")

    return formatted_output

###############################################################################
# 4. TOOL TO COMBINE ALL FEATURES INTO FINAL RESULT
###############################################################################
@tool
def combine_all_features(entity_features_batches: list = None) -> dict:
    """
    Combine all entity-specific features with existing universal features.

    Args:
        entity_features_batches: Optional list of batches of entity-specific features (not used in this implementation)

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
                import re
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

###############################################################################
# 5. COMBINE TOOLS AND INSTANTIATE THE AGENT
###############################################################################
tools = [
    UserInputTool(),
    generate_entity_features_batch,
    get_feature_names,
    format_features_for_display,
    combine_all_features
]

model = LiteLLMModel(
    model_id="gpt-4o-mini",
    api_key=os.getenv("OPENAI_API_KEY")
)

agent = CodeAgent(
    model=model,
    tools=tools,
    verbosity_level=0
)

###############################################################################
# 6. RUN THE AGENT AND OUTPUT THE JSON
###############################################################################
if __name__ == "__main__":
    print("Feature Extraction Assistant")
    print("----------------------------")
    print(f"Loaded {len(UNIVERSAL_FEATURES)} universal features")
    print("These features will be used as a reference and will be included in the final output")
    print()

    result = agent.run(task)

    try:
        # Check if result is already a dictionary
        if isinstance(result, dict):
            parsed = result
        else:
            # Try to extract just the JSON portion if it's a string
            if "```json" in result:
                result = result.split("```json")[1].split("```")[0].strip()
            elif "```" in result:
                result = result.split("```")[1].split("```")[0].strip()
            # Parse JSON
            parsed = json.loads(result)

        # Get entity name from the first input request since we don't have conversation_history
        entity_name = "Tripple negative breast"

        # Try to find entity name from first user input in a more direct way
        try:
            # Directly ask the user for the entity name to use in the filename
            print("\nWhat oncology entity name should be used for the config file?")
            direct_entity_input = input("Enter entity name: ").strip().lower()
            if direct_entity_input:
                entity_name = direct_entity_input.replace(" ", "_")
        except:
            # If direct input fails, use a default name
            print("Using default entity name")

        # Combine the features (universal + entity-specific)
        combined_features = combine_all_features()

        # Save combined features to the entity-specific config file
        save_entity_features(entity_name, combined_features["features"])

        print(f"\nFeatures saved to config/{entity_name}_feature_config.json")
        #print(f"Total features: {len(combined_features['features'])} ({len(UNIVERSAL_FEATURES)} universal + {len(ENTITY_FEATURES)} entity-specific)")
    except Exception as e:
        print(f"\nError processing result: {str(e)}")
        print("\nRaw result:\n", result)