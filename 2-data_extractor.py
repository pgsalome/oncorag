import spacy
import re
import os
import re
import json
import pandas as pd
from pathlib import Path
import subprocess
import getpass
from dotenv import load_dotenv
from langchain.docstore.document import Document
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_iris import IRISVector
import os
# Set the environment variable to allow iris import to work with containerized IRIS
os.environ['IRISINSTALLDIR'] = '/usr'


# Set API Keys (ensure your environment has these set)
os.environ['LANGCHAIN_TRACING_V2'] = 'true'
os.environ['LANGCHAIN_ENDPOINT'] = 'https://api.smith.langchain.com'

# Load environment variables
load_dotenv(override=True)

def convert_pdf_to_markdown(pdf_path, output_dir=None):
    """
    Converts a PDF to markdown using marker_single.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the markdown file.
                                   If None, uses the directory of the PDF.


    Returns:
        str: Path to the generated markdown file
    """
    # Setup paths
    pdf_path = Path(pdf_path)

    if output_dir is None:
        output_dir = pdf_path.parent
    else:
        output_dir = Path(output_dir)

    # Ensure output directory exists
    os.makedirs(output_dir, exist_ok=True)

    # Get the base filename without extension
    base_name = pdf_path.stem

    # Expected output markdown path
    markdown_path = output_dir / base_name/f"{base_name}.md"
    if os.path.isfile(markdown_path):
        return str(markdown_path)

    # Run marker_single to convert PDF to markdown
    cmd = ["marker_single", str(pdf_path), "--output_dir", str(output_dir)]
    print(f"Running command: {' '.join(cmd)}")

    try:
        result = subprocess.run(cmd, check=True, capture_output=True, text=True)
        print(f"Conversion successful for {pdf_path}")
    except subprocess.CalledProcessError as e:
        print(f"Error converting PDF: {e}")
        print(f"Error output: {e.stderr}")
        raise

    # Check if markdown file was created
    if not markdown_path.exists():
        raise FileNotFoundError(f"Expected markdown file not found at {markdown_path}")

    return str(markdown_path)


def extract_context_from_match(text, match, window_size=200):
    """Extract context around a regex match with specified window size."""
    start = max(0, match.start() - window_size)
    end = min(len(text), match.end() + window_size)

    context = text[start:end]

    # Try to expand to complete sentences
    if start > 0:
        # Find the first sentence start before our window
        prev_text = text[max(0, start - 100):start]
        sentence_starts = list(re.finditer(r'[.!?]\s+[A-Z]', prev_text))
        if sentence_starts:
            last_start = sentence_starts[-1]
            start = max(0, start - 100) + last_start.end() - 1
            context = text[start:end]

    if end < len(text):
        # Find the first sentence end after our window
        next_text = text[end:min(len(text), end + 100)]
        sentence_end = re.search(r'[.!?]\s+', next_text)
        if sentence_end:
            end = end + sentence_end.start() + 1
            context = text[start:end]

    return context.strip()


def extract_contexts_from_text(markdown_text, features, pdf_name):
    """Extract contexts using regex patterns for all features."""
    all_contexts = []
    all_context_data = []  # For CSV output

    for feature in features[:5]:
        feature_name = feature["name"]
        feature_desc = feature["description"]

        if "regex_patterns" in feature:
            for pattern in feature["regex_patterns"]:
                try:
                    for match in re.finditer(pattern, markdown_text, re.IGNORECASE):
                        context = extract_context_from_match(markdown_text, match)
                        matched_text = match.group(0)

                        if context:
                            # Create a document with metadata
                            doc = Document(
                                page_content=context,
                                metadata={
                                    "feature_name": feature_name,
                                    "feature_desc": feature_desc,
                                    "pattern": pattern,
                                    "matched_text": matched_text,
                                    "input_prompt": feature["input_prompt"],
                                    "expected_output_type": feature["expected_output_type"],
                                    "expected_range": feature["expected_range"],
                                    "pdf_source": pdf_name
                                }
                            )
                            all_contexts.append(doc)

                            # Add context data for CSV output
                            all_context_data.append({
                                "feature_name": feature_name,
                                "feature_desc": feature_desc,
                                "pattern": pattern,
                                "matched_text": matched_text,
                                "context": context,
                                "pdf_source": pdf_name
                            })
                except re.error as e:
                    print(f"Error in regex pattern for feature {feature_name}: {e}")
                    continue

    return all_contexts, all_context_data


def setup_iris_vector_store(collection_name="oncorag", connection_string=None, reset_collection=True):
    """
    Set up an IRIS vector store connection with option to reset the collection.

    Args:
        collection_name (str): Name of the vector collection
        connection_string (str): Connection string for IRIS
        reset_collection (bool): If True, delete the collection if it exists

    Returns:
        tuple: (IRISVector database instance, connection string)
    """
    # Default connection string if not provided
    if connection_string is None:
        username = '_SYSTEM'  # Changed from 'admin'
        password = 'SYS'  # Changed from 'supersecret'
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
            print(f"Deleting existing vector collection: {collection_name}")
            db.delete_collection()
            print(f"Successfully deleted collection: {collection_name}")
        except Exception as e:
            # If error occurs, collection might not exist
            print(f"No existing collection to delete or error occurred: {e}")

    # Create a new vector store
    try:
        print(f"Creating new vector store: {collection_name}")
        db = IRISVector.from_documents(
            embedding=embeddings,
            documents=[Document(page_content="Initialization document")],
            collection_name=collection_name,
            connection_string=connection_string,
        )

        # Try to remove the initialization document
        # Not all vector stores support this, so we'll catch exceptions
        try:
            # This might not work depending on the IRISVector implementation
            # We're making an educated guess based on common vector store patterns
            db.delete(["Initialization document"])
        except:
            print("Note: Could not remove initialization document, but collection was created.")
    except Exception as e:
        print(f"Error creating vector store: {e}")
        print("Attempting to connect to existing store instead...")

        # Connect to existing vector store
        db = IRISVector(
            embedding_function=embeddings,
            collection_name=collection_name,
            connection_string=connection_string,
        )

    return db, connection_string


def add_contexts_to_vector_store(db, contexts):
    """Add context documents to the vector store."""
    if contexts:
        db.add_documents(contexts)
        print(f"Added {len(contexts)} contexts to vector store")
    return db


def get_pdf_path_for_reference(patient_dir, reference_name):
    """
    Find the PDF file in a patient directory that matches the reference name.

    Args:
        patient_dir (str): Path to the patient directory
        reference_name (str): Reference name from feature JSON

    Returns:
        str: Path to the PDF file, or None if not found
    """
    # Check for exact filename match first
    pdf_exact = Path(patient_dir) / f"{reference_name}.pdf"
    if pdf_exact.exists():
        return str(pdf_exact)

    # Check for any PDF containing the reference name
    for file in os.listdir(patient_dir):
        if file.lower().startswith(reference_name.lower()) and file.endswith('.pdf'):
            return str(Path(patient_dir) / file)

    # If not found, just return any PDF in the directory as fallback
    for file in os.listdir(patient_dir):
        if file.endswith('.pdf'):
            return str(Path(patient_dir) / file)

    return None


def redact_names_from_markdown(markdown_text):
    """
    Redact names from markdown text using regex patterns.

    Args:
        markdown_text (str): The markdown text to redact names from

    Returns:
        str: Markdown text with names redacted
    """
    import re
    import spacy

    # Try to load spaCy model for NER
    try:
        nlp = spacy.load("en_core_web_sm")
    except:
        # If model not installed, attempt to download it
        import subprocess
        print("Downloading spaCy model for name detection...")
        subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
        nlp = spacy.load("en_core_web_sm")

    # Common patterns for names in medical documents
    name_patterns = [
        # Pattern for "Name: John Smith" or "Patient Name: John Smith"
        r'(?i)(patient\s+)?name\s*[:]\s*([A-Z][a-z]+\s+[A-Z][a-z]+)',
        # Pattern for "Dr. Smith" or similar
        r'(?i)(dr|doctor|md|physician|nurse|rn|pa|np)\s*\.?\s*([A-Z][a-z]+)',
        # Pattern for "Smith, John" format (last name first)
        r'([A-Z][a-z]+),\s*([A-Z][a-z]+)',
        # Simple pattern for potential names (two capitalized words in sequence)
        r'\b([A-Z][a-z]{1,20})\s+([A-Z][a-z]{1,20})\b'
    ]

    # First pass: use regex patterns to find common name formats
    potential_names = set()

    for pattern in name_patterns:
        matches = re.finditer(pattern, markdown_text)
        for match in matches:
            if len(match.groups()) >= 1:
                for group in match.groups()[1:]:  # Skip the first group as it's usually a title
                    if group and len(group) > 2:  # Avoid very short matches
                        potential_names.add(group)

    # Second pass: use spaCy for named entity recognition
    doc = nlp(markdown_text)

    # Extract person names from spaCy NER
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            potential_names.add(ent.text)

    # Sort names by length (longest first) to avoid partial replacements
    sorted_names = sorted(potential_names, key=len, reverse=True)

    # Replace each identified name with [REDACTED]
    redacted_text = markdown_text
    for name in sorted_names:
        # Ensure we're replacing complete words by using word boundaries
        pattern = r'\b' + re.escape(name) + r'\b'
        redacted_text = re.sub(pattern, '[REDACTED]', redacted_text)

    return redacted_text


# Update the existing convert_pdf_to_markdown function to include name redaction
def convert_pdf_to_markdown_with_redaction(pdf_path, output_dir=None):
    """
    Converts a PDF to markdown using marker_single and redacts names.

    Args:
        pdf_path (str): Path to the PDF file
        output_dir (str, optional): Directory to save the markdown file.
                                   If None, uses the directory of the PDF.

    Returns:
        str: Path to the generated markdown file
    """
    # First convert PDF to markdown using the existing function
    markdown_path = convert_pdf_to_markdown(pdf_path, output_dir)

    # Read the markdown content
    with open(markdown_path, 'r', encoding='utf-8') as f:
        markdown_text = f.read()

    # Redact names from the markdown text
    redacted_text = redact_names_from_markdown(markdown_text)

    # Write the redacted content back to the file
    with open(markdown_path, 'w', encoding='utf-8') as f:
        f.write(redacted_text)

    print(f"Names redacted in {markdown_path}")

    return markdown_path

# Update the process_patient_pdfs_to_vector_store function to use the redaction functionality
def process_patient_pdfs_to_vector_store(data_dir, features, db, patient_id, context_csv=None, raw_context_csv=None):
    """Process all relevant PDFs for a patient and add contexts to vector store."""
    patient_dir = Path(data_dir)
    processed_pdfs = set()
    markdown_cache = {}

    # Get unique references from features
    references = set()
    for feature in features:
        ref = feature.get("reference", "summary")
        fallback1 = feature.get("fallback_category_1", "")
        fallback2 = feature.get("fallback_category_2", "")

        references.add(ref)
        if fallback1:
            references.add(fallback1)
        if fallback2:
            references.add(fallback2)

    # Process each reference
    total_contexts = []
    all_context_data = []

    for reference in references:
        pdf_path = get_pdf_path_for_reference(patient_dir, reference)

        if not pdf_path or pdf_path in processed_pdfs:
            continue

        processed_pdfs.add(pdf_path)

        try:
            # Convert PDF to markdown with name redaction if not already in cache
            if pdf_path not in markdown_cache:
                # Use the new redaction function instead of the original
                markdown_path = convert_pdf_to_markdown_with_redaction(pdf_path, patient_dir)
                with open(markdown_path, 'r', encoding='utf-8') as f:
                    markdown_text = f.read()
                markdown_cache[pdf_path] = markdown_text
            else:
                markdown_text = markdown_cache[pdf_path]

            # Extract contexts from the markdown text
            pdf_name = Path(pdf_path).name
            contexts, context_data = extract_contexts_from_text(markdown_text, features, pdf_name)

            # Update metadata with patient_id
            for doc in contexts:
                doc.metadata["patient_id"] = patient_id
                doc.metadata["reference"] = reference

            # Update context data with patient_id
            for item in context_data:
                item["patient_id"] = patient_id
                item["reference"] = reference

            total_contexts.extend(contexts)
            all_context_data.extend(context_data)

            # Save raw markdown text if CSV is provided
            if raw_context_csv:
                raw_df = pd.DataFrame([{
                    "patient_id": patient_id,
                    "pdf_source": pdf_name,
                    "reference": reference,
                    "markdown_text": markdown_text
                }])

                file_exists = os.path.isfile(raw_context_csv)
                raw_df.to_csv(raw_context_csv, mode='a', header=not file_exists, index=False)

        except Exception as e:
            print(f"Error processing PDF {pdf_path}: {e}")

    # Add all contexts to the vector store
    add_contexts_to_vector_store(db, total_contexts)

    # Save context data to CSV if provided
    if context_csv and all_context_data:
        df = pd.DataFrame(all_context_data)
        file_exists = os.path.isfile(context_csv)
        df.to_csv(context_csv, mode='a', header=not file_exists, index=False)

    return len(total_contexts)


def retrieve_feature_data_from_vector_store(db, feature, patient_id):
    """
    Retrieve and process information for a specific feature using vector similarity search.
    Implements fallback logic to check alternative document references when primary search returns "Missing".

    Args:
        db: Vector database
        feature: Feature dictionary
        patient_id: Patient ID to filter results

    Returns:
        str: The extracted information
    """
    # Construct a query based on the feature description and input prompt
    query = f"{feature['description']} {feature['input_prompt']}"

    # Get feature name and reference documents
    feature_name = feature["name"]
    primary_reference = feature.get("reference", "summary")
    fallback1 = feature.get("fallback_category_1", "")
    fallback2 = feature.get("fallback_category_2", "")

    # Initialize LLM
    llm = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0)

    # Define the prompt template for RAG
    template = """Answer the task based only on the following contexts:
    {context}

    Feature Description: {feature_desc}

    Task: {query}

    Expected Output Type: {expected_output_type}
    Expected Output Range: {expected_range}

    If the information is not available in the provided contexts, respond only with "Missing"
    """
    prompt = ChatPromptTemplate.from_template(template)

    # Try primary reference first
    result = search_reference(db, query, feature, patient_id, primary_reference, llm, prompt)

    # If primary reference returns "Missing" and fallback1 is provided, try fallback1
    if result.strip().lower() == "missing" and fallback1:
        print(f"Primary reference returned Missing for {feature_name}. Trying fallback_category_1: {fallback1}")
        result = search_reference(db, query, feature, patient_id, fallback1, llm, prompt)

    # If fallback1 returns "Missing" and fallback2 is provided, try fallback2
    if result.strip().lower() == "missing" and fallback2:
        print(f"Fallback1 returned Missing for {feature_name}. Trying fallback_category_2: {fallback2}")
        result = search_reference(db, query, feature, patient_id, fallback2, llm, prompt)

    # If all attempts return "Missing", try one final search without reference filter
    if result.strip().lower() == "missing":
        print(f"All reference categories returned Missing for {feature_name}. Trying search across all documents.")
        result = search_all_documents(db, query, feature, patient_id, llm, prompt)

    return result


def search_reference(db, query, feature, patient_id, reference, llm, prompt):
    """
    Search for information in a specific reference document.

    Args:
        db: Vector database
        query: Search query
        feature: Feature dictionary
        patient_id: Patient ID
        reference: Document reference to search in
        llm: Language model instance
        prompt: Prompt template

    Returns:
        str: The extracted information or "Missing"
    """
    feature_name = feature["name"]

    try:
        # Retrieve relevant documents with both feature_name and reference filters
        docs_with_score = db.similarity_search_with_score(
            query,
            k=3,  # Get top 3 results
            filter={"patient_id": patient_id, "feature_name": feature_name, "reference": reference}
        )

        # If no documents found with feature_name filter, try just with reference filter
        if not docs_with_score:
            docs_with_score = db.similarity_search_with_score(
                query,
                k=3,
                filter={"patient_id": patient_id, "reference": reference}
            )

    except Exception as e:
        print(f"Error querying vector store for feature {feature_name} in reference {reference}: {e}")
        docs_with_score = []

    # If no results, return "Missing"
    if not docs_with_score:
        return "Missing"

    # Combine contexts from retrieved documents
    contexts = []
    for doc, score in docs_with_score:
        contexts.append(f"Context (similarity score: {score}):\n{doc.page_content}")

    combined_context = "\n\n" + "\n\n".join(contexts)

    # Format the prompt
    formatted_prompt = prompt.format(
        query=feature["input_prompt"],
        context=combined_context,
        feature_desc=feature["description"],
        expected_output_type=feature["expected_output_type"],
        expected_range=feature["expected_range"]
    )

    # Get the response from the language model
    try:
        response = llm.invoke(formatted_prompt).content
        return response
    except Exception as e:
        print(f"Error processing feature {feature_name} with reference {reference}: {e}")
        return "Error in processing"


def search_all_documents(db, query, feature, patient_id, llm, prompt):
    """
    Search across all documents as a last resort.

    Args:
        db: Vector database
        query: Search query
        feature: Feature dictionary
        patient_id: Patient ID
        llm: Language model instance
        prompt: Prompt template

    Returns:
        str: The extracted information or "Missing"
    """
    feature_name = feature["name"]

    try:
        # Retrieve relevant documents with only patient_id filter
        docs_with_score = db.similarity_search_with_score(
            query,
            k=5,  # Get top 5 results for broader search
            filter={"patient_id": patient_id}
        )
    except Exception as e:
        print(f"Error querying all documents for feature {feature_name}: {e}")
        docs_with_score = []

    # If no results, return "Missing"
    if not docs_with_score:
        return "Missing"

    # Combine contexts from retrieved documents
    contexts = []
    for doc, score in docs_with_score:
        contexts.append(f"Context (similarity score: {score}):\n{doc.page_content}")

    combined_context = "\n\n" + "\n\n".join(contexts)

    # Format the prompt
    formatted_prompt = prompt.format(
        query=feature["input_prompt"],
        context=combined_context,
        feature_desc=feature["description"],
        expected_output_type=feature["expected_output_type"],
        expected_range=feature["expected_range"]
    )

    # Get the response from the language model
    try:
        response = llm.invoke(formatted_prompt).content
        return response
    except Exception as e:
        print(f"Error processing feature {feature_name} across all documents: {e}")
        return "Error in processing"


def clean_extracted_data(data):
    """Clean up the extracted data."""
    cleaned_data = {}
    for key, value in data.items():
        if isinstance(value, str):
            # If the value looks like a tuple ('label', 'value'), clean it
            if value.startswith("('") and value.endswith("')"):
                value = value.split(",")[1].strip(" '")
            # Additional cleanup for any unwanted characters
            value = value.replace("('", "").replace("', '", "").replace("')", "").strip()
        cleaned_data[key] = value
    return cleaned_data


def save_to_csv(extracted_data, output_csv='extracted_data.csv'):
    """Save extracted data to CSV."""
    # Check if file exists to determine if header should be written
    file_exists = os.path.isfile(output_csv)

    df = pd.DataFrame([extracted_data])
    df.to_csv(output_csv, mode='a', header=not file_exists, index=False)
    print(f"Data saved to {output_csv}")


def process_patient_data(patient_dir, features_json, vector_db, output_csv='extracted_patient_data.csv'):
    """
    Process a patient's data directory.

    Args:
        patient_dir (str): Path to the patient directory
        features_json (str or list): Path to the features JSON file or loaded features
        vector_db: Vector database connection
        output_csv (str): Path to the output CSV file

    Returns:
        dict: Dictionary of extracted data
    """
    patient_dir = Path(patient_dir)
    patient_id = patient_dir.name
    print(f"Processing patient {patient_id}...")

    # Load features if path is provided
    if isinstance(features_json, str):
        with open(features_json, 'r') as f:
            features_data = json.load(f)
            features = features_data["features"]
    else:
        features = features_json

    # Process each feature using vector similarity search
    all_results = {"patient_id": patient_id}

    for feature in features:
        feature_name = feature["name"]
        print(f"Processing feature: {feature_name}")

        # Retrieve feature data using vector similarity search
        result = retrieve_feature_data_from_vector_store(vector_db, feature, patient_id)
        all_results[feature_name] = result

    # Save results to CSV
    save_to_csv(all_results, output_csv)

    return all_results




def process_directory(data_dir, features_json, output_csv='extracted_data.csv', context_csv='context_data.csv',
                      raw_context_csv='raw_context_data.csv', collection_name='patient_contexts'):
    """
    Process a directory containing patient data.

    Args:
        data_dir (str): Path to the directory containing patient subdirectories or PDF files
        features_json (str): Path to the features JSON file
        output_csv (str): Path to the output CSV file for extracted data
        context_csv (str): Path to the output CSV file for context data
        raw_context_csv (str): Path to the output CSV file for raw context data
        collection_name (str): Name of the vector collection to use

    Returns:
        list: List of dictionaries with extracted data
    """
    # Initialize output CSVs
    for csv_file in [output_csv, context_csv, raw_context_csv]:
        if csv_file and os.path.exists(csv_file):
            os.remove(csv_file)


    # Load features
    with open(features_json, 'r') as f:
        features_data = json.load(f)
        features = features_data["features"][:2]

    # Set up vector database
    vector_db, conn_string = setup_iris_vector_store(collection_name)

    all_results = []
    data_dir = Path(data_dir)

    # Check if data_dir is a directory with patient subdirectories or a directory with PDFs
    if not data_dir.is_dir():
        print(f"Error: {data_dir} is not a directory")
        return all_results

    # Check for PDF files directly in the data_dir
    pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]

    if pdf_files:
        # This is a directory with PDF files - treat as a single patient
        patient_id = data_dir.name

        # Process PDFs and add contexts to vector store
        num_contexts = process_patient_pdfs_to_vector_store(
            data_dir,
            features,
            vector_db,
            patient_id,
            context_csv,
            raw_context_csv
        )
        print(f"Processed {num_contexts} contexts for patient {patient_id}")

        # Process features
        results = process_patient_data(data_dir, features, vector_db, output_csv)
        all_results.append(results)
    else:
        # This is a directory with patient subdirectories
        patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

        # First pass: process all PDFs and add contexts to vector store
        for patient_dir in patient_dirs:
            full_patient_dir = os.path.join(data_dir, patient_dir)
            patient_id = Path(full_patient_dir).name

            num_contexts = process_patient_pdfs_to_vector_store(
                full_patient_dir,
                features,
                vector_db,
                patient_id,
                context_csv,
                raw_context_csv
            )
            print(f"Processed {num_contexts} contexts for patient {patient_id}")

        # Second pass: process features for each patient
        for patient_dir in patient_dirs:
            full_patient_dir = os.path.join(data_dir, patient_dir)
            results = process_patient_data(full_patient_dir, features, vector_db, output_csv)
            all_results.append(results)

    print(f"Processed {len(all_results)} patients. Results saved to {output_csv}")
    if context_csv:
        print(f"Context data saved to {context_csv}")
    if raw_context_csv:
        print(f"Raw context data saved to {raw_context_csv}")

    return all_results



# Define paths


features_json = '/home/patrick/projects/git/oncorag/config/feature_list_new.json' # Path to your features JSON file
data_dir = '/home/patrick/projects/git/oncorag/data'       # Directory with patient subdirectories or PDFs
output_csv = '/home/patrick/projects/git/oncorag/output/extracted_data120.csv'
context_csv = '/home/patrick/projects/git/oncorag/output/context_data.csv'                 # CSV for extracted contexts
raw_context_csv = '/home/patrick/projects/git/oncorag/output/raw_context_data.csv'         # CSV for raw markdown content

# Process the directory
process_directory(data_dir, features_json, output_csv, context_csv, raw_context_csv)


# Example usage
# if __name__ == "__main__":
#     import argparse
#
#     parser = argparse.ArgumentParser(
#         description="Extract patient data using regex patterns and vector similarity search")
#     parser.add_argument("--data_dir", required=True, help="Directory containing patient data")
#     parser.add_argument("--features_json", required=True, help="Path to features JSON file")
#     parser.add_argument("--output_csv", default="extracted_data.csv", help="Output CSV file for extracted data")
#     parser.add_argument("--context_csv", default="context_data.csv", help="Output CSV file for context data")
#     parser.add_argument("--raw_context_csv", default="raw_context_data.csv",
#                         help="Output CSV file for raw context data")
#     parser.add_argument("--collection_name", default="patient_contexts", help="Vector collection name")
#
#     args = parser.parse_args()
#
#     # Process the directory
#     results = process_directory(
#         args.data_dir,
#         args.features_json,
#         args.output_csv,
#         args.context_csv,
#         args.raw_context_csv,
#         args.collection_name
#     )