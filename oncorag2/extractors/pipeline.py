"""
Full extraction pipeline for oncology reports.

This module orchestrates the document processing, context extraction,
and feature extraction steps in a complete pipeline.
"""

import json
import logging
import os
from pathlib import Path
from typing import Dict, List, Optional, Union



from oncorag2.extractors.document_processor import DocumentProcessor
from oncorag2.extractors.context_extractor import ContextExtractor
from oncorag2.extractors.feature_extractor import FeatureExtractor
from oncorag2.utils.database import setup_iris_vector_store

logger = logging.getLogger(__name__)


class ExtractionPipeline:
    """
    End-to-end pipeline for extracting features from oncology reports.

    This class orchestrates the full workflow:
    1. Convert PDFs to markdown and redact sensitive info
    2. Extract context snippets based on feature definitions
    3. Store contexts in vector database
    4. Extract structured data using RAG techniques
    """

    def __init__(self,
                 features_json: Union[str, List[Dict]],
                 collection_name: str = "patient_contexts",
                 connection_string: Optional[str] = None,
                 reset_collection: bool = True):
        """
        Initialize the extraction pipeline.

        Args:
            features_json: Path to the features JSON file or loaded features
            collection_name: Name of the vector collection
            connection_string: Connection string for IRIS
            reset_collection: If True, delete the collection if it exists
        """
        self.collection_name = collection_name
        self.connection_string = connection_string

        # Load features
        if isinstance(features_json, str):
            with open(features_json, 'r') as f:
                features_data = json.load(f)
                self.features = features_data["features"]
        else:
            self.features = features_json

        # Initialize components
        self.document_processor = DocumentProcessor()
        self.context_extractor = ContextExtractor()

        # Set up vector database
        self.vector_db, self.conn_string = setup_iris_vector_store(
            collection_name=collection_name,
            connection_string=connection_string,
            reset_collection=reset_collection
        )

        # Initialize feature extractor with vector DB
        self.feature_extractor = FeatureExtractor(vector_db=self.vector_db)

        logger.info(f"Extraction pipeline initialized with {len(self.features)} features")

    def get_pdf_path_for_reference(self, patient_dir: Union[str, Path], reference_name: str) -> Optional[str]:
        """
        Find the PDF file in a patient directory that matches the reference name.

        Args:
            patient_dir: Path to the patient directory
            reference_name: Reference name from feature JSON

        Returns:
            Path to the PDF file, or None if not found
        """
        patient_dir = Path(patient_dir)

        # Check for exact filename match first
        pdf_exact = patient_dir / f"{reference_name}.pdf"
        if pdf_exact.exists():
            return str(pdf_exact)

        # Check for any PDF containing the reference name
        for file in os.listdir(patient_dir):
            if file.lower().startswith(reference_name.lower()) and file.endswith('.pdf'):
                return str(patient_dir / file)

        # If not found, just return any PDF in the directory as fallback
        for file in os.listdir(patient_dir):
            if file.endswith('.pdf'):
                return str(patient_dir / file)

        return None

    def process_patient_pdfs_to_vector_store(self,
                                             data_dir: Union[str, Path],
                                             patient_id: str,
                                             context_csv: Optional[str] = None,
                                             raw_context_csv: Optional[str] = None) -> int:
        """
        Process all relevant PDFs for a patient and add contexts to vector store.

        Args:
            data_dir: Path to the patient data directory
            patient_id: Patient ID
            context_csv: Path to the output CSV file for context data
            raw_context_csv: Path to the output CSV file for raw context data

        Returns:
            Number of contexts added to the vector store
        """
        patient_dir = Path(data_dir)
        processed_pdfs = set()
        markdown_cache = {}

        # Update context extractor with CSV paths
        self.context_extractor.context_csv = context_csv
        self.context_extractor.raw_context_csv = raw_context_csv

        # Get unique references from features
        references = set()
        for feature in self.features:
            ref = feature.get("reference", "")
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
            pdf_path = self.get_pdf_path_for_reference(patient_dir, reference)

            if not pdf_path or pdf_path in processed_pdfs:
                continue

            processed_pdfs.add(pdf_path)

            try:
                # Convert PDF to markdown with name redaction if not already in cache
                if pdf_path not in markdown_cache:
                    markdown_text = self.document_processor.get_markdown_text(pdf_path)
                    markdown_cache[pdf_path] = markdown_text
                else:
                    markdown_text = markdown_cache[pdf_path]

                # Extract contexts from the markdown text
                pdf_name = Path(pdf_path).name
                contexts, context_data = self.context_extractor.extract_contexts_from_text(
                    markdown_text, self.features, pdf_name
                )

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
                    self.context_extractor.save_raw_text(
                        markdown_text, pdf_name, patient_id, reference
                    )

            except Exception as e:
                logger.error(f"Error processing PDF {pdf_path}: {e}")

        # Add all contexts to the vector store
        if total_contexts:
            try:
                self.vector_db.add_documents(total_contexts)
                logger.info(f"Added {len(total_contexts)} contexts to vector store for patient {patient_id}")
            except Exception as e:
                logger.error(f"Error adding contexts to vector store: {e}")

        # Save context data to CSV if provided
        if context_csv and all_context_data:
            self.context_extractor.save_context_data(all_context_data)

        return len(total_contexts)

    def process_patient_data(self,
                             patient_dir: Union[str, Path],
                             output_csv: str = 'extracted_patient_data.csv') -> Dict:
        """
        Process a patient's data directory to extract features.

        Args:
            patient_dir: Path to the patient directory
            output_csv: Path to the output CSV file

        Returns:
            Dictionary of extracted data
        """
        patient_dir = Path(patient_dir)
        patient_id = patient_dir.name
        logger.info(f"Processing patient {patient_id}...")

        # Process features using vector similarity search
        all_results = self.feature_extractor.process_patient_features(
            self.features, patient_id
        )

        # Save results to CSV
        self.feature_extractor.save_to_csv(all_results, output_csv)

        return all_results

    def process_directory(self,
                          data_dir: Union[str, Path],
                          output_csv: str = 'extracted_data.csv',
                          context_csv: Optional[str] = 'context_data.csv',
                          raw_context_csv: Optional[str] = 'raw_context_data.csv') -> List[Dict]:
        """
        Process a directory containing patient data.

        Args:
            data_dir: Path to the directory containing patient subdirectories or PDF files
            output_csv: Path to the output CSV file for extracted data
            context_csv: Path to the output CSV file for context data
            raw_context_csv: Path to the output CSV file for raw context data

        Returns:
            List of dictionaries with extracted data
        """
        # Initialize output CSVs
        for csv_file in [output_csv, context_csv, raw_context_csv]:
            if csv_file and os.path.exists(csv_file):
                os.remove(csv_file)

        all_results = []
        data_dir = Path(data_dir)

        # Check if data_dir is a directory with patient subdirectories or a directory with PDFs
        if not data_dir.is_dir():
            logger.error(f"Error: {data_dir} is not a directory")
            return all_results

        # Check for PDF files directly in the data_dir
        pdf_files = [f for f in os.listdir(data_dir) if f.lower().endswith('.pdf')]

        if pdf_files:
            # This is a directory with PDF files - treat as a single patient
            patient_id = data_dir.name

            # Process PDFs and add contexts to vector store
            num_contexts = self.process_patient_pdfs_to_vector_store(
                data_dir,
                patient_id,
                context_csv,
                raw_context_csv
            )
            logger.info(f"Processed {num_contexts} contexts for patient {patient_id}")

            # Process features
            results = self.process_patient_data(data_dir, output_csv)
            all_results.append(results)
        else:
            # This is a directory with patient subdirectories
            patient_dirs = [d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))]

            # First pass: process all PDFs and add contexts to vector store
            for patient_dir in patient_dirs:
                full_patient_dir = os.path.join(data_dir, patient_dir)
                patient_id = Path(full_patient_dir).name

                num_contexts = self.process_patient_pdfs_to_vector_store(
                    full_patient_dir,
                    patient_id,
                    context_csv,
                    raw_context_csv
                )
                logger.info(f"Processed {num_contexts} contexts for patient {patient_id}")

            # Second pass: process features for each patient
            for patient_dir in patient_dirs:
                full_patient_dir = os.path.join(data_dir, patient_dir)
                results = self.process_patient_data(full_patient_dir, output_csv)
                all_results.append(results)

        logger.info(f"Processed {len(all_results)} patients. Results saved to {output_csv}")
        if context_csv:
            logger.info(f"Context data saved to {context_csv}")
        if raw_context_csv:
            logger.info(f"Raw context data saved to {raw_context_csv}")

        return all_results