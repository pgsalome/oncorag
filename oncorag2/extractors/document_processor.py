"""
Document processing utilities for oncology reports.

This module handles PDF conversion to markdown and document processing
with privacy-preserving redaction of patient identifiable information.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union

from oncorag2.utils.redaction import redact_names_from_markdown

logger = logging.getLogger(__name__)


class DocumentProcessor:
    """
    Processes PDF documents for feature extraction.

    This class handles converting PDFs to markdown and redacting sensitive information.
    """

    def __init__(self, output_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the document processor.

        Args:
            output_dir: Directory to save processed documents (default: same as input)
        """
        self.output_dir = Path(output_dir) if output_dir else None
        self.markdown_cache: Dict[str, str] = {}
        logger.info(f"Document processor initialized with output dir: {output_dir}")

    def convert_pdf_to_markdown(self, pdf_path: Union[str, Path]) -> str:
        """
        Convert a PDF file to markdown using marker_single.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Path to the generated markdown file
        """
        # Setup paths
        pdf_path = Path(pdf_path)

        if self.output_dir is None:
            output_dir = pdf_path.parent
        else:
            output_dir = self.output_dir

        # Ensure output directory exists
        os.makedirs(output_dir, exist_ok=True)

        # Get the base filename without extension
        base_name = pdf_path.stem

        # Expected output markdown path
        markdown_path = output_dir / base_name / f"{base_name}.md"
        if os.path.isfile(markdown_path):
            logger.info(f"Found existing markdown file: {markdown_path}")
            return str(markdown_path)

        # Run marker_single to convert PDF to markdown
        cmd = ["marker_single", str(pdf_path), "--output_dir", str(output_dir)]
        logger.info(f"Running command: {' '.join(cmd)}")

        try:
            result = subprocess.run(cmd, check=True, capture_output=True, text=True)
            logger.info(f"Conversion successful for {pdf_path}")
        except subprocess.CalledProcessError as e:
            logger.error(f"Error converting PDF: {e}")
            logger.error(f"Error output: {e.stderr}")
            raise

        # Check if markdown file was created
        if not markdown_path.exists():
            raise FileNotFoundError(f"Expected markdown file not found at {markdown_path}")

        return str(markdown_path)

    def convert_pdf_to_markdown_with_redaction(self, pdf_path: Union[str, Path]) -> str:
        """
        Convert a PDF to markdown and redact sensitive information.

        Args:
            pdf_path: Path to the PDF file

        Returns:
            Path to the redacted markdown file
        """
        # First convert PDF to markdown
        markdown_path = self.convert_pdf_to_markdown(pdf_path)

        # Read the markdown content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        # Redact names from the markdown text
        redacted_text = redact_names_from_markdown(markdown_text)

        # Write the redacted content back to the file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(redacted_text)

        logger.info(f"Names redacted in {markdown_path}")

        return markdown_path

    def get_markdown_text(self, pdf_path: Union[str, Path], use_cache: bool = True) -> str:
        """
        Get the markdown text from a PDF, using cache if available.

        Args:
            pdf_path: Path to the PDF file
            use_cache: Whether to use cached markdown text

        Returns:
            The markdown text content
        """
        pdf_path_str = str(pdf_path)

        # Return from cache if available and requested
        if use_cache and pdf_path_str in self.markdown_cache:
            logger.debug(f"Using cached markdown for {pdf_path_str}")
            return self.markdown_cache[pdf_path_str]

        # Convert and redact
        markdown_path = self.convert_pdf_to_markdown_with_redaction(pdf_path)

        # Read the content
        with open(markdown_path, 'r', encoding='utf-8') as f:
            markdown_text = f.read()

        # Cache the result
        self.markdown_cache[pdf_path_str] = markdown_text

        return markdown_text

    def process_directory(self, directory: Union[str, Path]) -> Dict[str, str]:
        """
        Process all PDF files in a directory.

        Args:
            directory: Directory containing PDF files

        Returns:
            Dictionary mapping PDF paths to markdown content
        """
        directory = Path(directory)
        result = {}

        # Find all PDF files
        pdf_files = list(directory.glob("**/*.pdf"))

        if not pdf_files:
            logger.warning(f"No PDF files found in {directory}")
            return result

        # Process each PDF
        for pdf_path in pdf_files:
            try:
                markdown_text = self.get_markdown_text(pdf_path)
                result[str(pdf_path)] = markdown_text
                logger.info(f"Processed {pdf_path}")
            except Exception as e:
                logger.error(f"Error processing {pdf_path}: {str(e)}")

        return result

    def clear_cache(self) -> None:
        """Clear the markdown cache."""
        self.markdown_cache.clear()
        logger.debug("Markdown cache cleared")