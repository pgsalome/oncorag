"""
PDF utilities for document processing.

This module provides utility functions for working with PDF files,
especially for oncology reports and medical documentation.
"""

import os
import subprocess
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Convert a PDF to markdown using marker_single.

    Args:
        pdf_path: Path to the PDF file
        output_dir: Directory to save the markdown file.
                   If None, uses the directory of the PDF.

    Returns:
        Path to the generated markdown file
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
    markdown_path = output_dir / base_name / f"{base_name}.md"
    if os.path.isfile(markdown_path):
        logger.info(f"Using existing markdown file: {markdown_path}")
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


def read_markdown_file(markdown_path: Union[str, Path]) -> str:
    """
    Read the content of a markdown file.

    Args:
        markdown_path: Path to the markdown file

    Returns:
        Content of the markdown file as text
    """
    try:
        with open(markdown_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        logger.error(f"Error reading markdown file {markdown_path}: {e}")
        raise