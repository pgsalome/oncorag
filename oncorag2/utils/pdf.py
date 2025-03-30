"""
PDF utilities for document processing.
This module provides utility functions for working with PDF files,
especially for oncology reports and medical documentation.
"""

import os
import logging
from pathlib import Path
from typing import Optional, Union

logger = logging.getLogger(__name__)


def convert_pdf_to_markdown(
        pdf_path: Union[str, Path],
        output_dir: Optional[Union[str, Path]] = None
) -> str:
    """
    Convert a PDF to markdown using pdfplumber.

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

    # Create subdirectory structure similar to the original
    base_dir = output_dir / base_name
    os.makedirs(base_dir, exist_ok=True)

    # Expected output markdown path
    markdown_path = base_dir / f"{base_name}.md"

    # If file already exists, just return the path
    if os.path.isfile(markdown_path):
        logger.info(f"Using existing markdown file: {markdown_path}")
        return str(markdown_path)

    # Try to import pdfplumber
    try:
        import pdfplumber
    except ImportError:
        logger.error("pdfplumber not installed. Install with: pip install pdfplumber")
        raise ImportError("pdfplumber package is required. Install with: pip install pdfplumber")

    # Extract text and tables from PDF
    try:
        logger.info(f"Converting PDF to markdown using pdfplumber: {pdf_path}")

        with pdfplumber.open(pdf_path) as pdf:
            text_content = f"# {base_name}\n\n"

            for i, page in enumerate(pdf.pages):
                text_content += f"## Page {i + 1}\n\n"

                # Extract text
                text = page.extract_text()
                if text:
                    text_content += text + "\n\n"

                # Extract tables
                tables = page.extract_tables()
                for table in tables:
                    if table and len(table) > 0 and any(table[0]):
                        # Create markdown table header
                        text_content += "| " + " | ".join(str(cell or "") for cell in table[0]) + " |\n"
                        text_content += "| " + " | ".join("---" for _ in table[0]) + " |\n"

                        # Create table rows
                        for row in table[1:]:
                            if row and any(row):
                                text_content += "| " + " | ".join(str(cell or "") for cell in row) + " |\n"

                        text_content += "\n"

        # Write to markdown file
        with open(markdown_path, 'w', encoding='utf-8') as f:
            f.write(text_content)

        logger.info(f"PDF successfully converted to markdown: {markdown_path}")
        return str(markdown_path)

    except Exception as e:
        logger.error(f"Error converting PDF to markdown with pdfplumber: {e}")

        # Fallback to simple PyPDF2 extraction if pdfplumber fails
        try:
            import PyPDF2
            logger.info(f"Falling back to PyPDF2 for {pdf_path}")

            text_content = f"# {base_name}\n\n"
            with open(pdf_path, 'rb') as file:
                reader = PyPDF2.PdfReader(file)
                for page_num in range(len(reader.pages)):
                    page = reader.pages[page_num]
                    text_content += f"## Page {page_num + 1}\n\n"
                    text_content += page.extract_text() + "\n\n"

            # Write to markdown file
            with open(markdown_path, 'w', encoding='utf-8') as f:
                f.write(text_content)

            logger.info(f"PDF converted to basic text with PyPDF2: {markdown_path}")
            return str(markdown_path)

        except ImportError:
            logger.error("Neither pdfplumber nor PyPDF2 is installed.")
            raise ImportError("Install pdfplumber or PyPDF2: pip install pdfplumber PyPDF2")
        except Exception as e2:
            logger.error(f"Both pdfplumber and PyPDF2 conversion failed: {e2}")
            raise


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