"""oncorag package initialization."""

__version__ = "1.0.0"
SOFTWARE_NAME = "OncoRAG"
REPOSITORY_URL = "https://github.com/pgsalome/oncorag"
PAPER_DOI = "10.1038/s41746-026-03170-8"
PAPER_URL = f"https://doi.org/{PAPER_DOI}"
PAPER_TITLE = (
    "OncoRAG: graph-based retrieval enabling clinical phenotyping from oncology "
    "notes using local mid-size language models"
)

def main():
    """Load the legacy CLI only when invoked, not during schema imports."""
    from .main import main as cli_main

    return cli_main()

__all__ = [
    "main",
    "__version__",
    "SOFTWARE_NAME",
    "REPOSITORY_URL",
    "PAPER_DOI",
    "PAPER_URL",
    "PAPER_TITLE",
]
