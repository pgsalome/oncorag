"""oncorag package initialization."""

def main():
    """Load the legacy CLI only when invoked, not during schema imports."""
    from .main import main as cli_main

    return cli_main()

__all__ = ["main"]
