"""Main entry point for DJ mixing analytics - uses CLI interface.

This module provides backward compatibility with the original main.py.
For full functionality, use: python -m src.cli
"""

import sys
from .cli import main as cli_main


def main():
    """Main entry point."""
    return cli_main()


if __name__ == "__main__":
    sys.exit(main())
