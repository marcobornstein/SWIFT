"""Allow ``python -m swift`` as well as the ``swift-train`` entry point."""

import sys

from .cli import main

if __name__ == "__main__":
    sys.exit(main())
