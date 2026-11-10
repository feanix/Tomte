"""
Run Tomte's command line from any folder:

    python path/to/tomte_cli.py --setup Body.tomte
    python path/to/tomte_cli.py SOURCE.fbx TARGET.fbx tex1.png [options]

Same options as `python -m uvtransfer` (see --help). The app's
File > Copy command line gives a ready-made command using this file.
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from uvtransfer.cli import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
