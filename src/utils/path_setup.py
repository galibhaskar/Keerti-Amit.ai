"""Setup Python path for imports."""

import sys
from pathlib import Path

# Add project root to Python path if not already there
_project_root = Path(__file__).parent.parent.parent
_project_root_str = str(_project_root)
if _project_root_str not in sys.path:
    sys.path.insert(0, _project_root_str)

