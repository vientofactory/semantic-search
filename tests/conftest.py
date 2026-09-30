"""Shared test bootstrap.

Makes the side project root importable so test modules can `import
lawcast_semantic` directly. This is the only place in the suite that adjusts
sys.path, whatever way pytest is invoked.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
