from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path

# Tests add documents, write log lines and rebuild calibration. Point all runtime
# state at a throwaway directory *before* the package is imported, so the test suite
# never changes the project's own vector store, request log or calibration file.
os.environ.setdefault("CRED_ARTIFACT_DIR", tempfile.mkdtemp(prefix="cred-tests-"))

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
