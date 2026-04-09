from __future__ import annotations

import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
APP_ROOT = REPO_ROOT / "multiagent_mvp"

if str(APP_ROOT) not in sys.path:
    sys.path.insert(0, str(APP_ROOT))
