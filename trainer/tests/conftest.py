import os
import sys
from pathlib import Path

# Ensure the package under trainer/src is importable without installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

# Keep tests from hitting the W&B cloud or blocking on credential prompts.
# Tests that exercise the active-logger path delete this var and inject a
# fake wandb module instead.
os.environ.setdefault("WANDB_MODE", "disabled")
