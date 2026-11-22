import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
# work without an editable install too
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "agent"))
