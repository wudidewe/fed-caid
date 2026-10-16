"""Small synthetic workflow, independent of the local datasets."""
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from iov_instruction import main


if __name__ == "__main__":
    arguments = [
        "--smoke", "--scope", "can", "--rounds", "1",
        "--base-epochs", "1", "--adapter-epochs", "2", "--device", "cpu",
    ]
    raise SystemExit(main(arguments + sys.argv[1:]))
