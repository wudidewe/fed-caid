"""Compatibility entry; implementation is maintained in src/main.py."""
from _iov_compat import load_implementation

_implementation = load_implementation("main")
if __name__ == "__main__" and hasattr(_implementation, "main"):
    raise SystemExit(_implementation.main())
