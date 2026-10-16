"""Import the canonical src module under its original, checkpoint-safe name."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def load_implementation(name):
    source_dir = Path(__file__).resolve().parent / "src"
    source_path = source_dir / f"{name}.py"
    if not source_path.is_file():
        raise ImportError(f"Missing IOV-instruction implementation: {source_path}")
    source_text = str(source_dir)
    if source_text not in sys.path:
        sys.path.insert(0, source_text)
    existing = sys.modules.get(name)
    if existing is not None and Path(getattr(existing, "__file__", "")).resolve() == source_path:
        return existing
    spec = importlib.util.spec_from_file_location(name, source_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Cannot load implementation: {name}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    try:
        spec.loader.exec_module(module)
    except BaseException:
        if existing is None:
            sys.modules.pop(name, None)
        else:
            sys.modules[name] = existing
        raise
    return module
