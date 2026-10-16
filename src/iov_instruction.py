"""Public command and inference facade for IOV-instruction-v1.0."""
from __future__ import annotations

__version__ = "1.0.0"


def main(argv=None):
    """Run the unchanged application options, including --check and --smoke."""
    from main import main as application_main
    return application_main(argv)


def predict_private_decisions(checkpoint, x, device="cpu"):
    """Load a private checkpoint using the existing prediction implementation."""
    from privacy_run import predict_private_decisions as predict
    return predict(checkpoint, x, device)


if __name__ == "__main__":
    raise SystemExit(main())
