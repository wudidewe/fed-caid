"""Test the FedCAID wheel from outside the checkout in a temporary environment."""

import argparse
import json
import os
import subprocess
import sys
from hashlib import file_digest
from pathlib import Path
from tempfile import TemporaryDirectory

CPU_TORCH_INDEX = "https://download.pytorch.org/whl/cpu"
INSTALL_CHECK = """
from importlib import import_module, resources
from importlib.metadata import distribution
from pathlib import Path
import json
import sys

import torch

from src._iov_layout import project_root

package = distribution('iov-instruction')
prefix = Path(sys.prefix).resolve()
assert package.read_text('top_level.txt').split() == ['src']
names = {
    '.'.join(path.parts[:-1] if path.name == '__init__.py' else (*path.parts[:-1], path.stem))
    for path in package.files
    if path.parts[0] == 'src' and path.suffix == '.py'
}
assert {'src', 'src.revision_next', 'src.vendor.tabm', 'src.fedcaid_resources'} <= names
modules = {name: Path(import_module(name).__file__).resolve() for name in names}
if any(not path.is_relative_to(prefix) for path in modules.values()):
    raise RuntimeError('Wheel imports resolved outside the isolated installation')
assert not package.entry_points
assert project_root(modules['src._iov_layout']) == Path.cwd().resolve()
data = resources.files('src.fedcaid_resources')
compatibility = (
    'frozen_format_compatibility.json',
    'package_import_compatibility.json',
    'source_normalization_compatibility.json',
)
assert all(data.joinpath(name).is_file() for name in compatibility)
expected = ('recipe.json', 'heldout_identities.csv', 'excluded_identities.csv')
datasets = ('CIC-IoV-2024', 'Car-Hacking', 'CIC-IDS-2017', 'CIC-UNSW-NB15')
checked = [f'{dataset}/{name}' for dataset in datasets for name in expected
           if data.joinpath('preparation', dataset, name).is_file()]
assert len(checked) == 12
print(json.dumps({
    'modules': sorted(modules), 'resources': checked, 'compatibility_resources': compatibility,
    'torch': torch.__version__,
}))
"""
SMOKE = """
import json
from pathlib import Path
from src.revision_next import main

arguments = ['--smoke', '--device', 'cpu', '--dataset', 'CIC-IDS-2017',
             '--operating-guard', '--single-donor', '--seeds', '42', '--out', 'smoke']
assert main(arguments) == 0
receipt = json.loads(Path('smoke/complete.json').read_text(encoding='utf-8'))
assert receipt['smoke'] and receipt['evaluated'] and not receipt['test_evaluated']
"""


def _parse_args():
    """Choose one wheel and whether to resolve dependencies from scratch."""
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--wheel", type=Path)
    group.add_argument("--wheel-dir", type=Path)
    parser.add_argument("--receipt", type=Path, default=Path("installed_wheel_check.json"))
    parser.add_argument("--resolve-dependencies", action="store_true")
    parser.add_argument("--cpu-torch", action="store_true")
    args = parser.parse_args()
    wheels = [args.wheel] if args.wheel else list(args.wheel_dir.glob("iov_instruction-*.whl"))
    if len(wheels) != 1 or not wheels[0].is_file():
        parser.error("Select exactly one built FedCAID wheel")
    if args.cpu_torch and not args.resolve_dependencies:
        parser.error("--cpu-torch requires --resolve-dependencies")
    args.wheel = wheels[0].resolve()
    return args


def _run(command, root):
    """Run outside the checkout with Python path overrides removed."""
    allowed = {key: value for key, value in os.environ.items() if key not in {"PYTHONPATH", "PYTHONHOME"}}
    allowed['PYTHONUTF8'] = '1'
    python, *arguments = command
    return subprocess.run(
        [python, '-X', 'utf8', *arguments],
        cwd=root,
        env=allowed,
        check=True,
        capture_output=True,
        encoding='utf-8',
    )


def _installation(root, args):
    """Install the wheel and return the isolated Python executable."""
    environment = root / "environment"
    shared = [] if args.resolve_dependencies else ["--system-site-packages"]
    _run([sys.executable, "-m", "venv", *shared, str(environment)], root)
    python = environment / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if args.cpu_torch:
        _run([str(python), "-m", "pip", "install", "torch==2.5.1", "--index-url", CPU_TORCH_INDEX], root)
    flags = [] if args.resolve_dependencies else ["--no-deps", "--ignore-installed"]
    _run([str(python), "-m", "pip", "install", *flags, str(args.wheel)], root)
    if args.resolve_dependencies:
        _run([str(python), "-m", "pip", "check"], root)
    return str(python)


def _verify(args, receipt):
    """Check dependencies, imports and synthetic inference, updating the receipt."""
    with TemporaryDirectory(prefix="fedcaid_installed_") as name:
        root = Path(name)
        python = _installation(root, args)
        receipt["dependency_resolution_tested"] = args.resolve_dependencies
        receipt["imports_and_resources"] = json.loads(_run([python, "-I", "-c", INSTALL_CHECK], root).stdout)
        _run([python, "-I", "-c", SMOKE], root)
        receipt["smoke"] = json.loads((root / "smoke" / "complete.json").read_text(encoding="utf-8"))


def main():
    """Write an install receipt with the exact tested wheel digest."""
    args = _parse_args()
    with args.wheel.open("rb") as stream:
        digest = file_digest(stream, "sha256").hexdigest()
    receipt = {"wheel": str(args.wheel), "wheel_sha256": digest, "outside_checkout": True, "complete": False}
    try:
        _verify(args, receipt)
        receipt["complete"] = True
    except subprocess.CalledProcessError as error:
        receipt["error"] = (error.stderr or str(error))[-4000:]
        raise RuntimeError(receipt["error"]) from error
    finally:
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2), encoding="utf-8")
    print(f"Wheel verified: {args.receipt.resolve()}")


if __name__ == "__main__":
    main()
