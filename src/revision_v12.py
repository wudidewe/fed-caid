"""Read, write and lock run receipts shared by the evidence pipeline."""

import json
import os
from contextlib import contextmanager
from hashlib import file_digest

if os.name == 'nt':
    from msvcrt import LK_NBLCK, locking
else:
    from fcntl import LOCK_EX, LOCK_NB, flock


OPTIONAL_CONFIRMATION_STAGES = frozenset({'external', 'lan'})


def _completion_summary(state, stages):
    """Summarize terminal stage records in the order declared by the run plan."""
    failed = [name for name in stages if state[name]['status'] == 'failed']
    pending = [name for name in stages if state[name]['status'] == 'pending']
    if failed:
        status = 'completed_with_failures'
    elif pending:
        status = 'completed_with_pending'
    else:
        status = 'completed'
    return {
        'status': status,
        'failed_steps': failed,
        'pending_steps': pending,
        'optional_pending_steps': [name for name in pending if name in OPTIONAL_CONFIRMATION_STAGES],
        'required_pending_steps': [name for name in pending if name not in OPTIONAL_CONFIRMATION_STAGES],
    }


def _read(path):
    """Read a UTF-8 JSON receipt."""
    return json.loads(path.read_text(encoding='utf-8'))


def _hash(path):
    """Return the SHA-256 digest of a file."""
    with path.open('rb') as stream:
        return file_digest(stream, 'sha256').hexdigest()


def _write(path, data):
    """Write a JSON receipt atomically within the destination directory."""
    temporary = path.with_suffix('.partial')
    contents = json.dumps(data, indent=2, sort_keys=True, allow_nan=False)
    temporary.write_text(contents + '\n', encoding='utf-8')
    temporary.replace(path)


def _bindings(directory, names):
    """Map artifact names within a receipt directory to SHA-256 digests."""
    return {name: _hash(directory / name) for name in names}


def _acquire_lock(stream):
    """Acquire the output directory's nonblocking platform lock."""
    if os.name == 'nt':
        locking(stream.fileno(), LK_NBLCK, 1)
    else:
        flock(stream.fileno(), LOCK_EX | LOCK_NB)


@contextmanager
def _output_lock(root, *, filename='.v12_lock'):
    """Hold exclusive ownership until the current output transaction ends."""
    with (root / filename).open('a+b') as stream:
        stream.seek(0)
        try:
            _acquire_lock(stream)
        except OSError as error:
            raise ValueError('Another V12 process owns this output directory') from error
        yield
