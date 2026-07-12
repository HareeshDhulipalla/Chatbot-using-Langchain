import json
import logging
import os
import re
from datetime import datetime
from pathlib import Path

# Cross‑platform file locking
try:
    from filelock import FileLock, Timeout
except ImportError:
    # If filelock is not available, fallback to a no‑op lock (not safe for concurrency)
    class FileLock:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            pass
        def __exit__(self, exc_type, exc_val, exc_tb):
            pass

FEEDBACK_FILE = os.getenv('FEEDBACK_FILE', 'feedback.jsonl')
FEEDBACK_MAX_BYTES = int(os.getenv('FEEDBACK_MAX_BYTES', '10485760'))  # 10 MB default
FEEDBACK_BACKUP_COUNT = int(os.getenv('FEEDBACK_BACKUP_COUNT', '5'))
logger = logging.getLogger(__name__)

def _sanitize_context(context: str) -> str:
    """Redact obvious PII patterns from feedback context."""
    # Credit card numbers (16 digits, optionally spaced or hyphenated)
    context = re.sub(r'\b(?:\d[ -]?){13,16}\b', '[REDACTED]', context)
    # Social Security Number pattern
    context = re.sub(r'\b\d{3}-\d{2}-\d{4}\b', '[REDACTED]', context)
    # Generic long digit sequences (potential account numbers)
    context = re.sub(r'\b\d{6,}\b', '[REDACTED]', context)
    # Email addresses
    context = re.sub(r'\b[\w\.-]+@[\w\.-]+\.\w+\b', '[REDACTED]', context)
    return context

def _rotate_feedback_file() -> None:
    """Rotate feedback.jsonl if it exceeds FEEDBACK_MAX_BYTES.
    Renames current file to feedback.jsonl.1, .2, ... up to BACKUP_COUNT.
    This function uses the same lock as `record_feedback` to avoid race conditions.
    """
    path = Path(FEEDBACK_FILE)
    if not path.exists():
        return

    lock_path = f"{FEEDBACK_FILE}.lock"
    try:
        with FileLock(lock_path, timeout=10):
            # Re‑check size inside lock
            try:
                if path.stat().st_size < FEEDBACK_MAX_BYTES:
                    return
            except OSError:
                return

            # Remove oldest backup if exists
            oldest = path.with_suffix(f'.{FEEDBACK_BACKUP_COUNT}')
            if oldest.exists():
                try:
                    oldest.unlink()
                except OSError as e:
                    logger.warning(f'Could not remove oldest feedback backup {oldest}: {e}')

            # Shift existing backups up by one
            for i in range(FEEDBACK_BACKUP_COUNT - 1, 0, -1):
                src = path.with_suffix(f'.{i}')
                dst = path.with_suffix(f'.{i + 1}')
                if src.exists():
                    try:
                        src.rename(dst)
                    except OSError as e:
                        logger.warning(f'Could not rotate feedback backup {src} -> {dst}: {e}')

            # Rotate current file to .1
            try:
                path.rename(path.with_suffix('.1'))
            except OSError as e:
                logger.error(f'Failed to rotate feedback file {path}: {e}')
    except Timeout:
        logger.warning('Could not acquire lock for feedback rotation; skipping rotation.')

def record_feedback(session_key: str, sentiment: str, context: str = '') -> bool:
    """Record feedback with sanitized context to avoid storing PII."""
    sanitized_context = _sanitize_context(context)

    entry = {
        'timestamp': datetime.utcnow().isoformat(),
        'session': session_key,
        'sentiment': sentiment,
        'context': sanitized_context
    }

    # Ensure the directory exists before writing
    feedback_dir = os.path.dirname(FEEDBACK_FILE)
    if feedback_dir:
        try:
            os.makedirs(feedback_dir, exist_ok=True)
        except (IOError, OSError) as e:
            logger.error(f'Failed to create feedback directory: {e}')
            return False

    # Rotate if needed before writing (uses same lock inside)
    _rotate_feedback_file()

    lock_path = f"{FEEDBACK_FILE}.lock"
    try:
        with FileLock(lock_path, timeout=10):
            # Open in append mode with restrictive permissions (0o600)
            with open(FEEDBACK_FILE, 'a', opener=lambda path, flags: os.open(path, flags, 0o600)) as f:
                f.write(json.dumps(entry) + '\n')
                f.flush()
                os.fsync(f.fileno())
    except (IOError, OSError, Timeout) as e:
        logger.error(f'Failed to write feedback: {e}')
        return False

    return True
