import logging
import json
import sys
import os
import traceback
from datetime import datetime
from typing import Optional
import logging.handlers

LOG_LEVEL = os.getenv('LOG_LEVEL', 'INFO')
LOG_FILE = os.getenv('LOG_FILE', 'app.log')

# Validate LOG_LEVEL against allowed levels
ALLOWED_LOG_LEVELS = ['DEBUG', 'INFO', 'WARNING', 'ERROR', 'CRITICAL']
if LOG_LEVEL not in ALLOWED_LOG_LEVELS:
    LOG_LEVEL = 'INFO'

class JsonFormatter(logging.Formatter):
    def format(self, record):
        # Extract extra fields from record.__dict__ (excluding standard LogRecord attributes)
        standard_attrs = {
            'name', 'msg', 'args', 'created', 'filename', 'funcName', 'levelname',
            'levelno', 'lineno', 'module', 'msecs', 'message', 'pathname',
            'process', 'processName', 'relativeCreated', 'thread',
            'threadName', 'exc_info', 'exc_text', 'stack_info', 'getMessage'
        }
        extra = {k: v for k, v in record.__dict__.items() if k not in standard_attrs}
        
        # Build the log entry
        log_entry = {
            'timestamp': datetime.utcnow().isoformat(),
            'level': record.levelname,
            'logger': record.name,
            **extra
        }
        # Include the log message if not part of extra
        if record.getMessage() and 'message' not in log_entry:
            log_entry['message'] = record.getMessage()
        
        # Include exception information if present
        if record.exc_info:
            exc_type, exc_value, exc_tb = record.exc_info
            formatted_exc = ''.join(traceback.format_exception(exc_type, exc_value, exc_tb))
            log_entry['exception'] = formatted_exc
        elif record.exc_text:
            log_entry['exception'] = record.exc_text

        return json.dumps(log_entry)

handler = logging.StreamHandler(sys.stdout)
handler.setFormatter(JsonFormatter())

# Create the log file with restrictive permissions (0600)
def _secure_opener(path, flags):
    # O_CREAT is handled by RotatingFileHandler; we ensure mode 0o600
    return os.open(path, flags, 0o600)

file_handler = logging.handlers.RotatingFileHandler(
    LOG_FILE,
    maxBytes=10 * 1024 * 1024,
    backupCount=5,
    opener=_secure_opener
)
file_handler.setFormatter(JsonFormatter())

logger = logging.getLogger('ai_chatbot')
logger.setLevel(getattr(logging, LOG_LEVEL))
logger.addHandler(handler)
logger.addHandler(file_handler)

def get_logger(name: str) -> logging.Logger:
    return logging.getLogger(f'ai_chatbot.{name}')

def log_event(event: str, session_id: Optional[str] = None, **kwargs):
    # Pass structured data via extra parameter to avoid double JSON encoding
    extra = {'event': event}
    if session_id is not None:
        extra['session_id'] = session_id
    extra.update(kwargs)
    logger.info('', extra=extra)
