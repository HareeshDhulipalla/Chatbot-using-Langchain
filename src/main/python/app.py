from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain_core.exceptions import LangChainException
import streamlit as st
import time
import os
import re
import mimetypes
import secrets
import hashlib
from datetime import datetime
from dotenv import load_dotenv

from config import get_llm
from conversation_manager import get_or_create_session, get_history, save_message
from document_rag import get_retriever, process_and_index
from security/security_utils.py import verify_otp
from logging_util import get_logger, log_event
from feedback_handler import record_feedback

# Load environment variables
load_dotenv()

# Logger
logger = get_logger(__name__)

# Helper to sanitize exception information for logging
def _sanitize_error(exc: Exception) -> str:
    """Return a non-sensitive identifier for an exception."""
    return exc.__class__.__name__

# -- OTP Verification Middleware --
if 'verified_otp' not in st.session_state:
    st.session_state.verified_otp = False

def check_otp():
   # Placeholder check logic (to be implemented)
    if len(st.secrets.get('OTP', '')) > 0 and st.session_state.user_input == st.secrets['OTP']:
        st.session_state.verified_otp = True
    else:
        logger.error('Invalid OTP provided')
        st.session_state.verified_otp = False

# -------------------------------------------------
# Helper function to sanitize context for prompt injection prevention
# and enforce token-based limits
def sanitize_context(context, max_tokens=10000):
    if not context or st.session_state.verified_otp is False:
        return ''
    
    # Remove control characters except newline and tab
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', context)
    
    # Remove known prompt injection patterns with more precise matching
    # Only match at start of string or after newline, with optional whitespace
    injection_patterns = [
        r'(^|\n)\s*ignore\s+previous\s+instructions',
