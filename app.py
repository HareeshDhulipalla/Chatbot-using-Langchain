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

# -------------------------------------------------
# Helper function to sanitize context for prompt injection prevention
# and enforce token-based limits
def sanitize_context(context, max_tokens=10000):
    if not context:
        return ''
    
    # Remove control characters except newline and tab
    sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', context)
    
    # Remove known prompt injection patterns with more precise matching
    # Only match at start of string or after newline, with optional whitespace
    injection_patterns = [
        r'(^|\n)\s*ignore\s+previous\s+instructions',
        r'(^|\n)\s*ignore\s+all\s+previous\s+instructions',
        r'(^|\n)\s*system\s*:',
        r'(^|\n)\s*###\s*$',  # Markdown header on its own line
        r'<<\s*[A-Z_]+\s*>>',  # Template markers like <<SYSTEM>>, <<INSTRUCTIONS>>
    ]
    for pattern in injection_patterns:
        sanitized = re.sub(pattern, '', sanitized, flags=re.IGNORECASE)
    
    # Estimate token count (rough approximation: 1 token ≈ 4 characters)
    estimated_tokens = len(sanitized) // 4
    
    # Truncate to fit within max_tokens
    if estimated_tokens > max_tokens:
        # Truncate at word boundary to avoid cutting mid-word
        target_chars = max_tokens * 4
        truncated = sanitized[:target_chars]
        # Find last space to truncate cleanly
        last_space = truncated.rfind(' ')
        if last_space > 0:
            truncated = truncated[:last_space]
        sanitized = truncated
    
    return sanitized

# -------------------------------------------------
# Authentication check – require user to provide a valid token
# that matches the expected CHATBOT_AUTH_TOKEN environment variable.
# This prevents unauthorized access to the banking assistant.
auth_token_expected = os.getenv("CHATBOT_AUTH_TOKEN")
if not auth_token_expected or len(auth_token_expected.strip()) == 0:
    st.error("Server configuration error: CHATBOT_AUTH_TOKEN not set.")
    st.stop()

# Retrieve token from session state or prompt user securely
if "auth_token" not in st.session_state:
    # Prompt user for token using a password field (masked input)
    st.session_state.auth_token = st.text_input(
        "Enter authentication token:", type="password"
    )

user_auth_token = st.session_state.get("auth_token", "")

# Validate the provided token against expected token using constant-time comparison
if not user_auth_token or not secrets.compare_digest(user_auth_token, auth_token_expected):
    st.error("Authentication required. Please provide a valid token.")
    st.stop()

# Store authenticated identity in session state
if 'authenticated_user' not in st.session_state:
    st.session_state.authenticated_user = hashlib.sha256(user_auth_token.encode()).hexdigest()[:16]

# -------------------------------------------------
# Session management – preserve session across pages and provider changes
if 'session_id' not in st.session_state:
    # Use cryptographically random session ID
    st.session_state.session_id = secrets.token_urlsafe(32)
session_id = st.session_state.session_id

# Initialise session in conversation manager
get_or_create_session(session_id)

# Initialize feedback secret early so it exists before any chat interaction
if 'feedback_secret' not in st.session_state:
    st.session_state.feedback_secret = secrets.token_urlsafe(16)

# -------------------------------------------------
# Sidebar control panel
st.sidebar.title('Control Panel')

# Provider selector – persist selection via session_state
provider_options = ['groq', 'openai', 'anthropic']
default_provider = st.session_state.get('llm_provider', 'groq')
provider_idx = provider_options.index(default_provider) if default_provider in provider_options else 0
provider = st.sidebar.selectbox('LLM Provider', provider_options, index=provider_idx)
# Persist the selection when it changes
if provider != default_provider:
    st.session_state['llm_provider'] = provider

# -------------------------------------------------
# File uploader with sanitization
# Track indexed file identifiers to avoid reprocessing on Streamlit reruns
if 'indexed_file_ids' not in st.session_state:
    st.session_state.indexed_file_ids = set()

uploaded_files = st.sidebar.file_uploader('Upload Document', accept_multiple_files=True)
if uploaded_files:
    # Define allowed file extensions and MIME types
    allowed_ext = {'.pdf', '.docx', '.txt'}
    allowed_mimetypes = {
        'application/pdf',
        'application/vnd.openxmlformats-officedocument.wordprocessingml.document',
        'text/plain'
    }
    valid_files = []
    new_file_ids = []
    attempted = 0
    skipped_existing = 0
    for file in uploaded_files:
        attempted += 1
        # Strip path components from filename
        safe_name = os.path.basename(file.name)
        # Stable identifier for this uploaded file across reruns
        file_id = getattr(file, 'file_id', f"{safe_name}_{getattr(file, 'size', 0)}")
        if file_id in st.session_state.indexed_file_ids:
            skipped_existing += 1
            continue
        # Validate extension
        if not any(safe_name.lower().endswith(ext) for ext in allowed_ext):
            st.sidebar.error(f'Unsupported file type: {file.name}')
            continue
        # Validate MIME type
        mime_type = mimetypes.guess_type(safe_name)[0]
        if mime_type not in allowed_mimetypes:
            st.sidebar.error(f'Invalid MIME type for {file.name}: {mime_type}')
            continue
        # Validate file content using magic bytes
        file_content = file.read(8)
        file.seek(0)  # Reset file pointer
        is_valid_content = False
        if safe_name.lower().endswith('.pdf'):
            # PDF magic bytes: %PDF
            is_valid_content = file_content.startswith(b'%PDF')
        elif safe_name.lower().endswith('.docx'):
            # DOCX magic bytes: PK (ZIP archive)
            is_valid_content = file_content.startswith(b'PK\x03\x04')
        elif safe_name.lower().endswith('.txt'):
            # TXT files - check if content is printable/text-like
            try:
                sample = file.read(1024)
                file.seek(0)
                sample.decode('utf-8')
                is_valid_content = True
            except UnicodeDecodeError:
                file.seek(0)
                is_valid_content = False
        if not is_valid_content:
            st.sidebar.error(f'File content validation failed for {file.name}. Possible malicious content.')
            continue
        # Generate unique filename to prevent overwrites and path traversal
        unique_prefix = secrets.token_hex(8)
        file_timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        file.unique_name = f"{file_timestamp}_{unique_prefix}_{safe_name}"
        # Use the unique name for processing to prevent overwrites/path traversal
        file.name = file.unique_name
        valid_files.append(file)
        new_file_ids.append(file_id)
    if not valid_files:
        if attempted > 0 and skipped_existing == attempted:
            st.sidebar.info('All uploaded documents are already indexed.')
        else:
            st.sidebar.error('No valid documents uploaded.')
    else:
        with st.spinner('Processing documents...'):
            try:
                process_and_index(valid_files)
                st.session_state.indexed_file_ids.update(new_file_ids)
                st.sidebar.success('Documents indexed')
            except ConnectionError as e:
                logger.error('', extra={'event':'document_processing_error','error':_sanitize_error(e)})
                st.sidebar.error('Connection failed. Please check document service.')
            except TimeoutError as e:
                logger.error('', extra={'event':'document_processing_error','error':_sanitize_error(e)})
                st.sidebar.error('Processing timed out. Please try again.')
            except Exception as e:
                logger.error('', extra={'event':'document_processing_error','error':_sanitize_error(e)})
                st.sidebar.error('Failed to process documents. Please try again.')

# -------------------------------------------------
# Main chat area
st.title('AI Banking Assistant (RAG + History)')

# Configuration for conversation history limit
MAX_HISTORY_TURNS = 10  # Limit to last 10 user-assistant exchanges

# Display chat history for the current session (limited to prevent context overflow)
history = get_history(session_id)
# Limit history to last MAX_HISTORY_TURNS * 2 (user + assistant) messages
limited_history = history[-(MAX_HISTORY_TURNS * 2):] if len(history) > MAX_HISTORY_TURNS * 2 else history
for idx, msg in enumerate(limited_history):
    role = 'user' if msg['role'] == 'user' else 'assistant'
    st.chat_message(role).write(msg['content'])
    # Render feedback button for each assistant message in the displayed history
    if role == 'assistant':
        feedback_key = f"{session_id}_{st.session_state.feedback_secret}_{idx}"
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('👍', key=f'up_{feedback_key}'):
                record_feedback(session_id, 'positive')
        with col2:
            if st.button('👎', key=f'down_{feedback_key}'):
                record_feedback(session_id, 'negative')

# User input
if input_text := st.chat_input('Type your banking question...'):
    # Display user message immediately
    st.chat_message('user').write(input_text)
    
    # Log request with session_id for traceability
    start = time.monotonic()
    log_event('request', session_id=session_id, provider=provider)
    
    # Retrieve context via RAG with limited results (wrapped in error handling)
    try:
        retriever = get_retriever()
        docs = retriever.invoke(input_text) if retriever else []
    except (ConnectionError, TimeoutError, LangChainException) as e:
        logger.error('', extra={'event':'rag_error','provider':provider,'error':_sanitize_error(e)})
        st.error('Failed to retrieve documents. Please try again.')
        docs = []
    except Exception as e:
        logger.error('', extra={'event':'rag_unexpected_error','provider':provider,'error':_sanitize_error(e)})
        st.error('An unexpected error occurred during document retrieval.')
        docs = []
    
    # Limit number of retrieved documents
    MAX_DOCS = 4
    docs = docs[:MAX_DOCS] if len(docs) > MAX_DOCS else docs
    context_lines = [d.page_content for d in docs if getattr(d, 'page_content', None)]
    context = '\n\n'.join(context_lines) if context_lines else ''
    
    # Sanitize and truncate context with token-aware limits
    context = sanitize_context(context, max_tokens=2000)
    
    # Build messages list using LangChain message objects
    messages = []
    if context:
        messages.append(SystemMessage(content=f"Context:\n{context}"))
    for h in limited_history:
        if h['role'] == 'user':
            messages.append(HumanMessage(content=h['content']))
        else:
            messages.append(AIMessage(content=h['content']))
    sanitized_input = sanitize_context(input_text, max_tokens=500)
    messages.append(HumanMessage(content=sanitized_input))
    
    # Invoke LLM
    try:
        llm = get_llm(provider=provider)
        response = llm.invoke(messages)
        elapsed = time.monotonic() - start
        
        # Handle response content safely
        content = getattr(response, 'content', str(response))
        
        # Extract token usage safely (provider-aware)
        usage = getattr(response, 'usage', None) or getattr(response, 'usage_metadata', None)
        token_usage = 0
        if usage:
            if isinstance(usage, dict):
                token_usage = usage.get('total_tokens', 0) or (usage.get('input_tokens', 0) + usage.get('output_tokens', 0))
            else:
                token_usage = getattr(usage, 'total_tokens', 0) or (
                    getattr(usage, 'input_tokens', 0) + getattr(usage, 'output_tokens', 0)
                )
        
        logger.debug('', extra={'event': 'token_usage_debug', 'provider': provider, 'raw_usage': str(usage)})
        log_event('response', session_id=session_id, latency=elapsed, token_usage=token_usage)
        
        # Persist messages
        save_message(session_id, 'user', input_text)
        save_message(session_id, 'assistant', content)
        st.chat_message('assistant').write(content)
        
        # Feedback UI for the newly generated assistant message
        feedback_key = f"{session_id}_{st.session_state.feedback_secret}_{len(limited_history)}"
        col1, col2 = st.columns([1, 1])
        with col1:
            if st.button('👍', key=f'up_{feedback_key}'):
                record_feedback(session_id, 'positive')
        with col2:
            if st.button('👎', key=f'down_{feedback_key}'):
                record_feedback(session_id, 'negative')
    except ConnectionError as e:
        logger.error('', extra={'event':'llm_connection_error','provider':provider,'error':_sanitize_error(e)})
        st.error('Connection to AI service failed. Please try again.')
    except TimeoutError as e:
        logger.error('', extra={'event':'llm_timeout_error','provider':provider,'error':_sanitize_error(e)})
        st.error('Request timed out. Please try again.')
    except LangChainException as e:
        logger.error('', extra={'event':'llm_langchain_error','provider':provider,'error':_sanitize_error(e)})
        st.error('AI service error. Please try again.')
    except Exception as e:
        logger.error('', extra={'event':'llm_unexpected_error','provider':provider,'error':_sanitize_error(e)})
        st.error('An unexpected error occurred. Please try again later.')
