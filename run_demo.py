import importlib
import sys

# Import the app module (ensures we load local app.py)
mod = importlib.import_module('app')

# Set response length to Very long for demo tests (guarded if session_state missing)
try:
    import streamlit as st
    if hasattr(st, 'session_state'):
        st.session_state['response_length'] = 'Very long'
except Exception:
    pass

# Run the demo search
res = mod.demo_search_response('suggest some cookies', response_length='Very long')

# Print UTF-8 safely to avoid console encoding errors on Windows
sys.stdout.buffer.write(('ANSWER:\n' + res['answer'] + '\n\n').encode('utf-8', errors='replace'))
for doc in res['source_documents']:
    sys.stdout.buffer.write(('SOURCE: ' + str(getattr(doc, 'page_content', '')) + '\n').encode('utf-8', errors='replace'))
