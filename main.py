import streamlit as st
import os
import tempfile
from rag import GovernmentSchemeRAG

# --- Page and Session Initialization ---
st.set_page_config(
    page_title="Scheme Chatbot",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Initialize session state for chat history.
if "messages" not in st.session_state:
    st.session_state.messages = []
if "json_path" not in st.session_state:
    st.session_state.json_path = "scheme_data.json"

# --- Caching the RAG System ---
@st.cache_resource
def load_rag_system(json_path, google_api_key, hf_token):
    """Loads and caches the RAG system to avoid reloading on each interaction."""
    try:
        return GovernmentSchemeRAG(json_path=json_path, google_api_key=google_api_key, hf_token=hf_token)
    except Exception as e:
        st.error(f"Failed to load RAG system: {e}")
        return None

# --- Sidebar Configuration ---
with st.sidebar:
    st.header("⚙️ Configuration")
    st.subheader("🔑 API Keys")
    google_api_key = st.text_input("Google Gemini API Key", type="password", placeholder="Enter Google API Key")
    hf_token = st.text_input("Hugging Face Token", type="password", placeholder="Enter Hugging Face Token")
    
    st.markdown("---")

# --- Main Chat Interface ---
st.title("🤖 Government Scheme Chatbot")
st.markdown("Ask me anything about Indian government schemes from the provided data.")

if not google_api_key and not hf_token:
    st.warning("Please enter your Google Gemini or Hugging Face API Key in the sidebar to start.")
    st.stop()

rag_system = load_rag_system(st.session_state.json_path, google_api_key, hf_token)

if rag_system:
    # Display chat messages from history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    # Accept user input
    if prompt := st.chat_input("Ask about a scheme..."):
        # Add user message to history and display it
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # --- HISTORY UPDATE ---
        # Format the chat history into a string to pass to the RAG system.
        # We'll take the last 4 messages (2 turns) to keep the context relevant and concise.
        history_str = "\n".join(
            [f"{msg['role']}: {msg['content']}" for msg in st.session_state.messages[-5:-1]]
        )
        
        # Display assistant response
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                # Pass the history string to the query method
                response = rag_system.query(prompt, history=history_str)
                answer = response.get("answer", "Sorry, something went wrong.")
                st.markdown(answer)
        
        # Add assistant response to chat history
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    st.error("The chatbot could not be initialized. Please check your configuration.")
