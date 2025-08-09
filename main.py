import streamlit as st
import rag
from langchain.schema.messages import HumanMessage, AIMessage

# --- Caching the heavy, key-independent parts of the RAG chain ---
@st.cache_resource
def load_vectorstore(limit: int): 
    """Load and cache the vector store which is slow and doesn't need an API key."""
    return rag.load_and_build_vectorstore(limit=limit) 

# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Indian Government Scheme Chatbot",
    page_icon="🇮🇳",
    layout="centered"
)

# --- Sidebar ---
st.sidebar.header("🔑 API Configuration")
gemini_api_key = st.sidebar.text_input(
    "Enter your Gemini API Key:", 
    type="password",
    help="You can get your free API key from [Google AI Studio](https://aistudio.google.com/app/apikey)."
).strip()

st.sidebar.header("⚙️ Options")

doc_limit = st.sidebar.slider(
    label="Documents to Load",
    min_value=10,
    max_value=1000, 
    value=100,      
    step=10,
    help="Set the number of documents to load into the knowledge base."
)

# --- ✨ NEW: Load Button to control data loading ---
if st.sidebar.button("Load Knowledge Base", type="primary"):
    with st.spinner("Loading documents into the vector store... Please wait."):
        # Store the loaded vector store and the state in the session
        st.session_state.vectorstore = load_vectorstore(limit=doc_limit)
        st.session_state.vectorstore_loaded = True
        st.sidebar.success("Knowledge base loaded successfully!")

if st.sidebar.button("Clear Chat History"):
    st.session_state.messages = []
    # Optionally, you could also clear the vectorstore state here if desired
    # st.session_state.vectorstore_loaded = False 
    st.rerun()

# --- Main Page Content ---
st.title("🇮🇳 Indian Government Scheme Chatbot")
st.markdown("""
Ask me about government schemes! I can provide details on **eligibility**, **benefits**, and more.
""")

# --- Main App Logic: Now conditional on loading state ---

# Check if the vector store has been loaded.
if not st.session_state.get("vectorstore_loaded", False):
    st.info("Please configure your settings in the sidebar and click 'Load Knowledge Base' to start.")

# Check for API Key after loading.
elif not gemini_api_key:
    st.warning("Knowledge base is loaded. Please enter your Gemini API key to start chatting.")

# Proceed with the chat interface if everything is ready.
else:
    try:
        # Retrieve the vector store from session state
        vectorstore = st.session_state.vectorstore
        if vectorstore is None:
             st.error("🚨 Apologies, the knowledge base failed to load. Please ensure 'scheme_data.json' is present.")
        else:
            rag_chain = rag.get_rag_chain(vectorstore, gemini_api_key)
            
            # Initialize chat history
            if "messages" not in st.session_state:
                st.session_state.messages = []
            
            if not st.session_state.messages:
                 st.session_state.messages.append(
                     {"role": "assistant", "content": "Knowledge base ready. How can I help you?"}
                 )

            # Display chat messages
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # Handle user input
            if prompt := st.chat_input("Ask about a scheme..."):
                st.session_state.messages.append({"role": "user", "content": prompt})
                with st.chat_message("user"):
                    st.markdown(prompt)

                with st.chat_message("assistant"):
                    with st.spinner("Searching for the best answer..."):
                        chat_history_for_chain = [
                            HumanMessage(content=m["content"]) if m["role"] == "user" 
                            else AIMessage(content=m["content"]) 
                            for m in st.session_state.messages[:-1]
                        ]
                        
                        result = rag_chain.invoke({
                            "input": prompt,
                            "chat_history": chat_history_for_chain
                        })
                        
                        answer = result.get("answer", "I'm sorry, I couldn't find an answer.")
                        st.markdown(answer)
                        
                        st.session_state.messages.append({"role": "assistant", "content": answer})

    except Exception as e:
        st.error(f"An error occurred: {e}. Please check if your API key is valid.")