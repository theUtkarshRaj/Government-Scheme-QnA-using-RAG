# # main.py
# ## Version 5 - Fixed input clearing error compatible with ver-4

# import streamlit as st
# from rag import GovernmentSchemeRAG # Assuming rag.py is in the same directory
# import os # Added to check for API keys

# # --- Caching the RAG system ---
# @st.cache_resource
# def load_rag_system(json_path):
#     """Loads the RAG system. Handles potential errors during initialization."""
#     try:
#         if isinstance(json_path, str) and not os.path.exists(json_path):
#              st.error(f"Default JSON file not found: {json_path}. Please upload a file.")
#              return None
#         rag_system = GovernmentSchemeRAG(json_path)
#         if not hasattr(rag_system, 'index') or rag_system.index is None or rag_system.index.ntotal == 0:
#              st.warning("RAG system loaded, but the vector index is empty. Check JSON content and embedding process.")
#         return rag_system
#     except ValueError as ve:
#         st.error(f"Failed to initialize RAG system: {ve}")
#         return None
#     except Exception as e:
#         st.error(f"An unexpected error occurred during RAG system loading: {e}")
#         return None


# def main():
#     # Set page configuration
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")

#     # --- Sidebar Configuration ---
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         # Model Selection
#         st.subheader("🤖 Select Model")
#         model_options = ['HuggingFace Flan-T5 Small']
#         google_key_present = bool(os.getenv("GOOGLE_API_KEY"))
#         if google_key_present:
#             model_options.append('Gemini 1.0 Pro')
#         else:
#             st.caption("Gemini model unavailable (GOOGLE_API_KEY not found in .env)")

#         if 'selected_model_name' not in st.session_state:
#              st.session_state.selected_model_name = model_options[0]

#         selected_model_name = st.radio(
#             "Choose the AI model for generating answers:",
#             model_options,
#             key='selected_model_name'
#         )
#         if selected_model_name == 'Gemini 1.0 Pro':
#             st.session_state.model_choice = 'Gemini'
#         else:
#             st.session_state.model_choice = 'HuggingFace'


#         # Theme toggle button
#         st.markdown("---")
#         st.subheader("🎨 Theme")
#         if "dark_mode" not in st.session_state:
#             st.session_state.dark_mode = False

#         if st.button("🌙 Toggle Dark/Light Mode"):
#             st.session_state.dark_mode = not st.session_state.dark_mode
#             # Note: Theme application might need more logic based on st.session_state.dark_mode
#             st.rerun()


#         # File upload
#         st.markdown("---")
#         st.subheader("📁 Data Source")
#         uploaded_file = st.file_uploader("Upload scheme JSON (Optional)", type=["json"])

#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#             st.caption(f"Using uploaded file: {uploaded_file.name}")
#         else:
#             default_path = "scheme_data.json"
#             st.session_state.json_path = default_path
#             st.caption(f"Using default file: {default_path}")


#         # History
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         history_display_limit = 5
#         for entry in reversed(st.session_state.history[-history_display_limit:]):
#              q_short = entry['question'][:40] + "..." if len(entry['question']) > 40 else entry['question']
#              with st.expander(f"❓ {q_short} ({entry.get('model_used', 'N/A')})"):
#                  st.markdown(f"**💬 Answer:**")
#                  st.markdown(entry['answer'])


#     # --- Main Page Logic ---

#     # Load RAG system using the determined path
#     rag_system = load_rag_system(st.session_state.json_path)

#     if rag_system:
#         st.success(f"✅ RAG System Ready. Using **{st.session_state.model_choice}** model.")

#         # ***** START: FIX FOR INPUT CLEARING *****
#         # Check if input needs clearing from previous run ('Get Answer' button click)
#         if st.session_state.get('clear_input_flag', False):
#             st.session_state.user_query_input = ""  # Clear the state variable
#             st.session_state.clear_input_flag = False # Reset the flag
#         # ***** END: FIX FOR INPUT CLEARING *****

#         # Unified Input Section
#         st.subheader("💡 Ask Your Question")
#         example_queries = [
#             "What schemes are available for women entrepreneurs?",
#             "Schemes related to education for girls?",
#             "Financial assistance for farmers?",
#             "Startup schemes in India?",
#             "Tell me about Pradhan Mantri Jan Dhan Yojana"
#         ]

#         # Initialize example query state if not present
#         if 'example_query' not in st.session_state:
#              st.session_state.example_query = ""

#         selected_example = st.selectbox(
#              "📌 Popular Queries (Select or type below):",
#              [""] + example_queries,
#              key="example_query" # Use the state key here
#         )


#         # Initialize user query input state if not present
#         if 'user_query_input' not in st.session_state:
#              st.session_state.user_query_input = st.session_state.example_query # Initialize with example if selected


#         # Update input field if a NEW example is selected
#         # Check if the selectbox value changed and differs from the text input state
#         if selected_example and selected_example != st.session_state.user_query_input:
#              st.session_state.user_query_input = selected_example # Update the text input state
#              st.rerun() # Rerun to reflect the change in the text_input below


#         user_query = st.text_input(
#             "🔍 Type your question here:",
#             key='user_query_input' # Bind to the session state variable
#         )


#         # Scheme filter
#         selected_ministry = None
#         if hasattr(rag_system, 'metadata') and rag_system.metadata:
#             try:
#                 all_ministries = sorted(list(set(meta.get("ministry", "Unknown") for meta in rag_system.metadata if meta)))
#                 all_ministries = [m for m in all_ministries if m != "Unknown"]
#                 selected_ministry = st.selectbox("🏛️ Filter by Ministry (Optional)", ["All"] + all_ministries)
#             except Exception as e:
#                  st.warning(f"Could not generate Ministry filter: {e}")

#         # Process the query
#         if st.button("Get Answer") and user_query.strip():
#             with st.spinner(f"🤔 Thinking using {st.session_state.model_choice}..."):
#                 # 1. Retrieve chunks
#                 results = rag_system.query(user_query, top_k=3)

#                 # 2. Filter by Ministry if selected
#                 if selected_ministry and selected_ministry != "All":
#                     filtered_results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#                     if not filtered_results:
#                          st.warning(f"No relevant results found for the ministry '{selected_ministry}'. Showing general results.")
#                     else:
#                          results = filtered_results

#                 # 3. Generate Answer
#                 if results:
#                     context = "\n\n---\n\n".join([r["chunk"] for r in results])
#                     generated_answer = rag_system.generate_answer(
#                         user_query,
#                         context,
#                         model_choice=st.session_state.model_choice
#                     )
#                 else:
#                     generated_answer = "I couldn't find relevant information in the provided data to answer your question."
#                     results = []

#             # Save to history
#             st.session_state.history.append({
#                 "question": user_query,
#                 "answer": generated_answer,
#                 "sources": results,
#                 "model_used": st.session_state.model_choice
#             })

#             # ***** START: FIX FOR INPUT CLEARING *****
#             # Set flag to clear input on the *next* run
#             st.session_state.clear_input_flag = True
#             # ***** END: FIX FOR INPUT CLEARING *****

#             # Reset example selection (if you want the dropdown cleared too)
#             st.session_state.example_query = ""

#             st.rerun() # Rerun to display answer and clear input (on next run)


#         # --- Display Latest Answer ---
#         if st.session_state.history:
#             latest = st.session_state.history[-1]
#             st.subheader(f"🧠 Answer (via {latest.get('model_used', 'N/A')})")
#             st.markdown(latest["answer"])

#             # Display Sources
#             if latest["sources"]:
#                 st.subheader("📄 Sources Used")
#                 for idx, result in enumerate(latest["sources"], 1):
#                     meta = result.get("metadata", {})
#                     scheme = meta.get('scheme_name', 'Unknown Scheme')
#                     ministry = meta.get('ministry', 'Unknown Ministry')
#                     title = f"{scheme} — {ministry}"
#                     with st.expander(f"Source {idx}: {title}"):
#                         st.markdown(result.get("chunk", "N/A"))
#             else:
#                  st.caption("No specific sources identified or used for this answer.")

#     else:
#         st.warning("RAG system could not be loaded. Please check the JSON file or configuration.")


# if __name__ == "__main__":
#     main()

### Version -4 

import streamlit as st
from rag import GovernmentSchemeRAG

@st.cache_resource
def load_rag_system(json_path):
    return GovernmentSchemeRAG(json_path)

def main():
    # Set page configuration
    st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
    st.title("🗂️ Government Scheme QnA")

    # Sidebar config
    with st.sidebar:
        st.header("⚙️ Configuration")

        # Theme toggle button
        if "dark_mode" not in st.session_state:
            st.session_state.dark_mode = False  # Default to light mode

        if st.button("🌙 Toggle Dark Mode"):
            st.session_state.dark_mode = not st.session_state.dark_mode
            # Inject JavaScript to toggle theme dynamically
            js_code = """
                const root = document.querySelector('html');
                if (root.classList.contains('dark')) {
                    root.classList.remove('dark');
                } else {
                    root.classList.add('dark');
                }
            """
            st.components.v1.html(f"<script>{js_code}</script>", height=0)

        # File upload
        uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
        if uploaded_file:
            st.session_state.json_path = uploaded_file
        else:
            st.session_state.json_path = "scheme_data.json"

        # History
        st.markdown("---")
        st.subheader("📜 History")
        if "history" not in st.session_state:
            st.session_state.history = []
        for entry in reversed(st.session_state.history):
            with st.expander(f"❓ {entry['question'][:40]}..."):
                st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

    # Load RAG system
    rag_system = load_rag_system(st.session_state.json_path)
    st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

    # Unified Input Section
    st.subheader("💡 Ask Your Question")
    example_queries = [
        "What schemes are available for women entrepreneurs?",
        "Schemes related to education for girls?",
        "Financial assistance for farmers?",
        "Startup schemes in India?"
    ]

    # Dropdown for suggestions
    selected_example = st.selectbox("📌 Popular Queries:", [""] + example_queries)

    # Text input field for typing or displaying selected suggestion
    user_query = st.text_input("🔍 Type your question here:", value=selected_example)

    # Scheme filter
    all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
    selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

    # Process the query
    if user_query.strip():  # Ensure the query is not empty
        with st.spinner("🤔 Thinking..."):
            results = rag_system.query(user_query, top_k=3)
            if selected_ministry != "All":
                results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
            context = "\n\n".join([r["chunk"] for r in results])
            generated_answer = rag_system.generate_answer(user_query, context)

        # Save to history
        st.session_state.history.append({
            "question": user_query,
            "answer": generated_answer,
            "sources": results
        })

    # Show latest answer
    if st.session_state.history:
        latest = st.session_state.history[-1]
        st.subheader("🧠 Answer")
        st.write(latest["answer"])

        col1, col2 = st.columns(2)
        with col1:
            if st.button("👍 Helpful", key="like"):
                st.success("Thanks for your feedback!")
        with col2:
            if st.button("👎 Not helpful", key="dislike"):
                st.warning("We’ll use this to improve.")

        st.subheader("📄 Sources")
        for idx, result in enumerate(latest["sources"], 1):
            meta = result["metadata"]
            title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
            with st.expander(f"Source {idx}: {title}"):
                st.markdown(result["chunk"])

if __name__ == "__main__":
    main()

## Version -4 compatible with ver-3 of rag.py

# import streamlit as st
# from rag import GovernmentSchemeRAG
# from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import speech_recognition as sr
# import av
# from io import BytesIO

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def transcribe_audio(audio):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio) as source:
#         audio_data = recognizer.record(source)
#         return recognizer.recognize_google(audio_data)

# def main():
#     # Set page configuration
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")

#     # Sidebar config
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         # Theme toggle button
#         if "dark_mode" not in st.session_state:
#             st.session_state.dark_mode = False  # Default to light mode

#         if st.button("🌙 Toggle Dark Mode"):
#             st.session_state.dark_mode = not st.session_state.dark_mode

#         # Apply theme based on session state
#         if st.session_state.dark_mode:
#             st.markdown("""
#                 <style>
#                     body { background-color: #0e1117; color: #fafafa; }
#                 </style>
#             """, unsafe_allow_html=True)
#         else:
#             st.markdown("""
#                 <style>
#                     body { background-color: #ffffff; color: #000000; }
#                 </style>
#             """, unsafe_allow_html=True)

#         # File upload
#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#         # History
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         for entry in reversed(st.session_state.history):
#             with st.expander(f"❓ {entry['question'][:40]}..."):
#                 st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Unified Input Section
#     st.subheader("💡 Ask Your Question")
#     example_queries = [
#         "What schemes are available for women entrepreneurs?",
#         "Schemes related to education for girls?",
#         "Financial assistance for farmers?",
#         "Startup schemes in India?"
#     ]

#     # Dropdown for suggestions
#     selected_example = st.selectbox("📌 Popular Queries:", [""] + example_queries)

#     # Text input field for typing or displaying selected suggestion/audio transcription
#     user_query = st.text_input("🔍 Type your question here:", value=selected_example)

#     # Voice input integration
#     st.markdown("🎙️ Use voice input (optional):")
#     voice_text = None
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=256,
#         media_stream_constraints={"audio": True, "video": False},
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     )

#     if webrtc_ctx.audio_receiver:
#         try:
#             audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             wav_buffer = BytesIO()
#             for frame in audio_frames:
#                 sound = av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")
#                 sound.sample_rate = frame.sample_rate
#                 sound.to_audio().export(wav_buffer, format="wav")
#             wav_buffer.seek(0)
#             voice_text = transcribe_audio(wav_buffer)
#             st.success(f"🗣️ Transcribed: {voice_text}")
#             user_query = voice_text  # Replace the text input with transcribed text
#         except Exception as e:
#             st.error(f"Error during transcription: {e}")

#     # Scheme filter
#     all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
#     selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

#     # Process the query
#     if user_query.strip():  # Ensure the query is not empty
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             if selected_ministry != "All":
#                 results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

# if __name__ == "__main__":
#     main()

# # # ### Improved version -3 

# import streamlit as st
# from rag import GovernmentSchemeRAG
# from streamlit_webrtc import webrtc_streamer, WebRtcMode
# import speech_recognition as sr
# import av
# from io import BytesIO

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def transcribe_audio(audio):
#     recognizer = sr.Recognizer()
#     with sr.AudioFile(audio) as source:
#         audio_data = recognizer.record(source)
#         return recognizer.recognize_google(audio_data)

# def main():
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")

#     # Sidebar config
#     with st.sidebar:
#         st.header("⚙️ Configuration")

#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#         dark_mode = st.checkbox("🌙 Dark Mode (manual)", value=False)
#         if dark_mode:
#             st.markdown("<style>body { background-color: #0e1117; color: #fafafa; }</style>", unsafe_allow_html=True)

#         # History
#         st.markdown("---")
#         st.subheader("📜 History")
#         if "history" not in st.session_state:
#             st.session_state.history = []
#         for entry in reversed(st.session_state.history):
#             with st.expander(f"❓ {entry['question'][:40]}..."):
#                 st.markdown(f"💬 **Answer:** {entry['answer'][:300]}{'...' if len(entry['answer']) > 300 else ''}")

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Popular queries
#     st.subheader("💡 Suggested Questions")
#     example_queries = [
#         "What schemes are available for women entrepreneurs?",
#         "Schemes related to education for girls?",
#         "Financial assistance for farmers?",
#         "Startup schemes in India?"
#     ]
#     selected_example = st.selectbox("📌 Pick a popular query or type your own:", [""] + example_queries)

#     # Voice input
#     st.markdown("🎙️ Use voice input (optional):")
#     voice_text = None
#     webrtc_ctx = webrtc_streamer(
#         key="speech-to-text",
#         mode=WebRtcMode.SENDONLY,
#         audio_receiver_size=256,
#         media_stream_constraints={"audio": True, "video": False},
#         rtc_configuration={"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]},
#     )

#     if webrtc_ctx.audio_receiver:
#         try:
#             audio_frames = webrtc_ctx.audio_receiver.get_frames(timeout=1)
#             wav_buffer = BytesIO()
#             for frame in audio_frames:
#                 sound = av.AudioFrame.from_ndarray(frame.to_ndarray(), layout="mono")
#                 sound.sample_rate = frame.sample_rate
#                 sound.to_audio().export(wav_buffer, format="wav")
#             wav_buffer.seek(0)
#             voice_text = transcribe_audio(wav_buffer)
#             st.success(f"🗣️ Transcribed: {voice_text}")
#         except:
#             pass

#     # Scheme filter
#     all_ministries = sorted(set([meta.get("ministry", "Unknown") for meta in rag_system.metadata]))
#     selected_ministry = st.selectbox("🏛️ Filter by Ministry", ["All"] + all_ministries)

#     # Determine query
#     user_query = voice_text or selected_example or st.text_input("🔍 Or type your own question:")

#     if user_query:
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             if selected_ministry != "All":
#                 results = [r for r in results if r["metadata"].get("ministry") == selected_ministry]
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

# if __name__ == "__main__":
#     main()


# ### Imporved UI and More features
# import streamlit as st
# from rag import GovernmentSchemeRAG

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def main():
#     st.set_page_config(page_title="🗂️ Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")
#     st.markdown("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

#     # Sidebar for upload
#     with st.sidebar:
#         st.header("⚙️ Configuration")
#         uploaded_file = st.file_uploader("📁 Upload scheme JSON", type=["json"])
#         if uploaded_file:
#             st.session_state.json_path = uploaded_file
#         else:
#             st.session_state.json_path = "scheme_data.json"

#     # Load RAG system
#     rag_system = load_rag_system(st.session_state.json_path)
#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     # Initialize history
#     if "history" not in st.session_state:
#         st.session_state.history = []

#     # User input
#     user_query = st.text_input("🔍 Enter your question:")
#     if user_query:
#         with st.spinner("🤔 Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         # Save to history
#         st.session_state.history.append({
#             "question": user_query,
#             "answer": generated_answer,
#             "sources": results
#         })

#     # Show latest answer
#     if st.session_state.history:
#         latest = st.session_state.history[-1]
#         st.subheader("🧠 Answer")
#         st.write(latest["answer"])

#         # Feedback
#         col1, col2 = st.columns(2)
#         with col1:
#             if st.button("👍 Helpful", key="like"):
#                 st.success("Thanks for your feedback!")
#         with col2:
#             if st.button("👎 Not helpful", key="dislike"):
#                 st.warning("We’ll use this to improve.")

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(latest["sources"], 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])

#     # Show past history
#     if len(st.session_state.history) > 1:
#         with st.expander("📜 Previous Questions"):
#             for entry in reversed(st.session_state.history[:-1]):
#                 st.markdown(f"**❓ Q:** {entry['question']}")
#                 st.markdown(f"**💡 A:** {entry['answer']}")
#                 st.markdown("---")

# if __name__ == "__main__":
#     main()

### Working Code -2(Giving Ans with sources -1)

# import streamlit as st
# from rag import GovernmentSchemeRAG


# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)


# def main():
#     st.set_page_config(page_title="Government Scheme QnA", layout="wide")
#     st.title("🗂️ Government Scheme QnA")
#     st.write("Ask a question about Indian government schemes. Example: _What schemes are available for women entrepreneurs?_")

#     # Set default path
#     if "json_path" not in st.session_state:
#         st.session_state.json_path = "scheme_data.json"

#     rag_system = load_rag_system(st.session_state.json_path)

#     st.success(f"✅ Loaded {len(rag_system.chunks)} chunks from {len(rag_system.metadata)} schemes.")

#     user_query = st.text_input("🔍 Enter your question:")
#     if user_query:
#         with st.spinner("Thinking..."):
#             results = rag_system.query(user_query, top_k=3)
#             context = "\n\n".join([r["chunk"] for r in results])
#             generated_answer = rag_system.generate_answer(user_query, context)

#         st.subheader("🧠 Answer")
#         st.write(generated_answer)

#         st.subheader("📄 Sources")
#         for idx, result in enumerate(results, 1):
#             meta = result["metadata"]
#             title = f"{meta.get('scheme_name', 'Unknown')} — {meta.get('ministry', '')}"
#             with st.expander(f"Source {idx}: {title}"):
#                 st.markdown(result["chunk"])


# if __name__ == "__main__":
#     main()



## Working Code -1

# # main.py - Streamlit interface for the RAG application
# import streamlit as st
# import os
# import json
# import tempfile
# from rag import GovernmentSchemeRAG

# st.set_page_config(
#     page_title="Government Scheme Q&A",
#     page_icon="🏛️",
#     layout="wide"
# )

# @st.cache_resource
# def load_rag_system(json_path):
#     return GovernmentSchemeRAG(json_path)

# def main():
#     st.title("🏛️ Government Scheme Question & Answer System")
#     st.write("Ask questions about government schemes from MyScheme portal")
    
#     # File upload section (first run only)
#     if 'data_loaded' not in st.session_state:
#         st.session_state.data_loaded = False
    
#     if not st.session_state.data_loaded:
#         # Option to use local file path
#         use_local_file = st.checkbox("Use local JSON file")
        
#         if use_local_file:
#             file_path = "/Users/amitanand/Desktop/Utkarsh/scheme_data.json"
#             st.text(f"Using local file: {file_path}")
            
#             if st.button("Load Data"):
#                 st.session_state.json_path = file_path
#                 st.session_state.data_loaded = True
#                 st.experimental_rerun()
#         else:
#             uploaded_file = st.file_uploader("Upload your schemes JSON file", type=["json"])
            
#             if uploaded_file:
#                 # Save the uploaded file to a temporary file
#                 with tempfile.NamedTemporaryFile(delete=False, suffix='.json') as tmp_file:
#                     tmp_file.write(uploaded_file.getvalue())
#                     temp_json_path = tmp_file.name
                
#                 st.session_state.json_path = temp_json_path
#                 st.session_state.data_loaded = True
#                 st.experimental_rerun()
    
#     # Once data is loaded, show the QA interface
#     if st.session_state.data_loaded:
#         with st.spinner("Loading RAG system... This may take a while for large JSON files."):
#             # Initialize RAG system
#             rag_system = load_rag_system(st.session_state.json_path)
        
#         # Check if RAG system was initialized successfully
#         if rag_system and hasattr(rag_system, 'chunks') and len(rag_system.chunks) > 0:
#             st.success(f"Successfully loaded {len(rag_system.chunks)} chunks from {len(rag_system.schemes_data)} schemes!")
            
#             # Input for user question
#             question = st.text_input("Ask a question about government schemes:", 
#                                     "What schemes are available for farmers in Maharashtra?")
            
#             # Number of sources to retrieve
#             top_k = st.slider("Number of sources to retrieve:", min_value=1, max_value=5, value=3)
            
#             # Process question on button click
#             if st.button("Get Answer"):
#                 with st.spinner("Searching and generating answer..."):
#                     result = rag_system.query(question, top_k=top_k)
                
#                 # Display answer
#                 st.subheader("Answer")
#                 st.write(result)
                
#                 # Display sources
#                 st.subheader("Sources")
#                 for idx, source in enumerate(result, 1):
#                     with st.expander(f"Source {idx}: {source.get('scheme_name', 'Unknown Scheme')}"):
#                         st.write(f"**Ministry:** {source.get('ministry', 'N/A')}")
#                         st.write(f"**Department:** {source.get('department', 'N/A')}")
#                         st.write(f"**Page:** {source.get('page', 'N/A')}")
#                         if 'type' in source:
#                             st.write(f"**Content Type:** {source['type']}")
#         else:
#             st.error("Failed to initialize RAG system. Please check the JSON file and try again.")

# if __name__ == "__main__":
#     main()