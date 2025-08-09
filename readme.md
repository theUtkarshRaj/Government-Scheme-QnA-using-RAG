# 🤖 Government Scheme RAG Chatbot

This project is a sophisticated, yet easy-to-use chatbot built with Streamlit and LangChain. It leverages the power of Retrieval-Augmented Generation (RAG) to answer questions about Indian government schemes using Google's Gemini large language model.

## ✨ Features

  * **Interactive Chat Interface**: A clean and user-friendly web UI built with Streamlit.
  * **Google Gemini Powered**: Directly connects to the Gemini API to provide high-quality language understanding and generation.
  * **Retrieval-Augmented Generation (RAG)**: Provides contextually-aware and accurate answers by retrieving relevant information from a local knowledge base before generating a response.
  * **User-Controlled Knowledge Base**:
      * **On-Demand Loading**: Users explicitly trigger the data loading process with a button, ensuring control over when heavy processing occurs.
      * **Adjustable Document Limit**: A sidebar slider allows users to control how many scheme documents are loaded, balancing knowledge depth with performance.
  * **Conversational Memory**: The chatbot considers the recent chat history to provide more relevant answers to follow-up questions.
  * **Efficient Search**: Uses **ChromaDB** for fast and effective local similarity searches on scheme documents.
  * **Modular Codebase**: The core RAG logic (`rag.py`) is decoupled from the Streamlit frontend (`main_streamlit.py`) for better maintainability.

-----

## ⚙️ How It Works: The RAG Architecture

The chatbot operates on a RAG pipeline, which is an intelligent way to make Large Language Models (LLMs) answer questions based on a specific set of documents.

1.  **Load & Chunk**: When triggered by the user, scheme data from the `scheme_data.json` file is loaded. Each scheme is formatted into a `Document` object and split into manageable chunks.
2.  **Embed & Store**: Each document chunk is converted into a numerical representation (embedding) using the `all-MiniLM-L6-v2` model. These embeddings are then stored in a **ChromaDB** vector store, which acts as a searchable local knowledge index.
3.  **Retrieve**: When you ask a question, your query is also converted into an embedding. The system then searches the ChromaDB index to find the document chunks with the most similar embeddings (i.e., the most relevant schemes).
4.  **Augment & Generate**: The relevant documents (context), your question, and the recent chat history are combined into a single, detailed prompt. This "augmented" prompt is then sent to the Gemini LLM, which generates a final, context-aware answer.

-----

## 🚀 Getting Started

Follow these steps to set up and run the project locally.

### Prerequisites

  * Python 3.8 or higher
  * A Google Gemini API key. You can get one from [Google AI Studio](https://aistudio.google.com/app/apikey).

### 1\. Clone the Repository

```bash
git clone https://github.com/theUtkarshRaj/Government-Scheme-QnA-using-RAG
cd Government-Scheme-QnA-using-RAG
```

### 2\. Set Up a Virtual Environment

It's highly recommended to use a virtual environment to manage project dependencies.

```bash
# Create a virtual environment
python -m venv venv

# Activate it
# On Windows
venv\Scripts\activate
# On macOS/Linux
source venv/bin/activate
```

### 3\. Install Dependencies

Install all the required Python packages from the `requirements.txt` file.

```bash
pip install -r requirements.txt
```

### 4\. Prepare Your Data File

The chatbot requires a `scheme_data.json` file in the root directory. You can use your own data, but it must follow this structure: a list of objects, where each object contains a single `"data"` key.

**`scheme_data.json` Example:**

```json
[
  {
    "data": {
      "scheme_name": "Pradhan Mantri Kisan Samman Nidhi",
      "ministry": "Ministry of Agriculture & Farmers Welfare",
      "department": "Department of Agriculture and Farmers Welfare",
      "details_content": [
        "This is a central sector scheme with 100% funding from the Government of India.",
        "It provides an income support of Rs. 6,000/- per year in three equal installments."
      ],
      "eligibility_content": [
        "All small and marginal farmer families are eligible.",
        "The farmer family is defined as a husband, wife, and minor children."
      ],
      "application_process": [
        "Farmers can self-register through the Farmers Corner in the portal.",
        "Registration is also available through Common Service Centers (CSCs)."
      ]
    }
  },
  {
    "data": {
      "scheme_name": "Atal Pension Yojana",
      "ministry": "Ministry of Finance",
      "department": "Department of Financial Services",
      "details_content": [
        "A pension scheme for citizens of India focused on the unorganized sector workers.",
        "Guarantees a minimum monthly pension of Rs. 1,000, 2,000, 3,000, 4,000 or 5,000 at the age of 60."
      ],
      "eligibility_content": [
        "Any citizen of India between 18-40 years of age.",
        "Must have a savings bank account."
      ],
      "application_process": [
        "Can be subscribed by visiting the bank branch/post office where the individual’s savings bank account is held."
      ]
    }
  }
]
```

### 5\. Run the Application

Launch the Streamlit app with the following command:

```bash
streamlit run main_streamlit.py
```

Your web browser should automatically open to the application's URL (usually `http://localhost:8501`).

-----

## 💬 Usage

1.  **Enter API Key**: On the sidebar, enter your Google Gemini API Key. The chatbot will not start until it's provided.
2.  **Configure Options**: Use the "Documents to Load" slider to set how many schemes you want the bot to know about.
3.  **Load Knowledge Base**: Click the **"Load Knowledge Base"** button. This will start the data processing and embedding. The chat will not be active until this step is complete.
4.  **Ask Questions**: Once the knowledge base is loaded and the API key is accepted, you can type your question into the chat input at the bottom of the page and press Enter.
5.  **Clear History**: Use the "Clear Chat History" button in the sidebar at any time to reset the conversation.

-----

## 📂 Project Structure

```
.
├── main.py             # The Streamlit frontend application
├── rag.py                # Core RAG logic and LangChain implementation
├── requirements.txt      # List of Python dependencies
├── scheme_data.json      # Default data file for government schemes
└── README.md             # This file
```