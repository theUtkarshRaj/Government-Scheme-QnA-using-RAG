import os
import json
from dotenv import load_dotenv

from langchain.schema import Document
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.chains import create_retrieval_chain, create_history_aware_retriever
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.llms.base import LLM
from pydantic import BaseModel, Field
from typing import Optional, List, Any
from langchain.schema import Generation, LLMResult
import google.generativeai as genai

# ---------- SETTINGS ----------
CHUNK_SIZE = 500
CHUNK_OVERLAP = 50
FILE_PATH = "scheme_data.json"

load_dotenv()

# ---------- Gemini LLM Wrapper ----------
class GeminiLLM(LLM, BaseModel):
    api_key: str = Field(..., exclude=True)
    model_name: str = "gemini-1.5-flash-latest"
    model: Any = None

    def __init__(self, **data: Any):
        super().__init__(**data)
        genai.configure(api_key=self.api_key)
        self.model = genai.GenerativeModel(self.model_name)

    @property
    def _llm_type(self) -> str:
        return "google-gemini"

    def _call(self, prompt: str, stop: Optional[List[str]] = None) -> str:
        try:
            config = {"stop_sequences": stop} if stop else None
            response = self.model.generate_content(prompt, generation_config=config)
            return response.text
        except Exception as e:
            print(f"Error calling Gemini API: {e}")
            return "Sorry, something went wrong on my end. Please try again."

    def _generate(self, prompts: List[str], stop: Optional[List[str]] = None) -> LLMResult:
        generations = []
        for prompt in prompts:
            text = self._call(prompt, stop=stop)
            generations.append([Generation(text=text)])
        return LLMResult(generations=generations)

# ---------- Data Loading & Processing  ----------
def load_data(file_path, limit=None):
    try:
        with open(file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
        return data[:limit] if limit else data
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Error loading data from {file_path}: {e}")
        return []

#-------  Langchain works good with documents --------
def prepare_documents(data):
    docs = []
    for entry in data:
        if "data" not in entry: continue
        d = entry["data"]
        page_content = (f"Scheme Name: {d.get('scheme_name', 'N/A')}\n"
                        f"Ministry/Department: {d.get('ministry', 'N/A')} / {d.get('department', 'N/A')}\n"
                        f"Benefits: {' '.join(d.get('details_content', []))}\n"
                        f"Eligibility: {' '.join(d.get('eligibility_content', []))}\n"
                        f"Application Process: {' '.join(d.get('application_process', []))}\n"
                        f"Tags: {', '.join(d.get('tags', []))}\n")
        metadata = {"scheme_name": str(d.get('scheme_name', 'Unknown')).lower(),
                    "ministry": str(d.get('ministry', 'Unknown')).lower(),
                    "department": str(d.get('department', 'Unknown')).lower()}
        docs.append(Document(page_content=page_content, metadata=metadata))
    return docs

# -------- Divide into chunks -----------
def split_documents(documents, chunk_size=CHUNK_SIZE, chunk_overlap=CHUNK_OVERLAP):

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap,
                                              separators=["\n\n", "\n", ". ", ", ", " "], length_function=len)
    return splitter.split_documents(documents)

# ---------- KEY-INDEPENDENT RAG COMPONENTS ----------
def load_and_build_vectorstore(limit: int = 100): 
    """
    Loads data, prepares documents, and builds the vector store.
    This function is slow and does not require an API key.
    """
    print(f"Loading data and building vector store with a limit of {limit} documents...")
    
    # Use the 'limit' parameter when loading data
    data = load_data(FILE_PATH, limit=limit) 
    
    if not data: return None
    docs = prepare_documents(data)
    if not docs: return None
    chunked_docs = split_documents(docs)
    
    embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2", model_kwargs={"device": "cpu"})
    vectorstore = Chroma.from_documents(documents=chunked_docs, embedding=embeddings)
    print("Vector store built successfully.")
    return vectorstore

# ---------- KEY-DEPENDENT RAG COMPONENTS ----------
def create_llm(api_key: str):
    """Creates the Gemini LLM instance using the provided API key."""
    if not api_key:
        raise ValueError("A valid Gemini API key must be provided.")
    return GeminiLLM(api_key=api_key)

def get_rag_chain(vectorstore, api_key: str):
    """
    Creates the final RAG chain using the pre-built vector store and the user-provided API key.
    """
    if not vectorstore or not api_key:
        return None

    llm = create_llm(api_key)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})

    contextualize_q_system_prompt = (
        "Given a chat history and the latest user question which might reference context in the chat history, "
        "formulate a standalone question which can be understood without the chat history. "
        "Do NOT answer the question, just reformulate it if needed and otherwise return it as is.")
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(llm, retriever, contextualize_q_prompt)

    qa_system_prompt = (
        "You are an expert assistant for Indian government schemes. Your goal is to provide brief, accurate, and direct answers based on the context below. "
        "Summarize the key information. Do not add conversational fillers, greetings, or long explanations. "
        "Get straight to the point. If a specific detail is not in the context, state that it is unavailable.\n\n"
        "Here is the information I found:\n{context}")
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", qa_system_prompt),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}"),
    ])
    qa_chain = create_stuff_documents_chain(llm, qa_prompt)
    
    return create_retrieval_chain(history_aware_retriever, qa_chain)