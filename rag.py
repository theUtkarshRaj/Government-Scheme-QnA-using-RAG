import os
import json
from typing import List, Dict, Any
from dotenv import load_dotenv

# LangChain core components
from langchain.docstore.document import Document
from langchain.prompts import ChatPromptTemplate
from langchain.schema.output_parser import StrOutputParser
from langchain.schema.runnable import RunnableParallel, RunnablePassthrough

# Embeddings and Vector Store
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint, ChatHuggingFace

# Google Generative AI
from langchain_google_genai import ChatGoogleGenerativeAI

# Load .env
load_dotenv()


class GovernmentSchemeRAG:
    def __init__(self, json_path: str, google_api_key: str, hf_token: str):
        if not google_api_key and not hf_token:
            raise ValueError("You must provide either a Google API key or a Hugging Face token.")

        self.json_path = json_path
        self.google_api_key = google_api_key
        self.hf_token = hf_token

        # Embeddings
        self.embedding_model = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

        # LLM
        self.llm = self._initialize_llm()

        # Load docs
        print(f"Loading documents from: {self.json_path}")
        documents = self._load_documents()
        if not documents:
            raise ValueError(f"No documents were loaded from {self.json_path}.")

        # Vector store
        vector_store = FAISS.from_documents(documents, self.embedding_model)
        self.retriever = vector_store.as_retriever(search_kwargs={"k": 5})
        print(f"Vector store created with {len(documents)} documents.")

        # Prompt
        self.prompt = self._create_prompt_template()

        # Chain
        self.rag_chain_with_sources = self._build_rag_chain()

    def _initialize_llm(self):
        """Initialize the LLM for Hugging Face or Google."""
        if self.google_api_key:
            print("Initializing model: Gemini 1.5 Flash")
            os.environ["GOOGLE_API_KEY"] = self.google_api_key
            return ChatGoogleGenerativeAI(model="gemini-1.5-flash-latest", temperature=0.3)

        if self.hf_token:
            print("Initializing model: mistralai/Mistral-7B-Instruct-v0.2")
            endpoint_llm = HuggingFaceEndpoint(
                repo_id="mistralai/Mistral-7B-Instruct-v0.2",
                temperature=0.3,
                max_new_tokens=256,
                top_p=0.9,
                huggingfacehub_api_token=self.hf_token
            )
            return ChatHuggingFace(llm=endpoint_llm) 

        return None

    def _load_documents(self) -> List[Document]:
        """Load JSON file into LangChain Document objects."""
        try:
            with open(self.json_path, "r", encoding="utf-8") as f:
                schemes_data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Error reading or parsing JSON file: {e}")
            return []

        documents = []
        for scheme in schemes_data:
            data = scheme.get("data", {})
            scheme_name = data.get("scheme_name", "Unknown Scheme")

            content_parts = [
                f"Scheme: {scheme_name}",
                f"Ministry: {data.get('ministry', 'N/A')}",
                f"Department: {data.get('department', 'N/A')}",
            ]
            for key in ["details_content", "eligibility_content", "application_process"]:
                content = data.get(key, [])
                if isinstance(content, list):
                    content_parts.extend([str(item) for item in content if item])
                elif content:
                    content_parts.append(str(content))

            page_content = "\n".join(content_parts).strip()

            doc = Document(
                page_content=page_content,
                metadata={
                    "scheme_name": scheme_name,
                    "ministry": data.get("ministry", "N/A"),
                    "department": data.get("department", "N/A"),
                    "source": os.path.basename(self.json_path)
                }
            )
            documents.append(doc)

        return documents

    @staticmethod
    def _create_prompt_template() -> ChatPromptTemplate:
        """Prompt template with history."""
        template = """
You are a friendly and helpful chatbot. Based on the context and chat history below,
provide a concise and direct answer to the user's question in 2-3 sentences.

Chat History:
{history}

Context from Documents:
{context}

User Question:
{question}

Concise Answer:
"""
        return ChatPromptTemplate.from_template(template)

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        """Format retrieved docs into a single string."""
        return "\n\n---\n\n".join([d.page_content for d in docs])

    def _build_rag_chain(self):
        """Build RAG chain with history support."""
        context_retriever_chain = (
            (lambda x: x["question"])
            | self.retriever
            | self._format_docs
        )

        rag_chain = (
            RunnablePassthrough.assign(context=context_retriever_chain)
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

        chain_with_sources = RunnableParallel(
            answer=rag_chain,
            sources=(lambda x: x["question"]) | self.retriever,
        )
        return chain_with_sources

    @staticmethod
    def _post_process_answer(answer: str) -> str:
        return answer.strip()

    def query(self, question: str, history: str = "") -> Dict[str, Any]:
        """Query the RAG system."""
        print(f"Invoking RAG chain for question: '{question}'")
        if not self.rag_chain_with_sources:
            return {"answer": "Error: RAG chain is not initialized.", "sources": []}

        input_data = {"question": question, "history": history}
        result = self.rag_chain_with_sources.invoke(input_data)
        result["answer"] = self._post_process_answer(result["answer"])
        return result
