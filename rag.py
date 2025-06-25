### APi key input 
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import re

class GovernmentSchemeRAG:
    def __init__(self, json_path, hf_token=""):
        self.json_path = json_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.dimension = None

        # API Key provided via parameter (from Streamlit input)
        self.hf_token = hf_token

        self.chunks, self.metadata = self.chunk_documents()
        if not self.chunks:
            raise ValueError("No chunks available to create embeddings.")

        self.create_index()

    def chunk_documents(self):
        chunks = []
        metadata = []
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:  # Specify encoding
                self.schemes_data = json.load(f)
        except FileNotFoundError:
            print(f"Error: JSON file not found at {self.json_path}")
            return [], []
        except json.JSONDecodeError:
            print(f"Error: Could not decode JSON from {self.json_path}. Check file format.")
            return [], []
        except Exception as e:
            print(f"An unexpected error occurred loading the JSON: {e}")
            return [], []

        for scheme in self.schemes_data:
            data = scheme.get("data", {})

            text_parts = []
            scheme_name = data.get("scheme_name", "Unknown Scheme")
            ministry = data.get("ministry", "Unknown Ministry")
            department = data.get("department", "Unknown Department")

            text_parts.append(f"Scheme: {scheme_name}")
            text_parts.append(f"Ministry: {ministry}")
            text_parts.append(f"Department: {department}")

            for key in ["details_content", "eligibility_content", "application_process"]:
                content = data.get(key, [])
                if isinstance(content, list):
                    # Clean up potential None values or non-string items if necessary
                    cleaned_content = [str(item) for item in content if item is not None]
                    text_parts.extend(cleaned_content)
                elif content is not None:  # Handle cases where it might be a single string
                    text_parts.append(str(content))

            chunk = "\n".join(text_parts).strip()
            if chunk:
                chunks.append(chunk)
                metadata.append({
                    "scheme_name": scheme_name,
                    "ministry": ministry,
                    "department": department
                })

        return chunks, metadata

    def create_index(self):
        if not self.chunks:
            print("Skipping index creation as no chunks were loaded.")
            return
        embeddings = np.array([self.embedding_model.encode(chunk) for chunk in self.chunks]).astype('float32')  # Ensure float32

        if embeddings.ndim == 1:
            if embeddings.shape[0] > 0:  # Check if the single dimension is not empty
                self.dimension = embeddings.shape[0]
                embeddings = embeddings.reshape(1, -1)
            else:
                print("Warning: Embeddings array is empty or invalid.")
                return  # Cannot create index with empty embeddings
        elif embeddings.shape[0] == 0:  # Check if the 2D array has no rows
            print("Warning: Embeddings array is empty.")
            return  # Cannot create index with empty embeddings
        else:
            self.dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        print(f"FAISS index created successfully with {self.index.ntotal} vectors.")

    def query(self, question, top_k=3):
        if not self.index or self.index.ntotal == 0:
            return []  # Return empty if index doesn't exist or is empty
        question_embedding = self.embedding_model.encode(question).reshape(1, -1).astype('float32')  # Ensure float32
        distances, indices = self.index.search(question_embedding, top_k)

        results = []
        for i in indices[0]:
            # Check index bounds robustly
            if 0 <= i < len(self.chunks):
                results.append({
                    "chunk": self.chunks[i],
                    "metadata": self.metadata[i]
                })
            else:
                print(f"Warning: Index {i} out of bounds for chunks list (length {len(self.chunks)}).")
        return results

    def generate_answer(self, question, context):
        prompt = f"""
Context about government schemes:
{context}

Question: {question}

Given the context below about a government scheme, answer the user's question concisely, focusing on the key details requested.

If available, mention:
- Scheme Name
- Purpose
- Eligibility
- Key Benefits
- Application Process Overview (briefly)
- Website Link (if explicitly found in context)

Highlight important section titles in **bold**.
If information is missing for a section, simply omit that section. Be clear and direct.
"""
        answer = "Could not generate answer using Hugging Face."  # Default error message

        if self.hf_token:
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-small"
            headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
            payload = {"inputs": prompt, "options": {"wait_for_model": True, "max_length": 450, "temperature": 0.1}}
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()  # Raise HTTPError for bad responses (4xx or 5xx)
                output = response.json()
                if output and isinstance(output, list) and 'generated_text' in output[0]:
                    answer = output[0].get("generated_text", "No answer returned by Flan-T5.")
                else:
                    answer = f"Unexpected response format from Flan-T5 API: {output}"
            except requests.exceptions.RequestException as e:
                print(f"Error calling Hugging Face API: {e}")
                answer = f"Error: Could not connect to Hugging Face API - {e}"
            except Exception as e:
                print(f"Error processing Hugging Face response: {e}")
                answer = f"Error processing Hugging Face response: {e}"
        else:
            answer = "Hugging Face model unavailable (check HUGGINGFACE_TOKEN input)."

        # Apply post-processing
        answer = answer.replace("Scheme Name:", "**Scheme Name:**")
        answer = answer.replace("Ministry/Department:", "**Ministry/Department:**")
        answer = answer.replace("Purpose:", "**Purpose:**")
        answer = answer.replace("Benefits:", "**Benefits:**")
        answer = answer.replace("Key Benefits:", "**Key Benefits:**")
        answer = answer.replace("Eligibility:", "**Eligibility:**")
        answer = answer.replace("Application Process:", "**Application Process:**")
        answer = answer.replace("Application Process Overview:", "**Application Process Overview:**")
        answer = answer.replace("Required Documents:", "**Required Documents:**")
        answer = answer.replace("Website Link:", "**Website Link:**")
        answer = answer.replace("Source:", "**Source:**")  # Keep this if your prompt might generate it

        # Make URLs clickable
        urls = re.findall(r'(https?://[^\s]+)', answer)
        for url in urls:
            # Basic check to avoid mangling markdown links if already formatted
            if f"[{url}]({url})" not in answer and f"**Website Link:** {url}" in answer:
                answer = answer.replace(url, f"[{url}]({url})")

        # Format application steps (basic newline formatting)
        if "**Application Process:**" in answer or "**Application Process Overview:**" in answer:
            lines = answer.split('\n')
            formatted_lines = []
            in_app_process = False
            for line in lines:
                if line.strip().startswith("**Application Process"):
                    in_app_process = True
                    formatted_lines.append(line)
                elif in_app_process and re.match(r'^\s*\d+\.\s+', line.strip()):
                    formatted_lines.append(line.strip())  # Keep numbered steps
                elif in_app_process and line.strip().startswith('- '):
                    formatted_lines.append(line.strip())  # Keep bullet points
                elif in_app_process and line.strip() == "":
                    # Stop adding newlines if the section seems to end
                    if len(formatted_lines) > 0 and formatted_lines[-1].strip() != "":
                        in_app_process = False  # Assume end of section on blank line
                    formatted_lines.append(line)  # Keep blank lines within reason
                elif in_app_process:
                    formatted_lines.append(line)  # Keep other lines in the section
                else:
                    formatted_lines.append(line)  # Add lines outside the section
            answer = "\n".join(formatted_lines)

        return answer
