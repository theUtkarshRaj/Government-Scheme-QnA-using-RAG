### APi key input 
import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import re
import google.generativeai as genai

class GovernmentSchemeRAG:
    def __init__(self, json_path, hf_token="", google_api_key=""):
        self.json_path = json_path
        self.embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.index = None
        self.dimension = None
        self.hf_token = hf_token
        self.google_api_key = google_api_key
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
You are an expert assistant for Indian government schemes.

Context:
{context}

User Question:
{question}

Instructions:
1. Search the context for a scheme whose name or description exactly matches what the user asked for.
2. If you find an exact match, provide a detailed answer ONLY about that scheme, including:
   - **Scheme Name**
   - **Ministry/Department**
   - **Purpose**
   - **Eligibility**
   - **Benefits/Assistance**
   - **Application Process** (with steps if available)
   - **Official Website Link** (if found)
   - Use clear sections with bold headers (Markdown: **Header:**).
3. If you do NOT find an exact match:
   - Clearly state: "No exact match found for your query."
   - List the names of the most relevant or related schemes (if any), but DO NOT provide their details.
   - Example: "Related schemes: Scheme A, Scheme B, Scheme C."
4. Never provide details for unrelated or only partially matching schemes.
5. Be concise, clear, and use Markdown formatting for readability.

Remember: Only answer about the exact scheme if found. If not, just list related scheme names, no details.
"""
        answer = "Could not generate answer using any model."  # Default error message

        # Try Gemini Flash if Google API key is provided
        if self.google_api_key:
            try:
                genai.configure(api_key=self.google_api_key)
                model = genai.GenerativeModel("gemini-1.5-flash-latest")
                response = model.generate_content(prompt)
                answer = response.text
            except Exception as e:
                print(f"Error calling Gemini API: {e}")
                answer = f"Error: Could not connect to Gemini API - {e}"
        # Otherwise, try Hugging Face API
        elif self.hf_token:
            api_url = "https://api-inference.huggingface.co/models/bigscience/bloomz-560m"
            headers = {"Authorization": f"Bearer {self.hf_token}", "Content-Type": "application/json"}
            payload = {
                "inputs": prompt,
                "options": {
                    "wait_for_model": True,
                    "max_length": 1000,
                    "temperature": 0.3,
                    "top_p": 0.9,
                    "do_sample": True
                }
            }
            try:
                response = requests.post(api_url, headers=headers, json=payload)
                response.raise_for_status()
                output = response.json()
                if output and isinstance(output, list) and 'generated_text' in output[0]:
                    answer = output[0].get("generated_text", "No answer returned by model.")
                else:
                    answer = f"Unexpected response format from API: {output}"
            except requests.exceptions.RequestException as e:
                print(f"Error calling Hugging Face API: {e}")
                answer = f"Error: Could not connect to Hugging Face API - {e}"
            except Exception as e:
                print(f"Error processing Hugging Face response: {e}")
                answer = f"Error processing Hugging Face response: {e}"
        else:
            answer = "No API key provided for Gemini or Hugging Face."

        # Post-processing for better formatting
        headers = [
            "Scheme Name:", "Ministry/Department:", "Purpose:", "Eligibility:",
            "Benefits:", "Key Benefits:", "Application Process:", "Application Steps:",
            "Required Documents:", "Website Link:", "Official Link:", "Source:",
            "Overview:", "Relevant Schemes:", "Scheme Details:"
        ]
        for header in headers:
            answer = answer.replace(header, f"**{header}**")

        # Format application steps
        if "**Application Process:**" in answer or "**Application Steps:**" in answer:
            lines = answer.split('\n')
            formatted_lines = []
            in_steps = False
            step_count = 0
            for line in lines:
                if line.strip().startswith("**Application Process:**") or line.strip().startswith("**Application Steps:**"):
                    in_steps = True
                    formatted_lines.append(line)
                elif in_steps:
                    if re.match(r'^\s*\d+[\.\)]\s+', line.strip()):
                        step_count += 1
                        formatted_lines.append(f"\n{step_count}. {line.strip().split('.', 1)[1].strip()}")
                    elif line.strip().startswith('- '):
                        formatted_lines.append(f"\n• {line.strip()[2:]}")
                    elif line.strip().startswith('**'):
                        in_steps = False
                        formatted_lines.append(line)
                    elif line.strip() and not line.strip().startswith('**'):
                        formatted_lines.append(line)
                else:
                    formatted_lines.append(line)
            answer = '\n'.join(formatted_lines)

        # Make URLs clickable
        urls = re.findall(r'(https?://[^\s]+)', answer)
        for url in urls:
            if f"[{url}]({url})" not in answer:
                if f"**Website Link:** {url}" in answer:
                    answer = answer.replace(f"**Website Link:** {url}", f"**Website Link:** [{url}]({url})")
                elif f"**Official Link:** {url}" in answer:
                    answer = answer.replace(f"**Official Link:** {url}", f"**Official Link:** [{url}]({url})")
                else:
                    answer = answer.replace(url, f"[{url}]({url})")
        answer = re.sub(r'\n{3,}', '\n\n', answer)
        return answer.strip()
