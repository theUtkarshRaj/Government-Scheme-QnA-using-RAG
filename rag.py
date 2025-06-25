import json
import numpy as np
import faiss
import requests
from sentence_transformers import SentenceTransformer
import re
from typing import List, Dict, Tuple, Optional

class ImprovedGovernmentSchemeRAG:
    def __init__(self, json_path: str, hf_token: str = "", chunk_size: int = 500, chunk_overlap: int = 50):
        self.json_path = json_path
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        
        # Use a better embedding model
        self.embedding_model = SentenceTransformer("all-mpnet-base-v2")
        self.index = None
        self.dimension = None
        self.hf_token = hf_token
        
        # Load and process data
        self.schemes_data = self.load_schemes_data()
        self.chunks, self.metadata = self.create_chunks()
        
        if not self.chunks:
            raise ValueError("No chunks available to create embeddings.")
        
        self.create_index()
    
    def load_schemes_data(self) -> List[Dict]:
        """Load schemes data with robust error handling"""
        try:
            with open(self.json_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            print(f"Successfully loaded {len(data)} schemes")
            return data
        except FileNotFoundError:
            raise FileNotFoundError(f"JSON file not found at {self.json_path}")
        except json.JSONDecodeError as e:
            raise ValueError(f"Invalid JSON format: {e}")
        except Exception as e:
            raise Exception(f"Error loading JSON: {e}")
    
    def create_chunks(self) -> Tuple[List[str], List[Dict]]:
        """Create focused, smaller chunks for better retrieval"""
        chunks = []
        metadata = []
        
        for scheme in self.schemes_data:
            data = scheme.get("data", {})
            scheme_name = data.get("scheme_name", "Unknown Scheme")
            ministry = data.get("ministry", "Unknown Ministry")
            department = data.get("department", "Unknown Department")
            
            base_metadata = {
                "scheme_name": scheme_name,
                "ministry": ministry,
                "department": department
            }
            
            # Create separate chunks for different content types
            content_types = {
                "overview": f"Scheme: {scheme_name}\nMinistry: {ministry}\nDepartment: {department}",
                "details": data.get("details_content", []),
                "eligibility": data.get("eligibility_content", []),
                "application_process": data.get("application_process", [])
            }
            
            for content_type, content in content_types.items():
                if content_type == "overview":
                    chunks.append(content)
                    metadata.append({**base_metadata, "content_type": content_type})
                else:
                    if isinstance(content, list) and content:
                        content_text = "\n".join([str(item) for item in content if item])
                        if content_text.strip():
                            # Create smaller chunks if content is too long
                            text_chunks = self.split_text(
                                f"Scheme: {scheme_name}\n{content_type.replace('_', ' ').title()}:\n{content_text}",
                                self.chunk_size,
                                self.chunk_overlap
                            )
                            
                            for chunk in text_chunks:
                                chunks.append(chunk)
                                metadata.append({**base_metadata, "content_type": content_type})
        
        print(f"Created {len(chunks)} chunks from {len(self.schemes_data)} schemes")
        return chunks, metadata
    
    def split_text(self, text: str, chunk_size: int, overlap: int) -> List[str]:
        """Split long text into overlapping chunks"""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        
        while start < len(text):
            end = start + chunk_size
            
            if end >= len(text):
                chunks.append(text[start:])
                break
            
            # Try to break at sentence boundary
            chunk = text[start:end]
            last_sentence = chunk.rfind('.')
            last_newline = chunk.rfind('\n')
            
            if last_sentence > start + chunk_size // 2:
                end = start + last_sentence + 1
            elif last_newline > start + chunk_size // 2:
                end = start + last_newline + 1
            
            chunks.append(text[start:end])
            start = end - overlap
        
        return chunks
    
    def create_index(self):
        """Create FAISS index with better error handling"""
        if not self.chunks:
            raise ValueError("No chunks available for indexing")
        
        print("Creating embeddings...")
        embeddings = []
        
        # Process in batches to avoid memory issues
        batch_size = 32
        for i in range(0, len(self.chunks), batch_size):
            batch = self.chunks[i:i + batch_size]
            batch_embeddings = self.embedding_model.encode(batch, convert_to_numpy=True)
            embeddings.extend(batch_embeddings)
        
        embeddings = np.array(embeddings).astype('float32')
        
        if embeddings.ndim != 2 or embeddings.shape[0] == 0:
            raise ValueError("Invalid embeddings shape")
        
        self.dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(self.dimension)
        self.index.add(embeddings)
        
        print(f"FAISS index created with {self.index.ntotal} vectors of dimension {self.dimension}")
    
    def query(self, question: str, top_k: int = 5, similarity_threshold: float = 0.7) -> List[Dict]:
        """Enhanced query with similarity filtering"""
        if not self.index or self.index.ntotal == 0:
            return []
        
        question_embedding = self.embedding_model.encode([question]).astype('float32')
        distances, indices = self.index.search(question_embedding, min(top_k * 2, self.index.ntotal))
        
        results = []
        for distance, idx in zip(distances[0], indices[0]):
            if 0 <= idx < len(self.chunks):
                # Convert L2 distance to similarity (0-1 scale)
                similarity = 1 / (1 + distance)
                
                if similarity >= similarity_threshold:
                    results.append({
                        "chunk": self.chunks[idx],
                        "metadata": self.metadata[idx],
                        "similarity": similarity,
                        "distance": distance
                    })
        
        # Sort by similarity and return top_k
        results.sort(key=lambda x: x['similarity'], reverse=True)
        return results[:top_k]
    
    def prepare_context(self, results: List[Dict], max_length: int = 2000) -> str:
        """Prepare context with length management"""
        if not results:
            return "No relevant information found."
        
        context_parts = []
        current_length = 0
        
        for result in results:
            chunk = result['chunk']
            estimated_length = len(chunk)
            
            if current_length + estimated_length > max_length:
                # Truncate the chunk if we're close to limit
                remaining_space = max_length - current_length
                if remaining_space > 100:  # Only add if meaningful space left
                    chunk = chunk[:remaining_space] + "..."
                    context_parts.append(f"[Similarity: {result['similarity']:.2f}]\n{chunk}")
                break
            
            context_parts.append(f"[Similarity: {result['similarity']:.2f}]\n{chunk}")
            current_length += estimated_length
        
        return "\n\n---\n\n".join(context_parts)
    
    def generate_answer(self, question: str, context: str) -> str:
        """Enhanced answer generation with better prompting"""
        if not context or context == "No relevant information found.":
            return "I couldn't find relevant information about your query in the government schemes database."
        
        prompt = f"""You are an expert assistant helping users understand Indian government schemes.

Context Information:
{context}

User Question: {question}

Instructions:
1. Answer based ONLY on the provided context
2. If the context doesn't contain sufficient information, clearly state this
3. Structure your response with clear sections using **bold** headers
4. Include relevant details like eligibility, benefits, and application process
5. Be accurate and helpful

Response:"""

        if not self.hf_token:
            return self.generate_fallback_answer(question, context)
        
        try:
            # Use a better model for generation
            api_url = "https://api-inference.huggingface.co/models/google/flan-t5-large"
            headers = {
                "Authorization": f"Bearer {self.hf_token}",
                "Content-Type": "application/json"
            }
            
            payload = {
                "inputs": prompt,
                "parameters": {
                    "max_new_tokens": 500,
                    "temperature": 0.1,
                    "do_sample": True,
                    "top_p": 0.9
                },
                "options": {"wait_for_model": True}
            }
            
            response = requests.post(api_url, headers=headers, json=payload, timeout=30)
            response.raise_for_status()
            
            result = response.json()
            if isinstance(result, list) and len(result) > 0:
                generated_text = result[0].get("generated_text", "")
                if generated_text:
                    return self.format_answer(generated_text)
            
            return "Unable to generate a proper response from the AI model."
            
        except requests.exceptions.RequestException as e:
            print(f"API Error: {e}")
            return f"Error connecting to AI service: {str(e)}"
        except Exception as e:
            print(f"Generation Error: {e}")
            return self.generate_fallback_answer(question, context)
    
    def generate_fallback_answer(self, question: str, context: str) -> str:
        """Generate a simple fallback answer when AI service is unavailable"""
        return f"""**Information Found:**

{context}

**Note:** AI-powered answer generation is currently unavailable. The above information from the government schemes database may help answer your question: "{question}"

Please review the relevant sections marked with similarity scores."""
    
    def format_answer(self, answer: str) -> str:
        """Format the generated answer with proper styling"""
        # Clean up the answer
        answer = answer.strip()
        
        # Add bold formatting to common section headers
        formatting_rules = {
            "Scheme Name:": "**Scheme Name:**",
            "Ministry:": "**Ministry:**",
            "Department:": "**Department:**",
            "Purpose:": "**Purpose:**",
            "Eligibility:": "**Eligibility:**",
            "Benefits:": "**Benefits:**",
            "Key Benefits:": "**Key Benefits:**",
            "Application Process:": "**Application Process:**",
            "Required Documents:": "**Required Documents:**",
            "Website:": "**Website:**",
            "Contact:": "**Contact:**"
        }
        
        for old, new in formatting_rules.items():
            answer = answer.replace(old, new)
        
        # Make URLs clickable
        url_pattern = r'(https?://[^\s\)]+)'
        urls = re.findall(url_pattern, answer)
        for url in urls:
            if f"]({url})" not in answer:  # Don't double-format
                answer = answer.replace(url, f"[{url}]({url})")
        
        return answer
    
    def get_scheme_stats(self) -> Dict:
        """Get statistics about the loaded schemes"""
        if not self.schemes_data:
            return {}
        
        ministries = set()
        departments = set()
        
        for scheme in self.schemes_data:
            data = scheme.get("data", {})
            ministries.add(data.get("ministry", "Unknown"))
            departments.add(data.get("department", "Unknown"))
        
        return {
            "total_schemes": len(self.schemes_data),
            "total_chunks": len(self.chunks) if hasattr(self, 'chunks') else 0,
            "unique_ministries": len(ministries),
            "unique_departments": len(departments),
            "embedding_dimension": self.dimension
        }
