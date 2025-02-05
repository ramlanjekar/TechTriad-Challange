from sentence_transformers import SentenceTransformer
import numpy as np
from PyPDF2 import PdfReader
from typing import List, Dict, Tuple, Optional, Union
import groq
from sklearn.metrics.pairwise import cosine_similarity
import os
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from tenacity import retry, stop_after_attempt, wait_exponential
import torch
import gc

@dataclass
class SearchResult:
    text: str
    similarity: float
    page_number: int
    source_file: str

class PDFProcessor:
    def __init__(
        self, 
        model_name: str = "all-MiniLM-L6-v2",
        cache_dir: Optional[str] = None,
        similarity_threshold: float = 0.3
    ):
        try:
            self.embedding_model = SentenceTransformer(model_name)
            self.groq_client = self._init_groq_client()
            self.cache_dir = Path(cache_dir) if cache_dir else None
            self.chunk_size = 400  # Fixed chunk size
            self.similarity_threshold = similarity_threshold

            self.logger = logging.getLogger(__name__) 
                
        except Exception as e:
            raise RuntimeError(f"Failed to initialize PDFProcessor: {str(e)}")

    def _init_groq_client(self) -> groq.Groq:
        """Initialize with Groq client."""
        # Get the API key from environment variables
        api_key = os.getenv(f"API_KEY_1")
        if not api_key:
            raise ValueError("GROQ_API_KEY not found in environment")
        return groq.Groq(api_key=api_key)


    def _validate_pdf(self, pdf_path: str) -> None:
        """Validate PDF file existence and readability."""
        path = Path(pdf_path)
        if not path.exists():
            raise FileNotFoundError(f"PDF file not found: {path}")
        if path.stat().st_size == 0:
            raise ValueError(f"PDF file is empty: {path}")
        if path.suffix.lower() != '.pdf':
            raise ValueError(f"Not a PDF file: {path}")

    def extract_text_from_pdfs(self, pdf_paths: List[str]) -> Dict[str, List[Tuple[int, str]]]:
        """Extract text from PDFs with page tracking and error handling."""
        results = {}

        for pdf_path in pdf_paths:
            self._validate_pdf(pdf_path)
            try:
                with open(pdf_path, 'rb') as file:
                    pdf = PdfReader(file)
                    texts = []
                    for i, page in enumerate(pdf.pages):
                        text = page.extract_text()
                        if text.strip():  # Only include non-empty pages
                            # Clean unnecessary newlines
                            cleaned_text = " ".join(line.strip() for line in text.splitlines())
                            texts.append((i + 1, cleaned_text))
                    if not texts:
                        self.logger.warning(f"No text extracted from {pdf_path}")
                    results[pdf_path] = texts
            except Exception as e:
                self.logger.error(f"Error processing {pdf_path}: {e}")

        if not results:
            raise ValueError("No text could be extracted from any PDF")
        return results

    def create_text_chunks(
    self, 
    pdf_texts: Dict[str, List[Tuple[int, str]]]
) -> List[Tuple[str, int, str]]:
        """Create chunks with source tracking and improved sentence handling."""
        chunks = []
        
        for pdf_path, pages in pdf_texts.items():
            for page_num, text in pages:
                sentences = [s.strip() + '.' for s in text.replace('\n', ' ').split('.') if s.strip()]
                
                if not sentences:
                    continue
                    
                current_chunk = []
                current_length = 0
                
                for sentence in sentences:
                    sentence_length = len(sentence.split())
                    
                    if current_length + sentence_length > self.chunk_size and current_chunk:
                        chunk_text = ' '.join(current_chunk)
                        chunks.append((chunk_text, page_num, pdf_path))
                        current_chunk = []
                        current_length = 0
                        
                    current_chunk.append(sentence)
                    current_length += sentence_length
                
                if current_chunk:
                    chunk_text = ' '.join(current_chunk)
                    chunks.append((chunk_text, page_num, pdf_path))
                    
        if not chunks:
            raise ValueError("No valid chunks created from input text")
            
        return chunks
    
    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def embed_texts(self, texts: List[str]) -> np.ndarray:
            """Create embeddings without using PyTorch or GPU."""
            try:
                # Generate embeddings using the embedding model
                embeddings = self.embedding_model.encode(
                    texts,
                    convert_to_tensor=False,  # Ensure output is not a PyTorch tensor
                    show_progress_bar=True,
                    batch_size=5
                )
                
                # Convert to numpy array if not already
                if not isinstance(embeddings, np.ndarray):
                    embeddings = np.array(embeddings)
                
                return embeddings
                
            except Exception as e:
                self.logger.error(f"Embedding creation failed: {str(e)}")
                raise

    def find_relevant_chunks(
        self, 
        query: str, 
        chunks: List[Tuple[str, int, str]], 
        top_k: int = 3
    ) -> List[SearchResult]:
        """Find relevant chunks with improved filtering and validation."""
        if not chunks:
            raise ValueError("No chunks provided for search")
        if not query.strip():
            raise ValueError("Empty query provided")
        if top_k < 1:
            raise ValueError("top_k must be positive")
            
        try:
            query_embedding = self.embedding_model.encode([query])[0]
            chunk_texts = [chunk[0] for chunk in chunks]
            chunk_embeddings = self.embed_texts(chunk_texts)
            
            similarities = cosine_similarity([query_embedding], chunk_embeddings)[0]
            
            # Filter by similarity threshold
            valid_indices = np.where(similarities >= self.similarity_threshold)[0]
            if len(valid_indices) == 0:
                self.logger.warning("No chunks met the similarity threshold")
                valid_indices = np.argsort(similarities)[-1:]  # Take best match anyway
                
            # Sort and limit results
            top_indices = valid_indices[np.argsort(similarities[valid_indices])[-min(top_k, len(valid_indices)):]][::-1]
            
            results = []
            for idx in top_indices:
                text, page_num, source_file = chunks[idx]
                results.append(SearchResult(
                    text=text,
                    similarity=similarities[idx],
                    page_number=page_num,
                    source_file=Path(source_file).name
                ))
            
            return results
            
        except Exception as e:
            self.logger.error(f"Error in similarity search: {str(e)}")
            raise

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=4, max=10))
    def get_llm_response(self, query: str, context: List[SearchResult]) -> str:
        """Get LLM response with improved prompt and error handling."""
        if not context:
            raise ValueError("No context provided for query")
            
        try:
            # Construct prompt with enhanced context
            context_text = "\n\n".join([
                f"Context {i+1} (Source: {result.source_file}, Page: {result.page_number}, "
                f"Relevance: {result.similarity:.2f}):\n{result.text}"
                for i, result in enumerate(context)
            ])
            
            prompt = f"""Answer the following question based on the provided context.


            Context:
            {context_text}

            Question: {query}

            Answer: """

            completion = self.groq_client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a helpful assistant focused on accurate information provider given the context from documents."},
                    {"role": "user", "content": prompt}
                ],
                temperature=1.0,
                max_tokens=1000,
                top_p=1,
                stream=False
            )
            
            return completion.choices[0].message.content
            
        except Exception as e:
            self.logger.error(f"Error getting LLM response: {str(e)}")
            raise

    def process_query(
        self, 
        query: str, 
        pdf_paths: List[str], 
        top_k: int = 3
    ) -> Dict:
        """Process query with comprehensive error handling and validation."""
        try:
            # Extract and process text
            self.logger.info("Extracting text from PDFs...")
            pdf_texts = self.extract_text_from_pdfs(pdf_paths)
            
            self.logger.info("Creating text chunks...")
            chunks = self.create_text_chunks(pdf_texts)
            
            self.logger.info(f"Processing query: {query}")
            relevant_chunks = self.find_relevant_chunks(query, chunks, top_k)
            
            if not relevant_chunks:
                return {
                    "query": query,
                    "response": "No relevant content found in the provided documents.",
                    "relevant_chunks": []
                }
            
            response = self.get_llm_response(query, relevant_chunks)
            
            return {
                "query": query,
                "response": response,
                "relevant_chunks": [
                    {
                        "text": chunk.text,
                        "similarity": float(chunk.similarity),
                        "page_number": chunk.page_number,
                        "source_file": chunk.source_file
                    }
                    for chunk in relevant_chunks
                ]
            }
            
        except Exception as e:
            self.logger.error(f"Query processing failed: {str(e)}")
            raise

def talk(filepath: str, question: str) -> str:
    """Process the query and return the response."""
    try:
        processor = PDFProcessor(
            cache_dir="./cache",
            similarity_threshold=0.3
        )
        
        pdf_paths = [filepath]
        result = processor.process_query(question, pdf_paths)
        
        return result['response']
        
    except Exception as e:
        logging.error(f"Processing failed: {str(e)}")
        raise
