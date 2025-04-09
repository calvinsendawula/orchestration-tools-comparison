"""
Query engine component for simple RAG pipeline
"""
import os
import json
import logging
import numpy as np
import sys
from typing import List, Dict, Any, Optional, Tuple

# Add root directory to path so we can import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config_loader import config
from ai_providers import embedding_provider, inference_provider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryEngine:
    """Handles retrieval and response generation for the RAG pipeline"""
    
    def __init__(self):
        """Initialize the query engine using configuration"""
        # Get processed directory from config
        self.processed_dir = config.get_processed_directory()
        
        # Load processed data
        self.chunks, self.embeddings = self._load_data()
        
        logger.info(f"Initialized query engine with {len(self.chunks)} chunks and {len(self.embeddings)} embeddings")
    
    def _load_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Load chunks and embeddings from disk
        
        Returns:
            Tuple of (chunked documents, embeddings dictionary)
        """
        chunks = []
        embeddings = {}
        
        # Load chunks
        chunks_file = os.path.join(self.processed_dir, "chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        else:
            logger.warning(f"Chunks file not found at {chunks_file}")
        
        # Load embeddings
        embeddings_file = os.path.join(self.processed_dir, "embeddings.json")
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
        else:
            logger.warning(f"Embeddings file not found at {embeddings_file}")
        
        logger.info(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings from {self.processed_dir}")
        return chunks, embeddings
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        if not self.chunks or not self.embeddings:
            logger.warning("No chunks or embeddings available for retrieval")
            return []
        
        # Embed the query using the same provider used for documents
        query_embedding = embedding_provider.embed_text(query)
        
        # Calculate similarity for each chunk
        chunk_similarities = []
        for chunk in self.chunks:
            embedding_id = chunk["metadata"].get("embedding_id")
            if embedding_id and embedding_id in self.embeddings:
                doc_embedding = self.embeddings[embedding_id]
                
                # Calculate cosine similarity
                query_vec = np.array(query_embedding)
                doc_vec = np.array(doc_embedding)
                similarity = np.dot(query_vec, doc_vec) / (np.linalg.norm(query_vec) * np.linalg.norm(doc_vec))
                
                # Add chunk with similarity score
                chunk_with_score = {
                    "chunk": chunk,
                    "similarity": float(similarity)
                }
                chunk_similarities.append(chunk_with_score)
        
        # Sort by similarity (highest first) and take top k
        relevant_chunks = sorted(chunk_similarities, key=lambda x: x["similarity"], reverse=True)[:top_k]
        
        logger.info(f"Retrieved {len(relevant_chunks)} relevant chunks for query: {query}")
        return relevant_chunks
    
    def generate_response(self, query: str, relevant_chunks: List[Dict[str, Any]]) -> str:
        """
        Generate a response based on the query and relevant chunks
        
        Args:
            query: User query
            relevant_chunks: List of relevant chunks with similarity scores
            
        Returns:
            Generated response string
        """
        if not relevant_chunks:
            return "I couldn't find any relevant information to answer your question."
        
        # Extract just the chunk content for the inference provider
        context_for_inference = [
            {
                "content": item["chunk"]["content"],
                "metadata": item["chunk"]["metadata"]
            }
            for item in relevant_chunks
        ]
        
        # Use the inference provider to generate a response
        response = inference_provider.generate(query, context_for_inference)
        
        return response
    
    def process_query(self, query: str, top_k: int = 3) -> str:
        """
        Process a query through the full RAG pipeline
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            Generated response string
        """
        # Retrieve relevant chunks
        relevant_chunks = self.retrieve_relevant_chunks(query, top_k=top_k)
        
        # Generate response
        response = self.generate_response(query, relevant_chunks)
        
        return response

if __name__ == "__main__":
    # Test the query engine
    query_engine = QueryEngine()
    
    # Try a sample query
    query = "What are the health benefits of proteins?"
    response = query_engine.process_query(query)
    
    print(f"Query: {query}")
    print(f"Response: {response}") 