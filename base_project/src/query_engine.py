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
from vector_store import vector_store

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QueryEngine:
    """Handles retrieval and response generation for the RAG pipeline"""
    
    def __init__(self):
        """Initialize the query engine using configuration"""
        # Get processed directory from config
        self.processed_dir = config.get_processed_directory()
        
        # Vector store now handles loading data as needed
        logger.info(f"Initialized query engine with {vector_store.provider_name} vector store")
    
    def retrieve_relevant_chunks(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Retrieve most relevant chunks for a query
        
        Args:
            query: User query
            top_k: Number of top chunks to retrieve
            
        Returns:
            List of relevant chunks with similarity scores
        """
        # Embed the query using the same provider used for documents
        query_embedding = embedding_provider.embed_text(query)
        
        # Search using the vector store's interface
        # With our updated vector_store implementation, the interface is now consistent
        # for both local and Qdrant backends
        return vector_store.search_embeddings(query_embedding, top_k=top_k)
    
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