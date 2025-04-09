"""
Document processing component for simple RAG pipeline
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
from ai_providers import embedding_provider

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DocumentProcessor:
    """Handles chunking and embedding documents for the RAG pipeline"""
    
    def __init__(self):
        """Initialize the document processor using configuration"""
        # Get processed directory from config
        self.output_dir = config.get_processed_directory()
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Get document processing settings
        doc_config = config.get_document_processing_config()
        self.chunk_size = doc_config.get("chunk_size", 512)
        self.chunk_overlap = doc_config.get("chunk_overlap", 50)
        
        logger.info(f"Initialized document processor with output directory: {self.output_dir}")
        logger.info(f"Using chunk size: {self.chunk_size}, overlap: {self.chunk_overlap}")
    
    def chunk_document(self, document: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Split document into smaller chunks with possible overlap
        
        Args:
            document: Document object containing content and metadata
            
        Returns:
            List of chunk objects with content and metadata
        """
        content = document["content"]
        chunks = []
        
        # Simple character-based chunking for demonstration
        # In a real implementation, you'd use more sophisticated chunking
        start = 0
        while start < len(content):
            end = min(start + self.chunk_size, len(content))
            
            # If not the beginning, include overlap
            if start > 0:
                start = max(0, start - self.chunk_overlap)
                
            chunk_content = content[start:end]
            
            # Create chunk with content and metadata
            chunk = {
                "content": chunk_content,
                "metadata": {
                    **document["metadata"],
                    "chunk_id": len(chunks),
                    "char_start": start,
                    "char_end": end
                }
            }
            
            chunks.append(chunk)
            
            # Move to start position for next chunk
            start = end
        
        logger.info(f"Split document into {len(chunks)} chunks")
        return chunks
    
    def process_documents(self, documents: List[Dict[str, Any]]) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Process documents: chunk and embed
        
        Args:
            documents: List of document objects to process
            
        Returns:
            Tuple of (chunked documents, embeddings dictionary)
        """
        all_chunks = []
        embeddings = {}
        
        for document in documents:
            # Chunk the document
            chunks = self.chunk_document(document)
            all_chunks.extend(chunks)
            
            # Generate embeddings for each chunk
            for chunk in chunks:
                chunk_id = f"{chunk['metadata']['filename']}_{chunk['metadata']['chunk_id']}"
                
                # Use our embedding provider
                embedding = embedding_provider.embed_text(chunk["content"])
                embeddings[chunk_id] = embedding
                
                # Add the embedding reference to the chunk metadata
                chunk["metadata"]["embedding_id"] = chunk_id
        
        # Save chunks and embeddings
        self._save_processed_data(all_chunks, embeddings)
        
        return all_chunks, embeddings
    
    def _save_processed_data(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[float]]) -> None:
        """
        Save processed chunks and embeddings to disk
        
        Args:
            chunks: List of processed document chunks
            embeddings: Dictionary mapping chunk IDs to embeddings
        """
        # Save chunks
        chunks_file = os.path.join(self.output_dir, "chunks.json")
        with open(chunks_file, 'w', encoding='utf-8') as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
        
        # Save embeddings
        embeddings_file = os.path.join(self.output_dir, "embeddings.json")
        with open(embeddings_file, 'w', encoding='utf-8') as f:
            json.dump(embeddings, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved {len(chunks)} chunks and {len(embeddings)} embeddings to {self.output_dir}")
    
    def load_processed_data(self) -> Tuple[List[Dict[str, Any]], Dict[str, List[float]]]:
        """
        Load previously processed chunks and embeddings from disk
        
        Returns:
            Tuple of (chunked documents, embeddings dictionary)
        """
        chunks = []
        embeddings = {}
        
        # Load chunks
        chunks_file = os.path.join(self.output_dir, "chunks.json")
        if os.path.exists(chunks_file):
            with open(chunks_file, 'r', encoding='utf-8') as f:
                chunks = json.load(f)
        else:
            logger.warning(f"Chunks file not found at {chunks_file}")
        
        # Load embeddings
        embeddings_file = os.path.join(self.output_dir, "embeddings.json")
        if os.path.exists(embeddings_file):
            with open(embeddings_file, 'r', encoding='utf-8') as f:
                embeddings = json.load(f)
        else:
            logger.warning(f"Embeddings file not found at {embeddings_file}")
        
        logger.info(f"Loaded {len(chunks)} chunks and {len(embeddings)} embeddings from {self.output_dir}")
        return chunks, embeddings

if __name__ == "__main__":
    # Test the document processor with some sample data
    from data_ingestion import DataIngestion
    
    # Create and load some sample documents
    ingestion = DataIngestion()
    documents = ingestion.load_documents()
    
    # Process the documents
    processor = DocumentProcessor()
    chunks, embeddings = processor.process_documents(documents)
    
    # Print what we processed
    print(f"Processed {len(chunks)} chunks with {len(embeddings)} embeddings")
    if chunks:
        print(f"First chunk: {chunks[0]['content'][:100]}...")
    if embeddings:
        first_key = next(iter(embeddings.keys()))
        print(f"First embedding shape: {len(embeddings[first_key])}") 