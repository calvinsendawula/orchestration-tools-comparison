"""
Data ingestion component for simple RAG pipeline
"""
import os
import json
import logging
import requests
import sys
from typing import List, Dict, Any, Optional

# Add root directory to path so we can import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config_loader import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataIngestion:
    """Handles loading and preparing documents for the RAG pipeline"""
    
    def __init__(self):
        """
        Initialize the data ingestion component using configuration
        """
        self.data_dir = config.get_raw_data_directory()
        if not os.path.exists(self.data_dir):
            raise ValueError(f"Data directory {self.data_dir} does not exist")
        logger.info(f"Using raw data from {os.path.abspath(self.data_dir)}")
    
    def load_documents(self) -> List[Dict[str, Any]]:
        """
        Load all documents from the data directory
        
        Returns:
            List of document objects with content and metadata
        """
        documents = []
        
        # Get all .txt files in the data directory
        txt_files = [f for f in os.listdir(self.data_dir) if f.endswith('.txt')]
        
        if not txt_files:
            raise ValueError(f"No .txt files found in {self.data_dir}")
        
        for filename in txt_files:
            try:
                filepath = os.path.join(self.data_dir, filename)
                with open(filepath, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Create document object with content and metadata
                document = {
                    "content": content,
                    "metadata": {
                        "source": filepath,
                        "filename": filename,
                        "size_bytes": os.path.getsize(filepath),
                        "created_at": os.path.getctime(filepath),
                        "embedding_provider": config.get_embedding_provider(),
                        "tool": config.get_active_tool()
                    }
                }
                
                documents.append(document)
                logger.info(f"Loaded document: {filename}")
                
            except Exception as e:
                logger.error(f"Error loading document {filename}: {str(e)}")
        
        logger.info(f"Loaded {len(documents)} documents")
        return documents

if __name__ == "__main__":
    # Simple test to ensure the module works
    ingestion = DataIngestion()
    
    # Load documents
    documents = ingestion.load_documents()
    
    # Print what we got
    for doc in documents:
        print(f"Document: {doc['metadata']['filename']}")
        print(f"Content: {doc['content'][:100]}...")
        print("---") 