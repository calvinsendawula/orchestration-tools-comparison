"""
Main pipeline module for simple RAG pipeline
"""
import os
import sys
import time
import logging
from typing import List, Dict, Any, Optional

# Add root directory to path so we can import shared modules
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config_loader import config

# Import pipeline components
from data_ingestion import DataIngestion
from document_processor import DocumentProcessor
from query_engine import QueryEngine

# Set up logging
log_level = getattr(logging, config.get("log_level", "INFO"))
logging.basicConfig(level=log_level, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RagPipeline:
    """Main pipeline class that orchestrates the entire RAG workflow"""
    
    def __init__(self):
        """Initialize the RAG pipeline using configuration"""
        # Get directory paths from config
        self.raw_dir = config.get_raw_data_directory()
        self.processed_dir = config.get_processed_directory()
        
        # Log configuration information
        logger.info(f"Active tool: {config.get_active_tool()}")
        logger.info(f"Embedding provider: {config.get_embedding_provider()}")
        logger.info(f"Inference provider: {config.get('inference.provider', 'local')}")
        logger.info(f"Raw data directory: {os.path.abspath(self.raw_dir)}")
        logger.info(f"Processed data directory: {os.path.abspath(self.processed_dir)}")
        
        # Initialize pipeline components
        self.data_ingestion = DataIngestion()
        self.document_processor = DocumentProcessor()
        self.query_engine = None  # Will be initialized after processing docs
    
    def run_ingestion(self) -> List[Dict[str, Any]]:
        """
        Run the data ingestion step
        
        Returns:
            List of ingested documents
        """
        logger.info("Starting data ingestion step")
        start_time = time.time()
        
        # Load documents from data directory
        documents = self.data_ingestion.load_documents()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data ingestion completed in {elapsed_time:.2f} seconds. Ingested {len(documents)} documents.")
        
        return documents
    
    def run_processing(self, documents: List[Dict[str, Any]]) -> None:
        """
        Run the document processing step
        
        Args:
            documents: List of documents to process
        """
        logger.info("Starting document processing step")
        start_time = time.time()
        
        # Process documents
        chunks, embeddings = self.document_processor.process_documents(documents)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document processing completed in {elapsed_time:.2f} seconds. "
                   f"Created {len(chunks)} chunks and {len(embeddings)} embeddings.")
    
    def initialize_query_engine(self) -> None:
        """Initialize the query engine after documents have been processed"""
        logger.info("Initializing query engine")
        self.query_engine = QueryEngine()
    
    def run_query(self, query: str) -> str:
        """
        Run a query through the query engine
        
        Args:
            query: User query string
            
        Returns:
            Response string
        """
        if not self.query_engine:
            logger.warning("Query engine not initialized. Initializing now.")
            self.initialize_query_engine()
        
        logger.info(f"Processing query: '{query}'")
        start_time = time.time()
        
        # Process query
        response = self.query_engine.process_query(query)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Query processing completed in {elapsed_time:.2f} seconds")
        
        return response
    
    def run_full_pipeline(self, query: str) -> str:
        """
        Run the full RAG pipeline
        
        Args:
            query: Query to process
            
        Returns:
            Response to the query
        """
        logger.info("Starting full RAG pipeline run")
        start_time = time.time()
        
        # Run ingestion
        documents = self.run_ingestion()
        
        # Run processing
        self.run_processing(documents)
        
        # Initialize query engine
        self.initialize_query_engine()
        
        # Run query
        response = self.run_query(query)
        logger.info(f"Query response generated")
        
        elapsed_time = time.time() - start_time
        logger.info(f"Full pipeline completed in {elapsed_time:.2f} seconds")
        
        return response

if __name__ == "__main__":
    # Run the full pipeline with default settings
    pipeline = RagPipeline()
    
    # Get query from command line argument or use default
    query = "What are good sources of protein?" if len(sys.argv) <= 1 else sys.argv[1]
    
    # Run full pipeline
    response = pipeline.run_full_pipeline(query)
    
    print("\n" + "="*50)
    print(f"Tool: {config.get_active_tool()}")
    print(f"Embedding: {config.get_embedding_provider()}")
    print(f"Inference: {config.get('inference.provider', 'local')}")
    print("-"*50)
    print(f"Query: {query}")
    print("-"*50)
    print(f"Response: {response}")
    print("="*50) 