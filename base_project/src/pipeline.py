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
from logging_utility import logging_util

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
        logging_util.log_stage("DATA_INGESTION", "Starting data ingestion")
        start_time = time.time()
        
        # Load documents from data directory
        documents = self.data_ingestion.load_documents()
        
        elapsed_time = time.time() - start_time
        logger.info(f"Data ingestion completed in {elapsed_time:.2f} seconds. Ingested {len(documents)} documents.")
        logging_util.log_stage("DATA_INGESTION", f"Completed in {elapsed_time:.2f}s. Ingested {len(documents)} documents")
        logging_util.log_metric("num_documents", len(documents))
        logging_util.log_metric("ingestion_time", elapsed_time)
        
        return documents
    
    def run_processing(self, documents: List[Dict[str, Any]]) -> None:
        """
        Run the document processing step
        
        Args:
            documents: List of documents to process
        """
        logger.info("Starting document processing step")
        logging_util.log_stage("DOCUMENT_PROCESSING", "Starting document processing")
        start_time = time.time()
        
        # Process documents
        chunks, embeddings = self.document_processor.process_documents(documents)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Document processing completed in {elapsed_time:.2f} seconds. "
                   f"Created {len(chunks)} chunks and {len(embeddings)} embeddings.")
        logging_util.log_stage("DOCUMENT_PROCESSING", f"Completed in {elapsed_time:.2f}s. Created {len(chunks)} chunks")
        logging_util.log_metric("num_chunks", len(chunks))
        logging_util.log_metric("num_embeddings", len(embeddings))
        logging_util.log_metric("processing_time", elapsed_time)
    
    def initialize_query_engine(self) -> None:
        """Initialize the query engine after documents have been processed"""
        logger.info("Initializing query engine")
        logging_util.log_stage("QUERY_ENGINE", "Initializing query engine")
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
        logging_util.log_stage("QUERY", f"Processing query: '{query}'")
        start_time = time.time()
        
        # Process query
        response = self.query_engine.process_query(query)
        
        elapsed_time = time.time() - start_time
        logger.info(f"Query processing completed in {elapsed_time:.2f} seconds")
        logging_util.log_stage("QUERY", f"Query processing completed in {elapsed_time:.2f}s")
        logging_util.log_metric("query_time", elapsed_time)
        logging_util.log_metric("response_length", len(response) if response else 0)
        
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
        # Start tracking the run
        logging_util.start_run(query)
        
        start_time = time.time()
        
        try:
            # Check if we need to run the ingestion and processing steps
            reset_collection = config.get("vector_store.reset_collection", False)
            
            # Check for existing data in the collection
            skip_processing = False
            if not reset_collection:
                try:
                    from vector_store import vector_store
                    # This is a simple check to see if collection exists and has data
                    collection_name = vector_store.collection_name
                    try:
                        collection_info = vector_store.client.get_collection(collection_name=collection_name)
                        # Check if vectors_count exists and has a value greater than 0
                        vectors_count = getattr(collection_info, 'vectors_count', None)
                        has_data = vectors_count is not None and vectors_count > 0
                        
                        if has_data:
                            logger.info(f"Collection {collection_name} exists with {vectors_count} vectors. Skipping document processing.")
                            skip_processing = True
                        else:
                            logger.info(f"Collection {collection_name} exists but has no vectors. Will process documents.")
                    except Exception as e:
                        # If the collection doesn't exist yet, it will throw an error
                        logger.info(f"Collection {collection_name} not found. Will create and process documents.")
                except Exception as e:
                    logger.info(f"Could not check collection status: {str(e)}. Will process documents.")
            
            if not skip_processing:
                # Run ingestion
                documents = self.run_ingestion()
                
                # Run processing
                self.run_processing(documents)
            else:
                logger.info("Using existing collection. Skipping document ingestion and processing.")
                logging_util.log_stage("PROCESSING", "Skipped - using existing collection")
                
            # Initialize query engine
            self.initialize_query_engine()
            
            # Run query
            response = self.run_query(query)
            logger.info(f"Query response generated")
            
            elapsed_time = time.time() - start_time
            logger.info(f"Full pipeline completed in {elapsed_time:.2f} seconds")
            logging_util.log_metric("total_time", elapsed_time)
            
            # End tracking the run
            logging_util.end_run("completed")
            
            return response
        except Exception as e:
            # Log error and end tracking
            logger.error(f"Pipeline failed: {str(e)}")
            logging_util.end_run("failed", str(e))
            raise

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