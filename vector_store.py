"""
Vector store module for handling different vector storage providers.
"""

import os
import json
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Tuple
from config_loader import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import Qdrant
try:
    from qdrant_client import QdrantClient
    from qdrant_client.http import models
    from qdrant_client.http.exceptions import UnexpectedResponse
    HAVE_QDRANT = True
except ImportError:
    HAVE_QDRANT = False
    logger.warning("Qdrant client not installed. Qdrant storage will not be available.")
    logger.warning("To install, use: pip install qdrant-client")

class VectorStore:
    """Base class for vector storage providers"""
    
    def __init__(self):
        """Initialize the vector store provider based on configuration"""
        self.provider_name = config.get("vector_store.provider", "qdrant")
        self.reset_collection = config.get("vector_store.reset_collection", False)
        
        # Get dimension from config
        self.embedding_dim = config.get("embedding.dimension", 768)
        
        # Initialize Qdrant
        if self.provider_name == "qdrant" and HAVE_QDRANT:
            self._init_qdrant()
        else:
            raise ValueError("Only Qdrant is supported as vector store provider. Please install qdrant-client.")
    
    def _init_qdrant(self):
        """Initialize Qdrant vector store provider"""
        # Get configuration
        self.qdrant_url = os.getenv("QDRANT_URL")
        self.qdrant_api_key = os.getenv("QDRANT_API_KEY")
        
        if not self.qdrant_url or not self.qdrant_api_key:
            raise ValueError("Qdrant URL or API key not set in environment. Please set QDRANT_URL and QDRANT_API_KEY in .env file.")
        
        # Get collection name from config and format appropriately
        format_string = config.get("vector_store.collection_name_format", "{tool}_{provider}")
        active_tool = config.get_active_tool()
        embedding_provider = config.get_embedding_provider()
        self.collection_name = format_string.format(tool=active_tool, provider=embedding_provider)
        
        # Default collection name if formatting fails
        if not self.collection_name or self.collection_name == format_string:
            self.collection_name = config.get("qdrant.default_collection_name", "nutrition_knowledge")
        
        # Initialize Qdrant client with proper error handling
        logger.info(f"Connecting to Qdrant at: {self.qdrant_url}")
        self.client = QdrantClient(
            url=self.qdrant_url,
            api_key=self.qdrant_api_key,
            timeout=60  # Increased timeout for better reliability
        )
        
        # Set up collection
        self._setup_collection()
        
        # Set functions
        self.store_embeddings = self._qdrant_store_embeddings
        self.search_embeddings = self._qdrant_search_embeddings
        self.load_embeddings = self._qdrant_load_embeddings
        
        logger.info(f"Initialized Qdrant vector store with collection: {self.collection_name}")
    
    def _setup_collection(self):
        """Set up Qdrant collection, creating or resetting as needed"""
        try:
            # Check if collection exists
            collections = self.client.get_collections().collections
            collection_exists = any(collection.name == self.collection_name for collection in collections)
            
            # Handle collection based on existence and reset flag
            if collection_exists:
                if self.reset_collection:
                    logger.info(f"Resetting existing collection: {self.collection_name}")
                    self.client.delete_collection(collection_name=self.collection_name)
                    self._create_collection()
                else:
                    logger.info(f"Using existing collection: {self.collection_name}")
                    # Verify collection has the correct settings
                    try:
                        collection_info = self.client.get_collection(collection_name=self.collection_name)
                        logger.info(f"Collection info: {collection_info.name}, vectors: {getattr(collection_info, 'vectors_count', 'unknown')}")
                    except Exception as e:
                        logger.warning(f"Could not get collection details: {e}")
            else:
                logger.info(f"Collection {self.collection_name} does not exist, creating it")
                self._create_collection()
                
        except Exception as e:
            logger.error(f"Error setting up collection: {e}")
            raise
    
    def _create_collection(self):
        """Create a new Qdrant collection"""
        try:
            # Get Qdrant settings from config
            vector_size = config.get("qdrant.vector_dimension", self.embedding_dim)
            on_disk = config.get("qdrant.on_disk", True)
            
            # Create the collection with explicit Cosine distance
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=models.VectorParams(
                    size=vector_size,
                    distance=models.Distance.COSINE,
                ),
                on_disk_payload=on_disk,
            )
            logger.info(f"Created collection {self.collection_name} with dimension {vector_size}")
        except Exception as e:
            logger.error(f"Error creating collection: {e}")
            raise
    
    def _qdrant_store_embeddings(self, chunks: List[Dict[str, Any]], embeddings: Dict[str, List[float]]) -> bool:
        """
        Store embeddings in Qdrant
        
        Args:
            chunks: List of document chunks
            embeddings: Dictionary mapping chunk IDs to embeddings
            
        Returns:
            True if successful, False otherwise
        """
        try:
            # Prepare points for batch upload
            points = []
            point_id = 1  # Start with ID 1
            
            for chunk in chunks:
                chunk_id = chunk["metadata"]["embedding_id"]
                
                if chunk_id in embeddings:
                    # Ensure embedding values are native Python floats for serialization
                    embedding_vector = [float(value) for value in embeddings[chunk_id]]
                    
                    # Create point with integer ID instead of string
                    points.append(
                        models.PointStruct(
                            id=point_id,  # Use integer ID
                            vector=embedding_vector,
                            payload={
                                "content": chunk["content"],
                                "metadata": chunk["metadata"],
                                "chunk_id": chunk_id  # Store original ID in payload
                            }
                        )
                    )
                    point_id += 1
            
            # Upload in batches to avoid timeouts
            batch_size = 50
            for i in range(0, len(points), batch_size):
                batch = points[i:i+batch_size]
                logger.info(f"Uploading batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1} to Qdrant ({len(batch)} points)")
                
                try:
                    self.client.upsert(
                        collection_name=self.collection_name,
                        points=batch
                    )
                    logger.info(f"Successfully uploaded batch {i//batch_size + 1}/{(len(points)-1)//batch_size + 1}")
                except Exception as e:
                    logger.error(f"Error uploading batch to Qdrant: {e}")
                    return False
            
            logger.info(f"Successfully stored {len(points)} embeddings in Qdrant collection {self.collection_name}")
            return True
        except Exception as e:
            logger.error(f"Error storing embeddings in Qdrant: {e}")
            return False
    
    def _qdrant_search_embeddings(self, query_embedding: List[float], top_k: int = 3) -> List[Dict[str, Any]]:
        """
        Search for similar embeddings in Qdrant
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of results to return
            
        Returns:
            List of chunks with similarity scores
        """
        try:
            # Ensure embedding is properly serializable
            query_vector = [float(value) for value in query_embedding]
            
            # Search for similar vectors
            search_result = self.client.search(
                collection_name=self.collection_name,
                query_vector=query_vector,
                limit=top_k
            )
            
            # Format results
            results = []
            for scored_point in search_result:
                # Extract payload and score
                payload = scored_point.payload
                score = scored_point.score
                
                # Format as chunk with similarity
                chunk_with_score = {
                    "chunk": {
                        "content": payload["content"],
                        "metadata": payload["metadata"]
                    },
                    "similarity": score
                }
                results.append(chunk_with_score)
            
            logger.info(f"Found {len(results)} similar chunks in Qdrant")
            return results
        except Exception as e:
            logger.error(f"Error searching embeddings in Qdrant: {str(e)}")
            return []
    
    def _qdrant_load_embeddings(self) -> Dict[str, List[float]]:
        """
        Load all embeddings from Qdrant
        
        Returns:
            Dictionary mapping chunk IDs to embeddings
        """
        try:
            # This is less efficient for Qdrant as we need to pull all vectors
            # Only used for compatibility with the existing code
            scroll_result = self.client.scroll(
                collection_name=self.collection_name,
                limit=10000,  # Adjust based on expected collection size
                with_payload=True,
                with_vectors=True,
            )
            
            # Extract points
            points = scroll_result[0]
            
            # Format as dictionary
            embeddings = {}
            for point in points:
                if hasattr(point, "id") and hasattr(point, "vector"):
                    embeddings[str(point.id)] = point.vector
            
            logger.info(f"Loaded {len(embeddings)} embeddings from Qdrant collection {self.collection_name}")
            return embeddings
        except Exception as e:
            logger.error(f"Error loading embeddings from Qdrant: {str(e)}")
            return {}

# Initialize global vector store
vector_store = VectorStore() 