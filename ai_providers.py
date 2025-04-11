"""
AI Provider Utilities for RAG Pipeline

This module provides utility functions for working with different AI providers
for embedding and inference tasks.
"""

import os
import logging
import numpy as np
import requests  # Add this import for direct API calls
from typing import List, Dict, Any, Optional, Union, Callable
from config_loader import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Get configured providers to determine which warnings to show
config_embedding_provider = config.get("embedding.provider", "local")
config_inference_provider = config.get("inference.provider", "local")

# Try to import optional dependencies for LangChain integration
try:
    from langchain.embeddings import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAVE_LANGCHAIN_GEMINI = True
except ImportError:
    HAVE_LANGCHAIN_GEMINI = False
    use_langchain = config.get("libraries.use_langchain", False)
    if use_langchain and (config_embedding_provider == "gemini" or config_inference_provider == "gemini"):
        logger.warning("LangChain GoogleGenerativeAI integrations not installed.")

class EmbeddingProvider:
    """Class for handling embeddings using different providers"""
    
    def __init__(self):
        """Initialize the embedding provider based on configuration"""
        self.provider_name = config.get("embedding.provider", "gemini")
        self.model_name = config.get("embedding.model", "models/embedding-001")
        self.embedding_dim = config.get("embedding.dimension", 768)
        self.use_langchain = config.get("libraries.use_langchain", False)
        
        # Initialize the appropriate provider
        if self.provider_name == "gemini":
            self._init_gemini()
        elif self.provider_name == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported embedding provider: {self.provider_name}. Use 'gemini' or 'openai'.")
        
        logger.info(f"Initialized embedding provider: {self.provider_name} (model: {self.model_name})")
    
    def _init_gemini(self):
        """Initialize Gemini embedding provider"""
        api_key = config.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not found. Please set it in .env or config.yaml")
        
        if self.use_langchain and HAVE_LANGCHAIN_GEMINI:
            # Use LangChain integration if configured
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key
            )
            self.embed_texts = self._langchain_embed_texts
        else:
            # Use direct API approach
            self.gemini_api_key = api_key
            self.embed_texts = self._gemini_embed_texts
            logger.info(f"Initialized Gemini embeddings with direct API access, model: {self.model_name}")
    
    def _init_openai(self):
        """Initialize OpenAI embedding provider"""
        raise NotImplementedError("OpenAI embedding provider not yet implemented")
    
    def _gemini_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using direct Gemini API
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        embeddings = []
        
        for text in texts:
            # Use the correct API format as shown in successful tool
            url = f"https://generativelanguage.googleapis.com/v1/models/embedding-001:embedContent?key={self.gemini_api_key}"
            
            # Trim text if too long (API has limits)
            if len(text) > 25000:
                text = text[:25000]
                logger.warning(f"Text truncated to 25000 characters")
            
            # Prepare the request payload in the correct format
            payload = {
                "model": "models/embedding-001",
                "content": {
                    "parts": [
                        {"text": text}
                    ]
                }
            }
            
            # Make the API request with timeout
            response = requests.post(url, json=payload, timeout=30)
            
            if response.status_code != 200:
                error_message = f"Error getting embedding: {response.text}"
                logger.error(error_message)
                raise ValueError(error_message)
            
            result = response.json()
            
            # Extract the embedding values using the correct path
            if "embedding" in result and "values" in result["embedding"]:
                embedding_values = result["embedding"]["values"]
                embeddings.append(embedding_values)
            else:
                error_message = f"Gemini API returned unexpected response format: {result}"
                logger.error(error_message)
                raise ValueError(error_message)
        
        return embeddings
    
    def _langchain_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings using LangChain integration
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        try:
            return self.embedding_model.embed_documents(texts)
        except Exception as e:
            error_message = f"Error generating LangChain embeddings: {str(e)}"
            logger.error(error_message)
            raise ValueError(error_message)
    
    def embed_text(self, text: str) -> List[float]:
        """
        Generate embedding for a single text string
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        embeddings = self.embed_texts([text])
        return embeddings[0] if embeddings else []
    
    def embed_batch(self, texts: List[str], batch_size: int = 0) -> List[List[float]]:
        """
        Generate embeddings for a batch of texts
        
        Args:
            texts: List of texts to embed
            batch_size: Batch size for processing (0=no batching)
            
        Returns:
            List of embedding vectors
        """
        if not batch_size or batch_size >= len(texts):
            return self.embed_texts(texts)
        
        # Process in batches
        all_embeddings = []
        for i in range(0, len(texts), batch_size):
            batch = texts[i:i+batch_size]
            batch_embeddings = self.embed_texts(batch)
            all_embeddings.extend(batch_embeddings)
            logger.info(f"Processed embedding batch {i//batch_size + 1}/{(len(texts)-1)//batch_size + 1}")
        
        return all_embeddings


class InferenceProvider:
    """Class for handling inference using different providers"""
    
    def __init__(self):
        """Initialize the inference provider based on configuration"""
        self.provider_name = config.get("inference.provider", "gemini")
        self.model_name = config.get("inference.model", "gemini-2.0-flash")
        self.temperature = config.get("inference.temperature", 0.2)
        self.max_tokens = config.get("inference.max_tokens", 1024)
        self.use_langchain = config.get("libraries.use_langchain", False)
        
        # Initialize the appropriate provider
        if self.provider_name == "gemini":
            self._init_gemini()
        elif self.provider_name == "openai":
            self._init_openai()
        else:
            raise ValueError(f"Unsupported inference provider: {self.provider_name}. Use 'gemini' or 'openai'.")
        
        logger.info(f"Initialized inference provider: {self.provider_name} (model: {self.model_name})")
    
    def _init_gemini(self):
        """Initialize Gemini inference provider"""
        api_key = config.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not found. Please set it in .env or config.yaml")
        
        if self.use_langchain and HAVE_LANGCHAIN_GEMINI:
            # Use LangChain integration if configured
            self.generate = self._langchain_generate
            
            # Initialize LangChain ChatGoogleGenerativeAI
            self.llm = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens,
            )
        else:
            # Store API key for direct use
            self.gemini_api_key = api_key
            self.generate = self._gemini_generate
            logger.info(f"Initialized Gemini inference with direct API access, model: {self.model_name}")
    
    def _init_openai(self):
        """Initialize OpenAI inference provider"""
        raise NotImplementedError("OpenAI inference provider not yet implemented")
    
    def _gemini_generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using Gemini
        
        Args:
            prompt: User query
            context: List of context chunks with content and metadata
            
        Returns:
            Generated response string
        """
        # Create a prompt that includes the context information
        full_prompt = prompt
        
        if context and len(context) > 0:
            full_prompt = "Context information:\n\n"
            
            for i, ctx in enumerate(context):
                content = ctx.get("content", "")
                metadata = ctx.get("metadata", {})
                source = metadata.get("filename", f"Source {i+1}")
                
                full_prompt += f"--- {source} ---\n{content}\n\n"
            
            full_prompt += f"Based on the above context, answer the following question:\n{prompt}"
        
        # Use direct REST API approach instead of the client library
        url = f"https://generativelanguage.googleapis.com/v1/models/{self.model_name}:generateContent?key={self.gemini_api_key}"
        
        # Set up generation parameters in the correct format
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": full_prompt}
                    ]
                }
            ],
            "generationConfig": {
                "temperature": self.temperature,
                "topP": 0.8,
                "topK": 40,
                "maxOutputTokens": self.max_tokens,
            }
        }
        
        # Make the API request with timeout
        response = requests.post(url, json=payload, timeout=60)
        
        if response.status_code != 200:
            error_message = f"Error generating Gemini response: {response.text}"
            logger.error(error_message)
            raise ValueError(error_message)
        
        result = response.json()
        
        # Extract the generated text from the response using correct path
        if "candidates" in result and len(result["candidates"]) > 0:
            candidate = result["candidates"][0]
            if "content" in candidate and "parts" in candidate["content"]:
                parts = candidate["content"]["parts"]
                if parts and "text" in parts[0]:
                    return parts[0]["text"]
        
        # If we couldn't extract the text, raise an error
        error_message = f"Failed to parse Gemini response: {result}"
        logger.error(error_message)
        raise ValueError(error_message)
    
    def _langchain_generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using LangChain integration
        
        Args:
            prompt: User query
            context: List of context chunks with content and metadata
            
        Returns:
            Generated response string
        """
        # Prepare context information
        context_text = ""
        if context and len(context) > 0:
            context_text = "Context information:\n\n"
            
            for i, ctx in enumerate(context):
                content = ctx.get("content", "")
                metadata = ctx.get("metadata", {})
                source = metadata.get("filename", f"Source {i+1}")
                
                context_text += f"--- {source} ---\n{content}\n\n"
        
        # Construct the full prompt
        if context_text:
            full_prompt = f"{context_text}\n\nBased on the above context, answer the following question:\n{prompt}"
        else:
            full_prompt = prompt
        
        # Generate response
        response = self.llm.invoke(full_prompt)
        
        # Extract content based on the response structure
        if hasattr(response, "content"):
            return response.content
        else:
            return str(response)

# Initialize global providers
embedding_provider = EmbeddingProvider()
inference_provider = InferenceProvider()

if __name__ == "__main__":
    # Test the providers
    test_texts = [
        "What are the health benefits of proteins?",
        "How does hydration affect athletic performance?",
        "What vitamins are essential for bone health?"
    ]
    
    print(f"Using embedding provider: {embedding_provider.provider_name}")
    embeddings = embedding_provider.embed_batch(test_texts, batch_size=2)
    print(f"Generated {len(embeddings)} embeddings of dimension {len(embeddings[0])}")
    
    print(f"\nUsing inference provider: {inference_provider.provider_name}")
    response = inference_provider.generate(
        "What are the main types of proteins and their functions?",
        context=[{
            "content": "Proteins are essential macronutrients made up of amino acids that serve as building blocks for tissues, enzymes, and hormones.",
            "metadata": {"filename": "test_doc.txt"}
        }]
    )
    print(f"Generated response:\n{response}") 