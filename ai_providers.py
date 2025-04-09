"""
AI Provider Utilities for RAG Pipeline

This module provides utility functions for working with different AI providers
for embedding and inference tasks.
"""

import os
import logging
import numpy as np
from typing import List, Dict, Any, Optional, Union, Callable
from config_loader import config

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Try to import optional dependencies
try:
    from google import genai
    HAVE_GEMINI = True
except ImportError:
    HAVE_GEMINI = False
    logger.warning("Google GenerativeAI not installed. Gemini provider will not be available.")

try:
    from langchain.embeddings import GoogleGenerativeAIEmbeddings
    from langchain_google_genai import ChatGoogleGenerativeAI
    HAVE_LANGCHAIN_GEMINI = True
except ImportError:
    HAVE_LANGCHAIN_GEMINI = False
    logger.warning("LangChain GoogleGenerativeAI integrations not installed.")

class EmbeddingProvider:
    """Class for handling embeddings using different providers"""
    
    def __init__(self):
        """Initialize the embedding provider based on configuration"""
        self.provider_name = config.get("embedding.provider", "local")
        self.model_name = config.get("embedding.model", "embedding-001")
        self.embedding_dim = config.get("embedding.dimension", 768)
        self.use_langchain = config.get("libraries.use_langchain", False)
        
        # Initialize the appropriate provider
        if self.provider_name == "gemini":
            self._init_gemini()
        elif self.provider_name == "openai":
            self._init_openai()
        else:
            self._init_local()
        
        logger.info(f"Initialized embedding provider: {self.provider_name} (model: {self.model_name})")
    
    def _init_gemini(self):
        """Initialize Gemini embedding provider"""
        if not HAVE_GEMINI:
            raise ImportError("Google GenerativeAI not installed. Please install with: pip install google-generativeai")
        
        api_key = config.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not found. Please set it in .env or config.yaml")
        
        # Initialize the Gemini API
        genai.configure(api_key=api_key)
        
        if self.use_langchain and HAVE_LANGCHAIN_GEMINI:
            # Use LangChain integration if configured
            self.embedding_model = GoogleGenerativeAIEmbeddings(
                model=self.model_name,
                google_api_key=api_key
            )
            self.embed_texts = self._langchain_embed_texts
        else:
            # Use direct Gemini API with the latest client approach
            self.gemini_client = genai.Client(api_key=api_key)
            self.embed_texts = self._gemini_embed_texts
    
    def _init_openai(self):
        """Initialize OpenAI embedding provider"""
        raise NotImplementedError("OpenAI embedding provider not yet implemented")
    
    def _init_local(self):
        """Initialize local (simulated) embedding provider"""
        logger.warning("Using local simulated embeddings - these are not semantically meaningful")
        self.embed_texts = self._local_embed_texts
    
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
            try:
                # Using the new client approach for embeddings
                embedding_result = self.gemini_client.embed_content(
                    model=self.model_name,
                    content=text,
                    task_type="retrieval_document",  # For document/chunk embedding
                )
                
                if embedding_result and hasattr(embedding_result, "embedding"):
                    embeddings.append(embedding_result.embedding)
                else:
                    # Fallback to local if API fails
                    logger.warning(f"Gemini embedding failed, using local fallback for: {text[:50]}...")
                    embeddings.append(self._get_local_embedding(text))
            except Exception as e:
                logger.error(f"Error generating Gemini embedding: {str(e)}")
                # Fallback to local
                embeddings.append(self._get_local_embedding(text))
        
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
            logger.error(f"Error generating LangChain embeddings: {str(e)}")
            # Fallback to local
            return [self._get_local_embedding(text) for text in texts]
    
    def _local_embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Generate simulated local embeddings
        
        Args:
            texts: List of texts to embed
            
        Returns:
            List of embedding vectors
        """
        return [self._get_local_embedding(text) for text in texts]
    
    def _get_local_embedding(self, text: str) -> List[float]:
        """
        Generate a simulated embedding vector for a text string
        
        Args:
            text: Text to embed
            
        Returns:
            Vector embedding as a list of floats
        """
        # Simple deterministic "hash" to generate consistent embeddings for the same text
        seed = sum(ord(c) for c in text)
        np.random.seed(seed)
        
        # Generate random embedding vector
        embedding = np.random.randn(self.embedding_dim).astype(float)
        # Normalize to unit length
        embedding = embedding / np.linalg.norm(embedding)
        
        return embedding.tolist()
    
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
        self.provider_name = config.get("inference.provider", "local")
        self.model_name = config.get("inference.model", "gemini-1.5-pro")
        self.temperature = config.get("inference.temperature", 0.2)
        self.max_tokens = config.get("inference.max_tokens", 1024)
        self.use_langchain = config.get("libraries.use_langchain", False)
        
        # Initialize the appropriate provider
        if self.provider_name == "gemini":
            self._init_gemini()
        elif self.provider_name == "openai":
            self._init_openai()
        else:
            self._init_local()
        
        logger.info(f"Initialized inference provider: {self.provider_name} (model: {self.model_name})")
    
    def _init_gemini(self):
        """Initialize Gemini inference provider"""
        if not HAVE_GEMINI:
            raise ImportError("Google GenerativeAI not installed. Please install with: pip install google-generativeai")
        
        api_key = config.get_api_key("gemini")
        if not api_key:
            raise ValueError("Gemini API key not found. Please set it in .env or config.yaml")
        
        if self.use_langchain and HAVE_LANGCHAIN_GEMINI:
            # Use LangChain integration if configured
            self.inference_model = ChatGoogleGenerativeAI(
                model=self.model_name,
                google_api_key=api_key,
                temperature=self.temperature,
                max_output_tokens=self.max_tokens
            )
            self.generate = self._langchain_generate
        else:
            # Use direct Gemini API with the latest client approach
            self.gemini_client = genai.Client(api_key=api_key)
            self.generate = self._gemini_generate
    
    def _init_openai(self):
        """Initialize OpenAI inference provider"""
        raise NotImplementedError("OpenAI inference provider not yet implemented")
    
    def _init_local(self):
        """Initialize local (template-based) inference provider"""
        logger.warning("Using local template-based responses - not using a real LLM")
        self.generate = self._local_generate
    
    def _gemini_generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using direct Gemini API
        
        Args:
            prompt: The query prompt
            context: Optional list of context documents
            
        Returns:
            Generated response text
        """
        try:
            if context:
                # Format context for inclusion in the prompt
                context_text = "\n\n".join([
                    f"Context {i+1}:\n{doc['content']}\nSource: {doc['metadata'].get('filename', 'unknown')}"
                    for i, doc in enumerate(context)
                ])
                
                full_prompt = f"""
                Please answer the following question based on the provided context.
                
                {context_text}
                
                Question: {prompt}
                
                Answer:"""
            else:
                full_prompt = prompt
            
            # Using the new client approach for content generation
            response = self.gemini_client.generate_content(
                model=self.model_name,
                contents=full_prompt,
                generation_config={
                    "temperature": self.temperature,
                    "max_output_tokens": self.max_tokens,
                }
            )
            
            if response and hasattr(response, "text"):
                return response.text
            else:
                logger.warning("Gemini returned an unexpected response format")
                return self._get_local_response(prompt, context)
                
        except Exception as e:
            logger.error(f"Error generating Gemini response: {str(e)}")
            # Fallback to local
            return self._get_local_response(prompt, context)
    
    def _langchain_generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a response using LangChain integration
        
        Args:
            prompt: The query prompt
            context: Optional list of context documents
            
        Returns:
            Generated response text
        """
        try:
            if context:
                # Format context for inclusion in the prompt
                context_text = "\n\n".join([
                    f"Context {i+1}:\n{doc['content']}\nSource: {doc['metadata'].get('filename', 'unknown')}"
                    for i, doc in enumerate(context)
                ])
                
                full_prompt = f"""
                Please answer the following question based on the provided context.
                
                {context_text}
                
                Question: {prompt}
                
                Answer:"""
            else:
                full_prompt = prompt
            
            return self.inference_model.invoke(full_prompt)
            
        except Exception as e:
            logger.error(f"Error generating LangChain response: {str(e)}")
            # Fallback to local
            return self._get_local_response(prompt, context)
    
    def _local_generate(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a template-based response locally
        
        Args:
            prompt: The query prompt
            context: Optional list of context documents
            
        Returns:
            Generated response text
        """
        return self._get_local_response(prompt, context)
    
    def _get_local_response(self, prompt: str, context: List[Dict[str, Any]] = None) -> str:
        """
        Generate a template-based response when real inference isn't available
        
        Args:
            prompt: The query prompt
            context: Optional list of context documents
            
        Returns:
            Template-based response text
        """
        if not context:
            return f"I don't have enough information to answer the question: '{prompt}'"
        
        # Start with a simple template response
        response = f"Here's what I found about '{prompt}':\n\n"
        
        # Add content from each context document
        for i, doc in enumerate(context):
            # Add chunk content to response
            response += f"Information {i+1}:\n"
            response += f"{doc['content']}\n\n"
            response += f"Source: {doc['metadata'].get('filename', 'unknown')}\n\n"
        
        response += "This information should help answer your question."
        
        return response

# Create global instances
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