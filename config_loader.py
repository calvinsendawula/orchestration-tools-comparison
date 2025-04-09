"""
Configuration loader for the RAG Orchestration Tools Comparison

This script loads configuration from:
1. YAML configuration file (config.yaml)
2. Environment variables (.env file)
"""

import os
import yaml
import logging
from pathlib import Path
from typing import Dict, Any, Optional
from dotenv import load_dotenv

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class ConfigLoader:
    """Loads and manages configuration from YAML and environment variables"""
    
    def __init__(self, config_path: str = "config.yaml", env_path: str = ".env"):
        """
        Initialize the configuration loader
        
        Args:
            config_path: Path to YAML configuration file
            env_path: Path to environment variables file
        """
        self.config_path = config_path
        self.env_path = env_path
        self.config = {}
        
        # Load environment variables
        if os.path.exists(env_path):
            load_dotenv(env_path)
            logger.info(f"Loaded environment variables from {env_path}")
        else:
            logger.warning(f"Environment file {env_path} not found. Using system environment variables.")
        
        # Load configuration from YAML
        self._load_yaml_config()
        
        # Override with environment variables where applicable
        self._override_from_env()
        
        logger.info("Configuration loaded successfully")
    
    def _load_yaml_config(self) -> None:
        """Load configuration from YAML file"""
        try:
            with open(self.config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Loaded configuration from {self.config_path}")
        except FileNotFoundError:
            logger.error(f"Configuration file {self.config_path} not found")
            self.config = {}
        except Exception as e:
            logger.error(f"Error loading configuration: {str(e)}")
            self.config = {}
    
    def _override_from_env(self) -> None:
        """Override configuration with environment variables"""
        # API Keys
        if os.getenv("GEMINI_API_KEY"):
            self.config.setdefault("api_keys", {})["gemini"] = os.getenv("GEMINI_API_KEY")
        
        if os.getenv("OPENAI_API_KEY"):
            self.config.setdefault("api_keys", {})["openai"] = os.getenv("OPENAI_API_KEY")
        
        # Override active orchestration tool if specified
        if os.getenv("ACTIVE_ORCHESTRATION_TOOL"):
            self.config.setdefault("orchestration", {})["active_tool"] = os.getenv("ACTIVE_ORCHESTRATION_TOOL")
        
        # Override model choices if specified
        if os.getenv("EMBEDDING_PROVIDER"):
            self.config.setdefault("embedding", {})["provider"] = os.getenv("EMBEDDING_PROVIDER")
        
        if os.getenv("INFERENCE_PROVIDER"):
            self.config.setdefault("inference", {})["provider"] = os.getenv("INFERENCE_PROVIDER")
    
    def get_config(self) -> Dict[str, Any]:
        """Get the complete configuration"""
        return self.config
    
    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """
        Get a specific configuration value
        
        Args:
            key: Configuration key (can use dot notation for nested keys)
            default: Default value if key is not found
            
        Returns:
            The configuration value or default
        """
        # Handle nested keys with dot notation
        if "." in key:
            parts = key.split(".")
            current = self.config
            for part in parts:
                if isinstance(current, dict) and part in current:
                    current = current[part]
                else:
                    return default
            return current
        
        return self.config.get(key, default)
    
    def get_api_key(self, provider: str) -> Optional[str]:
        """
        Get API key for a specific provider
        
        Args:
            provider: Provider name (e.g., "gemini", "openai")
            
        Returns:
            API key or None if not found
        """
        api_keys = self.config.get("api_keys", {})
        
        # First check if it's in the loaded config
        if provider in api_keys:
            return api_keys[provider]
        
        # Then check environment directly
        env_var = f"{provider.upper()}_API_KEY"
        return os.getenv(env_var)
    
    def get_active_tool(self) -> str:
        """Get the currently active orchestration tool"""
        return self.get("orchestration.active_tool", "base")
    
    def get_embedding_provider(self) -> str:
        """Get the active embedding provider"""
        return self.get("embedding.provider", "local")
    
    def get_data_directory(self) -> str:
        """Get the data directory path"""
        data_dir = self.get("data_directory", "./data")
        
        # Create directory if it doesn't exist
        os.makedirs(data_dir, exist_ok=True)
        
        return data_dir
    
    def get_raw_data_directory(self) -> str:
        """Get the raw data directory path"""
        raw_dir = os.path.join(self.get_data_directory(), "raw")
        
        # Create directory if it doesn't exist
        os.makedirs(raw_dir, exist_ok=True)
        
        return raw_dir
    
    def get_processed_directory(self) -> str:
        """
        Get the processed data directory path for the current tool and embedding provider
        This ensures each combination has its own isolated directory
        """
        tool = self.get_active_tool()
        provider = self.get_embedding_provider()
        
        # Use the format string from config if available, otherwise use default
        format_str = self.get("processed_directory_format", 
                             os.path.join(self.get_data_directory(), "processed", "{tool}_{provider}"))
        
        # Format the directory path
        processed_dir = format_str.format(tool=tool, provider=provider)
        
        # Create directory if it doesn't exist
        os.makedirs(processed_dir, exist_ok=True)
        logger.info(f"Using processed directory for {tool} with {provider}: {processed_dir}")
        
        return processed_dir
    
    def get_embedding_config(self) -> Dict[str, Any]:
        """Get embedding configuration"""
        return self.get("embedding", {})
    
    def get_inference_config(self) -> Dict[str, Any]:
        """Get inference configuration"""
        return self.get("inference", {})
    
    def get_document_processing_config(self) -> Dict[str, Any]:
        """Get document processing configuration"""
        return self.get("document_processing", {})
    
    def get_library_config(self) -> Dict[str, Any]:
        """Get library usage configuration"""
        return self.get("libraries", {})
    
    def get_tool_config(self, tool_name: str) -> Dict[str, Any]:
        """
        Get configuration for a specific orchestration tool
        
        Args:
            tool_name: Name of the orchestration tool
            
        Returns:
            Tool-specific configuration
        """
        return self.get(tool_name, {})

# Create a global instance
config = ConfigLoader()

if __name__ == "__main__":
    # Test the configuration loader
    print("Active orchestration tool:", config.get_active_tool())
    print("Active embedding provider:", config.get_embedding_provider())
    print("Raw data directory:", config.get_raw_data_directory())
    print("Processed directory:", config.get_processed_directory())
    
    # Change the tool and see how the processed directory changes
    config.config["orchestration"]["active_tool"] = "airflow"
    print("New processed directory:", config.get_processed_directory()) 