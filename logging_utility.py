"""
Logging utility for the RAG Orchestration Tools Comparison

This module provides functionality for:
1. Structured logging to tool-specific log files
2. Metrics tracking in CSV files for performance comparison
"""

import os
import csv
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional

# Import the configuration loader
from config_loader import config

class LoggingUtility:
    """Handles logging and metrics tracking for the RAG pipeline"""
    
    def __init__(self):
        """Initialize the logging utility"""
        # Get logging configuration
        self.log_dir = config.get("logging.log_directory", "./logs")
        self.metrics_dir = config.get("logging.metrics_directory", "./logs/metrics")
        self.log_format = config.get("logging.file_format", "{tool}_{provider}.log")
        self.metrics_format = config.get("logging.metrics_format", "{tool}_{provider}_metrics.csv")
        
        # Create directories if they don't exist
        os.makedirs(self.log_dir, exist_ok=True)
        os.makedirs(self.metrics_dir, exist_ok=True)
        
        # Set up default logging
        logging.basicConfig(
            level=getattr(logging, config.get("log_level", "INFO")),
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Current metrics for tracking
        self.current_run = {
            "start_time": None,
            "end_time": None,
            "duration": None,
            "tool": None,
            "embedding_provider": None,
            "embedding_model": None,
            "inference_provider": None,
            "inference_model": None,
            "query": None,
            "status": None
        }
        
        # Get the active tool and providers for this run
        self.active_tool = config.get_active_tool()
        self.embedding_provider = config.get_embedding_provider()
        self.embedding_model = config.get("embedding.model", "")
        self.inference_provider = config.get("inference.provider", "local")
        self.inference_model = config.get("inference.model", "")
        
        # Configure the logger
        self._setup_logger()
    
    def _setup_logger(self):
        """Set up the logger for the current tool and provider"""
        # Format the log file name
        log_file = os.path.join(
            self.log_dir, 
            self.log_format.format(
                tool=self.active_tool,
                provider=self.embedding_provider
            )
        )
        
        # Create a new file handler
        file_handler = logging.FileHandler(log_file, mode='a')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        
        # Get logger for the current tool
        self.logger = logging.getLogger(f"{self.active_tool}_pipeline")
        
        # Remove existing handlers to avoid duplicates
        if self.logger.handlers:
            self.logger.handlers.clear()
        
        # Add the file handler
        self.logger.addHandler(file_handler)
        self.logger.setLevel(getattr(logging, config.get("log_level", "INFO")))
        
        # Log the configuration
        self.logger.info(f"Starting RAG pipeline with tool={self.active_tool}, "
                        f"embedding={self.embedding_provider}, "
                        f"inference={self.inference_provider}")
    
    def start_run(self, query: str = ""):
        """
        Start tracking a new pipeline run
        
        Args:
            query: The query being processed (if applicable)
        """
        self.current_run = {
            "start_time": datetime.now(),
            "end_time": None,
            "duration": None,
            "tool": self.active_tool,
            "embedding_provider": self.embedding_provider,
            "embedding_model": self.embedding_model,
            "inference_provider": self.inference_provider,
            "inference_model": self.inference_model,
            "query": query,
            "status": "running"
        }
        
        self.logger.info(f"Starting pipeline run with query: {query}")
        return self.current_run["start_time"]
    
    def end_run(self, status: str = "completed", error: Optional[str] = None):
        """
        End tracking for the current pipeline run
        
        Args:
            status: The run status (completed, failed, etc.)
            error: Error message if the run failed
        """
        if not self.current_run["start_time"]:
            self.logger.warning("Attempted to end a run that wasn't started")
            return
        
        self.current_run["end_time"] = datetime.now()
        self.current_run["duration"] = (
            self.current_run["end_time"] - self.current_run["start_time"]
        ).total_seconds()
        self.current_run["status"] = status
        
        # Log completion
        self.logger.info(
            f"Pipeline run completed with status={status} in "
            f"{self.current_run['duration']:.2f} seconds"
        )
        
        if error:
            self.logger.error(f"Error during pipeline run: {error}")
            self.current_run["error"] = error
        
        # Save metrics
        self._save_metrics()
        
        return self.current_run["duration"]
    
    def log_stage(self, stage_name: str, message: str, level: str = "INFO"):
        """
        Log a pipeline stage event
        
        Args:
            stage_name: Name of the pipeline stage
            message: Log message
            level: Log level (INFO, WARNING, ERROR, etc.)
        """
        log_method = getattr(self.logger, level.lower(), self.logger.info)
        log_method(f"[{stage_name}] {message}")
    
    def log_metric(self, name: str, value: Any):
        """
        Log a specific metric for the current run
        
        Args:
            name: Metric name
            value: Metric value
        """
        self.current_run[name] = value
        self.logger.info(f"Metric: {name}={value}")
    
    def _save_metrics(self):
        """Save the current run metrics to CSV"""
        # Format the metrics file name
        metrics_file = os.path.join(
            self.metrics_dir, 
            self.metrics_format.format(
                tool=self.active_tool,
                provider=self.embedding_provider
            )
        )
        
        # Prepare the data to save
        data = {
            "run_id": datetime.now().strftime("%Y%m%d%H%M%S"),
            "start_time": self.current_run["start_time"].strftime("%Y-%m-%d %H:%M:%S"),
            "end_time": self.current_run["end_time"].strftime("%Y-%m-%d %H:%M:%S") if self.current_run["end_time"] else "",
            "duration_seconds": f"{self.current_run['duration']:.2f}" if self.current_run["duration"] else "",
            "tool": self.current_run["tool"],
            "embedding_provider": self.current_run["embedding_provider"],
            "embedding_model": self.current_run["embedding_model"],
            "inference_provider": self.current_run["inference_provider"],
            "inference_model": self.current_run["inference_model"],
            "query": self.current_run["query"],
            "status": self.current_run["status"]
        }
        
        # Add any additional metrics that were logged
        for key, value in self.current_run.items():
            if key not in data and key not in ["start_time", "end_time"]:
                data[key] = str(value)
        
        # Check if file exists to determine if we need to write headers
        file_exists = os.path.isfile(metrics_file)
        
        # Write to CSV
        with open(metrics_file, 'a', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=data.keys())
            
            # Write header if file is new
            if not file_exists:
                writer.writeheader()
            
            writer.writerow(data)
        
        self.logger.info(f"Metrics saved to {metrics_file}")

# Create a global instance
logging_util = LoggingUtility()

if __name__ == "__main__":
    # Test the logging utility
    logging_util.start_run("What are good sources of protein?")
    logging_util.log_stage("DATA_INGESTION", "Loading documents")
    logging_util.log_metric("num_documents", 10)
    logging_util.log_stage("EMBEDDING", "Generating embeddings")
    logging_util.log_metric("num_chunks", 50)
    time.sleep(1)  # Simulate processing
    logging_util.end_run() 