# Simple RAG Pipeline (Baseline)

This is a simple Retrieval-Augmented Generation (RAG) pipeline implementation in Python that processes nutrition and health-related documents. It demonstrates a basic RAG workflow without using any orchestration tools and serves as a baseline for comparison with various orchestration tools like Airflow, Dagster, Flyte, Kubeflow, and Mage.

## Project Structure

- `src/`: Python modules implementing the RAG pipeline components
  - `data_ingestion.py`: Loads and prepares nutrition documents from the data directory
  - `document_processor.py`: Chunks documents and generates embeddings
  - `query_engine.py`: Retrieves relevant chunks and generates responses
  - `pipeline.py`: Main module that ties all components together

## Features

- Document loading from nutrition and health-related text files
- Document chunking (based on configurable size and overlap)
- Configurable vector embedding via Gemini API or local simulation
- Semantic search using cosine similarity
- Response generation using Gemini LLM or template-based fallback
- Complete end-to-end pipeline execution
- Isolated processed directories for different embedding providers

## Configuration

This project uses the shared configuration system from the root directory:

- `config.yaml`: Central configuration file for all settings
- `.env`: Environment file for API keys and secrets
- `config_loader.py`: Utility to load and manage configuration

You can configure the following aspects:
- Embedding provider (Gemini, local)
- Inference provider (Gemini, local)
- Document processing parameters (chunk size, overlap)
- Directory paths

## Requirements

This project uses the shared virtual environment from the root directory. The required packages are:

- numpy
- requests
- python-dotenv
- pyyaml
- google-generativeai (optional, for Gemini embeddings/inference)
- langchain (optional, for LangChain integration)

## Usage

### Run using the PowerShell script (Windows)

```powershell
.\run.ps1 -Tool base -Query "What are the key principles of weight management?"
```

### Run using the Shell script (Linux/macOS)

```bash
# First, ensure the script has execution permissions (only needed once)
chmod +x run.sh
# Then run it
./run.sh -t base -q "What are the key principles of weight management?"
```

### Run directly

```bash
# Windows
python base_project/src/pipeline.py "What are good sources of protein?"

# Linux/macOS
python3 base_project/src/pipeline.py "What are good sources of protein?"
```

You can provide your own nutrition or health-related query as a command-line argument.

### Switching Embedding Providers

#### Local Embeddings (No API Key Required)
```yaml
# In config.yaml
embedding:
  provider: "local"
inference:
  provider: "local"
```

#### Gemini API (Requires API Key)
1. Get an API key from https://ai.google.dev/
2. Create a `.env` file with `GEMINI_API_KEY=your_key_here`
3. Update config.yaml:
```yaml
# In config.yaml
embedding:
  provider: "gemini"
inference:
  provider: "gemini"
```

### Sample Queries

Here are some example queries you can try:

1. Nutrition Basics:
   - "What are the main types of carbohydrates and their functions?"
   - "Explain the role of proteins in the body"
   - "What are the different types of fats and their health impacts?"

2. Dietary Management:
   - "How can I manage food allergies effectively?"
   - "What are the best practices for food safety?"
   - "What dietary changes help with digestive disorders?"

3. Life Stages and Special Conditions:
   - "What are the nutritional needs during pregnancy?"
   - "How does nutrition change with aging?"
   - "What dietary considerations are important for autoimmune conditions?"

4. Health and Wellness:
   - "What are the benefits of proper hydration?"
   - "How do vitamins and minerals support health?"
   - "What are the key principles of popular dietary patterns?"

### Use the components individually

```python
import sys
import os

# Add root directory to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))
from config_loader import config

from base_project.src.data_ingestion import DataIngestion
from base_project.src.document_processor import DocumentProcessor
from base_project.src.query_engine import QueryEngine

# Ingest documents
ingestion = DataIngestion()
documents = ingestion.load_documents()

# Process documents
processor = DocumentProcessor()
chunks, embeddings = processor.process_documents(documents)

# Query for information
query_engine = QueryEngine()
response = query_engine.process_query("What are the essential vitamins and their functions?")
print(response)
```

## Data Requirements

The pipeline expects text files (.txt) in the `data/raw` directory containing nutrition and health-related content. The current dataset includes information about:

- Macronutrients (carbohydrates, proteins, fats)
- Micronutrients (vitamins and minerals)
- Dietary patterns and weight management
- Life stage nutrition
- Special conditions (allergies, digestive disorders, autoimmune conditions)
- Food safety and preparation
- Hydration

## Directory Structure

The project uses an isolated directory structure to keep embeddings and processed data separate:

```
data/
  raw/                       # Original document files
  processed/
    base_local/              # Processed data for base project with local embeddings
    base_gemini/             # Processed data for base project with Gemini embeddings
    airflow_local/           # Processed data for Airflow with local embeddings
    ...
```

This ensures that different combinations of tools and embedding providers don't interfere with each other, enabling fair comparison.

## Why This Project

This project serves as a baseline implementation to compare with various orchestration tools:

1. **Learning Curve**: See how the complexity changes when using orchestration tools
2. **Ease of Use**: Compare how the implementation changes with orchestration
3. **Integration**: Understand how it fits with other tools like LangChain
4. **Visualization**: Compare with orchestration tools that offer UI dashboards
5. **Monitoring**: Observe how monitoring, retry logic, and error handling changes

## Limitations

This is a simplified implementation with the following limitations:

- Can use simulated embeddings instead of a real embedding model (configurable)
- Uses simple character-based chunking
- Can use a template-based response generator instead of an actual LLM (configurable)
- Doesn't handle complex document formats
- No error recovery, parallel processing, or sophisticated scheduling

These limitations are intentional to keep the baseline simple. The orchestrated versions will address some of these issues. 