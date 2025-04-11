# Orchestration Comparison for RAG Pipelines

This project compares different orchestration tools and techniques for Retrieval-Augmented Generation (RAG) pipelines. It implements the same RAG pipeline across various orchestration frameworks to evaluate their effectiveness.

## Project Structure

This monorepo contains multiple implementations of the same RAG pipeline using different orchestration tools:

- `base_project/`: Basic implementation without orchestration (baseline)
- `airflow_project/`: Implementation using Apache Airflow
- `dagster_project/`: Implementation using Dagster
- `flyte_project/`: Implementation using Flyte
- `kubeflow_project/`: Implementation using Kubeflow
- `mage_project/`: Implementation using Mage

Each project implements the same RAG pipeline with identical functionality but using different orchestration approaches.

## RAG Pipeline Overview

Each implementation processes nutrition and health-related documents through the following steps:

1. **Data Ingestion**: Load documents from the data directory
2. **Document Processing**: Chunk documents and generate embeddings
3. **Vector Storage**: Store document chunks and embeddings
4. **Query Processing**: Retrieve relevant chunks and generate responses

## Features

- Document loading from nutrition and health-related text files
- Document chunking (based on configurable size and overlap)
- Vector embeddings via Gemini API or local simulation
- Vector storage in Qdrant or local JSON files
- Semantic search using cosine similarity
- Response generation using Gemini or template-based fallback
- Complete end-to-end pipeline execution

## Command Reference

### PowerShell Script (run.ps1)

| Parameter | Type   | Default | Description |
|-----------|--------|---------|-------------|
| `-Tool`   | string | "base"  | Orchestration tool to use |
| `-Query`  | string | ""      | Query to run against the RAG pipeline |
| `-UI`     | switch | false   | Start the orchestration tool's UI if available |
| `-Reset`  | switch | false   | Reset the vector database collection |

#### Available Tools
- `base`: Baseline implementation without orchestration
- `airflow`: Apache Airflow implementation
- `dagster`: Dagster implementation
- `flyte`: Flyte implementation
- `kubeflow`: Kubeflow implementation
- `mage`: Mage implementation

#### Command Examples

```powershell
# Basic query with base project
.\run.ps1 -Tool base -Query "What are good sources of protein?"

# Reset Qdrant collection and run query
.\run.ps1 -Tool base -Reset -Query "What are good sources of protein?"

# Start Dagster UI
.\run.ps1 -Tool dagster -UI

# Reset collection and start UI
.\run.ps1 -Tool dagster -Reset -UI

# Specify all parameters
.\run.ps1 -Tool airflow -Query "What are the health benefits of vitamin C?" -Reset -UI
```

### Shell Script (run.sh)

| Parameter | Type   | Default | Description |
|-----------|--------|---------|-------------|
| `-t, --tool`   | string | "base"  | Orchestration tool to use |
| `-q, --query`  | string | ""      | Query to run against the RAG pipeline |
| `-u, --ui`     | flag   | false   | Start the orchestration tool's UI if available |
| `-r, --reset`  | flag   | false   | Reset the vector database collection |
| `-h, --help`   | flag   | false   | Show help message |

#### Command Examples

```bash
# Basic query with base project
./run.sh -t base -q "What are good sources of protein?"

# Reset Qdrant collection and run query
./run.sh -t base -r -q "What are good sources of protein?"

# Start Dagster UI
./run.sh -t dagster -u

# Reset collection and start UI
./run.sh -t dagster -r -u

# Using long-form parameters
./run.sh --tool airflow --query "What are the health benefits of vitamin C?" --reset --ui

# Show help
./run.sh --help
```

### Running With Different Configurations

| Scenario | PowerShell Command | Bash Command |
|----------|-------------------|--------------|
| Basic query | `.\run.ps1 -Tool base -Query "query"` | `./run.sh -t base -q "query"` |
| With Gemini embeddings | Edit config.yaml: `embedding.provider: "gemini"` | Edit config.yaml: `embedding.provider: "gemini"` |
| With local embeddings | Edit config.yaml: `embedding.provider: "local"` | Edit config.yaml: `embedding.provider: "local"` |
| With Qdrant storage | Edit config.yaml: `vector_store.provider: "qdrant"` | Edit config.yaml: `vector_store.provider: "qdrant"` |
| With local storage | Edit config.yaml: `vector_store.provider: "local"` | Edit config.yaml: `vector_store.provider: "local"` |
| Reset vector collection | `.\run.ps1 -Tool base -Reset` | `./run.sh -t base -r` |
| Use different tool | `.\run.ps1 -Tool dagster` | `./run.sh -t dagster` |
| Start tool UI | `.\run.ps1 -Tool dagster -UI` | `./run.sh -t dagster -u` |

### Command Examples for Common Tasks

```powershell
# --- PowerShell Commands ---

# 1. Run a query using base project with Gemini embeddings and Qdrant storage
.\run.ps1 -Tool base -Query "What are good sources of fiber?"

# 2. Run a query using base project with local embeddings and local storage
# (First edit config.yaml to set embedding.provider: "local" and vector_store.provider: "local")
.\run.ps1 -Tool base -Query "What are good sources of fiber?"

# 3. Reset Qdrant collection and run query
.\run.ps1 -Tool base -Reset -Query "What are good sources of fiber?"

# 4. Run Airflow implementation with UI
.\run.ps1 -Tool airflow -UI

# 5. Run Dagster job with UI and reset collection
.\run.ps1 -Tool dagster -UI -Reset
```

```bash
# --- Bash Commands ---

# 1. Run a query using base project with Gemini embeddings and Qdrant storage
./run.sh -t base -q "What are good sources of fiber?"

# 2. Run a query using base project with local embeddings and local storage
# (First edit config.yaml to set embedding.provider: "local" and vector_store.provider: "local")
./run.sh -t base -q "What are good sources of fiber?"

# 3. Reset Qdrant collection and run query
./run.sh -t base -r -q "What are good sources of fiber?"

# 4. Run Airflow implementation with UI
./run.sh -t airflow -u

# 5. Run Dagster job with UI and reset collection
./run.sh -t dagster -u -r
```

## Shared Resources

The following resources are shared across all implementations:

- `data/`: Raw nutrition and health documents
- `config.yaml`: Central configuration file
- `config_loader.py`: Configuration loading utility
- `.env`: Environment variables (API keys, etc.)

## Getting Started

### Prerequisites

- Python 3.11+ (recommended 3.11.5)
- Docker (for some orchestration tools)
- API keys (optional, for Gemini)

### Setup

1. Clone this repository
2. Create and activate a virtual environment

```bash
# Windows
python -m venv venv
.\venv\Scripts\activate

# Linux/macOS
python3 -m venv venv
source venv/bin/activate
```

3. Install dependencies

```bash
pip install -r requirements.txt
```

4. (Optional) Set up API keys

Create a `.env` file in the root directory with your API keys:

```
GEMINI_API_KEY=your_gemini_api_key
QDRANT_URL=your_qdrant_url
QDRANT_API_KEY=your_qdrant_api_key
```

### Configuration

Edit `config.yaml` to configure the RAG pipeline:

```yaml
# Main configuration
project:
  data_dir: "data"
  raw_dir: "raw"
  processed_dir: "processed"

# Document processing
document:
  chunk_size: 500
  chunk_overlap: 50

# Embedding configuration
embedding:
  provider: "gemini"  # Options: "gemini", "local"
  model: "models/embedding-001"

# Vector store configuration
vector_store:
  provider: "qdrant"  # Options: "qdrant", "local"
  collection_name_format: "{tool}_{provider}"
  reset_collection: false

# Inference configuration
inference:
  provider: "gemini"  # Options: "gemini", "local"
  model: "gemini-2.0-flash"
  temperature: 0.7
  top_p: 0.95
  top_k: 40
  max_output_tokens: 1024

# Qdrant settings
qdrant:
  default_collection_name: "nutrition_knowledge"
  distance: "Cosine"
  on_disk: true
  vector_dimension: 768
```

### Configuration Reference

#### General Settings

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `data_directory` | Shared data directory | "./data" | Any valid path |
| `raw_directory` | Directory for raw document files | "./data/raw" | Any valid path |
| `processed_directory_format` | Format for processed directories | "./data/processed/{tool}_{provider}" | String with {tool} and {provider} placeholders |
| `log_level` | Logging level | "INFO" | "DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL" |

#### Embedding Settings

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `embedding.provider` | Provider for text embeddings | "gemini" | "gemini", "local" |
| `embedding.model` | Model to use for embedding | "models/embedding-001" | Depends on provider |
| `embedding.dimension` | Vector dimension for local embeddings | 768 | Integer > 0 |
| `embedding.batch_size` | Batch size for embedding requests | 8 | Integer > 0 |
| `embedding.use_cache` | Cache embeddings to avoid redundant API calls | true | true, false |

#### Vector Store Settings

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `vector_store.provider` | Vector database provider | "qdrant" | "qdrant", "local" |
| `vector_store.collection_name_format` | Format for collection names | "{tool}_{provider}" | String with {tool} and {provider} placeholders |
| `vector_store.reset_collection` | Whether to reset the collection if it exists | false | true, false |
| `vector_store.create_collection` | Whether to create the collection if it doesn't exist | true | true, false |

#### Qdrant Settings

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `qdrant.default_collection_name` | Default collection name if format isn't specified | "nutrition_knowledge" | Any string |
| `qdrant.distance` | Distance function to use | "Cosine" | "Cosine", "Euclid", "Dot" |
| `qdrant.on_disk` | Whether to store vectors on disk (in Qdrant cloud) | true | true, false |
| `qdrant.vector_dimension` | Dimension of vectors to store | 768 | Integer > 0 |

#### Inference Settings

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `inference.provider` | Provider for text generation | "gemini" | "gemini", "local" |
| `inference.model` | Model to use for generation | "gemini-2.0-flash" | Depends on provider |
| `inference.temperature` | Temperature for response generation | 0.2 | 0.0-1.0 |
| `inference.max_tokens` | Maximum tokens in generated response | 1024 | Integer > 0 |
| `inference.stream` | Whether to stream responses | false | true, false |

#### Document Processing

| Configuration | Description | Default | Options |
|---------------|-------------|---------|---------|
| `document_processing.chunk_size` | Character size of chunks | 512 | Integer > 0 |
| `document_processing.chunk_overlap` | Character overlap between chunks | 50 | Integer >= 0 |
| `document_processing.include_metadata` | Include document metadata in chunks | true | true, false |

### Environment Variables

Required environment variables in `.env`:

| Variable | Required For | Description |
|----------|--------------|-------------|
| `GEMINI_API_KEY` | Gemini embeddings/inference | API key for Google's Gemini models |
| `QDRANT_URL` | Qdrant vector storage | URL to your Qdrant instance (cloud or self-hosted) |
| `QDRANT_API_KEY` | Qdrant vector storage | API key for authenticating with Qdrant |

### Running the Pipeline

#### Using the PowerShell script (Windows)

```powershell
# Run with base implementation
.\run.ps1 -Tool base -Query "What are the key principles of weight management?"

# Run with Airflow implementation
.\run.ps1 -Tool airflow -Query "What are the key principles of weight management?"

# Reset the vector collection before running
.\run.ps1 -Tool base -Reset -Query "What are the key principles of weight management?"
```

#### Using the Shell script (Linux/macOS)

```bash
# First, ensure the script has execution permissions
chmod +x run.sh

# Run with base implementation
./run.sh -t base -q "What are the key principles of weight management?"

# Run with Airflow implementation
./run.sh -t airflow -q "What are the key principles of weight management?"

# Reset the vector collection before running
./run.sh -t base -r -q "What are the key principles of weight management?"
```

### Sample Queries

Try these example queries:

1. Nutrition Basics:
   - "What are the main types of carbohydrates and their functions?"
   - "Explain the role of proteins in the body"

2. Dietary Management:
   - "How can I manage food allergies effectively?"
   - "What are the best practices for food safety?"

3. Life Stages:
   - "What are the nutritional needs during pregnancy?"
   - "How does nutrition change with aging?"

## Configuration Options

### Embedding Providers

#### Local Embeddings (No API Key Required)
```yaml
embedding:
  provider: "local"
```

#### Gemini API (Requires API Key)
```yaml
embedding:
  provider: "gemini"
  model: "models/embedding-001"
```

### Inference Providers

#### Local Inference (No API Key Required)
```yaml
inference:
  provider: "local"
```

#### Gemini API (Requires API Key)
```yaml
inference:
  provider: "gemini"
  model: "gemini-2.0-flash"
```

### Vector Storage Providers

#### Local JSON Storage
```yaml
vector_store:
  provider: "local"
```

#### Qdrant Vector Database
```yaml
vector_store:
  provider: "qdrant"
  collection_name_format: "{tool}_{provider}"
```

## Comparison Criteria

The project evaluates orchestration tools based on:

1. **Learning Curve**: How easy is it to get started?
2. **Ease of Use**: How intuitive is the implementation?
3. **Integration**: How well does it integrate with other tools?
4. **Visualization**: How good is the UI/dashboard?
5. **Monitoring**: How comprehensive are the monitoring capabilities?
6. **Error Handling**: How robust is the error handling?
7. **Deployment**: How easy is it to deploy to production?
8. **Documentation**: How good is the documentation?
9. **Versatility**: How versatile is it for different use cases?
10. **Scalability**: How well does it scale?

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- The nutrition dataset is a simplified version created for educational purposes.
