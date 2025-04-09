# Orchestration Tools Comparison

This repository contains implementations of a RAG (Retrieval-Augmented Generation) pipeline using different orchestration tools. The pipeline processes nutrition and health-related documents to provide information about various aspects of nutrition, dietary management, and health conditions. The goal is to evaluate and compare various workflow orchestration solutions to find the best fit for our needs.

## Project Structure

- `base_project/`: Baseline implementation without orchestration
- `airflow/`: Implementation using Apache Airflow
- `dagster/`: Implementation using Dagster
- `flyte/`: Implementation using Flyte
- `kubeflow/`: Implementation using Kubeflow
- `mage/`: Implementation using Mage

## Purpose

This project aims to evaluate different orchestration tools based on:
- Learning curve and ease of use
- Integration capabilities with other tools (e.g., LangChain)
- Visualization and monitoring features
- Error handling and retry mechanisms
- Scalability and performance
- Community support and documentation

## Dataset

The project uses a collection of nutrition and health-related documents covering:
- Macronutrients (carbohydrates, proteins, fats)
- Micronutrients (vitamins and minerals)
- Dietary patterns and weight management
- Life stage nutrition
- Special conditions (allergies, digestive disorders, autoimmune conditions)
- Food safety and preparation
- Hydration

Each implementation processes this same dataset to ensure fair comparison between orchestration tools.

## Common Requirements

All implementations use Python 3.11.5 and share a common virtual environment. Each project's specific requirements are listed in their respective directories, but they are all installed in the root virtual environment.

## Getting Started

1. Create and activate the virtual environment:
```bash
# Windows
python -m venv .venv
.venv\Scripts\activate

# Linux/macOS
python -m venv .venv
source .venv/bin/activate
```

2. Install base requirements:
```bash
pip install -r requirements.txt
```

3. Run the project:
```bash
# Windows (PowerShell)
.\run.ps1 -Tool base -Query "What are good sources of protein?"

# Linux/macOS (Shell)
# First, ensure the script has execution permissions (only needed once)
chmod +x run.sh
# Then run it
./run.sh -t base -q "What are good sources of protein?"
```

4. To use a different orchestration tool:
```bash
# Windows (PowerShell)
.\run.ps1 -Tool dagster -UI

# Linux/macOS (Shell)
# If you haven't already made the script executable
chmod +x run.sh
# Run with the specified tool
./run.sh -t dagster -u
```

## Configuration

The project uses a central configuration system:
- `config.yaml`: Central configuration file for all settings
- `.env`: Environment file for API keys and secrets (create from .env.example)

You can configure:
- The active orchestration tool
- Embedding and inference providers
- Document processing parameters
- Directory paths

## Evaluation Criteria

Each implementation will be evaluated based on:

1. **Setup and Configuration**
   - Ease of initial setup
   - Configuration complexity
   - Local development experience

2. **Development Experience**
   - Code organization and structure
   - Testing capabilities
   - Debugging tools
   - Development workflow

3. **Operational Features**
   - Scheduling capabilities
   - Monitoring and observability
   - Error handling and recovery
   - Logging and tracing

4. **Integration Capabilities**
   - Compatibility with other tools
   - API and extension points
   - Customization options

5. **Performance and Scalability**
   - Resource utilization
   - Parallel execution
   - Distributed processing
   - Scaling options

## Contributing

Each project directory contains its own README with specific instructions for that implementation. When adding new implementations:

1. Create a new directory for the tool
2. Implement the RAG pipeline using that tool
3. Add a README.md with setup and usage instructions
4. Add a requirements.txt with necessary dependencies (without version numbers)
5. Update this main README with any relevant information

## License

[Your License Here] 