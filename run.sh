#!/bin/bash

# Shell script to run the RAG pipeline with the specified orchestration tool
# For Linux and macOS systems

# Default parameters
TOOL="base"
QUERY=""
UI=false
RESET=false

# Help function
show_help() {
  echo -e "\nUsage: ./run.sh [-t|--tool <tool>] [-q|--query <query>] [-u|--ui] [-r|--reset]"
  echo "  -t, --tool   : Orchestration tool to use (base, airflow, dagster, flyte, kubeflow, mage)"
  echo "  -q, --query  : Query to run against the RAG pipeline"
  echo "  -u, --ui     : Start the orchestration tool's UI if available"
  echo "  -r, --reset  : Reset the vector database collection (will delete existing data)"
  echo -e "\nExamples:"
  echo "  ./run.sh -t base -q 'What are good sources of protein?'"
  echo "  ./run.sh -t dagster -u"
  echo "  ./run.sh -t base -r"
}

# Parse arguments
while [[ $# -gt 0 ]]; do
  case $1 in
    -t|--tool)
      TOOL="$2"
      shift 2
      ;;
    -q|--query)
      QUERY="$2"
      shift 2
      ;;
    -u|--ui)
      UI=true
      shift
      ;;
    -r|--reset)
      RESET=true
      shift
      ;;
    -h|--help)
      show_help
      exit 0
      ;;
    *)
      echo "Unknown option: $1"
      show_help
      exit 1
      ;;
  esac
done

# ANSI color codes
GREEN='\033[0;32m'
CYAN='\033[0;36m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Helper functions
show_header() {
  echo -e "${GREEN}\n======================================================"
  echo "  Nutrition RAG Pipeline - Orchestration Tools Comparison"
  echo -e "======================================================${NC}\n"
}

check_python() {
  if ! command -v python3 &> /dev/null; then
    echo -e "${RED}Python not found. Please install Python 3.6+ and try again.${NC}"
    return 1
  fi
  
  # Check Python version
  python_version=$(python3 --version)
  echo -e "${CYAN}Using $python_version${NC}"
  return 0
}

check_requirements() {
  if [[ ! -f "requirements.txt" ]]; then
    echo -e "${YELLOW}Main requirements.txt not found. Creating empty file.${NC}"
    touch requirements.txt
  fi
  
  # Check if python-dotenv and pyyaml are installed
  installed=$(python3 -c "try:
  import dotenv, yaml
  print('Dependencies found')
except ImportError:
  print('Missing dependencies')")

  if [[ "$installed" == "Missing dependencies" ]]; then
    echo -e "${YELLOW}Installing required dependencies for configuration...${NC}"
    python3 -m pip install python-dotenv pyyaml
  fi
}

update_config() {
  local tool="$1"
  local reset="$2"
  
  # Update the active tool in the config
  if [[ -f "config.yaml" ]]; then
    # Replace active_tool value in config.yaml
    sed -i.bak "s/active_tool: \".*\"/active_tool: \"$tool\"/" config.yaml
    
    # Update reset_collection flag
    if [[ "$reset" == true ]]; then
      sed -i.bak "s/reset_collection: false/reset_collection: true/" config.yaml
      echo -e "${YELLOW}Reset flag set. Vector database collection will be reset.${NC}"
    else
      sed -i.bak "s/reset_collection: true/reset_collection: false/" config.yaml
    fi
    
    rm -f config.yaml.bak 2>/dev/null
    echo -e "${CYAN}Updated active tool to: $tool${NC}"
  fi
}

run_base_project() {
  local query="$1"
  
  if [[ -z "$query" ]]; then
    query="What are the key principles of weight management?"
  fi
  
  echo -e "${CYAN}Running base project with query: '$query'${NC}"
  python3 base_project/src/pipeline.py "$query"
}

run_airflow() {
  local ui="$1"
  
  if [[ ! -d "airflow" ]]; then
    echo -e "${RED}Airflow implementation not found.${NC}"
    return
  fi
  
  if [[ "$ui" == true ]]; then
    echo -e "${CYAN}Starting Airflow webserver...${NC}"
    python3 -m airflow webserver -p 8080 &
    python3 -m airflow scheduler &
    sleep 2
    open http://localhost:8080 2>/dev/null || xdg-open http://localhost:8080 2>/dev/null || echo "Open http://localhost:8080 in your browser"
  else
    echo -e "${CYAN}Running Airflow DAG...${NC}"
    python3 -m airflow dags trigger nutrition_rag_dag
  fi
}

run_dagster() {
  local ui="$1"
  
  if [[ ! -d "dagster" ]]; then
    echo -e "${RED}Dagster implementation not found.${NC}"
    return
  fi
  
  if [[ "$ui" == true ]]; then
    echo -e "${CYAN}Starting Dagster UI...${NC}"
    python3 -m dagster dev &
    sleep 2
    open http://localhost:3000 2>/dev/null || xdg-open http://localhost:3000 2>/dev/null || echo "Open http://localhost:3000 in your browser"
  else
    echo -e "${CYAN}Running Dagster job...${NC}"
    python3 -m dagster job execute -f dagster/repository.py -n nutrition_rag_job
  fi
}

run_flyte() {
  local ui="$1"
  
  if [[ ! -d "flyte" ]]; then
    echo -e "${RED}Flyte implementation not found.${NC}"
    return
  fi
  
  if [[ "$ui" == true ]]; then
    echo -e "${CYAN}Starting Flyte UI is not supported in local mode.${NC}"
  else
    echo -e "${CYAN}Running Flyte workflow...${NC}"
    python3 flyte/workflow.py
  fi
}

run_kubeflow() {
  local ui="$1"
  
  if [[ ! -d "kubeflow" ]]; then
    echo -e "${RED}Kubeflow implementation not found.${NC}"
    return
  fi
  
  echo -e "${YELLOW}Kubeflow typically runs on Kubernetes and requires additional setup.${NC}"
  echo -e "${YELLOW}Please refer to the kubeflow/README.md for detailed instructions.${NC}"
}

run_mage() {
  local ui="$1"
  
  if [[ ! -d "mage" ]]; then
    echo -e "${RED}Mage implementation not found.${NC}"
    return
  fi
  
  if [[ "$ui" == true ]]; then
    echo -e "${CYAN}Starting Mage UI...${NC}"
    cd mage
    python3 -m mage start &
    cd ..
    sleep 2
    open http://localhost:6789 2>/dev/null || xdg-open http://localhost:6789 2>/dev/null || echo "Open http://localhost:6789 in your browser"
  else
    echo -e "${CYAN}Running Mage pipeline...${NC}"
    python3 mage/run_pipeline.py
  fi
}

# Main script execution
show_header

# Check Python is available
if ! check_python; then
  exit 1
fi

# Check and install basic requirements
check_requirements

# Update configuration if needed
update_config "$TOOL" "$RESET"

# Run the selected tool
case "${TOOL,,}" in
  "base")
    run_base_project "$QUERY"
    ;;
  "airflow")
    run_airflow $UI
    ;;
  "dagster")
    run_dagster $UI
    ;;
  "flyte")
    run_flyte $UI
    ;;
  "kubeflow")
    run_kubeflow $UI
    ;;
  "mage")
    run_mage $UI
    ;;
  *)
    echo -e "${RED}Unknown tool: $TOOL${NC}"
    show_help
    exit 1
    ;;
esac

echo -e "${GREEN}\nExecution completed.${NC}"

# Make the script executable
# Don't forget to run: chmod +x run.sh 