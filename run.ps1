# PowerShell script to run the RAG pipeline with the specified orchestration tool
param (
    [string]$Tool = "base",
    [string]$Query = "",
    [switch]$UI = $false
)

# Helper functions
function Write-ColorOutput($ForegroundColor) {
    $fc = $host.UI.RawUI.ForegroundColor
    $host.UI.RawUI.ForegroundColor = $ForegroundColor
    if ($args) {
        Write-Output $args
    }
    else {
        $input | Write-Output
    }
    $host.UI.RawUI.ForegroundColor = $fc
}

function Show-Header {
    Write-ColorOutput "Green" "`n======================================================"
    Write-ColorOutput "Green" "  Nutrition RAG Pipeline - Orchestration Tools Comparison"
    Write-ColorOutput "Green" "======================================================`n"
}

function Show-Usage {
    Write-ColorOutput "Yellow" "Usage: .\run.ps1 [-Tool <tool>] [-Query <query>] [-UI]"
    Write-ColorOutput "Yellow" "  -Tool   : Orchestration tool to use (base, airflow, dagster, flyte, kubeflow, mage)"
    Write-ColorOutput "Yellow" "  -Query  : Query to run against the RAG pipeline"
    Write-ColorOutput "Yellow" "  -UI     : Start the orchestration tool's UI if available"
    Write-ColorOutput "Yellow" "Examples:"
    Write-ColorOutput "Yellow" "  .\run.ps1 -Tool base -Query 'What are good sources of protein?'"
    Write-ColorOutput "Yellow" "  .\run.ps1 -Tool dagster -UI"
}

function Check-Python {
    try {
        $pythonVersion = python --version
        Write-ColorOutput "Cyan" "Using $pythonVersion"
        return $true
    } catch {
        Write-ColorOutput "Red" "Python not found. Please install Python 3.6+ and try again."
        return $false
    }
}

function Check-Requirements {
    if (-not (Test-Path -Path "requirements.txt")) {
        Write-ColorOutput "Yellow" "Main requirements.txt not found. Creating empty file."
        "" | Out-File -FilePath "requirements.txt"
    }
    
    # Check if python-dotenv and pyyaml are installed
    $installed = python -c "try:
    import dotenv, yaml
    print('Dependencies found')
except ImportError:
    print('Missing dependencies')"

    if ($installed -eq "Missing dependencies") {
        Write-ColorOutput "Yellow" "Installing required dependencies for configuration..."
        python -m pip install python-dotenv pyyaml
    }
}

function Update-Config {
    param (
        [string]$Tool
    )
    
    # Update the active tool in the config
    # This is a simple approach; in a more sophisticated setup we'd use the YAML parser
    if (Test-Path -Path "config.yaml") {
        $content = Get-Content -Path "config.yaml" -Raw
        $pattern = "active_tool: `"(.*?)`""
        $replacement = "active_tool: `"$Tool`""
        $newContent = $content -replace $pattern, $replacement
        $newContent | Out-File -FilePath "config.yaml" -Encoding utf8
        Write-ColorOutput "Cyan" "Updated active tool to: $Tool"
    }
}

function Run-BaseProject {
    param (
        [string]$Query
    )
    
    if (-not $Query) {
        $Query = "What are the key principles of weight management?"
    }
    
    Write-ColorOutput "Cyan" "Running base project with query: '$Query'"
    python base_project/src/pipeline.py "$Query"
}

function Run-Airflow {
    param (
        [switch]$UI
    )
    
    if (-not (Test-Path -Path "airflow")) {
        Write-ColorOutput "Red" "Airflow implementation not found."
        return
    }
    
    if ($UI) {
        Write-ColorOutput "Cyan" "Starting Airflow webserver..."
        Start-Process -FilePath "python" -ArgumentList "-m airflow webserver -p 8080" -NoNewWindow
        Start-Process -FilePath "python" -ArgumentList "-m airflow scheduler" -NoNewWindow
        Start-Process "http://localhost:8080"
    } else {
        Write-ColorOutput "Cyan" "Running Airflow DAG..."
        python -m airflow dags trigger nutrition_rag_dag
    }
}

function Run-Dagster {
    param (
        [switch]$UI
    )
    
    if (-not (Test-Path -Path "dagster")) {
        Write-ColorOutput "Red" "Dagster implementation not found."
        return
    }
    
    if ($UI) {
        Write-ColorOutput "Cyan" "Starting Dagster UI..."
        Start-Process -FilePath "python" -ArgumentList "-m dagster dev" -NoNewWindow
        Start-Process "http://localhost:3000"
    } else {
        Write-ColorOutput "Cyan" "Running Dagster job..."
        python -m dagster job execute -f dagster/repository.py -n nutrition_rag_job
    }
}

function Run-Flyte {
    param (
        [switch]$UI
    )
    
    if (-not (Test-Path -Path "flyte")) {
        Write-ColorOutput "Red" "Flyte implementation not found."
        return
    }
    
    if ($UI) {
        Write-ColorOutput "Cyan" "Starting Flyte UI is not supported in local mode."
    } else {
        Write-ColorOutput "Cyan" "Running Flyte workflow..."
        python flyte/workflow.py
    }
}

function Run-Kubeflow {
    param (
        [switch]$UI
    )
    
    if (-not (Test-Path -Path "kubeflow")) {
        Write-ColorOutput "Red" "Kubeflow implementation not found."
        return
    }
    
    Write-ColorOutput "Yellow" "Kubeflow typically runs on Kubernetes and requires additional setup."
    Write-ColorOutput "Yellow" "Please refer to the kubeflow/README.md for detailed instructions."
}

function Run-Mage {
    param (
        [switch]$UI
    )
    
    if (-not (Test-Path -Path "mage")) {
        Write-ColorOutput "Red" "Mage implementation not found."
        return
    }
    
    if ($UI) {
        Write-ColorOutput "Cyan" "Starting Mage UI..."
        Set-Location mage
        Start-Process -FilePath "python" -ArgumentList "-m mage start" -NoNewWindow
        Start-Process "http://localhost:6789"
        Set-Location ..
    } else {
        Write-ColorOutput "Cyan" "Running Mage pipeline..."
        python mage/run_pipeline.py
    }
}

# Main script execution
Show-Header

# Check Python is available
if (-not (Check-Python)) {
    exit 1
}

# Check and install basic requirements
Check-Requirements

# Update configuration if needed
Update-Config -Tool $Tool

# Run the selected tool
switch ($Tool.ToLower()) {
    "base" {
        Run-BaseProject -Query $Query
    }
    "airflow" {
        Run-Airflow -UI:$UI
    }
    "dagster" {
        Run-Dagster -UI:$UI
    }
    "flyte" {
        Run-Flyte -UI:$UI
    }
    "kubeflow" {
        Run-Kubeflow -UI:$UI
    }
    "mage" {
        Run-Mage -UI:$UI
    }
    default {
        Write-ColorOutput "Red" "Unknown tool: $Tool"
        Show-Usage
        exit 1
    }
}

Write-ColorOutput "Green" "`nExecution completed." 