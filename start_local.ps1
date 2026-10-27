[CmdletBinding()]
param(
    [string]$ModelsPath,
    [ValidateNotNullOrEmpty()]
    [string]$Model = 'qwen2.5:3b',
    [ValidateRange(1024, 65535)]
    [int]$Port = 8501
)

$ErrorActionPreference = 'Stop'
$baseUrl = 'http://127.0.0.1:11434'
$pythonPath = Join-Path $PSScriptRoot '.venv\Scripts\python.exe'
$environmentNames = @('OLLAMA_MODELS', 'OLLAMA_HOST', 'RAG_BACKEND', 'MODEL_ID', 'OLLAMA_BASE_URL')
$savedEnvironment = @{}
foreach ($name in $environmentNames) {
    $savedEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, 'Process')
}

function Read-OllamaModels {
    try {
        return Invoke-RestMethod -Uri "$baseUrl/api/tags" -TimeoutSec 2
    } catch {
        return $null
    }
}

Push-Location -LiteralPath $PSScriptRoot
try {
    if (-not (Test-Path -LiteralPath $pythonPath -PathType Leaf)) {
        throw 'Project Python is missing: create .venv and install requirements.txt first.'
    }
    & $pythonPath -c "import streamlit, sklearn, fastembed, jieba, rank_bm25"
    if ($LASTEXITCODE -ne 0) {
        throw 'Project dependencies are missing. Run .\.venv\Scripts\python.exe -m pip install -r requirements.txt first.'
    }
    if ($ModelsPath) {
        if (-not (Test-Path -LiteralPath $ModelsPath -PathType Container)) {
            throw "Model directory does not exist: $ModelsPath"
        }
        $ModelsPath = (Resolve-Path -LiteralPath $ModelsPath).Path
    }

    $service = Read-OllamaModels
    if ($null -eq $service) {
        $ollamaCommand = Get-Command ollama -ErrorAction SilentlyContinue
        if ($null -eq $ollamaCommand) {
            throw 'Ollama is not installed or is not on PATH. Install/start Ollama, then retry.'
        }
        if ($ModelsPath) {
            $env:OLLAMA_MODELS = $ModelsPath
        }
        $env:OLLAMA_HOST = '127.0.0.1:11434'
        $logDirectory = Join-Path $PSScriptRoot 'runs'
        New-Item -ItemType Directory -Path $logDirectory -Force | Out-Null
        $stamp = Get-Date -Format 'yyyyMMdd-HHmmss'
        $stdoutPath = Join-Path $logDirectory "ollama-$stamp.stdout.log"
        $stderrPath = Join-Path $logDirectory "ollama-$stamp.stderr.log"
        $ollamaProcess = Start-Process -FilePath $ollamaCommand.Source -ArgumentList 'serve' `
            -WindowStyle Hidden -RedirectStandardOutput $stdoutPath `
            -RedirectStandardError $stderrPath -PassThru
        Write-Host "Starting local Ollama (PID $($ollamaProcess.Id))..."
        for ($attempt = 0; $attempt -lt 20; $attempt++) {
            $service = Read-OllamaModels
            if ($null -ne $service) { break }
            if ($ollamaProcess.HasExited) {
                throw "Ollama exited. Check $stderrPath"
            }
            Start-Sleep -Milliseconds 500
        }
        if ($null -eq $service) {
            throw "Ollama did not become ready. Check $stderrPath"
        }
    } else {
        Write-Host 'Using the existing Ollama service at 127.0.0.1:11434.'
        if ($ModelsPath) {
            Write-Host '-ModelsPath applies only when starting a new service; the existing service keeps its current model directory.'
        }
    }

    $installedModels = @($service.models | ForEach-Object { $_.name })
    if ($installedModels -notcontains $Model) {
        $available = $installedModels -join ', '
        throw "Model '$Model' is not available in the running service. Available: $available. Check the model directory and restart Ollama yourself if needed. This script does not download weights or stop an existing service."
    }

    $env:RAG_BACKEND = 'ollama'
    $env:MODEL_ID = $Model
    $env:OLLAMA_BASE_URL = $baseUrl
    Write-Host "Starting the web app with $Model. Open http://127.0.0.1:$Port"
    Write-Host 'Press Ctrl+C to stop the web app. Ollama remains available for command-line evaluations.'
    & $pythonPath -m streamlit run app.py --server.address=127.0.0.1 --server.port=$Port --browser.gatherUsageStats=false
    if ($LASTEXITCODE -ne 0) {
        throw "Streamlit exited with code $LASTEXITCODE. If port $Port is occupied, stop the old web app or specify -Port 8502."
    }
} finally {
    foreach ($name in $environmentNames) {
        [Environment]::SetEnvironmentVariable($name, $savedEnvironment[$name], 'Process')
    }
    Pop-Location
}
