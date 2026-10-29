# Butterfly launcher for Windows.
#   .\start.ps1                 run with Ollama + llama3.1
#   .\start.ps1 -Model qwen2.5:14b
#   .\start.ps1 -Demo           run without a model
#   .\start.ps1 -Judge          let the model grade its own replies
#   .\start.ps1 -Rebuild        force reinstall and UI rebuild
param(
    [string]$Model = "",
    [switch]$Demo,
    [switch]$Judge,
    [switch]$Rebuild,
    [int]$Port = 8000
)

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

function Need($cmd, $hint) {
    if (-not (Get-Command $cmd -ErrorAction SilentlyContinue)) {
        Write-Host "Missing $cmd. $hint" -ForegroundColor Red
        exit 1
    }
}

Need "python" "Install Python 3.10+ from python.org and tick 'Add to PATH'."
Need "npm" "Install Node.js 18+ from nodejs.org."

$venvPy = Join-Path $PSScriptRoot ".venv\Scripts\python.exe"
$marker = Join-Path $PSScriptRoot ".venv\.installed"

if (-not (Test-Path $venvPy)) {
    Write-Host "Creating virtual environment..." -ForegroundColor Cyan
    python -m venv .venv
}

if ($Rebuild -or -not (Test-Path $marker)) {
    Write-Host "Installing Python packages..." -ForegroundColor Cyan
    & $venvPy -m pip install --quiet --upgrade pip
    & $venvPy -m pip install --quiet -e ".[dev]"
    New-Item -ItemType File -Path $marker -Force | Out-Null
}

if ($Rebuild -or -not (Test-Path "frontend\dist\index.html")) {
    Write-Host "Building the UI (first run only)..." -ForegroundColor Cyan
    Push-Location frontend
    npm install --no-audit --no-fund
    npm run build
    Pop-Location
}

$runArgs = @("-m", "butterfly", "--port", "$Port")
if ($Demo) { $runArgs += "--demo" }
elseif (-not (Get-Command "ollama" -ErrorAction SilentlyContinue) -and -not $env:BUTTERFLY_BACKEND) {
    Write-Host "Ollama isn't installed, so starting in demo mode. Install it from ollama.com for real replies." -ForegroundColor Yellow
    $runArgs += "--demo"
}
if ($Model) { $runArgs += @("--model", $Model) }
if ($Judge) { $runArgs += @("--evaluator", "llm") }

& $venvPy @runArgs
