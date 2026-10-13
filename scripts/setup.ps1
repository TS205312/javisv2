[CmdletBinding()]
param(
    [switch]$SkipVoskModel,
    [switch]$SkipPipUpgrade
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$BaseDrive = [IO.Path]::GetPathRoot("D:\AI_System")
$BrainDrive = [IO.Path]::GetPathRoot("D:\AI_Brain")
$SystemDirs = @(
    (Join-Path $BaseDrive "AI_System\cache"),
    (Join-Path $BaseDrive "AI_System\logs"),
    (Join-Path $BaseDrive "AI_System\audio"),
    (Join-Path $BaseDrive "AI_System\vision")
)
$BrainDirs = @(
    (Join-Path $BrainDrive "AI_Brain\models"),
    (Join-Path $BrainDrive "AI_Brain\faiss"),
    (Join-Path $BrainDrive "AI_Brain\huggingface")
)
$LegacyModelsDir = "D:\jarvis-data\models"
$VoskModelDir = Join-Path ($BrainDirs[0]) "vosk-model-small-vn-0.4"

function Resolve-Python {
    $python = Get-Command python.exe -ErrorAction SilentlyContinue
    if ($null -ne $python) {
        return $python.Source
    }

    $py = Get-Command py.exe -ErrorAction SilentlyContinue
    if ($null -ne $py) {
        return $py.Source
    }

    throw "Python 3.10 or newer was not found. Install Python and add it to PATH."
}

function Ensure-Directory {
    param([Parameter(Mandatory)][string]$Path)
    $item = New-Item -ItemType Directory -Path $Path -Force -ErrorAction Stop
    Write-Host "Created/verified: $($item.FullName)"
}

$pythonCommand = Resolve-Python
Write-Host "Using Python: $pythonCommand"

if (-not (Test-Path $VenvPython -PathType Leaf)) {
    Write-Host "Creating virtual environment at $VenvDir"
    & $pythonCommand -m venv $VenvDir
    if ($LASTEXITCODE -ne 0) {
        throw "Virtual environment creation failed."
    }
} else {
    Write-Host "Using virtual environment: $VenvDir"
}

if (-not $SkipPipUpgrade) {
    Write-Host "Upgrading pip, setuptools, and wheel"
    & $VenvPython -m pip install --upgrade pip setuptools wheel
    if ($LASTEXITCODE -ne 0) {
        throw "Dependency bootstrap failed."
    }
}

Write-Host "Installing CPU-only Python dependencies"
& $VenvPython -m pip install -r (Join-Path $ProjectRoot "requirements.txt")
if ($LASTEXITCODE -ne 0) {
    throw "Dependency installation failed."
}

foreach ($directory in ($SystemDirs + $BrainDirs)) {
    Ensure-Directory -Path $directory
}

if (Test-Path $LegacyModelsDir -PathType Container) {
    Write-Host "Copying existing models from $LegacyModelsDir"
    Get-ChildItem -LiteralPath $LegacyModelsDir -Force |
        Copy-Item -Destination ($BrainDirs[0]) -Recurse -Force -ErrorAction SilentlyContinue
} else {
    Write-Host "No legacy D:\jarvis-data\models directory to copy."
}

if (-not $SkipVoskModel) {
    $modelComplete = (Test-Path (Join-Path $VoskModelDir "am") -PathType Container) -and
        (Test-Path (Join-Path $VoskModelDir "conf") -PathType Container) -and
        (Test-Path (Join-Path $VoskModelDir "graph") -PathType Container)

    if ($modelComplete) {
        Write-Host "Vosk Vietnamese model is already present: $VoskModelDir"
    } else {
        Write-Host "Downloading the Vosk Vietnamese model"
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File `
            (Join-Path $PSScriptRoot "download_models.ps1") `
            -ModelName "vosk-model-small-vn-0.4" `
            -Destination $VoskModelDir `
            -Force
        if ($LASTEXITCODE -ne 0) {
            throw "Vosk model download failed."
        }
    }
}

Write-Host ""
Write-Host "Required Ollama models (run manually; setup does not pull them):"
Write-Host "  ollama pull qwen2.5:0.5b"
Write-Host "  ollama pull moondream:latest"
Write-Host ""
Write-Host "Run the voice assistant:"
Write-Host "  C:\javis\scripts\run_assistant.ps1"
