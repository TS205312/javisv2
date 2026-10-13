[CmdletBinding()]
param(
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$CliArgs
)

$ErrorActionPreference = "Stop"
$ProjectRoot = Split-Path -Parent $PSScriptRoot
$VenvDir = Join-Path $ProjectRoot ".venv"
$VenvPython = Join-Path $VenvDir "Scripts\python.exe"
$ActivateScript = Join-Path $VenvDir "Scripts\Activate.ps1"

Set-Location -LiteralPath $ProjectRoot

if (-not (Test-Path $VenvPython -PathType Leaf)) {
    Write-Host "Virtual environment not found. Running setup..."
    & (Join-Path $PSScriptRoot "setup.ps1")
    if ($LASTEXITCODE -ne 0) {
        throw "Setup failed."
    }
}

$activated = $false
if (Test-Path $ActivateScript -PathType Leaf) {
    try {
        & $ActivateScript
        $activated = $true
    } catch {
        Write-Warning "Could not activate the virtual environment; using its Python executable directly."
    }
}

try {
    if ($CliArgs.Count -eq 0) {
        $CliArgs = @("voice")
    } elseif ($CliArgs[0] -notin @("chat", "voice", "vision", "ingest", "status", "doctor", "config")) {
        $CliArgs = @("voice") + $CliArgs
    }

    & $VenvPython -m src.cli.main @CliArgs
    exit $LASTEXITCODE
}
finally {
    if ($activated) {
        $deactivate = Get-Command deactivate -ErrorAction SilentlyContinue
        if ($null -ne $deactivate) {
            & $deactivate
        }
    }
}
