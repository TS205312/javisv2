[CmdletBinding()]
param(
    [string]$ModelName = "vosk-model-small-vn-0.4",
    [string]$Destination = "D:\AI_Brain\models\vosk-model-small-vn-0.4",
    [string]$Url = "https://alphacephei.com/vosk/models/vosk-model-small-vn-0.4.zip",
    [switch]$Force
)

$ErrorActionPreference = "Stop"
[Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12

function Test-VoskModel {
    param([Parameter(Mandatory)][string]$Path)
    return (Test-Path (Join-Path $Path "am") -PathType Container) -and
        (Test-Path (Join-Path $Path "conf") -PathType Container) -and
        (Test-Path (Join-Path $Path "graph") -PathType Container)
}

if (Test-VoskModel -Path $Destination) {
    Write-Host "Model already complete: $Destination"
    exit 0
}

if (Test-Path $Destination) {
    Write-Warning "Replacing incomplete destination: $Destination"
    Remove-Item -LiteralPath $Destination -Recurse -Force -ErrorAction Stop
}

$destinationPath = [IO.Path]::GetFullPath($Destination)
$parentDir = [IO.Path]::GetDirectoryName($destinationPath)
if (-not (Test-Path $parentDir -PathType Container)) {
    New-Item -ItemType Directory -Path $parentDir -Force | Out-Null
}

$workDir = Join-Path $parentDir (".{0}.{1}.download" -f $ModelName, [guid]::NewGuid().ToString("N"))
$archivePath = Join-Path $workDir ($ModelName + ".zip")
$extractDir = Join-Path $workDir "extract"
New-Item -ItemType Directory -Path $workDir -Force | Out-Null

try {
    Write-Progress -Activity "Downloading $ModelName" -Status "Starting download" -PercentComplete 5
    Invoke-WebRequest -Uri $Url -OutFile $archivePath -UseBasicParsing -ErrorAction Stop
    if (-not (Test-Path $archivePath -PathType Leaf) -or ((Get-Item $archivePath).Length -eq 0)) {
        throw "Downloaded archive is missing or empty."
    }

    Write-Progress -Activity "Downloading $ModelName" -Status "Extracting archive" -PercentComplete 70
    New-Item -ItemType Directory -Path $extractDir -Force | Out-Null
    Expand-Archive -LiteralPath $archivePath -DestinationPath $extractDir -Force -ErrorAction Stop

    $candidate = Join-Path $extractDir $ModelName
    if (-not (Test-Path $candidate -PathType Container)) {
        $directories = Get-ChildItem -LiteralPath $extractDir -Directory -Recurse |
            Where-Object { Test-VoskModel -Path $_.FullName } |
            Select-Object -First 1
        if ($null -eq $directories) {
            throw "Archive does not contain a complete Vosk model."
        }
        $candidate = $directories.FullName
    }

    if (-not (Test-VoskModel -Path $candidate)) {
        throw "Extracted model is incomplete."
    }

    Write-Progress -Activity "Downloading $ModelName" -Status "Installing model" -PercentComplete 90
    Move-Item -LiteralPath $candidate -Destination $destinationPath -ErrorAction Stop
    if (-not (Test-VoskModel -Path $destinationPath)) {
        throw "Installed model failed validation."
    }

    Write-Progress -Activity "Downloading $ModelName" -Status "Complete" -PercentComplete 100
    Write-Host "Installed $ModelName at $destinationPath"
}
finally {
    Write-Progress -Activity "Downloading $ModelName" -Status "Cleaning temporary files" -PercentComplete 100
    if (Test-Path $workDir -PathType Container) {
        Remove-Item -LiteralPath $workDir -Recurse -Force -ErrorAction SilentlyContinue
    }
    Start-Sleep -Milliseconds 100
    Write-Progress -Activity "Downloading $ModelName" -Completed
}
