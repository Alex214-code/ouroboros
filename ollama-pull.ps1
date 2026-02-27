# ollama-pull.ps1 — Download Ollama models with resume support
# Workaround for TLS handshake timeout / max retries exceeded
#
# Usage: powershell -ExecutionPolicy Bypass -File ollama-pull.ps1 gpt-oss:20b
# Or:    powershell -ExecutionPolicy Bypass -File ollama-pull.ps1 minicpm-v:8b
#
# Requires: curl (built into Windows 10+)
#
# This script downloads model blobs directly from Ollama registry
# using curl with resume (-C -), bypassing ollama pull bugs.

param(
    [Parameter(Mandatory=$true, Position=0)]
    [string]$ModelSpec
)

$ErrorActionPreference = "Stop"

# Parse name:tag
if ($ModelSpec -match '^(.+):(.+)$') {
    $name = $Matches[1]
    $tag = $Matches[2]
} else {
    $name = $ModelSpec
    $tag = "latest"
}

Write-Host "=== Ollama Pull (with resume) ===" -ForegroundColor Cyan
Write-Host "Model: $name`:$tag"

# Find Ollama models directory
$ollamaModels = $env:OLLAMA_MODELS
if (-not $ollamaModels) {
    $ollamaModels = Join-Path $env:USERPROFILE ".ollama\models"
}

if (-not (Test-Path $ollamaModels)) {
    Write-Error "Ollama models directory not found: $ollamaModels"
    exit 1
}

$blobsDir = Join-Path $ollamaModels "blobs"
$manifestsDir = Join-Path $ollamaModels "manifests"

if (-not (Test-Path $blobsDir)) {
    New-Item -ItemType Directory -Path $blobsDir -Force | Out-Null
}

# Manifest directory for this model
$manifestModelDir = Join-Path $manifestsDir "registry.ollama.ai\library\$name"
$manifestFile = Join-Path $manifestModelDir $tag

if (Test-Path $manifestFile) {
    Write-Host "Model $name`:$tag already exists in manifest. Re-downloading blobs if needed..." -ForegroundColor Yellow
}

if (-not (Test-Path $manifestModelDir)) {
    New-Item -ItemType Directory -Path $manifestModelDir -Force | Out-Null
}

# Fetch manifest from registry
Write-Host "`nFetching manifest..." -ForegroundColor Cyan
$registryBase = "https://registry.ollama.ai"
$manifestUrl = "$registryBase/v2/library/$name/manifests/$tag"

try {
    $manifestJson = Invoke-RestMethod -Uri $manifestUrl -Method Get -TimeoutSec 30
} catch {
    Write-Error "Failed to fetch manifest: $_"
    exit 1
}

# Convert back to JSON string for saving (compact, no BOM)
$manifestText = $manifestJson | ConvertTo-Json -Depth 10 -Compress

# Check for errors
if ($manifestJson.errors) {
    Write-Error "Registry error: $($manifestJson.errors | ConvertTo-Json)"
    exit 1
}

Write-Host "Manifest OK" -ForegroundColor Green

# Download config blob
$configDigest = $manifestJson.config.digest
if ($configDigest) {
    $blobFile = $configDigest -replace ':', '-'
    $blobPath = Join-Path $blobsDir $blobFile
    $blobUrl = "$registryBase/v2/library/$name/blobs/$configDigest"

    Write-Host "`n[config] $configDigest" -ForegroundColor Cyan
    if (Test-Path $blobPath) {
        Write-Host "  Already exists, checking..." -ForegroundColor Yellow
    }
    # curl with resume support
    & curl.exe -#L -C - -o $blobPath $blobUrl
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "Config download may have failed (exit code: $LASTEXITCODE), retrying..."
        & curl.exe -#L -C - -o $blobPath $blobUrl
    }
}

# Download layer blobs
$layers = $manifestJson.layers
$total = $layers.Count
$current = 0

foreach ($layer in $layers) {
    $current++
    $digest = $layer.digest
    $size = $layer.size
    $sizeMB = [math]::Round($size / 1MB, 1)
    $sizeGB = [math]::Round($size / 1GB, 2)

    $blobFile = $digest -replace ':', '-'
    $blobPath = Join-Path $blobsDir $blobFile
    $blobUrl = "$registryBase/v2/library/$name/blobs/$digest"

    if ($sizeGB -ge 1) {
        $sizeStr = "${sizeGB} GB"
    } else {
        $sizeStr = "${sizeMB} MB"
    }

    Write-Host "`n[$current/$total] $digest ($sizeStr)" -ForegroundColor Cyan

    # Check if already fully downloaded
    if (Test-Path $blobPath) {
        $existing = (Get-Item $blobPath).Length
        if ($existing -ge $size) {
            Write-Host "  Already complete, skipping." -ForegroundColor Green
            continue
        }
        $existMB = [math]::Round($existing / 1MB, 1)
        Write-Host "  Resuming from ${existMB} MB..." -ForegroundColor Yellow
    }

    # Download with resume — retry loop
    $maxRetries = 50
    $retry = 0
    while ($true) {
        & curl.exe -#L -C - -o $blobPath $blobUrl
        $exitCode = $LASTEXITCODE

        # Check if complete
        if ((Test-Path $blobPath) -and ((Get-Item $blobPath).Length -ge $size)) {
            Write-Host "  Done!" -ForegroundColor Green
            break
        }

        $retry++
        if ($retry -ge $maxRetries) {
            Write-Error "  Failed after $maxRetries retries for $digest"
            exit 1
        }

        if ((Test-Path $blobPath)) {
            $downloaded = [math]::Round((Get-Item $blobPath).Length / 1MB, 1)
            $pct = [math]::Round((Get-Item $blobPath).Length / $size * 100, 1)
            Write-Host "  Retry $retry/$maxRetries (${downloaded} MB / ${sizeStr}, ${pct}%)..." -ForegroundColor Yellow
        } else {
            Write-Host "  Retry $retry/$maxRetries..." -ForegroundColor Yellow
        }

        Start-Sleep -Seconds 2
    }
}

# Save manifest (UTF-8 without BOM — critical for Ollama!)
$utf8NoBom = [System.Text.UTF8Encoding]::new($false)
[System.IO.File]::WriteAllText($manifestFile, $manifestText, $utf8NoBom)
Write-Host "`n=== Model $name`:$tag downloaded successfully! ===" -ForegroundColor Green
Write-Host "Manifest saved to: $manifestFile"
Write-Host "`nRun: ollama run $name`:$tag"
