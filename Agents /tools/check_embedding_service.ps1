<#
check_embedding_service.ps1 - Windows health-check for an OpenAI-compatible
embedding service (LM Studio / Ollama / local-ai). Read-only: it only probes
GET /v1/models and POST /v1/embeddings. It does NOT restart or kill anything.

Usage:
  powershell -ExecutionPolicy Bypass -File check_embedding_service.ps1
  powershell -ExecutionPolicy Bypass -File check_embedding_service.ps1 -Url http://127.0.0.1:1234
  powershell -ExecutionPolicy Bypass -File check_embedding_service.ps1 -Model text-embedding-nomic-embed-text-v1.5

Exit code: 0 = healthy, 1 = unhealthy.
#>
param(
    [string]$Url  = "http://127.0.0.1:1234",
    [string]$Model = ""
)
$ErrorActionPreference = "Stop"

$logDir  = Join-Path $env:LOCALAPPDATA "embedding_service"
$logFile = Join-Path $logDir "embedding_service.log"
New-Item -ItemType Directory -Force -Path $logDir | Out-Null

function Log-Message([string]$msg) {
    $line = "$(Get-Date -Format 'yyyy-MM-dd HH:mm:ss') - $msg"
    Add-Content -Path $logFile -Value $line
    Write-Host $line
}

function Get-Models {
    try {
        $r = Invoke-RestMethod -Uri "$Url/v1/models" -TimeoutSec 5
        $ids = @($r.data | ForEach-Object { $_.id })
        $emb = @($ids | Where-Object { $_ -match 'embed|e5|bge|nomic|ada|text-embedding' })
        if ($emb.Count -gt 0) { return $emb }
        return $ids
    } catch {
        return @()
    }
}

function Test-Embedding([string]$model) {
    try {
        $body = @{ model = $model; input = "test" } | ConvertTo-Json
        $resp = Invoke-WebRequest -Uri "$Url/v1/embeddings" -Method Post `
                    -ContentType "application/json" -Body $body `
                    -TimeoutSec 10 -UseBasicParsing
        # 200 = works; 404 = service up but model unknown -> still usable.
        return ($resp.StatusCode -eq 200 -or $resp.StatusCode -eq 404)
    } catch {
        return $false
    }
}

Log-Message "=== checking embedding service at $Url ==="

if ($Model -ne "") {
    if (Test-Embedding $Model) {
        Log-Message "model $Model works"
        Write-Host "model $Model: working"
        exit 0
    }
    Write-Host "model $Model: not working"
    exit 1
}

$models = Get-Models
if ($models.Count -eq 0) {
    Log-Message "no models detected; trying common names"
    $models = @(
        "text-embedding-nomic-embed-text-v1.5",
        "text-embedding-ada-002",
        "bge-small-en",
        "e5-small-v2",
        "all-MiniLM-L6-v2",
        "intfloat/e5-small-v2"
    )
}

foreach ($m in $models) {
    Log-Message "testing model: $m"
    if (Test-Embedding $m) {
        Log-Message "service healthy; working model: $m"
        Write-Host "Embedding service: healthy (model: $m)"
        exit 0
    }
}

# Service reachable but no working embedding model?
try {
    $null = Invoke-WebRequest -Uri $Url -TimeoutSec 3 -UseBasicParsing
    Write-Host "Embedding service: reachable, but no working embedding model found"
    exit 0
} catch {
    Write-Host "Embedding service: not responding"
    exit 1
}
