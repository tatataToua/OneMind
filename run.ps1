<#
.SYNOPSIS
  Task runner for OneMind. Stands in for a Makefile, since `make` is not
  available on this machine.

.EXAMPLE
  ./run.ps1 setup      # install everything and pull the model
  ./run.ps1 demo       # start API and UI together
  ./run.ps1 test       # offline test suite
  ./run.ps1 eval       # routing evaluation (needs Ollama)
  ./run.ps1 check      # pre-demo readiness check
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'api', 'ui', 'demo', 'test', 'eval', 'fixtures', 'lint', 'check')]
    [string]$Task = 'check'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Model = 'qwen3.5:4b'

function Invoke-Uv {
    param([string[]]$UvArgs)
    Push-Location $Backend
    try { & python -m uv @UvArgs; if ($LASTEXITCODE -ne 0) { throw "uv failed: $($UvArgs -join ' ')" } }
    finally { Pop-Location }
}

switch ($Task) {

    'setup' {
        Write-Host '==> installing uv' -ForegroundColor Cyan
        & python -m pip install --quiet --disable-pip-version-check uv

        Write-Host '==> syncing python dependencies (3.12)' -ForegroundColor Cyan
        Invoke-Uv @('python', 'install', '3.12')
        Invoke-Uv @('sync', '--extra', 'dev')

        Write-Host '==> generating synthetic fixtures' -ForegroundColor Cyan
        Invoke-Uv @('run', 'python', 'fixtures/generate.py')

        Write-Host "==> pulling $Model" -ForegroundColor Cyan
        & ollama pull $Model

        Write-Host '==> installing frontend dependencies' -ForegroundColor Cyan
        Push-Location $Frontend
        try { & npm install } finally { Pop-Location }

        Write-Host ''
        Write-Host 'Setup complete. Run: ./run.ps1 demo' -ForegroundColor Green
    }

    'api' {
        Invoke-Uv @('run', 'uvicorn', 'onemind.api.main:app', '--host', '127.0.0.1', '--port', '8080', '--reload')
    }

    'ui' {
        Push-Location $Frontend
        try { & npm run dev } finally { Pop-Location }
    }

    'demo' {
        # Keeping the model resident matters more than it sounds: a cold load
        # mid-demo is the worst failure mode this system has.
        $env:OLLAMA_KEEP_ALIVE = '-1'
        $env:OLLAMA_NUM_PARALLEL = '4'

        Write-Host '==> starting API on :8080' -ForegroundColor Cyan
        Start-Process -FilePath 'powershell' -ArgumentList @(
            '-NoExit', '-Command', "Set-Location '$Backend'; python -m uv run uvicorn onemind.api.main:app --host 127.0.0.1 --port 8080"
        )

        Write-Host '==> warming the model' -ForegroundColor Cyan
        try { & ollama run $Model 'ok' | Out-Null } catch { Write-Warning 'could not warm the model' }

        Write-Host '==> starting UI on :5173' -ForegroundColor Cyan
        Push-Location $Frontend
        try { & npm run dev } finally { Pop-Location }
    }

    'test' { Invoke-Uv @('run', 'pytest', '-q') }

    'eval' { Invoke-Uv @('run', 'python', '../evals/run_eval.py', '--json', '../evals/report.json') }

    'fixtures' { Invoke-Uv @('run', 'python', 'fixtures/generate.py') }

    'lint' {
        Invoke-Uv @('run', 'ruff', 'check', 'src', 'tests')
        Invoke-Uv @('run', 'ruff', 'format', '--check', 'src', 'tests')
    }

    'check' {
        Write-Host ''
        Write-Host 'OneMind pre-demo check' -ForegroundColor Cyan
        Write-Host ('-' * 46)

        $ok = $true

        function Report {
            param([string]$Label, [bool]$Pass, [string]$Detail)
            $mark = if ($Pass) { 'PASS' } else { 'FAIL' }
            $colour = if ($Pass) { 'Green' } else { 'Red' }
            Write-Host ("  {0,-24} " -f $Label) -NoNewline
            Write-Host $mark -ForegroundColor $colour -NoNewline
            if ($Detail) { Write-Host "  $Detail" } else { Write-Host '' }
        }

        # Ollama reachable
        $ollamaUp = $false
        try {
            $v = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 4
            $ollamaUp = $true
            Report 'ollama' $true "v$($v.version)"
        } catch { Report 'ollama' $false 'not reachable on :11434'; $ok = $false }

        # Model present
        if ($ollamaUp) {
            $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 6
            $has = $tags.models | Where-Object { $_.name -eq $Model }
            Report 'model' ([bool]$has) $Model
            if (-not $has) { $ok = $false }
        }

        # GPU headroom.
        #
        # Low free VRAM is only a problem if the model is NOT already loaded.
        # Once it is resident, low free VRAM is the desired state - it means the
        # weights are on the card and the first demo prompt will not pay a cold
        # load. So check residency first and interpret the number accordingly.
        $resident = $false
        if ($ollamaUp) {
            try {
                $ps = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/ps' -TimeoutSec 5
                $resident = [bool]($ps.models | Where-Object { $_.name -eq $Model })
            } catch { }
        }

        try {
            $smi = & nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits 2>$null
            if ($smi) {
                $free = [int]($smi | Select-Object -First 1).ToString().Trim()
                if ($resident) {
                    Report 'gpu' $true "$Model resident, $free MiB free - warm, no cold start"
                } else {
                    $enough = $free -gt 4500
                    Report 'gpu' $enough "$free MiB free, model not loaded (needs ~4.5 GB)"
                    if (-not $enough) { $ok = $false }
                }
            }
        } catch { Report 'gpu' $false 'nvidia-smi unavailable' }

        # Fixtures
        $fx = Join-Path $Backend 'fixtures/patients.json'
        Report 'fixtures' (Test-Path $fx) $(if (Test-Path $fx) { 'generated' } else { 'run ./run.ps1 fixtures' })
        if (-not (Test-Path $fx)) { $ok = $false }

        # Python env
        $venv = Join-Path $Backend '.venv'
        Report 'python env' (Test-Path $venv) $(if (Test-Path $venv) { '.venv present' } else { 'run ./run.ps1 setup' })
        if (-not (Test-Path $venv)) { $ok = $false }

        # Frontend deps
        $nm = Join-Path $Frontend 'node_modules'
        Report 'frontend deps' (Test-Path $nm) $(if (Test-Path $nm) { 'installed' } else { 'run npm install' })
        if (-not (Test-Path $nm)) { $ok = $false }

        Write-Host ('-' * 46)
        if ($ok) {
            Write-Host '  ready. run ./run.ps1 demo' -ForegroundColor Green
        } else {
            Write-Host '  not ready. run ./run.ps1 setup' -ForegroundColor Yellow
        }
        Write-Host ''
    }
}
