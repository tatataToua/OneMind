<#
.SYNOPSIS
  Task runner for OneMind. Stands in for a Makefile, since `make` is not
  available on this machine.

.DESCRIPTION
  Tasks invoke the virtualenv interpreter directly rather than going through
  `uv run`. See the "Windows Smart App Control" note below - `uv run` cannot
  work on a machine with Smart App Control enabled, because uv launches the
  interpreter through a small trampoline executable it generates locally, and
  Windows blocks binaries it has never seen before.

  uv is still used for dependency resolution and installation. Only venv
  creation and process launching moved to stock CPython.

.EXAMPLE
  ./run.ps1 setup      # install everything and pull the model
  ./run.ps1 demo       # start API and UI together
  ./run.ps1 test       # offline test suite
  ./run.ps1 eval       # routing evaluation (needs Ollama)
  ./run.ps1 conv       # multi-turn eval: memory + two-wave dispatch
  ./run.ps1 compare    # router vs monolith baseline (needs Ollama)
  ./run.ps1 check      # pre-demo readiness check
  ./run.ps1 deploy     # build and ship to Cloud Run (needs gcloud)
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'api', 'ui', 'demo', 'test', 'eval', 'conv', 'compare', 'fixtures', 'lint', 'check', 'deploy')]
    [string]$Task = 'check'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Model = 'qwen3.5:4b'
$VenvPy = Join-Path $Backend '.venv\Scripts\python.exe'

# --- interpreter discovery -------------------------------------------------
#
# Windows Smart App Control (and WDAC in general) refuses to execute binaries
# that are neither signed nor known to Microsoft's reputation service. A
# python-build-standalone interpreter - what `uv python install` fetches - is
# unsigned, and its bundled libcrypto/libssl DLLs get blocked, so `import ssl`
# fails even when the interpreter itself starts. The python.org installer is
# Authenticode-signed by the Python Software Foundation and is not affected.
#
# So: require a signed 3.12. Failing that, warn loudly rather than build an
# environment that dies on the first HTTPS call.

function Resolve-BasePython {
    $candidates = @()
    try {
        $viaLauncher = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
        if ($LASTEXITCODE -eq 0 -and $viaLauncher) { $candidates += $viaLauncher.Trim() }
    } catch { }
    $candidates += "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe"
    $candidates += "C:\Python312\python.exe"

    foreach ($c in $candidates) {
        if (-not (Test-Path $c)) { continue }
        $sig = (Get-AuthenticodeSignature $c).Status
        if ($sig -eq 'Valid') { return $c }
        Write-Warning "$c is not signed ($sig); Smart App Control may block it."
    }

    throw @"
No signed Python 3.12 found.

Install the official build from python.org (Authenticode-signed, works under
Smart App Control):

  https://www.python.org/downloads/release/python-31210/

Do NOT use ``uv python install`` on this machine - those interpreters are
unsigned and Windows blocks their SSL libraries.
"@
}

function Resolve-Uv {
    foreach ($c in @(
            "$env:APPDATA\Python\Python314\Scripts\uv.exe",
            "$env:LOCALAPPDATA\Programs\uv\uv.exe",
            "$env:USERPROFILE\.local\bin\uv.exe")) {
        if (Test-Path $c) { return $c }
    }
    $cmd = Get-Command uv -ErrorAction SilentlyContinue
    if ($cmd) { return $cmd.Source }
    throw "uv not found. Install with: py -3 -m pip install uv"
}

function Invoke-Py {
    <#  Run a command with the venv interpreter, from the backend directory. #>
    param([string[]]$PyArgs)
    if (-not (Test-Path $VenvPy)) { throw "no virtualenv; run ./run.ps1 setup" }
    Push-Location $Backend
    try {
        & $VenvPy @PyArgs
        if ($LASTEXITCODE -ne 0) { throw "failed: python $($PyArgs -join ' ')" }
    } finally { Pop-Location }
}

switch ($Task) {

    'setup' {
        $base = Resolve-BasePython
        $uv = Resolve-Uv
        Write-Host "==> base interpreter: $base" -ForegroundColor Cyan

        # Stock `venv`, not `uv venv`: uv writes a locally-generated trampoline
        # as .venv\Scripts\python.exe, which Smart App Control blocks.
        Write-Host '==> creating virtualenv' -ForegroundColor Cyan
        if (Test-Path (Join-Path $Backend '.venv')) {
            Remove-Item (Join-Path $Backend '.venv') -Recurse -Force
        }
        & $base -m venv (Join-Path $Backend '.venv')
        if ($LASTEXITCODE -ne 0) { throw 'venv creation failed' }

        Write-Host '==> installing pinned dependencies from uv.lock' -ForegroundColor Cyan
        $req = Join-Path $env:TEMP 'onemind-requirements.txt'
        Push-Location $Backend
        try {
            & $uv export --frozen --extra dev --no-hashes --no-emit-project -o $req
            if ($LASTEXITCODE -ne 0) { throw 'uv export failed' }
            & $uv pip install --python $VenvPy -r $req
            if ($LASTEXITCODE -ne 0) { throw 'dependency install failed' }
            & $uv pip install --python $VenvPy -e . --no-deps
            if ($LASTEXITCODE -ne 0) { throw 'project install failed' }
        } finally { Pop-Location; Remove-Item $req -ErrorAction SilentlyContinue }

        Write-Host '==> generating synthetic fixtures' -ForegroundColor Cyan
        Invoke-Py @('fixtures/generate.py')

        Write-Host "==> pulling $Model" -ForegroundColor Cyan
        & ollama pull $Model

        Write-Host '==> installing frontend dependencies' -ForegroundColor Cyan
        Push-Location $Frontend
        try { & npm install } finally { Pop-Location }

        Write-Host ''
        Write-Host 'Setup complete. Run: ./run.ps1 demo' -ForegroundColor Green
    }

    'api' {
        Invoke-Py @('-m', 'uvicorn', 'onemind.api.main:app', '--host', '127.0.0.1', '--port', '8080', '--reload', '--reload-include', '*.json')
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
            '-NoExit', '-Command', "Set-Location '$Backend'; & '$VenvPy' -m uvicorn onemind.api.main:app --host 127.0.0.1 --port 8080 --reload --reload-include *.json"
        )

        Write-Host '==> warming the model' -ForegroundColor Cyan
        try { & ollama run $Model 'ok' | Out-Null } catch { Write-Warning 'could not warm the model' }

        Write-Host '==> starting UI on :5173' -ForegroundColor Cyan
        Push-Location $Frontend
        try { & npm run dev } finally { Pop-Location }
    }

    # Ship to Cloud Run. Source-based: Cloud Build builds the Dockerfile, so
    # Docker is not needed locally - which matters on the Smart App Control box
    # this file already works around elsewhere.
    'deploy' {
        # gcloud writes its "Your active configuration is: [...]" banner and all
        # build progress to stderr. Under the $ErrorActionPreference = 'Stop'
        # set at the top of this file, PowerShell 5.1 wraps any native stderr
        # line in a NativeCommandError and throws - so a completely successful
        # deploy dies on its own status message. Exit codes are checked
        # explicitly in this block instead, which is the stronger signal anyway.
        $ErrorActionPreference = 'Continue'

        $Service = 'onemind'
        # us-central1 is not arbitrary: Cloud Run's always-free allowance only
        # applies in a handful of US regions, and this is one of them.
        $Region = if ($env:ONEMIND_GCP_REGION) { $env:ONEMIND_GCP_REGION } else { 'us-central1' }
        $SecretName = 'onemind-groq-key'

        # `--verbosity=error` silences the configuration banner; without it the
        # banner lands in $project and every later --project flag is malformed.
        $project = (& gcloud config get-value project --verbosity=error 2>$null | Select-Object -Last 1)
        if (-not $project -or $project -eq '(unset)') {
            throw 'No gcloud project set. Run: gcloud init'
        }
        $project = $project.ToString().Trim()
        Write-Host "==> project $project / region $Region" -ForegroundColor Cyan

        # The key lives in Secret Manager rather than in --set-env-vars, so it
        # is not recorded in the service's revision history or in shell history.
        $key = $env:ONEMIND_GROQ_API_KEY
        if (-not $key) {
            $envFile = Join-Path $Backend '.env'
            if (Test-Path $envFile) {
                $line = Select-String -Path $envFile -Pattern '^ONEMIND_GROQ_API_KEY=(.+)$'
                if ($line) { $key = $line.Matches[0].Groups[1].Value.Trim() }
            }
        }
        if (-not $key) { throw "No Groq key. Set ONEMIND_GROQ_API_KEY or put it in backend/.env" }

        $exists = (& gcloud secrets describe $SecretName --project $project 2>$null)
        if (-not $exists) {
            Write-Host "==> creating secret $SecretName" -ForegroundColor Cyan
            & gcloud secrets create $SecretName --replication-policy=automatic --project $project
            if ($LASTEXITCODE -ne 0) { throw 'secret creation failed' }
        }

        Write-Host '==> storing key version' -ForegroundColor Cyan
        $tmp = Join-Path $env:TEMP 'onemind-groq-key.txt'
        try {
            # -NoNewline: a trailing newline becomes part of the secret and the
            # Authorization header then carries it, which fails as a 401 that
            # looks nothing like its cause.
            Set-Content -Path $tmp -Value $key -NoNewline -Encoding ascii
            & gcloud secrets versions add $SecretName --data-file=$tmp --project $project
            if ($LASTEXITCODE -ne 0) { throw 'storing the key failed' }
        } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }

        # Cloud Run runs as the default compute service account, which cannot
        # read secrets until told it may.
        $number = (& gcloud projects describe $project --format='value(projectNumber)' --verbosity=error | Select-Object -Last 1).ToString().Trim()
        $sa = "$number-compute@developer.gserviceaccount.com"
        Write-Host "==> granting $sa access to $SecretName" -ForegroundColor Cyan
        & gcloud secrets add-iam-policy-binding $SecretName `
            --member="serviceAccount:$sa" --role='roles/secretmanager.secretAccessor' `
            --project $project 2>&1 | Out-Null

        # Source-based deploys upload a zip to a staging bucket and have Cloud
        # Build read it back. On projects created after mid-2024 the default
        # compute service account is not granted that access, and the failure
        # arrives as a 403 on `storage.objects.get` for a bucket you never named
        # - which reads like a bug rather than a missing role.
        Write-Host '==> granting the build service account its role' -ForegroundColor Cyan
        & gcloud projects add-iam-policy-binding $project `
            --member="serviceAccount:$sa" --role='roles/cloudbuild.builds.builder' `
            --condition=None --verbosity=error 2>&1 | Out-Null

        Write-Host '==> building and deploying' -ForegroundColor Cyan
        & gcloud run deploy $Service `
            --source . `
            --region $Region `
            --project $project `
            --allow-unauthenticated `
            --port 8080 `
            --memory 1Gi `
            --timeout 300 `
            --min-instances 0 `
            --max-instances 1 `
            --set-env-vars 'ONEMIND_LLM_PROVIDER=groq' `
            --set-secrets "ONEMIND_GROQ_API_KEY=${SecretName}:latest"
        if ($LASTEXITCODE -ne 0) { throw 'deploy failed' }

        $url = (& gcloud run services describe $Service --region $Region --project $project --format='value(status.url)' --verbosity=error | Select-Object -Last 1).ToString().Trim()
        Write-Host ''
        Write-Host "  deployed: $url" -ForegroundColor Green
        Write-Host "  health:   $url/api/health" -ForegroundColor Green
        Write-Host ''
        Write-Host '  min-instances is 0, so the first hit pays a cold start.' -ForegroundColor Yellow
        Write-Host '  Load the URL once before demoing.' -ForegroundColor Yellow
        Write-Host ''
    }

    'test' { Invoke-Py @('-m', 'pytest', '-q') }

    'eval' { Invoke-Py @('../evals/run_eval.py', '--json', '../evals/report.json') }

    # Both architectures on the same dataset, scored by the same function.
    # Slow - 102 cases x3 attempts x2 arms - and the only run that answers
    # "did the orchestrator need to exist?".
    'compare' {
        Invoke-Py @('../evals/run_eval.py', '--arm', 'both', '--repeat', '3',
            '--json', '../evals/comparison_report.json')
    }

    # Multi-turn: session memory and two-wave dispatch. Separate from 'eval'
    # because it is stateful and slower - six conversations, run in sequence.
    'conv' {
        Invoke-Py @('../evals/conversations.py', '--json', '../evals/conversations_report.json')
    }

    'fixtures' { Invoke-Py @('fixtures/generate.py') }

    'lint' {
        Invoke-Py @('-m', 'ruff', 'check', 'src', 'tests', '../evals')
        Invoke-Py @('-m', 'ruff', 'format', '--check', 'src', 'tests', '../evals')
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

        # Python env.
        #
        # Presence is not enough. Under Smart App Control an interpreter can sit
        # on disk and still refuse to launch, and `ssl` can fail to import even
        # when it does - both of which surface mid-demo rather than here unless
        # this check actually executes something.
        if (-not (Test-Path $VenvPy)) {
            Report 'python env' $false 'no .venv; run ./run.ps1 setup'
            $ok = $false
        } else {
            try {
                $probe = & $VenvPy -c "import ssl, onemind; print('ok')" 2>&1
                if ($LASTEXITCODE -eq 0 -and $probe -match 'ok') {
                    Report 'python env' $true '.venv runs, ssl + onemind import'
                } else {
                    Report 'python env' $false "$probe"
                    $ok = $false
                }
            } catch {
                Report 'python env' $false "interpreter blocked: $($_.Exception.Message)"
                $ok = $false
            }
        }

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
