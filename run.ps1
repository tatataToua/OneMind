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
  ./run.ps1 ollama status   # is the local Ollama server up? (also: start, stop)
  ./run.ps1 deploy     # build and ship to Cloud Run (needs gcloud)
  ./run.ps1 tunnel     # point the hosted app at this machine's Ollama
#>
param(
    [Parameter(Position = 0)]
    [ValidateSet('setup', 'api', 'ui', 'demo', 'test', 'eval', 'conv', 'compare', 'fixtures', 'lint', 'check', 'deploy', 'tunnel', 'ollama')]
    [string]$Task = 'check',

    # Only read by the 'ollama' task; ignored by the rest.
    [Parameter(Position = 1)]
    [ValidateSet('start', 'stop', 'status')]
    [string]$Action = 'status'
)

$ErrorActionPreference = 'Stop'
$Root = $PSScriptRoot
$Backend = Join-Path $Root 'backend'
$Frontend = Join-Path $Root 'frontend'
$Model = 'qwen3.5:4b'
# Secret Manager names, shared by `deploy` and `tunnel`.
$KeySecret = 'onemind-groq-key'
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

function Deploy-Service {
    <#  Build from source and ship it, with whichever provider wiring the caller
        asked for.

        Both tasks go through here, because a source build is the only thing
        that guarantees the running image contains the code being demonstrated.
        `gcloud run services update` changes environment variables and nothing
        else, so a service pointed at a tunnel while still running an older
        image sends no bearer token and its own gateway refuses it. #>
    param(
        [Parameter(Mandatory)][string]$Service,
        [Parameter(Mandatory)][string]$Project,
        [Parameter(Mandatory)][string]$Region,
        [Parameter(Mandatory)][string]$ServiceAccount,
        [Parameter(Mandatory)][string]$EnvVars,
        [Parameter(Mandatory)][string]$Secrets,
        [Parameter(Mandatory)][string]$Root
    )

    # Source-based deploys upload a zip to a staging bucket and have Cloud Build
    # read it back. On projects created after mid-2024 the default compute
    # service account is not granted that access, and the failure arrives as a
    # 403 on `storage.objects.get` for a bucket you never named - which reads
    # like a bug rather than a missing role.
    Write-Host '==> granting the build service account its role' -ForegroundColor Cyan
    & gcloud projects add-iam-policy-binding $Project `
        --member="serviceAccount:$ServiceAccount" --role='roles/cloudbuild.builds.builder' `
        --condition=None --verbosity=error 2>&1 | Out-Null

    Write-Host '==> building and deploying' -ForegroundColor Cyan
    Push-Location $Root
    try {
        & gcloud run deploy $Service `
            --source . `
            --region $Region `
            --project $Project `
            --allow-unauthenticated `
            --port 8080 `
            --memory 1Gi `
            --timeout 300 `
            --min-instances 0 `
            --max-instances 1 `
            --set-env-vars $EnvVars `
            --set-secrets $Secrets
        if ($LASTEXITCODE -ne 0) { throw 'deploy failed' }
    } finally { Pop-Location }
}

function Get-GroqKey {
    <#  The Groq API key, from the environment or from backend/.env.

        Both tasks need it now: `deploy` runs on it, and `tunnel` wires it in
        behind Ollama as the fallback. It goes to Secret Manager rather than
        into --set-env-vars, so it stays out of the service's revision history
        and out of shell history. #>
    param([Parameter(Mandatory)][string]$Backend)

    if ($env:ONEMIND_GROQ_API_KEY) { return $env:ONEMIND_GROQ_API_KEY }

    $envFile = Join-Path $Backend '.env'
    if (Test-Path $envFile) {
        $line = Select-String -Path $envFile -Pattern '^ONEMIND_GROQ_API_KEY=(.+)$'
        if ($line) { return $line.Matches[0].Groups[1].Value.Trim() }
    }
    return ''
}

function Get-RuntimeServiceAccount {
    <#  Cloud Run runs as the project's default compute service account. Both
        the deploy and the tunnel have to name it to grant it a secret. #>
    param([Parameter(Mandatory)][string]$Project)
    $number = (& gcloud projects describe $Project --format='value(projectNumber)' --verbosity=error | Select-Object -Last 1).ToString().Trim()
    return "$number-compute@developer.gserviceaccount.com"
}

function Publish-Secret {
    <#  Store a value in Secret Manager and let Cloud Run read it.

        Used for the Groq key and for the tunnel token, for the same reason in
        both cases: a secret passed with --set-env-vars is recorded in the
        service's revision history and in shell history, and neither forgets. #>
    param(
        [Parameter(Mandatory)][string]$Name,
        [Parameter(Mandatory)][string]$Value,
        [Parameter(Mandatory)][string]$Project,
        [Parameter(Mandatory)][string]$ServiceAccount
    )

    $exists = (& gcloud secrets describe $Name --project $Project 2>$null)
    if (-not $exists) {
        Write-Host "==> creating secret $Name" -ForegroundColor Cyan
        & gcloud secrets create $Name --replication-policy=automatic --project $Project 2>$null | Out-Null
        if ($LASTEXITCODE -ne 0) {
            # A non-zero exit is not proof the secret is missing. gcloud retries
            # internally, so a create whose first attempt lands slowly gets
            # ALREADY_EXISTS on its own retry and reports the failure it just
            # caused. Ask what is true rather than trusting the exit code - the
            # state this wants is "the secret exists", and it does either way.
            if (-not (& gcloud secrets describe $Name --project $Project 2>$null)) {
                throw "creating $Name failed"
            }
        }
    }

    Write-Host "==> storing a version of $Name" -ForegroundColor Cyan
    $tmp = Join-Path $env:TEMP "$Name.txt"
    try {
        # -NoNewline: a trailing newline becomes part of the secret and the
        # Authorization header then carries it, which fails as a 401 that looks
        # nothing like its cause.
        Set-Content -Path $tmp -Value $Value -NoNewline -Encoding ascii
        & gcloud secrets versions add $Name --data-file=$tmp --project $Project | Out-Null
        if ($LASTEXITCODE -ne 0) { throw "storing $Name failed" }
    } finally { Remove-Item $tmp -ErrorAction SilentlyContinue }

    Write-Host "==> granting $ServiceAccount access to $Name" -ForegroundColor Cyan
    & gcloud secrets add-iam-policy-binding $Name `
        --member="serviceAccount:$ServiceAccount" --role='roles/secretmanager.secretAccessor' `
        --project $Project 2>&1 | Out-Null
}

function Get-OrCreateTunnelToken {
    <#  The bearer token the two ends of the tunnel share.

        Generated once and kept in backend/.env, which is gitignored. Stable
        across demos on purpose: the hostname changes every run, the secret does
        not, so Cloud Run only ever has to be told the new address. #>
    param([Parameter(Mandatory)][string]$EnvFile)

    if (Test-Path $EnvFile) {
        $line = Select-String -Path $EnvFile -Pattern '^ONEMIND_OLLAMA_AUTH_TOKEN=(.+)$'
        if ($line) { return $line.Matches[0].Groups[1].Value.Trim() }
    }

    # 32 bytes from the OS CSPRNG, hex encoded. Nothing derives this, so its
    # only job is to be long enough that finding the URL buys nothing.
    $bytes = New-Object byte[] 32
    ([System.Security.Cryptography.RandomNumberGenerator]::Create()).GetBytes($bytes)
    $token = -join ($bytes | ForEach-Object { $_.ToString('x2') })

    $existing = ''
    if (Test-Path $EnvFile) { $existing = Get-Content -Raw -Path $EnvFile }
    if ($existing -and -not $existing.EndsWith("`n")) { $existing += "`r`n" }
    $existing += "ONEMIND_OLLAMA_AUTH_TOKEN=$token`r`n"
    Set-Content -Path $EnvFile -Value $existing -NoNewline -Encoding ascii

    Write-Host '==> generated ONEMIND_OLLAMA_AUTH_TOKEN into backend/.env' -ForegroundColor Cyan
    return $token
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

        # `--verbosity=error` silences the configuration banner; without it the
        # banner lands in $project and every later --project flag is malformed.
        $project = (& gcloud config get-value project --verbosity=error 2>$null | Select-Object -Last 1)
        if (-not $project -or $project -eq '(unset)') {
            throw 'No gcloud project set. Run: gcloud init'
        }
        $project = $project.ToString().Trim()
        Write-Host "==> project $project / region $Region" -ForegroundColor Cyan

        # Ollama first here too, with nothing for it to find: inside the
        # container `ollama_host` is loopback, so the connection is refused at
        # once and Groq answers. That keeps one wiring for both tasks - the only
        # difference between a hosted demo and a borrowed-GPU one is whether
        # `ONEMIND_OLLAMA_HOST` points anywhere.
        $key = Get-GroqKey -Backend $Backend
        if (-not $key) { throw "No Groq key. Set ONEMIND_GROQ_API_KEY or put it in backend/.env" }

        $sa = Get-RuntimeServiceAccount -Project $project
        Publish-Secret -Name $KeySecret -Value $key -Project $project -ServiceAccount $sa

        Deploy-Service -Service $Service -Project $project -Region $Region -ServiceAccount $sa `
            -EnvVars 'ONEMIND_LLM_PROVIDER=ollama,ONEMIND_LLM_FALLBACK=groq' `
            -Secrets "ONEMIND_GROQ_API_KEY=${KeySecret}:latest" -Root $Root

        $url = (& gcloud run services describe $Service --region $Region --project $project --format='value(status.url)' --verbosity=error | Select-Object -Last 1).ToString().Trim()
        Write-Host ''
        Write-Host "  deployed: $url" -ForegroundColor Green
        Write-Host "  health:   $url/api/health" -ForegroundColor Green
        Write-Host ''
        Write-Host '  min-instances is 0, so the first hit pays a cold start.' -ForegroundColor Yellow
        Write-Host '  Load the URL once before demoing.' -ForegroundColor Yellow
        Write-Host ''
    }

    # Point the hosted service at this laptop's GPU for the length of a demo.
    #
    # Cloud Run has no free GPU, so the deployed build normally runs on Groq
    # (decision 25) and lives inside a free tier's token budget. During a live
    # walkthrough that budget is the wrong constraint, and the machine giving
    # the demo already has the weights resident. This opens a tunnel and
    # repoints the running service at it, so the hosted link runs the same
    # qwen3.5:4b the eval numbers were measured on, with no per-minute cap.
    #
    # What is exposed is `llm/gateway.py`, never Ollama: two routes, behind a
    # bearer token. Ollama stays on loopback.
    #
    # Reverting is `./run.ps1 deploy`, which uses --set-env-vars and so drops
    # every variable this task added along with the tunnel binding.
    'tunnel' {
        # gcloud and cloudflared both log to stderr; see the note in 'deploy'.
        $ErrorActionPreference = 'Continue'

        $Service = 'onemind'
        $Region = if ($env:ONEMIND_GCP_REGION) { $env:ONEMIND_GCP_REGION } else { 'us-central1' }
        $TokenSecret = 'onemind-ollama-token'
        $EnvFile = Join-Path $Backend '.env'
        $GatewayPort = 11435

        # --- the local half --------------------------------------------------

        try {
            $null = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/version' -TimeoutSec 4
        } catch {
            throw 'Ollama is not answering on :11434. Run: ./run.ps1 ollama start'
        }

        $tags = Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/tags' -TimeoutSec 6
        if (-not ($tags.models | Where-Object { $_.name -eq $Model })) {
            throw "$Model is not pulled. Run: ollama pull $Model"
        }

        $cf = Get-Command cloudflared -ErrorAction SilentlyContinue
        if (-not $cf) {
            throw 'cloudflared not found. Install it once: winget install --id Cloudflare.cloudflared'
        }

        $token = Get-OrCreateTunnelToken -EnvFile $EnvFile
        $headers = @{ Authorization = "Bearer $token" }

        # A cold load is this system's worst failure mode, and a tunnel does not
        # make it faster. keep_alive travels in the request rather than in the
        # environment, because the server is usually already running by now and
        # would never see a variable set here.
        Write-Host '==> warming the model' -ForegroundColor Cyan
        $warm = @{
            model      = $Model
            messages   = @(@{ role = 'user'; content = 'ok' })
            stream     = $false
            keep_alive = -1
        } | ConvertTo-Json -Depth 4
        try {
            Invoke-RestMethod -Uri 'http://127.0.0.1:11434/api/chat' -Method Post `
                -Body $warm -ContentType 'application/json' -TimeoutSec 240 | Out-Null
        } catch { Write-Warning 'could not warm the model; the first hosted turn will pay for it' }

        Write-Host "==> starting the gateway on :$GatewayPort" -ForegroundColor Cyan
        Start-Process -FilePath 'powershell' -ArgumentList @(
            '-NoExit', '-Command', "Set-Location '$Backend'; & '$VenvPy' -m onemind.llm.gateway"
        )

        $gatewayUp = $false
        foreach ($attempt in 1..20) {
            try {
                $null = Invoke-RestMethod -Uri "http://127.0.0.1:$GatewayPort/api/version" `
                    -Headers $headers -TimeoutSec 3
                $gatewayUp = $true
                break
            } catch { Start-Sleep -Milliseconds 500 }
        }
        if (-not $gatewayUp) { throw "the gateway did not answer on :$GatewayPort" }

        # --- the tunnel ------------------------------------------------------
        #
        # A quick tunnel: no Cloudflare account, no domain, and a fresh hostname
        # every run - which is why this task ends by telling Cloud Run the new
        # one rather than assuming a stable address.

        $log = Join-Path $env:TEMP 'onemind-cloudflared.log'
        Remove-Item $log -ErrorAction SilentlyContinue
        Write-Host '==> opening the tunnel' -ForegroundColor Cyan
        Start-Process -FilePath $cf.Source -WindowStyle Minimized -RedirectStandardError $log `
            -ArgumentList @('tunnel', '--no-autoupdate', '--url', "http://127.0.0.1:$GatewayPort")

        $public = ''
        foreach ($attempt in 1..40) {
            Start-Sleep -Milliseconds 500
            if (-not (Test-Path $log)) { continue }
            $hit = $null
            try {
                $hit = Select-String -Path $log -Pattern 'https://[a-z0-9-]+[.]trycloudflare[.]com' |
                    Select-Object -First 1
            } catch { continue }   # cloudflared still holds the file open
            if ($hit) { $public = $hit.Matches[0].Value; break }
        }
        if (-not $public) { throw "cloudflared reported no URL. See $log" }

        # --- prove both halves before handing the address to the internet ----

        Write-Host "==> verifying $public" -ForegroundColor Cyan

        # A fresh quick-tunnel hostname takes a few seconds to resolve, so this
        # is retried rather than raced. The first version of this check ran once
        # with $ErrorActionPreference set to Continue, which meant a DNS failure
        # printed an error and carried on to publish a tunnel it had never
        # reached.
        $version = $null
        foreach ($attempt in 1..20) {
            try {
                $version = Invoke-RestMethod -Uri "$public/api/version" -Headers $headers `
                    -TimeoutSec 20 -ErrorAction Stop
                break
            } catch { Start-Sleep -Seconds 2 }
        }
        if (-not $version) { throw "no answer from $public; the tunnel is not carrying traffic" }
        Write-Host "    ollama $($version.version) answered through the tunnel" -ForegroundColor Green

        # The safety argument for a public URL is that it is useless without the
        # token, so the refusal has to be observed rather than assumed. A network
        # error is not a refusal - counting one as proof is how an open tunnel
        # gets published on a hostname that simply had not resolved yet.
        $status = 0
        try {
            Invoke-WebRequest -Uri "$public/api/version" -TimeoutSec 20 -UseBasicParsing `
                -ErrorAction Stop | Out-Null
            throw 'the tunnel answered an unauthenticated request; refusing to publish it'
        } catch [System.Net.WebException] {
            if ($_.Exception.Response) { $status = [int]$_.Exception.Response.StatusCode }
        }
        if ($status -ne 401) {
            throw "an unauthenticated request returned $status rather than 401; refusing to publish"
        }
        Write-Host '    an unauthenticated request is refused with 401' -ForegroundColor Green

        # --- the hosted half -------------------------------------------------

        $project = (& gcloud config get-value project --verbosity=error 2>$null | Select-Object -Last 1)
        if (-not $project -or $project -eq '(unset)') { throw 'No gcloud project set. Run: gcloud init' }
        $project = $project.ToString().Trim()

        $sa = Get-RuntimeServiceAccount -Project $project
        Publish-Secret -Name $TokenSecret -Value $token -Project $project -ServiceAccount $sa

        # A source build, not `gcloud run services update`. Updating environment
        # variables leaves the image alone, and an image built before the
        # provider learned to send a bearer token gets a 401 from its own
        # gateway - which is what happened the first time this ran.
        #
        # The Groq key rides along rather than being dropped. It is the fallback
        # now (decision 27), so a tunnel that dies mid-demo degrades to the
        # hosted model instead of to an error. Published here as well as in
        # `deploy`, because `tunnel` may be the first thing anyone runs.
        $secrets = "ONEMIND_OLLAMA_AUTH_TOKEN=${TokenSecret}:latest"
        $key = Get-GroqKey -Backend $Backend
        if ($key) {
            Publish-Secret -Name $KeySecret -Value $key -Project $project -ServiceAccount $sa
            $secrets = "$secrets,ONEMIND_GROQ_API_KEY=${KeySecret}:latest"
        } else {
            Write-Warning 'no Groq key found - deploying with no fallback, so a dead tunnel is a dead demo'
        }

        Deploy-Service -Service $Service -Project $project -Region $Region -ServiceAccount $sa `
            -EnvVars "ONEMIND_LLM_PROVIDER=ollama,ONEMIND_LLM_FALLBACK=groq,ONEMIND_OLLAMA_HOST=$public" `
            -Secrets $secrets -Root $Root
        $url = (& gcloud run services describe $Service --region $Region --project $project --format='value(status.url)' --verbosity=error | Select-Object -Last 1).ToString().Trim()

        # One real turn, hosted service to this machine and back. /api/health
        # only reports which provider was *selected*; it does not prove the
        # container can reach the tunnel, so a wrong image or a stale token
        # would still read as healthy. This is the check that fails here rather
        # than in front of an audience.
        Write-Host '==> asking the hosted service a real question' -ForegroundColor Cyan
        $probe = @{ message = 'Which patients are enrolled in remote monitoring?' } | ConvertTo-Json
        $answer = Invoke-WebRequest -Uri "$url/api/chat/stream" -Method Post -Body $probe `
            -ContentType 'application/json' -TimeoutSec 300 -UseBasicParsing
        if ($answer.Content -match 'event: error') {
            $detail = ($answer.Content -split "`n" | Select-String -Pattern '"message"' |
                Select-Object -First 1)
            throw "the hosted service could not complete a turn: $detail"
        }
        Write-Host '    a full turn completed through the tunnel' -ForegroundColor Green

        # A completed turn is no longer proof on its own. With Groq wired in
        # behind Ollama (decision 27), a dead tunnel produces a perfectly good
        # answer from the wrong machine - which is the failure this whole task
        # exists to make impossible. Health reports the provider that actually
        # answered, so it is the assertion that matters.
        $live = (Invoke-RestMethod "$url/api/health" -TimeoutSec 120).provider
        if ($live -ne 'ollama') {
            throw "the turn completed, but on '$live' - the fallback answered, so the tunnel is not carrying the demo"
        }
        Write-Host '    and ollama answered it, not the fallback' -ForegroundColor Green

        Write-Host ''
        Write-Host "  hosted:  $url" -ForegroundColor Green
        Write-Host "  health:  $url/api/health   (expect provider=ollama)" -ForegroundColor Green
        Write-Host "  tunnel:  $public" -ForegroundColor Green
        Write-Host ''
        Write-Host '  Leave the gateway window and the cloudflared window open.' -ForegroundColor Yellow
        Write-Host '  Closing either ends the demo, and the URL is not reusable.' -ForegroundColor Yellow
        Write-Host '  Fan-out is only real if Ollama was started with' -ForegroundColor Yellow
        Write-Host '  OLLAMA_NUM_PARALLEL=4; otherwise four specialists queue.' -ForegroundColor Yellow
        Write-Host '  Put Groq back with: ./run.ps1 deploy' -ForegroundColor Yellow
        Write-Host ''
    }

    # Start, stop, or check the local Ollama server - the thing that has to be
    # listening on :11434 before `demo`, `eval`, `compare`, or `tunnel` will
    # work. On Windows the server is owned by a tray app ("ollama app.exe"),
    # not by the CLI: `ollama pull` and `ollama run` talk to the server but do
    # not reliably start it, which is the "timed out waiting for server to
    # start" you hit when nothing is serving :11434 yet.
    'ollama' {
        $OllamaBase = 'http://127.0.0.1:11434'

        function Test-OllamaUp {
            try { return Invoke-RestMethod -Uri "$OllamaBase/api/version" -TimeoutSec 3 }
            catch { return $null }
        }

        switch ($Action) {

            'status' {
                $v = Test-OllamaUp
                if (-not $v) {
                    Write-Host 'ollama: not running' -ForegroundColor Yellow
                    Write-Host '  start it with: ./run.ps1 ollama start'
                    break
                }
                $pulled = try { (Invoke-RestMethod "$OllamaBase/api/tags" -TimeoutSec 5).models } catch { @() }
                $loaded = try { (Invoke-RestMethod "$OllamaBase/api/ps" -TimeoutSec 5).models } catch { @() }
                Write-Host "ollama: running (v$($v.version))" -ForegroundColor Green
                Write-Host "  models pulled:  $($pulled.Count)   ($Model $(if ($pulled.name -contains $Model) { 'present' } else { 'MISSING - run ollama pull ' + $Model }))"
                if ($loaded) {
                    Write-Host "  loaded in VRAM: $($loaded.name -join ', ')  - warm, no cold start" -ForegroundColor Green
                } else {
                    Write-Host '  loaded in VRAM: none - the first request pays a cold load'
                }
            }

            'start' {
                $up = Test-OllamaUp
                if ($up) {
                    Write-Host "ollama: already running (v$($up.version))" -ForegroundColor Green
                    break
                }

                # Launch the same tray app the Start menu does - it starts the
                # server as a child and keeps it alive. Its path sits next to
                # the CLI, so derive it rather than hard-coding an install dir.
                $cli = (Get-Command ollama -ErrorAction Stop).Source
                $app = Join-Path (Split-Path $cli) 'ollama app.exe'
                if (-not (Test-Path $app)) { throw "cannot find 'ollama app.exe' next to $cli" }

                Write-Host "==> launching $app" -ForegroundColor Cyan
                Start-Process -FilePath $app

                foreach ($attempt in 1..30) {
                    Start-Sleep -Milliseconds 500
                    $up = Test-OllamaUp
                    if ($up) { break }
                }
                if (-not $up) { throw 'ollama was launched but did not answer on :11434 within 15s' }
                Write-Host "ollama: running (v$($up.version))" -ForegroundColor Green
                Write-Host "  warm the model with: ollama run $Model ok"
            }

            'stop' {
                # The tray app owns the serving process, so stopping it stops
                # both; a bare `ollama serve` (the debug way) is caught too.
                $procs = Get-Process -Name 'ollama app', 'ollama' -ErrorAction SilentlyContinue
                if (-not $procs) {
                    Write-Host 'ollama: not running' -ForegroundColor Green
                    break
                }
                $procs | Stop-Process -Force
                Write-Host "ollama: stopped ($($procs.Count) process$(if ($procs.Count -eq 1) { '' } else { 'es' }))" -ForegroundColor Green
            }
        }
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
        } catch { Report 'ollama' $false 'not on :11434 - run ./run.ps1 ollama start'; $ok = $false }

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
