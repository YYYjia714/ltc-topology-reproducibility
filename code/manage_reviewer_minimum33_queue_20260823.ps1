param([switch]$StatusOnly)

$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$OutputRoot = Join-Path $Project 'runs\formal_reviewer_minimum33_revision_20260823'
$PlanPath = Join-Path $OutputRoot 'formal_reviewer_minimum33_plan.json'
$AuthorizationPath = Join-Path $OutputRoot 'formal_start_authorization.json'
$Runner = Join-Path $Project 'run_reviewer_minimum33_stage_20260823.ps1'
$PostRunner = Join-Path $Project 'run_reviewer_minimum33_postanalysis_20260823.ps1'
$StateDir = Join-Path $OutputRoot 'stage_state'
$QueueStatePath = Join-Path $OutputRoot 'queue_state.json'
$PostStatePath = Join-Path $OutputRoot 'postanalysis_state.json'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-Payload($Payload, [int]$ExitCode = 0) {
    $Temporary = "$QueueStatePath.tmp"
    $Payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $QueueStatePath -Force
    $Payload | ConvertTo-Json -Depth 20
    exit $ExitCode
}

function Test-StageComplete($Stage) {
    $ExitPath = Join-Path $StateDir "$($Stage.id)_exit.json"
    if (-not (Test-Path -LiteralPath $ExitPath -PathType Leaf)) { return $false }
    try { $ExitState = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json } catch { return $false }
    if ($ExitState.status -ne 'completed' -or [int]$ExitState.exit_code -ne 0) { return $false }
    if ([string]$ExitState.plan_sha256 -ne $PlanHash) { return $false }
    $RunDir = Join-Path $OutputRoot ([string]$Stage.output_rel -replace '/', '\')
    foreach ($Name in @($Stage.required_outputs)) {
        $Path = Join-Path $RunDir ([string]$Name)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        $Artifact = @($ExitState.artifacts | Where-Object { $_.name -eq [string]$Name }) | Select-Object -First 1
        if ($null -eq $Artifact) { return $false }
        if ([string]$Artifact.sha256 -ne (Get-Sha256 $Path)) { return $false }
        if ([int64]$Artifact.size -ne (Get-Item -LiteralPath $Path).Length) { return $false }
    }
    return $true
}

if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) { throw "Missing formal plan: $PlanPath" }
$Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$PlanHash = Get-Sha256 $PlanPath
$Stages = @($Plan.stages)
if ($Stages.Count -ne 33) { throw "Formal plan must contain exactly 33 training stages, got $($Stages.Count)" }
New-Item -ItemType Directory -Force -Path $OutputRoot, $StateDir | Out-Null

$Completed = @($Stages | Where-Object { Test-StageComplete $_ })
$Next = $Stages | Where-Object { -not (Test-StageComplete $_) } | Select-Object -First 1

foreach ($Stage in $Stages) {
    $ExitPath = Join-Path $StateDir "$($Stage.id)_exit.json"
    if (Test-Path -LiteralPath $ExitPath -PathType Leaf) {
        try { $ExitState = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json } catch { continue }
        if ($ExitState.status -eq 'failed' -and -not (Test-StageComplete $Stage)) {
            Write-Payload ([ordered]@{
                status = 'FAILED'
                current_stage = [string]$Stage.id
                completed_training_stages = $Completed.Count
                total_training_stages = 33
                error = [string]$ExitState.error
                exit_state = $ExitState
                updated_utc = (Get-Date).ToUniversalTime().ToString('o')
            }) 1
        }
    }
}

$Active = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like '*E:\AMASS_LNN_Project*' -and $_.CommandLine -match '(train|evaluate_reviewer|benchmark_reviewer|plot_reviewer|audit_reviewer).*\.py') -or
    ($_.Name -match '^powershell(\.exe)?$' -and $_.CommandLine -match 'run_reviewer_minimum33_stage_20260823\.ps1') -or
    ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like '*E:\AMASS_LNN_Project*' -and $_.CommandLine -match '(evaluate|benchmark|plot|audit)_reviewer_minimum33.*20260823\.py') -or
    ($_.Name -match '^powershell(\.exe)?$' -and $_.CommandLine -match 'run_reviewer_minimum33_postanalysis_20260823\.ps1')
} | Select-Object -First 1
if ($null -ne $Active) {
    $IsAnalysis = $Active.CommandLine -match '(evaluate|benchmark|plot|audit)_reviewer_minimum|run_reviewer_minimum33_postanalysis'
    $ProgressFiles = Get-ChildItem (Join-Path $OutputRoot 'progress') -Filter '*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    $Progress = if ($ProgressFiles) {
        try { Get-Content -LiteralPath $ProgressFiles[0].FullName -Raw | ConvertFrom-Json } catch { $null }
    } else { $null }
    Write-Payload ([ordered]@{
        status = if ($IsAnalysis) { 'ANALYSIS_RUNNING' } else { 'RUNNING' }
        process_id = $Active.ProcessId
        completed_training_stages = $Completed.Count
        total_training_stages = 33
        current_stage = if ($IsAnalysis) {
            if (Test-Path -LiteralPath $PostStatePath) { [string](Get-Content -LiteralPath $PostStatePath -Raw | ConvertFrom-Json).phase } else { 'postanalysis' }
        } elseif ($null -eq $Next) { $null } else { [string]$Next.id }
        progress_file = if ($ProgressFiles) { $ProgressFiles[0].FullName } else { $null }
        progress = $Progress
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    })
}

if ($null -eq $Next) {
    $AuditPass = Join-Path $OutputRoot 'audit\reviewer_minimum_audit_pass.json'
    if (Test-Path -LiteralPath $PostStatePath -PathType Leaf) {
        $PostState = Get-Content -LiteralPath $PostStatePath -Raw | ConvertFrom-Json
        if ($PostState.status -eq 'failed') {
            Write-Payload ([ordered]@{
                status = 'FAILED'
                current_stage = [string]$PostState.phase
                completed_training_stages = 33
                total_training_stages = 33
                error = [string]$PostState.error
                log_file = [string]$PostState.log_file
                updated_utc = (Get-Date).ToUniversalTime().ToString('o')
            }) 1
        }
        if ($PostState.status -eq 'completed' -and
            (Test-Path -LiteralPath $AuditPass -PathType Leaf) -and
            [string]$PostState.audit_pass_sha256 -eq (Get-Sha256 $AuditPass)) {
            Write-Payload ([ordered]@{
                status = 'ALL_DONE'
                completed_training_stages = 33
                total_training_stages = 33
                post_training_completed = @($Plan.post_training_analyses)
                audit_pass_path = $AuditPass
                audit_pass_sha256 = Get-Sha256 $AuditPass
                updated_utc = (Get-Date).ToUniversalTime().ToString('o')
            })
        }
    }
    if ($StatusOnly) {
        Write-Payload ([ordered]@{
            status = 'ANALYSIS_READY'
            completed_training_stages = 33
            total_training_stages = 33
            post_training_required = @($Plan.post_training_analyses)
            updated_utc = (Get-Date).ToUniversalTime().ToString('o')
        })
    }
    $PostArguments = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $PostRunner + '"'))
    $PostProcess = Start-Process -FilePath 'powershell.exe' -ArgumentList $PostArguments -WindowStyle Hidden -PassThru
    Write-Payload ([ordered]@{
        status = 'ANALYSIS_STARTED'
        process_id = $PostProcess.Id
        current_stage = 'sequence_evaluation'
        completed_training_stages = 33
        total_training_stages = 33
        state_file = $PostStatePath
        log_file = Join-Path $OutputRoot 'logs\postanalysis.log'
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    })
}

$Authorized = $false
$AuthorizationReason = 'Missing authorization file.'
if (Test-Path -LiteralPath $AuthorizationPath -PathType Leaf) {
    $Authorization = Get-Content -LiteralPath $AuthorizationPath -Raw | ConvertFrom-Json
    try {
        $Authorized = [bool]$Authorization.approved -and
            [string]$Authorization.plan_sha256 -eq $PlanHash -and
            (Test-Path -LiteralPath ([string]$Authorization.review_path) -PathType Leaf) -and
            [string]$Authorization.review_sha256 -eq (Get-Sha256 ([string]$Authorization.review_path)) -and
            (Test-Path -LiteralPath ([string]$Authorization.inventory_path) -PathType Leaf) -and
            [string]$Authorization.inventory_sha256 -eq (Get-Sha256 ([string]$Authorization.inventory_path)) -and
            (Test-Path -LiteralPath ([string]$Authorization.smoke_report_path) -PathType Leaf) -and
            [string]$Authorization.smoke_report_sha256 -eq (Get-Sha256 ([string]$Authorization.smoke_report_path)) -and
            (Test-Path -LiteralPath ([string]$Authorization.preflight_path) -PathType Leaf) -and
            [string]$Authorization.preflight_sha256 -eq (Get-Sha256 ([string]$Authorization.preflight_path))
        if ($Authorized) {
            $Review = Get-Content -LiteralPath ([string]$Authorization.review_path) -Raw | ConvertFrom-Json
            $Inventory = Get-Content -LiteralPath ([string]$Authorization.inventory_path) -Raw | ConvertFrom-Json
            $SmokeReport = Get-Content -LiteralPath ([string]$Authorization.smoke_report_path) -Raw | ConvertFrom-Json
            $Preflight = Get-Content -LiteralPath ([string]$Authorization.preflight_path) -Raw | ConvertFrom-Json
            $Authorized = [string]$Review.status -eq 'PASS' -and [string]$Review.plan_sha256 -eq $PlanHash -and
                [string]$Inventory.plan_sha256 -eq $PlanHash -and
                [string]$SmokeReport.status -eq 'passed' -and [string]$SmokeReport.plan_sha256 -eq $PlanHash -and
                [string]$Preflight.status -eq 'PASS' -and [bool]$Preflight.formal_ready -and
                [string]$Preflight.plan_sha256 -eq $PlanHash
        }
        $AuthorizationReason = if ($Authorized) { 'Authorized' } else { 'Authorization bindings are incomplete or changed.' }
    }
    catch {
        $Authorized = $false
        $AuthorizationReason = $_.Exception.Message
    }
}
if (-not $Authorized) {
    Write-Payload ([ordered]@{
        status = 'LOCKED'
        reason = $AuthorizationReason
        next_stage = [string]$Next.id
        completed_training_stages = $Completed.Count
        total_training_stages = 33
        plan_path = $PlanPath
        plan_sha256 = $PlanHash
        authorization_path = $AuthorizationPath
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    })
}

if ($StatusOnly) {
    Write-Payload ([ordered]@{
        status = 'READY'
        next_stage = [string]$Next.id
        completed_training_stages = $Completed.Count
        total_training_stages = 33
        updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    })
}

$Arguments = @(
    '-NoProfile',
    '-ExecutionPolicy', 'Bypass',
    '-File', ('"' + $Runner + '"'),
    '-StageId', ([string]$Next.id)
)
$Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $Arguments -WindowStyle Hidden -PassThru
$RunDir = Join-Path $OutputRoot ([string]$Next.output_rel -replace '/', '\')
Write-Payload ([ordered]@{
    status = 'STARTED'
    process_id = $Process.Id
    current_stage = [string]$Next.id
    completed_training_stages = $Completed.Count
    total_training_stages = 33
    run_dir = $RunDir
    progress_file = Join-Path $OutputRoot "progress\$($Next.id).json"
    log_file = Join-Path $OutputRoot "logs\$($Next.id).log"
    started_utc = (Get-Date).ToUniversalTime().ToString('o')
})


