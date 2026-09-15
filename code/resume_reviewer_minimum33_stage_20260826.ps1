param(
    [Parameter(Mandatory = $true)]
    [string]$StageId,
    [switch]$ValidateOnly
)

$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$OutputRoot = Join-Path $Project 'runs\formal_reviewer_minimum33_revision_20260823'
$PlanPath = Join-Path $OutputRoot 'formal_reviewer_minimum33_plan.json'
$AuthorizationPath = Join-Path $OutputRoot 'formal_start_authorization.json'
$Runner = Join-Path $Project 'run_reviewer_minimum33_stage_20260823.ps1'
$StateDir = Join-Path $OutputRoot 'stage_state'
$RecoveryStatePath = Join-Path $StateDir "$StageId`_recovery.json"

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic([string]$Path, $Payload) {
    $Temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Test-StageComplete($Stage, [string]$PlanHash) {
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

foreach ($RequiredPath in @($PlanPath, $AuthorizationPath, $Runner)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) { throw "Missing required file: $RequiredPath" }
}
$Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$PlanHash = Get-Sha256 $PlanPath
$Stages = @($Plan.stages)
if ($Stages.Count -ne 33) { throw "Formal plan must contain 33 stages, got $($Stages.Count)" }
$Stage = @($Stages | Where-Object { $_.id -eq $StageId }) | Select-Object -First 1
if ($null -eq $Stage) { throw "Unknown stage: $StageId" }

$Authorization = Get-Content -LiteralPath $AuthorizationPath -Raw | ConvertFrom-Json
if (-not [bool]$Authorization.approved -or [string]$Authorization.plan_sha256 -ne $PlanHash) {
    throw 'Formal authorization is missing or does not bind the frozen plan.'
}
foreach ($Binding in @(
    @([string]$Authorization.review_path, [string]$Authorization.review_sha256, 'review'),
    @([string]$Authorization.inventory_path, [string]$Authorization.inventory_sha256, 'inventory'),
    @([string]$Authorization.smoke_report_path, [string]$Authorization.smoke_report_sha256, 'smoke report'),
    @([string]$Authorization.preflight_path, [string]$Authorization.preflight_sha256, 'preflight')
)) {
    if (-not (Test-Path -LiteralPath $Binding[0] -PathType Leaf)) { throw "Missing authorized $($Binding[2]): $($Binding[0])" }
    if ([string]$Binding[1] -ne (Get-Sha256 $Binding[0])) { throw "Authorized $($Binding[2]) hash changed." }
}
$Review = Get-Content -LiteralPath ([string]$Authorization.review_path) -Raw | ConvertFrom-Json
$Inventory = Get-Content -LiteralPath ([string]$Authorization.inventory_path) -Raw | ConvertFrom-Json
$Smoke = Get-Content -LiteralPath ([string]$Authorization.smoke_report_path) -Raw | ConvertFrom-Json
$Preflight = Get-Content -LiteralPath ([string]$Authorization.preflight_path) -Raw | ConvertFrom-Json
if ([string]$Review.status -ne 'PASS' -or [string]$Review.plan_sha256 -ne $PlanHash) { throw 'Recovery code review is not approved.' }
if ([string]$Inventory.plan_sha256 -ne $PlanHash) { throw 'Inventory does not bind the frozen plan.' }
if ([string]$Smoke.status -ne 'passed' -or [string]$Smoke.plan_sha256 -ne $PlanHash) { throw 'Recovery smoke test is not approved.' }
if ([string]$Preflight.status -ne 'PASS' -or -not [bool]$Preflight.formal_ready -or
    [string]$Preflight.plan_sha256 -ne $PlanHash) { throw 'Formal preflight is not ready.' }
foreach ($Item in @($Inventory.files)) {
    if (-not (Test-Path -LiteralPath ([string]$Item.path) -PathType Leaf)) { throw "Reviewed source missing: $($Item.path)" }
    if ([string]$Item.sha256 -ne (Get-Sha256 ([string]$Item.path))) { throw "Reviewed source changed: $($Item.path)" }
}

$Completed = @($Stages | Where-Object { Test-StageComplete $_ $PlanHash })
$Next = $Stages | Where-Object { -not (Test-StageComplete $_ $PlanHash) } | Select-Object -First 1
if ($null -eq $Next -or [string]$Next.id -ne $StageId) {
    throw "Requested recovery stage is not the next incomplete stage: requested=$StageId, next=$([string]$Next.id)"
}
$ExitPath = Join-Path $StateDir "$StageId`_exit.json"
if (-not (Test-Path -LiteralPath $ExitPath -PathType Leaf)) { throw "Missing failed exit state: $ExitPath" }
$FailedExit = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json
if ([string]$FailedExit.status -ne 'failed' -or [int]$FailedExit.exit_code -eq 0) {
    throw "Stage exit state is not a failure: $ExitPath"
}
$RunDir = Join-Path $OutputRoot ([string]$Stage.output_rel -replace '/', '\')
$CheckpointPath = Join-Path $RunDir 'last.pt'
$HistoryPath = Join-Path $RunDir 'train_history.json'
foreach ($RequiredPath in @($CheckpointPath, $HistoryPath)) {
    if (-not (Test-Path -LiteralPath $RequiredPath -PathType Leaf)) { throw "Missing recovery artifact: $RequiredPath" }
}
$Active = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like '*E:\AMASS_LNN_Project*' -and $_.CommandLine -match '(train|evaluate|benchmark|plot|audit).*\.py') -or
    ($_.Name -match '^powershell(\.exe)?$' -and $_.CommandLine -match 'run_reviewer_minimum33_(stage|postanalysis)_20260823\.ps1')
} | Select-Object -First 1
if ($null -ne $Active) { throw "Another formal process is active: PID $($Active.ProcessId)" }

$ReadyPayload = [ordered]@{
    status = 'RECOVERY_READY'
    stage_id = $StageId
    completed_training_stages = $Completed.Count
    total_training_stages = 33
    checkpoint_path = $CheckpointPath
    checkpoint_sha256 = Get-Sha256 $CheckpointPath
    failed_exit_path = $ExitPath
    failed_exit_sha256 = Get-Sha256 $ExitPath
    plan_sha256 = $PlanHash
    validated_utc = (Get-Date).ToUniversalTime().ToString('o')
}
if ($ValidateOnly) {
    $ReadyPayload | ConvertTo-Json -Depth 20
    exit 0
}

$ArchiveDir = Join-Path $StateDir 'recovery_archive'
New-Item -ItemType Directory -Force -Path $ArchiveDir | Out-Null
$ArchivePath = Join-Path $ArchiveDir "$StageId`_failed_$((Get-Date).ToString('yyyyMMdd_HHmmss')).json"
Move-Item -LiteralPath $ExitPath -Destination $ArchivePath
if ((Get-Sha256 $ArchivePath) -ne [string]$ReadyPayload.failed_exit_sha256) {
    Copy-Item -LiteralPath $ArchivePath -Destination $ExitPath
    throw 'Archived failure state hash verification failed.'
}

$StdoutPath = Join-Path $OutputRoot "logs\$StageId`_recovery_runner_stdout.log"
$StderrPath = Join-Path $OutputRoot "logs\$StageId`_recovery_runner_stderr.log"
$Arguments = @(
    '-NoProfile', '-ExecutionPolicy', 'Bypass',
    '-File', ('"' + $Runner + '"'),
    '-StageId', $StageId,
    '-Resume'
)
$Process = Start-Process -FilePath 'powershell.exe' -ArgumentList $Arguments -WorkingDirectory $Project `
    -WindowStyle Hidden -RedirectStandardOutput $StdoutPath -RedirectStandardError $StderrPath -PassThru
Start-Sleep -Seconds 5
if ($Process.HasExited) {
    if (-not (Test-Path -LiteralPath $ExitPath -PathType Leaf)) {
        Copy-Item -LiteralPath $ArchivePath -Destination $ExitPath
    }
    $FailurePayload = $ReadyPayload.Clone()
    $FailurePayload.status = 'RECOVERY_LAUNCH_FAILED'
    $FailurePayload.process_id = $Process.Id
    $FailurePayload.archive_path = $ArchivePath
    $FailurePayload.stdout_path = $StdoutPath
    $FailurePayload.stderr_path = $StderrPath
    $FailurePayload.updated_utc = (Get-Date).ToUniversalTime().ToString('o')
    Write-JsonAtomic $RecoveryStatePath $FailurePayload
    throw "Recovery process exited during startup; inspect $StderrPath"
}

$StartedPayload = $ReadyPayload.Clone()
$StartedPayload.status = 'RECOVERY_STARTED'
$StartedPayload.process_id = $Process.Id
$StartedPayload.archive_path = $ArchivePath
$StartedPayload.stdout_path = $StdoutPath
$StartedPayload.stderr_path = $StderrPath
$StartedPayload.updated_utc = (Get-Date).ToUniversalTime().ToString('o')
Write-JsonAtomic $RecoveryStatePath $StartedPayload
$StartedPayload | ConvertTo-Json -Depth 20
