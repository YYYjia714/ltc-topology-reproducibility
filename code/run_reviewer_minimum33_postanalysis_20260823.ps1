$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$RunRoot = Join-Path $Project 'runs\formal_reviewer_minimum33_revision_20260823'
$PlanPath = Join-Path $RunRoot 'formal_reviewer_minimum33_plan.json'
$AuthorizationPath = Join-Path $RunRoot 'formal_start_authorization.json'
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$StatePath = Join-Path $RunRoot 'postanalysis_state.json'
$LogPath = Join-Path $RunRoot 'logs\postanalysis.log'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic([string]$Path, $Payload) {
    $Temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 30 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Test-StageComplete($Stage) {
    $ExitPath = Join-Path $RunRoot "stage_state\$($Stage.id)_exit.json"
    if (-not (Test-Path -LiteralPath $ExitPath -PathType Leaf)) { return $false }
    $ExitState = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json
    if ($ExitState.status -ne 'completed' -or [int]$ExitState.exit_code -ne 0 -or [bool]$ExitState.smoke) { return $false }
    if ([string]$ExitState.plan_sha256 -ne (Get-Sha256 $PlanPath)) { return $false }
    $RunDir = Join-Path $RunRoot ([string]$Stage.output_rel -replace '/', '\')
    foreach ($Name in @($Stage.required_outputs)) {
        $Path = Join-Path $RunDir ([string]$Name)
        $Record = @($ExitState.artifacts | Where-Object { $_.name -eq [string]$Name }) | Select-Object -First 1
        if ($null -eq $Record -or -not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        if ([string]$Record.sha256 -ne (Get-Sha256 $Path)) { return $false }
        if ([int64]$Record.size -ne (Get-Item -LiteralPath $Path).Length) { return $false }
    }
    return $true
}

if (-not (Test-Path -LiteralPath $Python -PathType Leaf)) { throw "Missing project Python: $Python" }
if (-not (Test-Path -LiteralPath $PlanPath -PathType Leaf)) { throw "Missing formal plan: $PlanPath" }
$Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
if (@($Plan.stages).Count -ne 33) { throw 'Post-analysis requires exactly 33 planned training stages.' }
if (-not (Test-Path -LiteralPath $AuthorizationPath -PathType Leaf)) { throw "Missing authorization: $AuthorizationPath" }
$Authorization = Get-Content -LiteralPath $AuthorizationPath -Raw | ConvertFrom-Json
if (-not [bool]$Authorization.approved -or [string]$Authorization.plan_sha256 -ne (Get-Sha256 $PlanPath)) {
    throw 'Post-analysis authorization is missing or does not match the frozen plan.'
}
foreach ($Binding in @(
    @([string]$Authorization.review_path, [string]$Authorization.review_sha256, 'review'),
    @([string]$Authorization.inventory_path, [string]$Authorization.inventory_sha256, 'inventory'),
    @([string]$Authorization.smoke_report_path, [string]$Authorization.smoke_report_sha256, 'smoke report'),
    @([string]$Authorization.preflight_path, [string]$Authorization.preflight_sha256, 'formal preflight')
)) {
    if (-not (Test-Path -LiteralPath $Binding[0] -PathType Leaf)) { throw "Missing authorized $($Binding[2]): $($Binding[0])" }
    if ([string]$Binding[1] -ne (Get-Sha256 ([string]$Binding[0]))) { throw "Authorized $($Binding[2]) hash changed." }
}
$Inventory = Get-Content -LiteralPath ([string]$Authorization.inventory_path) -Raw | ConvertFrom-Json
if ([string]$Inventory.plan_sha256 -ne (Get-Sha256 $PlanPath)) { throw 'Frozen inventory does not bind this plan.' }
$Review = Get-Content -LiteralPath ([string]$Authorization.review_path) -Raw | ConvertFrom-Json
if ([string]$Review.status -ne 'PASS' -or [string]$Review.plan_sha256 -ne (Get-Sha256 $PlanPath)) {
    throw 'Authorized code review did not pass this frozen plan.'
}
$SmokeReport = Get-Content -LiteralPath ([string]$Authorization.smoke_report_path) -Raw | ConvertFrom-Json
if ([string]$SmokeReport.status -ne 'passed' -or [string]$SmokeReport.plan_sha256 -ne (Get-Sha256 $PlanPath)) {
    throw 'Authorized smoke report did not pass this frozen plan.'
}
$Preflight = Get-Content -LiteralPath ([string]$Authorization.preflight_path) -Raw | ConvertFrom-Json
if ([string]$Preflight.status -ne 'PASS' -or -not [bool]$Preflight.formal_ready -or
    [string]$Preflight.plan_sha256 -ne (Get-Sha256 $PlanPath)) {
    throw 'Authorized formal preflight is not ready for this frozen plan.'
}
foreach ($Item in @($Inventory.files)) {
    if (-not (Test-Path -LiteralPath ([string]$Item.path) -PathType Leaf)) { throw "Reviewed source missing: $($Item.path)" }
    if ([string]$Item.sha256 -ne (Get-Sha256 ([string]$Item.path))) { throw "Reviewed source changed: $($Item.path)" }
}
$Incomplete = @($Plan.stages | Where-Object { -not (Test-StageComplete $_) })
if ($Incomplete.Count -ne 0) { throw "Post-analysis blocked: $($Incomplete.Count) formal stages are incomplete or changed." }

$AnalysisRoot = Join-Path $RunRoot 'analysis'
$SequenceDir = Join-Path $AnalysisRoot 'sequence'
$EfficiencyDir = Join-Path $AnalysisRoot 'efficiency'
$CurvesDir = Join-Path $AnalysisRoot 'curves'
$AuditDir = Join-Path $RunRoot 'audit'
New-Item -ItemType Directory -Force -Path (Split-Path $LogPath), $SequenceDir, $EfficiencyDir, $CurvesDir, $AuditDir | Out-Null

$Started = (Get-Date).ToUniversalTime().ToString('o')
Write-JsonAtomic $StatePath ([ordered]@{
    status = 'running'
    phase = 'sequence_evaluation'
    started_utc = $Started
    plan_sha256 = Get-Sha256 $PlanPath
    log_file = $LogPath
})

try {
    $Jobs = @(
        [ordered]@{
            phase = 'sequence_evaluation'
            script = Join-Path $Project 'scripts\evaluate_reviewer_minimum33_20260823.py'
            arguments = @('--plan', $PlanPath, '--output-dir', $SequenceDir, '--device', 'cuda')
        },
        [ordered]@{
            phase = 'efficiency_benchmark'
            script = Join-Path $Project 'scripts\benchmark_reviewer_minimum33_efficiency_20260823.py'
            arguments = @('--plan', $PlanPath, '--output-dir', $EfficiencyDir, '--device', 'cuda')
        },
        [ordered]@{
            phase = 'curve_generation'
            script = Join-Path $Project 'scripts\plot_reviewer_minimum33_curves_20260823.py'
            arguments = @('--plan', $PlanPath, '--output-dir', $CurvesDir)
        },
        [ordered]@{
            phase = 'complete_audit'
            script = Join-Path $Project 'scripts\audit_reviewer_minimum33_20260823.py'
            arguments = @('--plan', $PlanPath, '--output-dir', $AuditDir)
        }
    )
    foreach ($Job in $Jobs) {
        Write-JsonAtomic $StatePath ([ordered]@{
            status = 'running'
            phase = [string]$Job.phase
            started_utc = $Started
            updated_utc = (Get-Date).ToUniversalTime().ToString('o')
            plan_sha256 = Get-Sha256 $PlanPath
            log_file = $LogPath
        })
        "[$((Get-Date).ToUniversalTime().ToString('o'))] START $($Job.phase)" | Add-Content -LiteralPath $LogPath -Encoding UTF8
        $JobArguments = @($Job.arguments)
        & $Python ([string]$Job.script) @JobArguments *>> $LogPath
        if ($LASTEXITCODE -ne 0) { throw "$($Job.phase) exited with code $LASTEXITCODE" }
        "[$((Get-Date).ToUniversalTime().ToString('o'))] COMPLETE $($Job.phase)" | Add-Content -LiteralPath $LogPath -Encoding UTF8
    }
    $PassPath = Join-Path $AuditDir 'reviewer_minimum_audit_pass.json'
    if (-not (Test-Path -LiteralPath $PassPath -PathType Leaf)) { throw "Audit pass artifact was not created: $PassPath" }
    $Payload = [ordered]@{
        status = 'completed'
        phase = 'all_done'
        started_utc = $Started
        finished_utc = (Get-Date).ToUniversalTime().ToString('o')
        plan_sha256 = Get-Sha256 $PlanPath
        audit_pass_path = $PassPath
        audit_pass_sha256 = Get-Sha256 $PassPath
        log_file = $LogPath
    }
    Write-JsonAtomic $StatePath $Payload
    $Payload | ConvertTo-Json -Depth 20
}
catch {
    $Payload = [ordered]@{
        status = 'failed'
        phase = if (Test-Path -LiteralPath $StatePath) { (Get-Content -LiteralPath $StatePath -Raw | ConvertFrom-Json).phase } else { 'preflight' }
        error = $_.Exception.Message
        started_utc = $Started
        finished_utc = (Get-Date).ToUniversalTime().ToString('o')
        plan_sha256 = Get-Sha256 $PlanPath
        log_file = $LogPath
    }
    Write-JsonAtomic $StatePath $Payload
    $Payload | ConvertTo-Json -Depth 20
    throw
}

