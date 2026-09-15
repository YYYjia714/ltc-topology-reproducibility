param(
    [Parameter(Mandatory = $true)]
    [string]$StageId,
    [switch]$Resume,
    [switch]$Smoke,
    [string]$SmokeRoot = ''
)

$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$FormalRoot = Join-Path $Project 'runs\formal_reviewer_minimum33_revision_20260823'
$PlanPath = Join-Path $FormalRoot 'formal_reviewer_minimum33_plan.json'
$AuthorizationPath = Join-Path $FormalRoot 'formal_start_authorization.json'
$Python = Join-Path $Project '.venv\Scripts\python.exe'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

function Write-JsonAtomic([string]$Path, $Payload) {
    $Temporary = "$Path.tmp"
    $Payload | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $Temporary -Encoding UTF8
    Move-Item -LiteralPath $Temporary -Destination $Path -Force
}

function Test-ArtifactSet($ExitState, [string]$RunDir, $RequiredOutputs) {
    if ($ExitState.status -ne 'completed' -or [int]$ExitState.exit_code -ne 0) { return $false }
    if ([string]$ExitState.plan_sha256 -ne $PlanHash) { return $false }
    foreach ($Name in @($RequiredOutputs)) {
        $Path = Join-Path $RunDir ([string]$Name)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { return $false }
        $Record = @($ExitState.artifacts | Where-Object { $_.name -eq [string]$Name }) | Select-Object -First 1
        if ($null -eq $Record) { return $false }
        if ([string]$Record.sha256 -ne (Get-Sha256 $Path)) { return $false }
        if ([int64]$Record.size -ne (Get-Item -LiteralPath $Path).Length) { return $false }
    }
    return $true
}

if (-not (Test-Path -LiteralPath $Python)) { throw "Missing project Python: $Python" }
if (-not (Test-Path -LiteralPath $PlanPath)) { throw "Missing formal plan: $PlanPath" }
$Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$PlanHash = Get-Sha256 $PlanPath
$Stage = @($Plan.stages | Where-Object { $_.id -eq $StageId }) | Select-Object -First 1
if ($null -eq $Stage) { throw "Unknown training stage: $StageId" }

if ($Smoke) {
    if ([string]::IsNullOrWhiteSpace($SmokeRoot)) { throw '-SmokeRoot is required with -Smoke.' }
    $OutputRoot = [System.IO.Path]::GetFullPath($SmokeRoot)
}
else {
    $OutputRoot = $FormalRoot
    if (-not (Test-Path -LiteralPath $AuthorizationPath)) { throw "Missing authorization: $AuthorizationPath" }
    $Authorization = Get-Content -LiteralPath $AuthorizationPath -Raw | ConvertFrom-Json
    if (-not [bool]$Authorization.approved) { throw 'Formal revision queue is locked.' }
    if ([string]$Authorization.plan_sha256 -ne $PlanHash) { throw 'Authorized plan hash does not match.' }
    foreach ($Binding in @(
        @([string]$Authorization.review_path, [string]$Authorization.review_sha256, 'review'),
        @([string]$Authorization.inventory_path, [string]$Authorization.inventory_sha256, 'inventory'),
        @([string]$Authorization.smoke_report_path, [string]$Authorization.smoke_report_sha256, 'smoke report'),
        @([string]$Authorization.preflight_path, [string]$Authorization.preflight_sha256, 'formal preflight')
    )) {
        if (-not (Test-Path -LiteralPath $Binding[0] -PathType Leaf)) { throw "Missing authorized $($Binding[2]): $($Binding[0])" }
        if ($Binding[1] -ne (Get-Sha256 $Binding[0])) { throw "Authorized $($Binding[2]) hash changed." }
    }
    $Review = Get-Content -LiteralPath ([string]$Authorization.review_path) -Raw | ConvertFrom-Json
    if ([string]$Review.status -ne 'PASS' -or [string]$Review.plan_sha256 -ne $PlanHash) {
        throw 'Authorized code review did not pass this frozen plan.'
    }
    $SmokeReport = Get-Content -LiteralPath ([string]$Authorization.smoke_report_path) -Raw | ConvertFrom-Json
    if ([string]$SmokeReport.status -ne 'passed' -or [string]$SmokeReport.plan_sha256 -ne $PlanHash) {
        throw 'Authorized smoke report did not pass this frozen plan.'
    }
    $Preflight = Get-Content -LiteralPath ([string]$Authorization.preflight_path) -Raw | ConvertFrom-Json
    if ([string]$Preflight.status -ne 'PASS' -or -not [bool]$Preflight.formal_ready -or
        [string]$Preflight.plan_sha256 -ne $PlanHash) {
        throw 'Formal preflight is not ready for this frozen plan.'
    }
    $Inventory = Get-Content -LiteralPath ([string]$Authorization.inventory_path) -Raw | ConvertFrom-Json
    if ([string]$Inventory.plan_sha256 -ne $PlanHash) { throw 'Frozen inventory does not bind this plan.' }
    foreach ($Item in @($Inventory.files)) {
        if (-not (Test-Path -LiteralPath ([string]$Item.path) -PathType Leaf)) { throw "Reviewed source missing: $($Item.path)" }
        if ([string]$Item.sha256 -ne (Get-Sha256 ([string]$Item.path))) { throw "Reviewed source changed: $($Item.path)" }
    }
}

$StateDir = Join-Path $OutputRoot 'stage_state'
$ProgressDir = Join-Path $OutputRoot 'progress'
$LogDir = Join-Path $OutputRoot 'logs'
$RunDir = Join-Path $OutputRoot ([string]$Stage.output_rel -replace '/', '\')
$ExitPath = Join-Path $StateDir "$StageId`_exit.json"
$ProgressPath = Join-Path $ProgressDir "$StageId.json"
$LogPath = Join-Path $LogDir "$StageId.log"
New-Item -ItemType Directory -Force -Path $OutputRoot, $StateDir, $ProgressDir, $LogDir | Out-Null

$RequiredOutputs = @($Stage.required_outputs)
if (Test-Path -LiteralPath $ExitPath -PathType Leaf) {
    try { $Existing = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json } catch { $Existing = $null }
    if ($null -ne $Existing -and (Test-ArtifactSet $Existing $RunDir $RequiredOutputs)) {
        $Existing | ConvertTo-Json -Depth 20
        exit 0
    }
}

$Started = Get-Date
$CommandRecord = @()
$ScriptPath = $null
$CommandArgs = @()
$ExitCode = 1
try {
    foreach ($DependencyId in @($Stage.depends_on)) {
        $Dependency = @($Plan.stages | Where-Object { $_.id -eq [string]$DependencyId }) | Select-Object -First 1
        if ($null -eq $Dependency) { throw "Unknown dependency: $DependencyId" }
        $DependencyRunDir = Join-Path $OutputRoot ([string]$Dependency.output_rel -replace '/', '\')
        $DependencyExitPath = Join-Path $StateDir "$DependencyId`_exit.json"
        if (-not (Test-Path -LiteralPath $DependencyExitPath -PathType Leaf)) { throw "Dependency has no exit state: $DependencyId" }
        $DependencyExit = Get-Content -LiteralPath $DependencyExitPath -Raw | ConvertFrom-Json
        if (-not (Test-ArtifactSet $DependencyExit $DependencyRunDir @($Dependency.required_outputs))) {
            throw "Dependency artifacts are incomplete or changed: $DependencyId"
        }
    }

    $Active = Get-CimInstance Win32_Process | Where-Object {
        $_.Name -match '^python(\.exe)?$' -and
        $_.CommandLine -like '*E:\AMASS_LNN_Project*' -and
        $_.CommandLine -match '(train|evaluate_protocol|benchmark_protocol).*\.py'
    } | Select-Object -First 1
    if ($null -ne $Active) { throw "Another formal project process is active: PID $($Active.ProcessId)" }

    if ((Test-Path -LiteralPath $RunDir) -and (Get-ChildItem -LiteralPath $RunDir -Force -ErrorAction SilentlyContinue) -and -not $Resume) {
        throw "Run directory is non-empty; preserve it and use a reviewed recovery action: $RunDir"
    }
    New-Item -ItemType Directory -Force -Path $RunDir | Out-Null
    $DataRoot = Join-Path ([string]$Plan.data_root) "$($Stage.dataset)\common18"
    $Metadata = Join-Path $DataRoot 'metadata.json'
    $Defaults = $Plan.defaults
    $ManifestHash = [string]$Plan.split_manifest_sha256
    $Epochs = if ($Smoke) { '1' } else { [string]$Defaults.epochs }
    $MaxTrain = if ($Smoke) { '16' } else { '0' }
    $MaxVal = if ($Smoke) { '8' } else { '0' }
    $Device = if ($Smoke -and -not [bool](Get-Command nvidia-smi -ErrorAction SilentlyContinue)) { 'cpu' } else { 'cuda' }
    $InventoryHash = if ($Smoke) { '' } else { [string]$Authorization.inventory_sha256 }
    $BatchSize = if ($Smoke) { '2' } elseif ($Stage.PSObject.Properties.Name -contains 'batch_size_override') {
        [string]$Stage.batch_size_override
    } else { [string]$Defaults.batch_size }

    if ([string]$Stage.kind -eq 'ltc') {
        $Profile = [string]$Stage.compatibility_profile
        $Lambda26To50 = [double]$Defaults.lambda_26_50
        $Lambda51To75 = [double]$Defaults.lambda_51_75
        $LambdaBone = [double]$Defaults.lambda_bone
        $LambdaAnchor = [double]$Defaults.lambda_anchor
        $LambdaVelocity = [double]$Defaults.lambda_velocity
        switch ([string]$Stage.loss_profile) {
            'regularizers_half' {
                $LambdaBone *= 0.5; $LambdaAnchor *= 0.5; $LambdaVelocity *= 0.5
            }
            'regularizers_double' {
                $LambdaBone *= 2.0; $LambdaAnchor *= 2.0; $LambdaVelocity *= 2.0
            }
            'no_velocity' { $LambdaVelocity = 0.0 }
            'equal_stage_weights' { $Lambda26To50 = 1.0; $Lambda51To75 = 1.0 }
            'default' { }
            default { throw "Unknown loss profile: $($Stage.loss_profile)" }
        }
        $WarmPath = $null
        if ($Stage.PSObject.Properties.Name -contains 'warm_start_id') {
            $WarmStage = @($Plan.stages | Where-Object { $_.id -eq [string]$Stage.warm_start_id }) | Select-Object -First 1
            $WarmPath = Join-Path (Join-Path $OutputRoot ([string]$WarmStage.output_rel -replace '/', '\')) 'best.pt'
            if (-not (Test-Path -LiteralPath $WarmPath -PathType Leaf)) { throw "Missing warm start: $WarmPath" }
        }
        elseif ($Stage.PSObject.Properties.Name -contains 'warm_start_reuse_id') {
            $ReuseInventoryPath = Join-Path $FormalRoot 'reused\reused_inventory.json'
            if (-not (Test-Path -LiteralPath $ReuseInventoryPath -PathType Leaf)) { throw "Missing reuse inventory: $ReuseInventoryPath" }
            $ReuseInventory = Get-Content -LiteralPath $ReuseInventoryPath -Raw | ConvertFrom-Json
            $Reuse = @($ReuseInventory.records | Where-Object { $_.id -eq [string]$Stage.warm_start_reuse_id }) | Select-Object -First 1
            if ($null -eq $Reuse -or $null -eq $Reuse.compatible_checkpoint) { throw "Missing compatible reuse checkpoint: $($Stage.warm_start_reuse_id)" }
            $WarmPath = [string]$Reuse.compatible_checkpoint.path
            if (-not (Test-Path -LiteralPath $WarmPath -PathType Leaf)) { throw "Missing compatible warm start: $WarmPath" }
            if ([string]$Reuse.compatible_checkpoint.sha256 -ne (Get-Sha256 $WarmPath)) { throw "Compatible warm-start hash changed: $WarmPath" }
        }

        if ([string]$Stage.objective -eq 'no_rollout') {
            $ScriptPath = Join-Path $Project 'train.py'
            $CommandArgs = @(
                '--data-root', $DataRoot,
                '--metadata', $Metadata,
                '--output-dir', $RunDir,
                '--model', [string]$Stage.model,
                '--hidden-size', [string]$Defaults.hidden_size,
                '--num-layers', [string]$Defaults.num_layers,
                '--dropout', [string]$Defaults.dropout,
                '--batch-size', $BatchSize,
                '--epochs', $Epochs,
                '--lr', [string]$Defaults.ltc_lr,
                '--weight-decay', [string]$Defaults.weight_decay,
                '--optimizer', 'adam',
                '--normalize',
                '--stats-file', (Join-Path $DataRoot 'normalization_stats.npz'),
                '--seed', [string]$Stage.seed,
                '--expected-manifest-sha256', $ManifestHash,
                '--max-train-samples', $MaxTrain,
                '--max-val-samples', $MaxVal,
                '--max-test-samples', $MaxVal,
                '--num-workers', [string]$Defaults.num_workers,
                '--eval-every', '1',
                '--patience', '0',
                '--min-delta', '0',
                '--device', $Device,
                '--progress-file', $ProgressPath,
                '--progress-update-interval', '50'
            )
            if ([string]$Stage.model -eq 'ltc') {
                $CommandArgs += @('--disable-topology-encoder', '--disable-topology-decoder', '--disable-root-guidance')
            }
            elseif (-not [bool]$Stage.anchor_guidance) { $CommandArgs += '--disable-root-guidance' }
            if ($Resume) {
                $ResumePath = Join-Path $RunDir 'last.pt'
                if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) { throw "Missing resume checkpoint: $ResumePath" }
                $CommandArgs += @('--resume-from', $ResumePath)
            }
        }
        elseif ($Profile -eq 'legacy_seed42_exact') {
            if ($Resume) { throw 'The exact legacy rollout trainer has no reviewed resume path.' }
            if ([string]$Stage.model -ne 'ltc_topology' -or $null -eq $WarmPath) {
                throw 'Exact legacy rollout requires LTC-Topology and an audited no-rollout warm start.'
            }
            $ScriptPath = Join-Path $Project 'train_ltc_topology_rollout_consistency.py'
            $CommandArgs = @(
                '--dataset-name', [string]$Stage.dataset_name,
                '--data-root', $DataRoot,
                '--metadata', $Metadata,
                '--output-dir', $RunDir,
                '--warm-start', $WarmPath,
                '--progress-file', $ProgressPath,
                '--history', [string]$Defaults.history,
                '--future-step', [string]$Defaults.future_step,
                '--horizon', [string]$Defaults.validation_horizon,
                '--stride', [string]$Defaults.stride,
                '--max-train-windows', $MaxTrain,
                '--max-val-windows', $MaxVal,
                '--batch-size', $BatchSize,
                '--num-workers', [string]$Defaults.num_workers,
                '--epochs', $Epochs,
                '--lr', [string]$Defaults.ltc_lr,
                '--weight-decay', [string]$Defaults.weight_decay,
                '--hidden-size', [string]$Defaults.hidden_size,
                '--num-layers', [string]$Defaults.num_layers,
                '--dropout', [string]$Defaults.dropout,
                '--lambda-26-50', [string]$Lambda26To50,
                '--lambda-51-75', [string]$Lambda51To75,
                '--lambda-bone', [string]$LambdaBone,
                '--lambda-root', [string]$LambdaAnchor,
                '--lambda-velocity', [string]$LambdaVelocity,
                '--seed', [string]$Stage.seed,
                '--expected-manifest-sha256', $ManifestHash,
                '--device', $Device
            )
        }
        elseif ($Profile -eq 'legacy_ablation_exact') {
            if ($Resume) { throw 'Reviewer ablation recovery requires a separately reviewed recovery plan.' }
            if ($null -eq $WarmPath) { throw 'Reviewer rollout/compute ablations require an audited warm start.' }
            $ScriptPath = Join-Path $Project 'train_reviewer_ltc_legacy_compatible_20260822.py'
            $CommandArgs = @(
                '--dataset-name', [string]$Stage.dataset_name,
                '--data-root', $DataRoot,
                '--metadata', $Metadata,
                '--output-dir', $RunDir,
                '--warm-start', $WarmPath,
                '--progress-file', $ProgressPath,
                '--training-objective', [string]$Stage.objective,
                '--model', [string]$Stage.model,
                '--history', [string]$Defaults.history,
                '--future-step', [string]$Defaults.future_step,
                '--horizon', [string]$Defaults.validation_horizon,
                '--stride', [string]$Defaults.stride,
                '--max-train-windows', $MaxTrain,
                '--max-val-windows', $MaxVal,
                '--batch-size', $BatchSize,
                '--num-workers', [string]$Defaults.num_workers,
                '--epochs', $Epochs,
                '--lr', [string]$Defaults.ltc_lr,
                '--weight-decay', [string]$Defaults.weight_decay,
                '--hidden-size', [string]$Defaults.hidden_size,
                '--num-layers', [string]$Defaults.num_layers,
                '--dropout', [string]$Defaults.dropout,
                '--lambda-26-50', [string]$Lambda26To50,
                '--lambda-51-75', [string]$Lambda51To75,
                '--lambda-bone', [string]$LambdaBone,
                '--lambda-root', [string]$LambdaAnchor,
                '--lambda-velocity', [string]$LambdaVelocity,
                '--seed', [string]$Stage.seed,
                '--expected-manifest-sha256', $ManifestHash,
                '--device', $Device
            )
            if ([string]$Stage.model -eq 'ltc') {
                $CommandArgs += @('--disable-topology-encoder', '--disable-topology-decoder', '--disable-root-guidance')
            }
            elseif (-not [bool]$Stage.anchor_guidance) { $CommandArgs += '--disable-root-guidance' }
        }
        else {
            throw "Unsupported LTC compatibility profile/objective: $Profile / $($Stage.objective)"
        }
    }
    elseif ([string]$Stage.kind -eq 'hisrep') {
        $ScriptPath = Join-Path $Project 'train_hisrep_cmu_clean.py'
        $HisrepBatchSize = if ($Smoke) { '16' } else { $BatchSize }
        $HisrepMaxTrain = if ($Smoke) { '64' } else { $MaxTrain }
        $HisrepMaxVal = if ($Smoke) { '32' } else { $MaxVal }
        $CommandArgs = @(
            '--dataset-name', [string]$Stage.dataset_name,
            '--metadata', $Metadata,
            '--run-dir', $RunDir,
            '--progress-path', $ProgressPath,
            '--input-n', '25',
            '--output-n', '25',
            '--model-output-n', '10',
            '--itera', '3',
            '--dct-n', '20',
            '--epochs', $Epochs,
            '--batch-size', $HisrepBatchSize,
            '--test-batch-size', $HisrepBatchSize,
            '--lr', [string]$Defaults.hisrep_lr,
            '--seed', [string]$Stage.seed,
            '--stride', [string]$Defaults.stride,
            '--num-workers', [string]$Defaults.num_workers,
            '--max-train-windows', $HisrepMaxTrain,
            '--max-val-windows', $HisrepMaxVal,
            '--expected-manifest-sha256', $ManifestHash,
            '--disable-tqdm',
            '--device', $Device
        )
        if ($Resume) {
            $ResumePath = Join-Path $RunDir 'last.pt'
            if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) { throw "Missing resume checkpoint: $ResumePath" }
            $CommandArgs += @('--resume-from', $ResumePath)
        }
    }
    elseif ([string]$Stage.kind -eq 'baseline') {
        $ScriptPath = Join-Path $Project 'train_protocol_v2_reviewer_baseline_20260822.py'
        $CommandArgs = @(
            '--dataset-name', [string]$Stage.dataset_name,
            '--data-root', $DataRoot,
            '--metadata', $Metadata,
            '--output-dir', $RunDir,
            '--progress-file', $ProgressPath,
            '--baseline', [string]$Stage.baseline,
            '--history', [string]$Defaults.history,
            '--future-step', [string]$Defaults.future_step,
            '--validation-horizon', [string]$Defaults.validation_horizon,
            '--stride', [string]$Defaults.stride,
            '--max-train-windows', $MaxTrain,
            '--max-val-windows', $MaxVal,
            '--batch-size', $BatchSize,
            '--num-workers', [string]$Defaults.num_workers,
            '--epochs', $Epochs,
            '--lr', [string]$Defaults.baseline_lr,
            '--weight-decay', [string]$Defaults.weight_decay,
            '--hidden-size', [string]$Defaults.hidden_size,
            '--num-layers', [string]$Defaults.num_layers,
            '--dropout', [string]$Defaults.dropout,
            '--seed', [string]$Stage.seed,
            '--expected-manifest-sha256', $ManifestHash,
            '--device', $Device
        )
        if (-not $Smoke) { $CommandArgs += @('--frozen-inventory-sha256', $InventoryHash) }
        if ($Resume) {
            $ResumePath = Join-Path $RunDir 'last.pt'
            if (-not (Test-Path -LiteralPath $ResumePath -PathType Leaf)) { throw "Missing resume checkpoint: $ResumePath" }
            $CommandArgs += @('--resume-from', $ResumePath)
        }
    }
    else { throw "Unsupported training kind: $($Stage.kind)" }

    $CommandRecord = @($Python, $ScriptPath) + $CommandArgs
    $env:PYTHONUNBUFFERED = '1'
    $env:CUBLAS_WORKSPACE_CONFIG = ':4096:8'
    $SavedPreference = $ErrorActionPreference
    $ErrorActionPreference = 'Continue'
    & $Python $ScriptPath @CommandArgs *>> $LogPath
    $ExitCode = $LASTEXITCODE
    $ErrorActionPreference = $SavedPreference
    if ($ExitCode -ne 0) { throw "Stage command exited with code $ExitCode" }

    $Artifacts = @()
    foreach ($Name in $RequiredOutputs) {
        $Path = Join-Path $RunDir ([string]$Name)
        if (-not (Test-Path -LiteralPath $Path -PathType Leaf)) { throw "Missing required output: $Name" }
        $Artifacts += [ordered]@{
            name = [string]$Name
            path = $Path
            size = (Get-Item -LiteralPath $Path).Length
            sha256 = Get-Sha256 $Path
        }
    }
    $ExitPayload = [ordered]@{
        stage_id = $StageId
        dataset = [string]$Stage.dataset
        kind = [string]$Stage.kind
        status = 'completed'
        exit_code = 0
        smoke = [bool]$Smoke
        started_at = $Started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        run_dir = $RunDir
        progress_file = $ProgressPath
        log_file = $LogPath
        command = $CommandRecord
        script_sha256 = Get-Sha256 $ScriptPath
        plan_sha256 = $PlanHash
        artifacts = $Artifacts
    }
    Write-JsonAtomic $ExitPath $ExitPayload
}
catch {
    $Failure = [ordered]@{
        stage_id = $StageId
        dataset = [string]$Stage.dataset
        kind = [string]$Stage.kind
        status = 'failed'
        exit_code = if ($ExitCode -eq 0) { 1 } else { $ExitCode }
        smoke = [bool]$Smoke
        error = $_.Exception.Message
        started_at = $Started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        run_dir = $RunDir
        progress_file = $ProgressPath
        log_file = $LogPath
        command = $CommandRecord
        plan_sha256 = $PlanHash
    }
    Write-JsonAtomic $ExitPath $Failure
    throw
}

Get-Content -LiteralPath $ExitPath -Raw
