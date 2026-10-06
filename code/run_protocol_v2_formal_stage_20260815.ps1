param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('cmu', 'kit', 'bmlmovi')]
    [string]$Dataset,

    [Parameter(Mandatory = $true)]
    [ValidateSet('native25_no_rollout', 'native25_full_rollout', 'common18_no_rollout', 'common18_full_rollout', 'hisrep_common18', 'audit')]
    [string]$Stage
)

$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$Python = Join-Path $Project '.venv\Scripts\python.exe'
$ProtocolRoot = Join-Path $Project 'data\processed\protocol_v2_subject_grouped_rootrel_pose'
$FormalRoot = Join-Path $Project 'runs\formal_protocol_v2_subject_grouped_rootrel_pose_20260815'
$ManifestHash = '2e4ed5afb524c328efdd1246fa61fb96e104b39484e17d55784fa3ef2aec879e'
$DatasetNames = @{ cmu = 'CMU'; kit = 'KIT'; bmlmovi = 'BMLmovi' }
$DatasetName = $DatasetNames[$Dataset]
$RunRoot = Join-Path $FormalRoot $Dataset
$RunDir = Join-Path $RunRoot $Stage
$ProgressDir = Join-Path $FormalRoot 'progress'
$LogDir = Join-Path $FormalRoot 'logs'
$StateDir = Join-Path $FormalRoot 'stage_state'
$ProgressPath = Join-Path $ProgressDir ($Dataset + '_' + $Stage + '.json')
$LogPath = Join-Path $LogDir ($Dataset + '_' + $Stage + '.log')
$ExitPath = Join-Path $StateDir ($Dataset + '_' + $Stage + '_exit.json')

New-Item -ItemType Directory -Force -Path $RunDir, $ProgressDir, $LogDir, $StateDir | Out-Null
$env:PYTHONUNBUFFERED = '1'
$started = Get-Date

try {
    switch ($Stage) {
        'native25_no_rollout' {
            $DataRoot = Join-Path $ProtocolRoot "$Dataset\native25"
            & $Python (Join-Path $Project 'train.py') `
                --data-root $DataRoot `
                --metadata (Join-Path $DataRoot 'metadata.json') `
                --output-dir $RunDir `
                --model ltc_topology `
                --hidden-size 256 --num-layers 2 --dropout 0.0 `
                --batch-size 128 --epochs 30 `
                --lr 0.0002 --weight-decay 0.0001 --optimizer adam `
                --normalize --stats-file (Join-Path $DataRoot 'normalization_stats.npz') `
                --seed 42 --expected-manifest-sha256 $ManifestHash `
                --num-workers 0 --eval-every 1 --patience 0 --min-delta 0 `
                --device cuda --progress-file $ProgressPath --progress-update-interval 50 *>> $LogPath
        }
        'native25_full_rollout' {
            $DataRoot = Join-Path $ProtocolRoot "$Dataset\native25"
            $WarmStart = Join-Path $RunRoot 'native25_no_rollout\best.pt'
            if (-not (Test-Path -LiteralPath $WarmStart)) { throw "Missing warm start: $WarmStart" }
            & $Python (Join-Path $Project 'train_ltc_topology_rollout_consistency.py') `
                --dataset-name $DatasetName --data-root $DataRoot --metadata (Join-Path $DataRoot 'metadata.json') `
                --output-dir $RunDir --warm-start $WarmStart --progress-file $ProgressPath `
                --history 25 --future-step 25 --horizon 75 --stride 5 `
                --max-train-windows 0 --max-val-windows 0 `
                --batch-size 128 --num-workers 0 --epochs 30 `
                --lr 0.0002 --weight-decay 0.0001 `
                --hidden-size 256 --num-layers 2 --dropout 0.0 `
                --lambda-26-50 0.7 --lambda-51-75 0.5 --lambda-bone 0.1 --lambda-root 0.1 --lambda-velocity 0.05 `
                --seed 42 --expected-manifest-sha256 $ManifestHash --device cuda *>> $LogPath
        }
        'common18_no_rollout' {
            $DataRoot = Join-Path $ProtocolRoot "$Dataset\common18"
            & $Python (Join-Path $Project 'train.py') `
                --data-root $DataRoot `
                --metadata (Join-Path $DataRoot 'metadata.json') `
                --output-dir $RunDir `
                --model ltc_topology `
                --hidden-size 256 --num-layers 2 --dropout 0.0 `
                --batch-size 128 --epochs 30 `
                --lr 0.0002 --weight-decay 0.0001 --optimizer adam `
                --normalize --stats-file (Join-Path $DataRoot 'normalization_stats.npz') `
                --seed 42 --expected-manifest-sha256 $ManifestHash `
                --num-workers 0 --eval-every 1 --patience 0 --min-delta 0 `
                --device cuda --progress-file $ProgressPath --progress-update-interval 50 *>> $LogPath
        }
        'common18_full_rollout' {
            $DataRoot = Join-Path $ProtocolRoot "$Dataset\common18"
            $WarmStart = Join-Path $RunRoot 'common18_no_rollout\best.pt'
            if (-not (Test-Path -LiteralPath $WarmStart)) { throw "Missing warm start: $WarmStart" }
            & $Python (Join-Path $Project 'train_ltc_topology_rollout_consistency.py') `
                --dataset-name $DatasetName --data-root $DataRoot --metadata (Join-Path $DataRoot 'metadata.json') `
                --output-dir $RunDir --warm-start $WarmStart --progress-file $ProgressPath `
                --history 25 --future-step 25 --horizon 75 --stride 5 `
                --max-train-windows 0 --max-val-windows 0 `
                --batch-size 128 --num-workers 0 --epochs 30 `
                --lr 0.0002 --weight-decay 0.0001 `
                --hidden-size 256 --num-layers 2 --dropout 0.0 `
                --lambda-26-50 0.7 --lambda-51-75 0.5 --lambda-bone 0.1 --lambda-root 0.1 --lambda-velocity 0.05 `
                --seed 42 --expected-manifest-sha256 $ManifestHash --device cuda *>> $LogPath
        }
        'hisrep_common18' {
            $DataRoot = Join-Path $ProtocolRoot "$Dataset\common18"
            $ResumeArgs = @()
            $LastCheckpoint = Join-Path $RunDir 'last.pt'
            $ExistingResults = Join-Path $RunDir 'results.json'
            if ((Test-Path -LiteralPath $LastCheckpoint) -and (Test-Path -LiteralPath $ExistingResults)) {
                $ExistingPayload = Get-Content -LiteralPath $ExistingResults -Raw | ConvertFrom-Json
                if ($ExistingPayload.status -ne 'completed' -and
                    [int]$ExistingPayload.epochs_completed -gt 0 -and
                    [int]$ExistingPayload.epochs_completed -lt 30) {
                    $ResumeArgs = @('--resume-from', $LastCheckpoint)
                }
            }
            & $Python (Join-Path $Project 'train_hisrep_cmu_clean.py') `
                --dataset-name $DatasetName --metadata (Join-Path $DataRoot 'metadata.json') `
                --input-n 25 --output-n 25 --model-output-n 10 --itera 3 --dct-n 20 `
                --epochs 30 --batch-size 128 --test-batch-size 256 --lr 0.0005 `
                --stride 5 --seed 42 --expected-manifest-sha256 $ManifestHash `
                --num-workers 0 --disable-tqdm --device cuda `
                --run-dir $RunDir --progress-path $ProgressPath @ResumeArgs *>> $LogPath
        }
        'audit' {
            $AuditDir = Join-Path $RunRoot 'audit'
            New-Item -ItemType Directory -Force -Path $AuditDir | Out-Null
            # PowerShell 5 can turn the first native stderr line into a
            # terminating error when ErrorActionPreference is Stop. Capture
            # the complete Python output before checking the process code.
            $SavedErrorActionPreference = $ErrorActionPreference
            $ErrorActionPreference = 'Continue'
            & $Python (Join-Path $Project 'scripts\evaluate_protocol_v2_cmu_pilot.py') `
                --protocol-root $ProtocolRoot --pilot-root $RunRoot --output-dir $AuditDir `
                --manifest-sha256 $ManifestHash --dataset-slug $Dataset --dataset-name $DatasetName `
                --max-windows 0 --batch-size 128 --seed 2026 --device cuda 2>&1 | Out-File -LiteralPath $LogPath -Append -Encoding utf8
            $AuditExitCode = $LASTEXITCODE
            $ErrorActionPreference = $SavedErrorActionPreference
            if ($AuditExitCode -ne 0) { throw "Audit command exited with code $AuditExitCode. See $LogPath" }
            if ($AuditExitCode -eq 0) {
                $MetricJson = Join-Path $AuditDir ($Dataset + '_physical_metrics.json')
                if (-not (Test-Path -LiteralPath $MetricJson)) { throw "Missing audit metrics: $MetricJson" }
                $MetricPayload = Get-Content -LiteralPath $MetricJson -Raw | ConvertFrom-Json
                $FiniteText = Get-Content -LiteralPath $MetricJson -Raw
                if ($FiniteText -match 'NaN|Infinity') { throw 'Non-finite value in dataset audit' }
                [ordered]@{
                    status = 'PASS'
                    dataset = $DatasetName
                    manifest_sha256 = $ManifestHash
                    evaluated_at = (Get-Date).ToString('o')
                    metric_file = $MetricJson
                    test_windows_native25 = $MetricPayload.audits[0].test_windows
                    test_windows_common18 = $MetricPayload.audits[1].test_windows
                    window_fingerprint_match = ($MetricPayload.audits[0].test_window_records_sha256 -eq $MetricPayload.audits[1].test_window_records_sha256)
                } | ConvertTo-Json -Depth 4 | Set-Content -LiteralPath (Join-Path $AuditDir 'dataset_audit_pass.json') -Encoding UTF8
            }
        }
    }
    $code = $LASTEXITCODE
    if ($null -eq $code) { $code = 0 }
    if ($code -ne 0) { throw "Stage command exited with code $code" }
    [ordered]@{
        dataset = $Dataset
        stage = $Stage
        status = 'completed'
        exit_code = 0
        started_at = $started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        log = $LogPath
    } | ConvertTo-Json | Set-Content -LiteralPath $ExitPath -Encoding UTF8
}
catch {
    [ordered]@{
        dataset = $Dataset
        stage = $Stage
        status = 'failed'
        exit_code = if ($null -eq $LASTEXITCODE) { 1 } else { $LASTEXITCODE }
        error = $_.Exception.Message
        started_at = $started.ToString('o')
        finished_at = (Get-Date).ToString('o')
        log = $LogPath
    } | ConvertTo-Json | Set-Content -LiteralPath $ExitPath -Encoding UTF8
    throw
}
