$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$FormalRoot = Join-Path $Project 'runs\formal_protocol_v2_subject_grouped_rootrel_pose_20260815'
$Runner = Join-Path $Project 'run_protocol_v2_formal_stage_20260815.ps1'
$StateDir = Join-Path $FormalRoot 'stage_state'
$QueueState = Join-Path $FormalRoot 'queue_state.json'
$Mirror = 'C:\Users\1\Desktop\LTC拓扑75\NEWsanwei75\PROTOCOL_V2_FORMAL_RETRAIN_20260815'
New-Item -ItemType Directory -Force -Path $FormalRoot, $StateDir, $Mirror | Out-Null

$stages = @(
    foreach ($dataset in @('cmu', 'kit', 'bmlmovi')) {
        foreach ($stage in @('native25_no_rollout', 'native25_full_rollout', 'common18_no_rollout', 'common18_full_rollout', 'hisrep_common18', 'audit')) {
            [pscustomobject]@{ dataset = $dataset; stage = $stage }
        }
    }
)

function Stage-Complete($item) {
    $exitPath = Join-Path $StateDir "$($item.dataset)_$($item.stage)_exit.json"
    if (-not (Test-Path -LiteralPath $exitPath)) { return $false }
    try {
        $exitState = Get-Content -LiteralPath $exitPath -Raw | ConvertFrom-Json
    }
    catch {
        return $false
    }
    if ($exitState.status -ne 'completed' -or [int]$exitState.exit_code -ne 0) { return $false }

    if ($item.stage -eq 'audit') {
        return Test-Path -LiteralPath (Join-Path $FormalRoot "$($item.dataset)\audit\dataset_audit_pass.json")
    }
    $dir = Join-Path $FormalRoot "$($item.dataset)\$($item.stage)"
    return (Test-Path -LiteralPath (Join-Path $dir 'best.pt')) -and
           (Test-Path -LiteralPath (Join-Path $dir 'results.json')) -and
           (Test-Path -LiteralPath (Join-Path $dir 'train_history.json'))
}

function Exit-State($item) {
    $path = Join-Path $StateDir "$($item.dataset)_$($item.stage)_exit.json"
    if (Test-Path -LiteralPath $path) { return Get-Content -LiteralPath $path -Raw | ConvertFrom-Json }
    return $null
}

$active = Get-CimInstance Win32_Process | Where-Object {
    ($_.Name -match '^python(\.exe)?$' -and $_.CommandLine -like "*$FormalRoot*") -or
    ($_.Name -match '^powershell(\.exe)?$' -and $_.CommandLine -match 'run_protocol_v2_formal_stage_20260815\.ps1')
} | Select-Object -First 1

$completed = @($stages | Where-Object { Stage-Complete $_ })
$next = $stages | Where-Object { -not (Stage-Complete $_) } | Select-Object -First 1

if ($null -ne $active) {
    $progressFiles = Get-ChildItem (Join-Path $FormalRoot 'progress') -Filter '*.json' -ErrorAction SilentlyContinue | Sort-Object LastWriteTime -Descending
    $progress = if ($progressFiles) { Get-Content $progressFiles[0].FullName -Raw | ConvertFrom-Json } else { $null }
    $payload = [ordered]@{
        status = 'RUNNING'
        process_id = $active.ProcessId
        completed_stages = $completed.Count
        total_stages = $stages.Count
        current = if ($null -eq $next) { $null } else { "$($next.dataset)/$($next.stage)" }
        progress_file = if ($progressFiles) { $progressFiles[0].FullName } else { $null }
        progress = $progress
        updated_at = (Get-Date).ToString('o')
    }
    $payload | ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $QueueState -Encoding UTF8
    $payload | ConvertTo-Json -Depth 8
    exit 0
}

foreach ($item in $stages) {
    $exit = Exit-State $item
    if ($null -ne $exit -and $exit.status -eq 'failed' -and -not (Stage-Complete $item)) {
        $payload = [ordered]@{ status='FAILED'; current="$($item.dataset)/$($item.stage)"; exit_state=$exit; updated_at=(Get-Date).ToString('o') }
        $payload | ConvertTo-Json -Depth 6 | Set-Content -LiteralPath $QueueState -Encoding UTF8
        $payload | ConvertTo-Json -Depth 6
        exit 1
    }
}

if ($null -eq $next) {
    $payload = [ordered]@{ status='ALL_DONE'; completed_stages=$stages.Count; total_stages=$stages.Count; updated_at=(Get-Date).ToString('o') }
    $payload | ConvertTo-Json | Set-Content -LiteralPath $QueueState -Encoding UTF8
    Copy-Item -LiteralPath $QueueState -Destination $Mirror -Force
    $payload | ConvertTo-Json
    exit 0
}

$argumentList = @('-NoProfile', '-ExecutionPolicy', 'Bypass', '-File', ('"' + $Runner + '"'), '-Dataset', $next.dataset, '-Stage', $next.stage)
$process = Start-Process -FilePath 'powershell.exe' -ArgumentList $argumentList -WindowStyle Hidden -PassThru
$payload = [ordered]@{
    status = 'STARTED'
    process_id = $process.Id
    current = "$($next.dataset)/$($next.stage)"
    completed_stages = $completed.Count
    total_stages = $stages.Count
    run_root = (Join-Path $FormalRoot "$($next.dataset)\$($next.stage)")
    progress_file = (Join-Path $FormalRoot "progress\$($next.dataset)_$($next.stage).json")
    log_file = (Join-Path $FormalRoot "logs\$($next.dataset)_$($next.stage).log")
    started_at = (Get-Date).ToString('o')
}
$payload | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath $QueueState -Encoding UTF8
Copy-Item -LiteralPath $QueueState -Destination $Mirror -Force
$payload | ConvertTo-Json -Depth 5
