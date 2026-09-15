param([string]$SmokeRoot = '')

$ErrorActionPreference = 'Stop'
$Project = 'E:\AMASS_LNN_Project'
$FormalRoot = Join-Path $Project 'runs\formal_reviewer_minimum33_revision_20260823'
$PlanPath = Join-Path $FormalRoot 'formal_reviewer_minimum33_plan.json'
$Runner = Join-Path $Project 'run_reviewer_minimum33_stage_20260823.ps1'
$Python = Join-Path $Project '.venv\Scripts\python.exe'

function Get-Sha256([string]$Path) {
    return (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash.ToLowerInvariant()
}

if ([string]::IsNullOrWhiteSpace($SmokeRoot)) {
    $Stamp = Get-Date -Format 'yyyyMMdd_HHmmss'
    $SmokeRoot = Join-Path $FormalRoot "smoke\$Stamp"
}
$SmokeRoot = [System.IO.Path]::GetFullPath($SmokeRoot)
New-Item -ItemType Directory -Force -Path $SmokeRoot | Out-Null

$Plan = Get-Content -LiteralPath $PlanPath -Raw | ConvertFrom-Json
$Stages = @($Plan.stages | ForEach-Object { [string]$_.id })
if ($Stages.Count -ne 33) { throw "Smoke requires exactly 33 planned stages, got $($Stages.Count)." }

$Records = @()
$Started = Get-Date
foreach ($StageId in $Stages) {
    try {
        & powershell.exe -NoProfile -ExecutionPolicy Bypass -File $Runner `
            -StageId $StageId -Smoke -SmokeRoot $SmokeRoot | Out-Null
        $ExitPath = Join-Path $SmokeRoot "stage_state\$StageId`_exit.json"
        if (-not (Test-Path -LiteralPath $ExitPath -PathType Leaf)) {
            throw "Missing smoke exit state: $ExitPath"
        }
        $Exit = Get-Content -LiteralPath $ExitPath -Raw | ConvertFrom-Json
        if ($Exit.status -ne 'completed' -or [int]$Exit.exit_code -ne 0) {
            throw "Smoke stage failed: $StageId"
        }
        $Records += [ordered]@{
            stage_id = $StageId
            status = 'passed'
            exit_path = $ExitPath
            exit_sha256 = Get-Sha256 $ExitPath
            log_path = [string]$Exit.log_file
        }
    }
    catch {
        $Records += [ordered]@{
            stage_id = $StageId
            status = 'failed'
            error = $_.Exception.Message
        }
        $Report = [ordered]@{
            status = 'failed'
            plan_path = $PlanPath
            plan_sha256 = Get-Sha256 $PlanPath
            smoke_root = $SmokeRoot
            started_at = $Started.ToString('o')
            finished_at = (Get-Date).ToString('o')
            records = $Records
        }
        $Report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath (Join-Path $SmokeRoot 'smoke_report.json') -Encoding UTF8
        throw
    }
}

$Report = [ordered]@{
    status = 'passed'
    plan_path = $PlanPath
    plan_sha256 = Get-Sha256 $PlanPath
    smoke_root = $SmokeRoot
    coverage = 'all_33_formal_stage_ids_with_smoke_limits'
    started_at = $Started.ToString('o')
    finished_at = (Get-Date).ToString('o')
    records = $Records
}
$ReportPath = Join-Path $SmokeRoot 'smoke_report.json'
$Report | ConvertTo-Json -Depth 20 | Set-Content -LiteralPath $ReportPath -Encoding UTF8
$Report | ConvertTo-Json -Depth 20
