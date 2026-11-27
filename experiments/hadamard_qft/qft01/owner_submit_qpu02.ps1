# Owner step for HAD-QFT-QPU02. Run on the machine with the saved IBM account.
# Usage: .\owner_submit_qpu02.ps1 -Dir results_local/qpu02_b5_r0 -Hash <plan_sha256 printed by build>
# The authorization file is written only if -Hash equals compiled.json's plan_sha256 and the
# estimate is within -Ceiling. Then submit and collect run. Success is judged by job.json and
# exit codes, because Python writes IBM warnings to stderr. Never rerun submit for a stage:
# if job.json exists, run collect only.
param([Parameter(Mandatory)][string]$Dir, [Parameter(Mandatory)][string]$Hash,
      [double]$Ceiling = 30, [string]$Backend = 'ibm_kingston')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
$m = Get-Content "$Dir/compiled.json" -Raw | ConvertFrom-Json
if ($m.plan_sha256 -ne $Hash) { throw "Hash mismatch: compiled plan is $($m.plan_sha256)" }
if ($m.estimated_qpu_seconds -gt $Ceiling) { throw 'Estimate exceeds ceiling' }
if (Test-Path "$Dir/job.json") { throw 'job.json exists: run collect only' }
$tag = Split-Path $Dir -Leaf
$authPath = "results_local/auth_$tag.json"
$auth = [ordered]@{ authorized = $true; backend = $Backend; plan_sha256 = $Hash; max_qpu_seconds = $Ceiling;
    authorized_by = 'owner'; checked_current_allocation = $true; acknowledge_estimate_not_hard_limit = $true }
[System.IO.File]::WriteAllText((Join-Path (Get-Location) $authPath), ($auth | ConvertTo-Json -Compress), (New-Object System.Text.UTF8Encoding($false)))
function Invoke-Py { $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; python @args 2>$null | Out-Host; $c = $LASTEXITCODE; $ErrorActionPreference = $old; return $c }
$null = Invoke-Py qpu02.py submit --out $Dir --authorization $authPath
if (-not (Test-Path "$Dir/job.json")) { throw 'no job.json after submit; inspect IBM job history before any retry' }
Get-Content "$Dir/job.json"
$null = Invoke-Py qpu02.py collect --out $Dir
if (-not (Test-Path "$Dir/counts.json")) { throw 'no counts.json; rerun collect only, never submit' }
