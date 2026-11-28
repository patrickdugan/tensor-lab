# Owner step for HAD-QFT-QPU01. Run on the machine with the saved IBM account.
# Usage:  .\owner_submit.ps1 -B 3 -Hash <plan_sha256 printed by build>
# The hash you type is compared with compiled.json; the authorization file is written
# only on an exact match, then submit and collect run. Rebuild first if the build is
# older than 15 minutes (submit refuses stale builds).
param([Parameter(Mandatory)][int]$B, [Parameter(Mandatory)][string]$Hash, [string]$Backend = 'ibm_kingston')
$ErrorActionPreference = 'Stop'
Set-Location $PSScriptRoot
# Python writes IBM warnings to stderr; judge success by exit code only.
function Invoke-Py { $old = $ErrorActionPreference; $ErrorActionPreference = 'Continue'; python @args 2>$null | Out-Host; $code = $LASTEXITCODE; $ErrorActionPreference = $old; return $code }
$out = "results_local/qft$B"
$m = Get-Content "$out/compiled.json" -Raw | ConvertFrom-Json
if ($m.plan_sha256 -ne $Hash) { throw "Hash mismatch: compiled plan is $($m.plan_sha256)" }
if ($m.estimated_qpu_seconds -gt 15) { throw 'Estimate exceeds 15 s ceiling' }
$auth = [ordered]@{ authorized = $true; backend = $Backend; plan_sha256 = $Hash; max_qpu_seconds = 15;
    authorized_by = 'owner'; checked_current_allocation = $true; acknowledge_estimate_not_hard_limit = $true }
$json = ($auth | ConvertTo-Json -Compress)
[System.IO.File]::WriteAllText((Join-Path (Get-Location) "results_local/auth_qft$B.json"), $json, (New-Object System.Text.UTF8Encoding($false)))
$code = Invoke-Py qpu01.py submit --out $out --authorization "results_local/auth_qft$B.json"
if (Test-Path "$out/job.json") { Get-Content "$out/job.json" } elseif ($code -ne 0) { throw 'submit failed with no job.json; check IBM job history before any retry' }
$code = Invoke-Py qpu01.py collect --out $out
if ($code -ne 0) { throw 'collect failed; rerun collect only, never submit' }
