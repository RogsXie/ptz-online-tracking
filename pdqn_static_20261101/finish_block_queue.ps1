$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/EI/ptz_ue/pdqn_static_20261101'
$ptzPython = 'C:/Users/12879/.conda/envs/50mamba/python.exe'
$trainingOwnerPid = 60112
while (Get-Process -Id $trainingOwnerPid -ErrorAction SilentlyContinue) {
    Start-Sleep -Seconds 2
}
$trainingStatus = Get-Content -Raw -LiteralPath "$taskRoot/online-interval4/status.json" | ConvertFrom-Json
if ($trainingStatus.completed -ne 320) {
    throw 'The owned training process ended before episode320. Inspect its failure before continuing.'
}
& $ptzPython -B "$taskRoot/calibrate_corners.py"
if ($LASTEXITCODE -ne 0) { throw 'Actual corner-layout calibration failed.' }
$cornerResult = Get-Content -Raw -LiteralPath "$taskRoot/corner-layout-calibration/summary.json" | ConvertFrom-Json
if (!$cornerResult.passed) { throw 'Corner layouts need correction before broader evaluation.' }
& $ptzPython -B "$taskRoot/evaluate_audited.py" --checkpoint "$taskRoot/online-interval4/checkpoint-0320.pt" --output "$taskRoot/audited-development-0320" --split development --count 80
if ($LASTEXITCODE -ne 0) { throw 'Audited development ended with an error. Preserve and inspect the output.' }
