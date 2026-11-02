$ErrorActionPreference = 'Stop'
$taskPython = 'C:/Users/12879/.conda/envs/50mamba/python.exe'
$taskRoot = 'D:/EI/ptz_ue/pdqn_static_20261101/speed_revision_20261101'
Set-Location -LiteralPath $taskRoot
$taskStatus = Get-Content -LiteralPath 'unified_motion_20261101/pdqn_training_v1/status.json' -Raw | ConvertFrom-Json
if ($taskStatus.completed_segments -ne 48) { throw 'Training has not completed; refusing concurrent GPU/UE jobs.' }
& $taskPython prove_unified_training.py *> unified_motion_20261101/training-proof-console.log
if ($LASTEXITCODE -ne 0) { throw 'Training evidence inspection failed.' }
foreach ($taskCase in @('known9','variable','staticcorner','staticlarge')) {
    $taskSeconds = if ($taskCase.StartsWith('static')) { 20 } else { 30 }
    $taskOutput = "unified_motion_20261101/live-$taskCase-v1"
    & $taskPython live_unified_motion.py --case $taskCase --checkpoint unified_motion_20261101/pdqn_training_v1/last.pt --output $taskOutput --seconds $taskSeconds *> "unified_motion_20261101/live-$taskCase-console.log"
    if ($LASTEXITCODE -ne 0) { throw "Live run failed: $taskCase" }
    & $taskPython render_unified_video.py $taskOutput *> "unified_motion_20261101/render-$taskCase-console.log"
    if ($LASTEXITCODE -ne 0) { throw "Rendering failed: $taskCase" }
    Write-Output "FOCUSED VIDEO COMPLETE $taskCase"
}
& $taskPython summarize_unified_experiment.py *> unified_motion_20261101/summary-console.log
if ($LASTEXITCODE -ne 0) { throw 'Focused summary failed.' }
