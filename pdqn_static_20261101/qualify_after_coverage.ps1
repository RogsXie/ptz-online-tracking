$ErrorActionPreference = 'Stop'
$taskRoot = 'D:/EI/ptz_ue/pdqn_static_20261101'
$ptzPython = 'C:/Users/12879/.conda/envs/50mamba/python.exe'
$trainingOwnerPid = 73436
$statusPath = "$taskRoot/queued-qualification-status.json"

function Write-QueueState($phase, $detail) {
    @{ phase = $phase; detail = $detail; utc_seconds = [DateTimeOffset]::UtcNow.ToUnixTimeSeconds() } |
        ConvertTo-Json -Depth 8 | Set-Content -LiteralPath $statusPath -Encoding utf8
}

function Invoke-Ptz($entry, $arguments) {
    $ptzArguments = @('-B', "$taskRoot/$entry") + $arguments
    & $ptzPython @ptzArguments
    if ($LASTEXITCODE -ne 0) { throw "$entry ended with an error; preserve its output and inspect it." }
}

function Get-SourceManifest {
    $paths = @(
        "$taskRoot/train.py", "$taskRoot/environment.py", "$taskRoot/scenes.py",
        "$taskRoot/evaluation_scenes.py", "$taskRoot/evaluate_audited.py",
        "$taskRoot/evaluate_corner_development.py", "$taskRoot/independent_validation.py",
        "$taskRoot/../paper_reproduction/author_aligned/source.py",
        "$taskRoot/../paper_reproduction/author_aligned/environment.py",
        "$taskRoot/../paper_reproduction/integration/runtime.py",
        "$taskRoot/../paper_reproduction/integration/transport.py",
        "$taskRoot/../paper_reproduction/integration/focused_environment.py",
        "$taskRoot/../paper_reproduction/integration/pretrained_detector.py",
        "$taskRoot/../paper_reproduction/control_model_v13/detector.py",
        "$taskRoot/../paper_reproduction/hascam/reid.py", "$taskRoot/../paper_reproduction/hascam/cfff.py",
        "$taskRoot/../paper_reproduction/reference/HAS-CAM/agent.py",
        "$taskRoot/../paper_reproduction/reference/HAS-CAM/model.py",
        "$taskRoot/../paper_reproduction/reference/HAS-CAM/memory.py"
    )
    $manifest = @{}
    foreach ($path in $paths) { $manifest[$path] = (Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash }
    return $manifest
}

function Assert-FrozenSources($manifest) {
    foreach ($path in $manifest.Keys) {
        if ((Get-FileHash -LiteralPath $path -Algorithm SHA256).Hash -ne $manifest[$path]) {
            throw "An imported evaluation/control source changed during the frozen evaluation: $path"
        }
    }
}

try {
    Write-QueueState 'waiting_for_owned_training' @{ owned_python_pid = $trainingOwnerPid }
    while (Get-Process -Id $trainingOwnerPid -ErrorAction SilentlyContinue) { Start-Sleep -Seconds 2 }
    $trainingStatus = Get-Content -Raw -LiteralPath "$taskRoot/coverage-support-0320-0480/status.json" | ConvertFrom-Json
    if ($trainingStatus.completed -ne 480) { throw 'The owned coverage-training block did not finish480episodes.' }
    $manifest = Get-SourceManifest
    $manifest | ConvertTo-Json -Depth 5 | Set-Content -LiteralPath "$taskRoot/qualification-source-freeze.json" -Encoding utf8

    # Model selection uses development only. Final-test results never select a checkpoint.
    $candidates = @()
    foreach ($number in @(400,480)) {
        $resultPath = "$taskRoot/coverage-support-0320-0480/development-$('{0:D4}' -f $number)/results.json"
        $rows = @(Get-Content -Raw -LiteralPath $resultPath | ConvertFrom-Json)
        $passed = @($rows | Where-Object { $_.success }).Count
        $candidates += @{ episode = $number; passed = $passed; count = $rows.Count }
    }
    $candidates = @($candidates | Sort-Object -Property @{Expression={$_.passed};Descending=$true}, @{Expression={$_.episode};Descending=$true})
    $selected = $null
    foreach ($candidate in $candidates) {
        if ($candidate.count -ne 80 -or $candidate.passed -lt 76) { continue }
        $number = $candidate.episode
        $suffix = '{0:D4}' -f $number
        $checkpoint = "$taskRoot/coverage-support-0320-0480/checkpoint-$suffix.pt"
        $cornerOut = "$taskRoot/audited-corner-development-$suffix-support"
        Write-QueueState 'auditing_development_corners' @{ episode=$number; candidates=$candidates }
        Invoke-Ptz 'evaluate_corner_development.py' @('--checkpoint',$checkpoint,'--output',$cornerOut,'--count','32')
        Assert-FrozenSources $manifest
        $corner = Get-Content -Raw -LiteralPath "$cornerOut/independent-results.json" | ConvertFrom-Json
        if (!$corner.qualification.qualified) { continue }

        $devOut = "$taskRoot/audited-development-$suffix-support"
        Write-QueueState 'auditing_full_development' @{ episode=$number }
        Invoke-Ptz 'evaluate_audited.py' @('--checkpoint',$checkpoint,'--output',$devOut,'--split','development','--count','80')
        Assert-FrozenSources $manifest
        $dev = Get-Content -Raw -LiteralPath "$devOut/independent-results.json" | ConvertFrom-Json
        if (!$dev.qualification.qualified) { continue }
        $selected = @{ episode=$number; checkpoint=$checkpoint; corner_report="$cornerOut/independent-results.json"; development_report="$devOut/independent-results.json" }
        break
    }
    if ($null -eq $selected) {
        Write-QueueState 'development_not_qualified' @{ candidates=$candidates; next='Inspect failure traces and continue original-core learning; no validation or final test executed.' }
        exit 0
    }

    $suffix = '{0:D4}' -f $selected.episode
    $validationOut = "$taskRoot/validation-$suffix-support"
    Write-QueueState 'independent_validation' $selected
    Invoke-Ptz 'evaluate_audited.py' @('--checkpoint',$selected.checkpoint,'--output',$validationOut,'--split','validation','--count','160')
    Assert-FrozenSources $manifest
    $validationPath = "$validationOut/independent-results.json"
    $validation = Get-Content -Raw -LiteralPath $validationPath | ConvertFrom-Json
    if (!$validation.qualification.qualified) {
        Write-QueueState 'validation_not_qualified' @{ selected=$selected; validation_report=$validationPath; final_test_executed=$false }
        exit 0
    }

    $finalOut = "$taskRoot/final-test-$suffix-support"
    Write-QueueState 'independent_final_test' @{ selected=$selected; validation_report=$validationPath }
    Invoke-Ptz 'evaluate_audited.py' @('--checkpoint',$selected.checkpoint,'--output',$finalOut,'--split','final_test','--count','160','--validation-report',$validationPath)
    Assert-FrozenSources $manifest
    $finalPath = "$finalOut/independent-results.json"
    $final = Get-Content -Raw -LiteralPath $finalPath | ConvertFrom-Json
    Write-QueueState 'independent_tests_complete' @{ selected=$selected; validation_report=$validationPath; final_test_report=$finalPath; final_test_qualified=$final.qualification.qualified; goal_complete=$false; remaining='Fresh real-time representative videos, independent video checks, report and PyCharm release.' }
} catch {
    Write-QueueState 'error_requires_inspection' @{ error=$_.Exception.Message }
    throw
}
