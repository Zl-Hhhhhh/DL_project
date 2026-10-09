param(
    [string]$Goal = 'samples/goals/boat.jpg',
    [string]$SourceDir = 'samples/materials/newspaper',
    [int]$SourceSampleSize = 30,
    [switch]$SmokeTest,
    [string]$Python = ''
)

$ErrorActionPreference = 'Stop'
Push-Location $PSScriptRoot
try {
    $pythonExe = if ($Python) { $Python } else { Join-Path (Split-Path $PSScriptRoot -Parent) '.venv/Scripts/python.exe' }
    if (-not (Test-Path -LiteralPath $pythonExe)) {
        throw 'Create the Python 3.10 environment first; see ENVIRONMENT.md.'
    }
    if (-not (Test-Path -LiteralPath $Goal -PathType Leaf)) {
        throw "Goal image does not exist: $Goal"
    }
    if (-not (Test-Path -LiteralPath $SourceDir -PathType Container)) {
        throw "Material folder does not exist: $SourceDir. Add JPG/PNG images or pass -SourceDir."
    }
    $materials = @(Get-ChildItem -LiteralPath $SourceDir -File | Where-Object { $_.Extension -cmatch '^\.(jpg|png|JPG|PNG)$' })
    if ($materials.Count -eq 0) {
        throw "No JPG/PNG material images found in $SourceDir."
    }
    if ($SourceSampleSize -lt 1) { throw 'SourceSampleSize must be positive.' }

    # infer.py splits image names on forward slashes.
    $inferArgs = @(
        'infer.py', '--scale', 'small',
        '--weight_path', 'weights/agents/dtd2imgnet10S_random0110',
        '--num_multi_env', '1', '--wmin', '0.1', '--wmax', '1.0',
        '--hmin', '0.1', '--hmax', '1.0', '--learned_max_steps', '10',
        '--goal', $Goal.Replace('\', '/'),
        '--source_dir', $SourceDir.Replace('\', '/').TrimEnd('/'),
        '--source_load_limit', '100', '--source_sample_size', "$SourceSampleSize",
        '--model_based', '--algo', 'sac', '--noop',
        '--window_ratio', '0.5', '--fixed_t', '9', '--video_fps', '30', '--paper_like'
    )
    if ($SmokeTest) {
        $inferArgs += @(
            '--goal_resolution', '128', '--goal_resolution_fit', 'square',
            '--source_resolution_ratio', '1.0', '--min_source_complexity', '0',
            '--scale_order', '64', '--num_cycles', '2',
            '--min_scrap_size', '0', '--sensitivity', '3.0'
        )
    } else {
        $inferArgs += @(
            '--goal_resolution', '512', '--goal_resolution_fit', 'horizontal',
            '--source_resolution_ratio', '0.6', '--min_source_complexity', '20',
            '--scale_order', '512', '256', '128', '64', '32', '--num_cycles', '8',
            '--min_scrap_size', '0.01', '--sensitivity', '3.0',
            '--complexity_aware', '--skip_negative_reward', '--disallow_duplicate'
        )
    }
    & $pythonExe -X utf8 @inferArgs
    if ($LASTEXITCODE -ne 0) { throw "Inference failed (exit code $LASTEXITCODE)." }
} finally {
    Pop-Location
}
