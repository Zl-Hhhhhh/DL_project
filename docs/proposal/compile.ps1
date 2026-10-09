param([string]$OutputDirectory)
$ErrorActionPreference = 'Stop'
$ProjectRoot = Split-Path (Split-Path $PSScriptRoot -Parent) -Parent
if (-not $OutputDirectory) { $OutputDirectory = Join-Path $ProjectRoot 'output\pdf' }
New-Item -ItemType Directory -Path $OutputDirectory -Force | Out-Null
$OutputDirectory = (Resolve-Path -LiteralPath $OutputDirectory).Path
$PortableTectonic = Join-Path $ProjectRoot '.cache\latex-tools\tectonic\tectonic.exe'
$PreviousCache = $env:TECTONIC_CACHE_DIR
Push-Location $PSScriptRoot
try {
    $env:TECTONIC_CACHE_DIR = Join-Path $ProjectRoot '.cache\latex-tools\tectonic-cache'
    if (Test-Path -LiteralPath $PortableTectonic) {
        & $PortableTectonic --untrusted --keep-logs --outdir $OutputDirectory project_proposal.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed.' }
    } elseif (Get-Command tectonic -ErrorAction SilentlyContinue) {
        & tectonic --untrusted --keep-logs --outdir $OutputDirectory project_proposal.tex
        if ($LASTEXITCODE -ne 0) { throw 'Tectonic compilation failed.' }
    } elseif (Get-Command pdflatex -ErrorAction SilentlyContinue) {
        for ($Pass = 0; $Pass -lt 2; $Pass++) {
            & pdflatex -interaction=nonstopmode -halt-on-error "-output-directory=$OutputDirectory" project_proposal.tex
            if ($LASTEXITCODE -ne 0) { throw 'pdfLaTeX compilation failed.' }
        }
    } else {
        throw 'Install Tectonic or a TeX distribution, or upload the source ZIP to Overleaf.'
    }
} finally {
    Pop-Location
    if ($null -eq $PreviousCache) { Remove-Item Env:TECTONIC_CACHE_DIR -ErrorAction SilentlyContinue }
    else { $env:TECTONIC_CACHE_DIR = $PreviousCache }
}
Write-Host "PDF: $(Join-Path $OutputDirectory 'project_proposal.pdf')"
