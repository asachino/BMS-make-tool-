$ErrorActionPreference = "Stop"
$projectRoot = Split-Path -Parent $MyInvocation.MyCommand.Path
Push-Location $projectRoot
try {
    python -m PyInstaller --clean --noconfirm bms-reuse.spec `
        --distpath (Join-Path $projectRoot "dist") `
        --workpath (Join-Path $projectRoot "build\pyinstaller")
} finally {
    Pop-Location
}
