$ErrorActionPreference = "Stop"

python -m pip install ".[gui]"
python -m PyInstaller --noconfirm --clean bms-reuse-gui.spec
Write-Host "GUI executable: dist\bms-reuse-gui.exe"
