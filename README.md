# BMS Stem Reuse Analyzer

CLI MVP for finding reusable hits in PCM WAV stems.

## Build the Windows executable

From the project directory, with Python and PyInstaller installed:

```powershell
python -m PyInstaller --version
.\build_exe.ps1
```

The reproducible output is `dist\bms-reuse.exe`. The script invokes PyInstaller
through Python, so the PyInstaller executable does not need to be on `PATH`.

## Run

```powershell
dist\bms-reuse.exe analyze kick_stem.wav `
  --output project.bra.json --export-dir keysounds --csv events.csv
```
