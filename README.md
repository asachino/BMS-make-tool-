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

## Windows GUI

GUI依存を入れて起動する場合:

```powershell
python -m pip install ".[gui]"
python gui_launcher.py
```

配布用GUI exeは次で作成できます。

```powershell
.\build_gui_exe.ps1
```

出力は `dist\bms-reuse-gui.exe` です。GUIはWAVのドラッグ＆ドロップ、非同期解析、進捗とキャンセル、分類フィルタ、代表WAV/JSON/CSV出力に対応しています。`Ctrl+O`で入力選択、`Ctrl+Enter`で解析、`Esc`でキャンセル、`Space`で代表WAVを再生できます。

類似度判定は位置合わせ後のゲイン正規化波形とスペクトルを優先し、微小ノイズや音量差を許容します。判定プロファイル、実効しきい値、位置合わせ幅、重なり警告は解析JSONに記録されます。

PySide6はQt for PythonのLGPLv3/GPLv3または商用ライセンスの対象です。再配布時は利用するライセンスの条件とPySide6の著作権表示を確認してください。

## Run

```powershell
dist\bms-reuse.exe analyze kick_stem.wav `
  --output project.bra.json --export-dir keysounds --csv events.csv
```

比較順を特徴量の近い代表から始める明示的な高速モードは
`--fast-compare` で有効化できます。代表順と判定結果が変わる可能性があるため、指定しない場合は通常モードです。
