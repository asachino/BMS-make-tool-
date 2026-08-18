# BMS Stem Reuse Analyzer

CLI/GUI tool for finding reusable hits in PCM WAV stems and preparing BMS key sounds.

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

出力は `dist\bms-reuse-gui.exe` です。GUIはWAVのドラッグ＆ドロップ、BPMグリッド付きタイムライン、非同期解析、要確認レビュー、代表WAV/JSON/CSV/BMS/BMSON出力に対応しています。`Ctrl+O`で入力選択、`Ctrl+Enter`で解析、`Esc`でキャンセル、`Space`で代表WAVを再生、`S/G/D/I`でレビュー確定できます。設定プリセットとフォルダ一括解析も利用できます。

類似度判定は位置合わせ後のゲイン正規化波形とスペクトルを優先し、微小ノイズや音量差を許容します。判定プロファイル、実効しきい値、位置合わせ幅、重なり警告は解析JSONに記録されます。

PySide6はQt for PythonのLGPLv3/GPLv3または商用ライセンスの対象です。再配布時は利用するライセンスの条件とPySide6の著作権表示を確認してください。

## Run

```powershell
dist\bms-reuse.exe analyze kick_stem.wav `
  --output project.bra.json --export-dir keysounds --csv events.csv
```

比較順を特徴量の近い代表から始める明示的な高速モードは
`--fast-compare` で有効化できます。代表順と判定結果が変わる可能性があるため、指定しない場合は通常モードです。

BMS/BMSON出力には`--bpm`が必須です。解析だけならBPMなしで実行できます。

```powershell
dist\bms-reuse.exe analyze kick_stem.wav `
  --output project.bra.json --export-dir keysounds --csv events.csv `
  --bpm 174 --bms chart.bms --bmson keysounds\chart.bmson
```

BMSON/BMSの音声名はチャートから見た相対パスです。代表WAVを別フォルダへ出力しても、CLI・batch・GUIはそのフォルダへの相対参照を自動生成します。BMSは`--bms-channel 01`（BGM互換、警告付き）が既定で、プレイ可能なキー音には`11`以降を指定できます。

フォルダ一括解析:

```powershell
dist\bms-reuse.exe batch .\stems --recursive --output-dir .\batch-out --bpm 174 --bms --bmson
```

各入力のサブフォルダと`manifest.json`が生成され、壊れた入力があっても残りを継続します。

## 解析後の再クラスタリング

保存済みの比較レポートだけを使い、音声再解析・FFTなしで使い回し度を変更できます。

```powershell
dist\bms-reuse.exe recluster project.bra.json `
  --output project.aggressive.bra.json --reuse-level aggressive
```

`--reuse-level` は `strict` / `balanced` / `aggressive` または `0.0`〜`1.0` の連続閾値です。`--threshold` と `--spectral-threshold` で個別指定もできます。既存のS/G/D/Iレビューは優先され、I除外はヒット・比較・クラスタ・イベント・代表WAV数に反映されます。

GUIから呼ぶ最小APIは `bms_reuse.recluster_result(result, reuse_level="balanced")` です。同じ`AnalysisResult`を更新して返し、`result.plan.clusters`（代表1つ/クラスタ）、`result.plan.events`（非除外ヒット1件/イベント）、`result.summary`、`result.settings["exports"]`、`result.settings["recluster_thresholds"]`を同期します。レビュー操作は `bms_reuse.set_review_state(result, hit_id, "S|G|D|I", target_cluster=...)` に集約でき、`None`（または`ACTIVE`）でIを復元します。`reexport=False`ならJSON/メタデータだけ更新します。

schema v2 JSONには、`recluster.profile`、`recluster.thresholds`（waveform/spectral/gain_tolerance_db）、`review`、`exports`、`validation`が保存されます。再クラスタリングは保存された`comparisons`のスコアとヒットの`features`/座標だけをデータ契約として受け取ります。

## 打楽器プロファイルとループ切断

`--instrument kick|snare|hihat|other` は帯域比・トランジェント・減衰などの特徴量を保存し、異なる楽器同士が自動的に同じクラスタへ入ることを防ぎます。解析JSONの`settings.instrument_profile`と各ヒットの`features.instrument`で再現できます。

ループ／刻みは既定の自動オンセット検出を保ったまま、秒・拍・小節・手動点・反復パターンへ切り替えられます。

```powershell
dist\bms-reuse.exe analyze stem.wav --instrument hihat `
  --cut-plan grid --bpm 174 --loop-beats 1
dist\bms-reuse.exe analyze loop.wav --cut-plan manual --loop-points 0,0.25,0.5,0.75
dist\bms-reuse.exe analyze roll.wav --cut-plan pattern --loop-pattern 0.125,0.125,0.25
```

各ヒットには音量・音色・密度（切り刻み）・テール・ステレオ・グリッド外（`OFF_GRID`）変化の`automation.variations`が保存されます。これはレビュー用の警告で、類似クラスタを自動的に拒否するものではありません。旧APIの`loop_rule`／`loop_seconds`等も引き続き利用できます。
