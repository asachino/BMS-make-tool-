"""Qt desktop interface for the BMS stem reuse analyzer.

The GUI is intentionally a client of :func:`analyze_file`; it owns only
presentation, export orchestration, and the cancellation boundary.
"""

from __future__ import annotations

import os
import sys
import tempfile
import threading
import time
import traceback
import wave
from pathlib import Path

try:
    from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, QSize, QThread, QTimer, Qt, Signal, Slot
    from PySide6.QtGui import QAction, QColor, QFont, QFontDatabase, QKeySequence, QPainter, QPen, QShortcut
    from PySide6.QtWidgets import (
        QApplication,
        QCheckBox,
        QComboBox,
        QDoubleSpinBox,
        QFileDialog,
        QFormLayout,
        QFrame,
        QGridLayout,
        QGroupBox,
        QHBoxLayout,
        QHeaderView,
        QLabel,
        QLineEdit,
        QListWidget,
        QListWidgetItem,
        QMainWindow,
        QMessageBox,
        QPlainTextEdit,
        QProgressBar,
        QPushButton,
        QScrollArea,
        QSizePolicy,
        QSpinBox,
        QSplitter,
        QSlider,
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by the CLI-only install
    raise RuntimeError("GUI requires PySide6. Install with: pip install .[gui]") from exc

from .application import AnalysisCancelled, AnalysisResult, analyze_file, exclude_hit, recluster_result, record_output_timing, refresh_reproducibility
from .batch import run_batch
from ._numeric import np
from .audio.loader import load_audio, mono_signal
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.wav_exporter import write_hit_wavs
from .export.bms_exporter import relative_sample_prefix, write_bms
from .export.bmson_exporter import write_bmson
from .export.quality import validate_exports
from .project.presets import load_preset, save_preset
from .clustering.reuse_plan import Cluster


APP_NAME = "StemReuse"
APP_VERSION = "0.3.0"

CLASS_COLORS = {
    "BASE": "#60a5fa",
    "SAME": "#34d399",
    "GAIN_VARIANT": "#fbbf24",
    "DIFFERENT": "#fb7185",
    "UNSURE": "#c084fc",
    "OVERLAP": "#f97316",
    "IGNORED": "#64748b",
}

CLASS_LABELS = {
    "BASE": "基準サンプル",
    "SAME": "同一",
    "GAIN_VARIANT": "音量違い",
    "DIFFERENT": "別音",
    "UNSURE": "判定保留",
    "OVERLAP": "音の重なり",
    "IGNORED": "除外",
}

FILTER_LABELS = {
    "All": "すべて",
    "BASE": "基準サンプル",
    "SAME": "同一",
    "GAIN_VARIANT": "音量違い",
    "DIFFERENT": "別音",
    "UNSURE": "判定保留",
    "OVERLAP": "音の重なり",
    "IGNORED": "除外",
    "REVIEW": "要確認",
}

INSTRUMENT_LABELS = {
    "kick": "キック",
    "snare": "スネア",
    "clap": "クラップ",
    "hihat": "ハイハット",
    "other": "その他",
}


def localize_progress(message: str) -> str:
    """Translate worker stage messages without changing core/JSON identifiers."""
    if message == "Loading audio":
        return "音声を読み込み中"
    if message == "Detecting onsets":
        return "オンセットを検出中"
    if message.startswith("Extracting hits "):
        return f"ヒットを切り出し中 {message[16:]}"
    if message.startswith("Extracting ") and message.endswith(" hits"):
        return f"{message[11:-5]}個のヒットを切り出し中"
    if message == "Extracting features":
        return "特徴量を抽出中"
    if message.startswith("Extracting features "):
        return f"特徴量を抽出中 {message[20:]}"
    if message == "Comparing and clustering hits":
        return "比較・クラスタリング中"
    if message.startswith("Comparing and clustering hits "):
        detail = message[30:].replace(" comparisons", "件比較").replace(" cache hits", "件キャッシュ再利用")
        return f"比較・クラスタリング中 {detail}"
    if message == "Writing analysis JSON":
        return "解析JSONを書き出し中"
    if message == "Writing representative samples":
        return "代表サンプルを書き出し中"
    if message == "Writing event CSV":
        return "イベントCSVを書き出し中"
    if message == "Analysis complete":
        return "解析完了"
    return message

DARK_STYLE = """
QMainWindow, QWidget { background: #101317; color: #f5f7fb; font-family: "Yu Gothic UI", "Meiryo UI", "Segoe UI"; font-size: 10pt; }
QLabel, QCheckBox { background: transparent; }
QFrame#Panel { background: #1b1f26; border: 1px solid #303640; border-radius: 8px; }
QGroupBox { background: transparent; border: 1px solid #303640; border-radius: 8px; }
QFrame#DropZone { border: 1px dashed #66707f; background: #1c222b; border-radius: 8px; }
QFrame#DropZone[dragActive="true"] { background: #242a33; border: 1px solid #8a95a5; }
QWidget#WaveformView { background: transparent; border: none; }
QLabel#Brand { color: #dce2ea; font-size: 17pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Kicker { color: #9aa4b2; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Title { color: #ffffff; font-size: 20pt; font-weight: 700; }
QLabel#Subtle, QLabel#Status { color: #9aa4b2; }
QLabel#Value { color: #ffffff; font-size: 17pt; font-weight: 700; }
QLabel#MetricCaption { color: #9aa4b2; font-size: 8pt; font-weight: 600; text-transform: uppercase; }
QLabel#DropTitle { color: #e8edf3; font-size: 12pt; font-weight: 700; }
QLabel#DropHint { color: #a9b1bd; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #1b1f26; border: 1px solid #35404e; border-radius: 6px; padding: 7px 9px; color: #f5f7fb; min-height: 20px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #007aff; padding: 6px 8px; }
QComboBox QAbstractItemView { background: #1b1f26; color: #f5f7fb; selection-background-color: #343c47; }
QPushButton { background: #242a33; color: #e9edf4; border: 1px solid #35404e; border-radius: 7px; padding: 8px 14px; min-height: 20px; }
QPushButton:hover { background: #2d3541; border-color: #59687d; }
QPushButton:pressed { background: #374354; }
QPushButton:disabled { color: #66707f; background: #1b1f26; }
QPushButton#Primary { background: #007aff; border-color: #007aff; color: #ffffff; font-weight: 700; }
QPushButton#Primary:hover { background: #0a84ff; border-color: #0a84ff; }
QPushButton#Primary:pressed { background: #006ee6; border-color: #006ee6; }
QPushButton#Danger { background: #42222b; border-color: #ff375f; color: #ffd7df; }
QCheckBox { spacing: 8px; padding: 2px 0; color: #d7dce5; }
QCheckBox:hover { color: #ffffff; }
QCheckBox:pressed { color: #d7dce5; }
QGroupBox { margin-top: 10px; padding: 16px 12px 12px 12px; font-weight: 700; color: #c4cad3; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QTableWidget { background: #1b1f26; alternate-background-color: #20252d; border: 1px solid #303640; border-radius: 8px; gridline-color: #2a3039; selection-background-color: #343c47; selection-color: #ffffff; }
QHeaderView::section { background: #242a33; color: #c4cad3; border: none; border-right: 1px solid #303640; border-bottom: 1px solid #303640; padding: 8px; font-weight: 700; }
QListWidget { background: #1b1f26; border: 1px solid #303640; border-radius: 7px; }
QListWidget::item { padding: 6px; }
QListWidget::item:selected { background: #343c47; }
QPlainTextEdit { background: #1b1f26; border: 1px solid #303640; border-radius: 7px; color: #9aa4b2; font-family: "Cascadia Mono", Consolas; font-size: 9pt; }
QProgressBar { background: #1b1f26; border: 1px solid #35404e; border-radius: 4px; text-align: center; color: #d7dce5; height: 12px; }
QProgressBar::chunk { background: #007aff; border-radius: 3px; }
QStatusBar { background: #111419; color: #9aa4b2; }
QSplitter::handle { background: #242b35; }
QToolTip { background: #222a35; color: #ffffff; border: 1px solid #4b5768; padding: 6px; }
"""

LIGHT_STYLE = """
QMainWindow, QWidget { background: #f3f5f8; color: #1c2430; font-family: "Yu Gothic UI", "Meiryo UI", "Segoe UI"; font-size: 10pt; }
QLabel, QCheckBox { background: transparent; }
QFrame#Panel { background: #ffffff; border: 1px solid #dfe4eb; border-radius: 8px; }
QGroupBox { background: transparent; border: 1px solid #dfe4eb; border-radius: 8px; }
QFrame#DropZone { border: 1px dashed #aab7c7; background: #f8fafc; border-radius: 8px; }
QFrame#DropZone[dragActive="true"] { background: #eef2f6; border: 1px solid #8d9aaa; }
QWidget#WaveformView { background: transparent; border: none; }
QLabel#Brand { color: #344054; font-size: 17pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Kicker { color: #667085; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Title { color: #102033; font-size: 20pt; font-weight: 700; }
QLabel#Subtle, QLabel#Status { color: #667085; }
QLabel#Value { color: #102033; font-size: 17pt; font-weight: 700; }
QLabel#MetricCaption { color: #667085; font-size: 8pt; font-weight: 600; text-transform: uppercase; }
QLabel#DropTitle { color: #344054; font-size: 12pt; font-weight: 700; }
QLabel#DropHint { color: #667085; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 6px; padding: 7px 9px; color: #1c2430; min-height: 20px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 2px solid #007aff; padding: 6px 8px; }
QComboBox QAbstractItemView { background: #ffffff; color: #1c2430; selection-background-color: #e8edf3; }
QPushButton { background: #ffffff; color: #1c2430; border: 1px solid #d7dee8; border-radius: 7px; padding: 8px 14px; min-height: 20px; }
QPushButton:hover { background: #f4f6f8; border-color: #b8c2cf; }
QPushButton:pressed { background: #e8edf3; }
QPushButton:disabled { color: #a0a8b4; background: #f3f5f8; }
QPushButton#Primary { background: #007aff; border-color: #007aff; color: #ffffff; font-weight: 700; }
QPushButton#Primary:hover { background: #0a84ff; border-color: #0a84ff; }
QPushButton#Primary:pressed { background: #006ee6; border-color: #006ee6; }
QPushButton#Danger { background: #fff0f3; border-color: #ff375f; color: #b4233f; }
QCheckBox { spacing: 8px; padding: 2px 0; color: #334155; }
QCheckBox:hover { color: #102033; }
QCheckBox:pressed { color: #475467; }
QGroupBox { margin-top: 10px; padding: 16px 12px 12px 12px; font-weight: 700; color: #475467; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QTableWidget { background: #ffffff; alternate-background-color: #f7f9fc; border: 1px solid #dfe4eb; border-radius: 8px; gridline-color: #e8edf3; selection-background-color: #e6edf5; selection-color: #102033; }
QHeaderView::section { background: #f3f5f8; color: #475467; border: none; border-right: 1px solid #dfe4eb; border-bottom: 1px solid #dfe4eb; padding: 8px; font-weight: 700; }
QListWidget { background: #ffffff; border: 1px solid #dfe4eb; border-radius: 7px; }
QListWidget::item { padding: 6px; }
QListWidget::item:selected { background: #e6edf5; }
QPlainTextEdit { background: #ffffff; border: 1px solid #dfe4eb; border-radius: 7px; color: #667085; font-family: "Cascadia Mono", Consolas; font-size: 9pt; }
QProgressBar { background: #ffffff; border: 1px solid #cbd5e1; border-radius: 4px; text-align: center; color: #344054; height: 12px; }
QProgressBar::chunk { background: #007aff; border-radius: 3px; }
QStatusBar { background: #f8fafc; color: #667085; }
QSplitter::handle { background: #e5eaf1; }
QToolTip { background: #ffffff; color: #1c2430; border: 1px solid #cbd5e1; padding: 6px; }
"""


def format_seconds(value: float) -> str:
    """Format an audio time without hiding sub-second precision."""
    value = max(0.0, float(value))
    minutes, seconds = divmod(value, 60.0)
    return f"{int(minutes):02d}:{seconds:05.2f}"


def classify_hits(result: AnalysisResult) -> list[dict]:
    """Flatten the core result into rows suitable for a review table."""
    report_by_hit = {report.candidate_id: report for report in result.comparisons}
    sample_by_hit = {
        hit_id: f"sample_{cluster.id:03d}"
        for cluster in result.plan.clusters
        for hit_id in cluster.hit_ids
    }
    cluster_by_hit = {
        hit_id: cluster.id
        for cluster in result.plan.clusters
        for hit_id in cluster.hit_ids
    }
    rows: list[dict] = []
    overrides = result.settings.get("review_overrides", {}) if isinstance(result.settings, dict) else {}
    for hit in result.hits:
        report = report_by_hit.get(hit.id)
        classification = report.classification if report else "BASE"
        override = overrides.get(str(hit.id)) if isinstance(overrides, dict) else None
        if override:
            classification = "IGNORED" if override == "I" else {"S": "SAME", "G": "GAIN_VARIANT", "D": "DIFFERENT"}.get(override, override)
        rows.append(
            {
                "id": hit.id,
                "time": hit.time,
                "classification": classification,
                "confidence": report.confidence if report else 100.0,
                "gain_db": report.gain_db if report else 0.0,
                "raw": report.raw_similarity if report else 1.0,
                "spectral": report.spectral_similarity if report else 1.0,
                "waveform": report.gain_normalized_similarity if report else 1.0,
                "attack": report.attack_similarity if report else 1.0,
                "body": report.body_similarity if report else 1.0,
                "tail": report.tail_similarity if report else 1.0,
                "overlap": bool(hit.overlap_warning or (report and report.overlap_warning)),
                "sample_id": sample_by_hit.get(hit.id, "—"),
                "cluster_id": cluster_by_hit.get(hit.id),
                "start": hit.source_start / max(1, result.sample_rate),
                "end": hit.source_end / max(1, result.sample_rate),
                "review_override": override,
            }
        )
    return rows


class AnalysisWorker(QThread):
    """Run the existing analyzer away from the UI thread."""

    progress = Signal(int, str)
    result_ready = Signal(object, object)
    failed = Signal(str)
    canceled = Signal()

    def __init__(self, input_path: Path, settings: dict, outputs: dict, parent: QObject | None = None):
        super().__init__(parent)
        self.input_path = input_path
        self.settings = settings
        self.outputs = outputs
        self.cancel_event = threading.Event()
        self._last_progress_at = 0.0
        self._last_progress_stage = ""

    def cancel(self) -> None:
        self.cancel_event.set()

    def _on_progress(self, percent: int, message: str) -> None:
        localized = localize_progress(message)
        stage = localized.split(" ", 1)[0]
        now = time.monotonic()
        if (
            self._last_progress_stage == stage
            and now - self._last_progress_at < 0.15
            and percent < 100
        ):
            return
        self._last_progress_at = now
        self._last_progress_stage = stage
        self.progress.emit(max(0, min(100, percent)), localized)

    def run(self) -> None:  # noqa: D401 - Qt thread entry point
        try:
            result = analyze_file(
                self.input_path,
                progress=self._on_progress,
                is_cancelled=self.cancel_event.is_set,
                **self.settings,
            )
            if self.cancel_event.is_set():
                raise AnalysisCancelled()
            exported: dict[str, object] = {}
            output_started = time.perf_counter()
            self._on_progress(94, "Writing analysis JSON")
            json_path = self.outputs.get("json")
            if json_path:
                exported["json"] = str(write_json(json_path, result.to_dict()))
            if self.cancel_event.is_set():
                raise AnalysisCancelled()
            export_dir = self.outputs.get("samples")
            if export_dir:
                self._on_progress(96, "Writing representative samples")
                audio = load_audio(self.input_path)
                exported["samples_dir"] = str(export_dir)
                exported["samples"] = [
                    str(path)
                    for path in write_hit_wavs(
                        export_dir,
                        audio,
                        result.hits,
                        result.plan,
                        fade_in_ms=float(self.settings.get("fade_in_ms", 0.0)),
                        fade_out_ms=float(self.settings.get("fade_out_ms", 0.0)),
                    )
                ]
            csv_path = self.outputs.get("csv")
            if csv_path:
                self._on_progress(98, "Writing event CSV")
                exported["csv"] = str(write_hits_csv(csv_path, result.hits, result.plan.events))
            bms_path = self.outputs.get("bms")
            if bms_path:
                exported["bms"] = str(write_bms(
                    bms_path,
                    result.plan,
                    bpm=self.settings.get("bpm"),
                    offset=float(self.settings.get("offset", 0.0)),
                    subdivision=int(self.settings.get("subdivision", 16)),
                    channel=str(self.settings.get("bms_channel", "01")),
                    wav_prefix=relative_sample_prefix(bms_path, self.outputs.get("samples")),
                ))
            bmson_path = self.outputs.get("bmson")
            if bmson_path:
                exported["bmson"] = str(write_bmson(
                    bmson_path,
                    result.plan,
                    bpm=self.settings.get("bpm"),
                    offset=float(self.settings.get("offset", 0.0)),
                ))
            exported["validation"] = validate_exports(result, exported)
            result.settings["validation"] = exported["validation"]
            result.settings["exports"] = {key: value for key, value in exported.items() if key != "validation"}
            record_output_timing(result, time.perf_counter() - output_started)
            if json_path:
                exported["json"] = str(write_json(json_path, result.to_dict()))
            self._on_progress(100, "Analysis complete")
            self.result_ready.emit(result, exported)
        except AnalysisCancelled:
            self.canceled.emit()
        except Exception:
            self.failed.emit(traceback.format_exc())


class BatchWorker(QThread):
    """Run folder analysis off the UI thread and keep failed inputs in a manifest."""

    progress = Signal(int, str)
    result_ready = Signal(object)
    failed = Signal(str)

    def __init__(self, folder: Path, output_dir: Path, settings: dict, parent: QObject | None = None):
        super().__init__(parent)
        self.folder = folder
        self.output_dir = output_dir
        self.settings = settings

    def run(self) -> None:
        try:
            manifest = run_batch(
                self.folder,
                output_dir=self.output_dir,
                recursive=True,
                progress=lambda percent, message: self.progress.emit(percent, message),
                export_bms=True,
                export_bmson=True,
                **self.settings,
            )
            self.result_ready.emit(manifest)
        except Exception:
            self.failed.emit(traceback.format_exc())


class DropZone(QFrame):
    path_dropped = Signal(str)
    browse_requested = Signal()

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("DropZone")
        self.setAcceptDrops(True)
        self.setFocusPolicy(Qt.StrongFocus)
        self.setMinimumHeight(132)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.setSpacing(5)
        self.icon = QLabel("◈")
        self.icon.setAlignment(Qt.AlignCenter)
        self.icon.setStyleSheet("color:#38bdf8;font-size:25pt;font-weight:700;")
        self.title = QLabel("WAVステムをここにドロップ")
        self.title.setObjectName("DropTitle")
        self.title.setAlignment(Qt.AlignCenter)
        self.hint = QLabel("またはEnterで選択  ·  PCM WAV")
        self.hint.setObjectName("DropHint")
        self.hint.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.icon)
        layout.addWidget(self.title)
        layout.addWidget(self.hint)

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls() or event.mimeData().hasText():
            event.acceptProposedAction()
            self.setProperty("dragActive", True)
            self.style().unpolish(self)
            self.style().polish(self)
        else:
            event.ignore()

    def dragLeaveEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        event.accept()

    def dropEvent(self, event) -> None:
        self.setProperty("dragActive", False)
        self.style().unpolish(self)
        self.style().polish(self)
        urls = event.mimeData().urls()
        if urls:
            self.path_dropped.emit(urls[0].toLocalFile())
        elif event.mimeData().hasText():
            self.path_dropped.emit(event.mimeData().text().strip().strip('"'))
        event.acceptProposedAction()

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton:
            self.browse_requested.emit()
        super().mousePressEvent(event)

    def keyPressEvent(self, event) -> None:
        if event.key() in (Qt.Key_Return, Qt.Key_Enter, Qt.Key_Space):
            self.browse_requested.emit()
        else:
            super().keyPressEvent(event)


def _waveform_envelope(audio, max_points: int = 1600) -> list[tuple[float, ...]]:
    """Shrink the real source samples to min/max buckets for QPainter."""
    frame_count = audio.frame_count
    if frame_count <= 0:
        return []
    bucket_count = min(max_points, frame_count)
    samples = audio.samples
    stereo = int(getattr(audio, "channels", 1)) >= 2 and hasattr(samples, "shape") and len(samples.shape) > 1
    mono = None if stereo else mono_signal(audio)
    points: list[tuple[float, ...]] = []
    for index in range(bucket_count):
        start = index * frame_count // bucket_count
        end = max(start + 1, (index + 1) * frame_count // bucket_count)
        chunk = samples[start:end]
        if stereo:
            values = []
            for channel in (0, 1):
                channel_chunk = chunk[:, channel]
                values.extend((float(np.min(channel_chunk)), float(np.max(channel_chunk))))
            points.append(tuple(values))
        else:
            mono_chunk = mono[start:end]
            if np is not None and hasattr(mono_chunk, "shape"):
                low = float(np.min(mono_chunk))
                high = float(np.max(mono_chunk))
            else:
                values = [float(value) for value in mono_chunk]
                low = min(values, default=0.0)
                high = max(values, default=0.0)
            points.append((low, high))
    return points


class PreviewLoadWorker(QThread):
    loaded = Signal(str, object, object)
    failed = Signal(str, str)

    def __init__(self, path: Path, parent: QObject | None = None):
        super().__init__(parent)
        self.path = path

    def run(self) -> None:
        try:
            audio = load_audio(self.path)
            self.loaded.emit(str(self.path), audio, _waveform_envelope(audio))
        except Exception as exc:
            self.failed.emit(str(self.path), str(exc))


class PlaybackSegmentWorker(QThread):
    ready = Signal(str, float)
    failed = Signal(str)

    def __init__(self, source_path: Path, start_seconds: float, parent: QObject | None = None):
        super().__init__(parent)
        self.source_path = source_path
        self.start_seconds = max(0.0, float(start_seconds))

    def run(self) -> None:
        target_path: Path | None = None
        try:
            with wave.open(str(self.source_path), "rb") as source:
                rate = source.getframerate()
                start_frame = min(source.getnframes(), round(self.start_seconds * rate))
                source.setpos(start_frame)
                handle = tempfile.NamedTemporaryFile(prefix="bms-reuse-preview-", suffix=".wav", delete=False)
                target_path = Path(handle.name)
                handle.close()
                with wave.open(str(target_path), "wb") as target:
                    target.setparams(source.getparams())
                    while not self.isInterruptionRequested():
                        chunk = source.readframes(65536)
                        if not chunk:
                            break
                        target.writeframesraw(chunk)
                    target.writeframes(b"")
            if self.isInterruptionRequested():
                target_path.unlink(missing_ok=True)
                return
            self.ready.emit(str(target_path), self.start_seconds)
        except Exception as exc:
            if target_path:
                target_path.unlink(missing_ok=True)
            self.failed.emit(str(exc))


class WaveformCanvas(QWidget):
    clicked = Signal(float)

    def __init__(self, owner: "WaveformView", parent: QWidget | None = None):
        super().__init__(parent)
        self.owner = owner
        self.setMinimumHeight(155)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def mousePressEvent(self, event) -> None:
        if event.button() == Qt.LeftButton and self.owner.duration > 0:
            rect = self.rect().adjusted(10, 8, -10, -28)
            fraction = max(0.0, min(1.0, (event.position().x() - rect.left()) / max(1, rect.width())))
            start, visible = self.owner._view_range()
            position = start + fraction * visible
            self.clicked.emit(position)
            if self.owner.rows:
                target = min(self.owner.rows, key=lambda row: abs(float(row["time"]) - position))
                self.owner.selected_id = target["id"]
                self.owner.hit_selected.emit(target["id"])
                self.update()
        super().mousePressEvent(event)

    def paintEvent(self, event) -> None:
        owner = self.owner
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.fillRect(self.rect(), QColor("#b7b9bb"))
        rect = self.rect().adjusted(10, 8, -10, -30)
        duration = owner.duration
        if duration <= 0:
            painter.setPen(QColor("#4f555a"))
            painter.drawText(rect, Qt.AlignCenter, "解析後に表示")
            return

        start, visible = owner._view_range()
        end = start + visible
        painter.setPen(QPen(QColor("#9da1a4"), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.left() + round(rect.width() * fraction)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            painter.setPen(QColor("#4f555a"))
            painter.drawText(x + 3, self.height() - 8, format_seconds(start + visible * fraction))
            painter.setPen(QPen(QColor("#9da1a4"), 1))

        # Thin musical grid lines sit behind the real source waveform.
        if owner.bpm and owner.bpm > 0:
            beat_step = 60.0 / owner.bpm / max(1, owner.subdivision)
            grid_step = beat_step
            while visible / max(grid_step, 1e-9) > max(200, rect.width() * 2):
                grid_step *= 2.0
            grid_time = owner.offset + max(0, int((start - owner.offset) / grid_step) - 1) * grid_step
            while grid_time <= end + grid_step:
                if grid_time >= start:
                    fraction = (grid_time - start) / max(visible, 1e-9)
                    x = rect.left() + round(fraction * rect.width())
                    beat_index = round((grid_time - owner.offset) / beat_step)
                    if beat_index % (owner.subdivision * 4) == 0:
                        color, width, label = QColor("#777d82"), 2, f"小節{beat_index // (owner.subdivision * 4) + 1}"
                    elif beat_index % owner.subdivision == 0:
                        color, width, label = QColor("#8f9599"), 1, "拍"
                    else:
                        color, width, label = QColor("#a5a9ac"), 1, ""
                    painter.setPen(QPen(color, width))
                    painter.drawLine(x, rect.top(), x, rect.bottom())
                    if label:
                        painter.setPen(color)
                        painter.drawText(x + 3, rect.top() + 13, label)
                grid_time += grid_step

        points = owner.waveform_points
        stereo = bool(points and len(points[0]) >= 4)
        if stereo:
            channel_height = max(1, (rect.height() - 8) // 2)
            channel_rects = (
                QRect(rect.left(), rect.top(), rect.width(), channel_height),
                QRect(rect.left(), rect.top() + channel_height + 8, rect.width(), channel_height),
            )
        else:
            channel_rects = (rect,)
        painter.setPen(QPen(QColor("#8f9599"), 1))
        for channel_rect in channel_rects:
            center = channel_rect.center().y()
            painter.drawLine(channel_rect.left(), center, channel_rect.right(), center)

        if points:
            painter.setPen(QPen(QColor("#20262b"), 1))
            denominator = max(1, len(points) - 1)
            for index, point in enumerate(points):
                point_time = duration * index / denominator
                if point_time < start or point_time > end:
                    continue
                x = rect.left() + round((point_time - start) / max(visible, 1e-9) * rect.width())
                if len(point) >= 4:
                    for channel_index, (low, high) in enumerate(((point[0], point[1]), (point[2], point[3]))):
                        channel_rect = channel_rects[channel_index]
                        center = channel_rect.center().y()
                        half_height = max(1, channel_rect.height() // 2 - 3)
                        top = center - round(max(-1.0, min(1.0, high)) * half_height)
                        bottom = center - round(max(-1.0, min(1.0, low)) * half_height)
                        painter.drawLine(x, top, x, bottom)
                else:
                    low, high = point[0], point[1]
                    channel_rect = channel_rects[0]
                    center = channel_rect.center().y()
                    half_height = max(1, channel_rect.height() // 2 - 3)
                    top = center - round(max(-1.0, min(1.0, high)) * half_height)
                    bottom = center - round(max(-1.0, min(1.0, low)) * half_height)
                    painter.drawLine(x, top, x, bottom)
        elif not owner.source_path:
            painter.setPen(QColor("#4f555a"))
            painter.drawText(rect, Qt.AlignCenter, "入力WAVを選択すると波形を表示")

        nearest_id = None
        if owner.rows and owner.position >= 0:
            nearest = min(owner.rows, key=lambda row: abs(float(row["time"]) - owner.position))
            if abs(float(nearest["time"]) - owner.position) <= max(0.04, visible / max(1, len(owner.rows) * 2)):
                nearest_id = nearest["id"]
        for marker_index, row in enumerate(owner.rows):
            marker_time = float(row["time"])
            if marker_time < start or marker_time > end:
                continue
            x = rect.left() + round((marker_time - start) / max(visible, 1e-9) * rect.width())
            selected = row["id"] == owner.selected_id or row["id"] == nearest_id
            color = QColor(CLASS_COLORS.get(row["classification"], "#94a3b8"))
            painter.setPen(QPen(color, 3 if selected else 1))
            painter.drawLine(x, rect.top(), x, rect.bottom())
            label = f"C{row.get('cluster_id') or '?'} {CLASS_LABELS.get(row['classification'], row['classification'])}"
            painter.setPen(color)
            painter.drawText(x + 3, rect.top() + 15 + (marker_index % 3) * 15, label)
            if selected:
                painter.setBrush(color)
                painter.drawEllipse(QPoint(x, rect.top() - 1), 3, 3)
                painter.setBrush(Qt.NoBrush)

        if owner.position <= duration:
            x = rect.left() + round((owner.position - start) / max(visible, 1e-9) * rect.width())
            if rect.left() <= x <= rect.right():
                painter.setPen(QPen(QColor("#ffffff"), 2))
                painter.drawLine(x, rect.top(), x, rect.bottom())


class WaveformView(QWidget):
    hit_selected = Signal(int)
    playback_status = Signal(str)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(235)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        self.duration = 0.0
        self.sample_rate = 0
        self.bpm: float | None = None
        self.offset = 0.0
        self.subdivision = 16
        self.position = 0.0
        self.rows: list[dict] = []
        self.selected_id: int | None = None
        self.source_path: Path | None = None
        self.audio = None
        self.waveform_points: list[tuple[float, ...]] = []
        self._viewport_seconds = 12.0
        self._view_start = 0.0
        self._follow_playhead = True
        self._playing = False
        self._playback_started_at = 0.0
        self._playback_offset = 0.0
        self._segment_path: Path | None = None
        self._pending_start: float | None = None
        self._preview_worker: PreviewLoadWorker | None = None
        self._playback_worker: PlaybackSegmentWorker | None = None
        self._slider_dragging = False
        self.setToolTip("波形をクリックしてシーク、ヒットマーカーをクリックして詳細を確認")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(5)
        self.canvas = WaveformCanvas(self)
        layout.addWidget(self.canvas, 1)
        controls = QHBoxLayout()
        self.playback_play_button = QPushButton("再生")
        self.playback_pause_button = QPushButton("一時停止")
        self.playback_stop_button = QPushButton("停止")
        for button, name in (
            (self.playback_play_button, "元WAVを再生"),
            (self.playback_pause_button, "元WAVを一時停止"),
            (self.playback_stop_button, "元WAVを停止"),
        ):
            button.setAccessibleName(name)
            button.setMinimumWidth(64)
        self.playback_play_button.clicked.connect(self.play_playback)
        self.playback_pause_button.clicked.connect(self.pause_playback)
        self.playback_stop_button.clicked.connect(self.stop_playback)
        controls.addWidget(self.playback_play_button)
        controls.addWidget(self.playback_pause_button)
        controls.addWidget(self.playback_stop_button)
        self.seek_slider = QSlider(Qt.Horizontal)
        self.seek_slider.setRange(0, 10000)
        self.seek_slider.setAccessibleName("元WAV再生位置")
        self.seek_slider.setToolTip("再生位置を変更（波形クリックでもシーク）")
        self.seek_slider.sliderPressed.connect(self._slider_pressed)
        self.seek_slider.sliderMoved.connect(self._slider_moved)
        self.seek_slider.sliderReleased.connect(self._slider_released)
        controls.addWidget(self.seek_slider, 1)
        self.playback_time_label = QLabel("00:00.00 / 00:00.00")
        self.playback_time_label.setMinimumWidth(125)
        controls.addWidget(self.playback_time_label)
        layout.addLayout(controls)
        view_controls = QHBoxLayout()
        view_controls.addWidget(QLabel("表示幅"))
        self.zoom_slider = QSlider(Qt.Horizontal)
        self.zoom_slider.setRange(1, 120)
        self.zoom_slider.setValue(12)
        self.zoom_slider.setAccessibleName("波形表示幅")
        self.zoom_slider.setToolTip("波形を拡大・縮小（表示秒数）")
        self.zoom_slider.valueChanged.connect(self._on_zoom_changed)
        view_controls.addWidget(self.zoom_slider, 1)
        self.zoom_label = QLabel("12.0秒")
        self.zoom_label.setMinimumWidth(58)
        view_controls.addWidget(self.zoom_label)
        view_controls.addWidget(QLabel("横位置"))
        self.pan_slider = QSlider(Qt.Horizontal)
        self.pan_slider.setRange(0, 10000)
        self.pan_slider.setAccessibleName("波形横位置")
        self.pan_slider.setToolTip("波形の表示位置を横スクロール")
        self.pan_slider.valueChanged.connect(self._on_pan_changed)
        view_controls.addWidget(self.pan_slider, 1)
        layout.addLayout(view_controls)
        self.playback_hint_label = QLabel("解析後に表示")
        self.playback_hint_label.setObjectName("Subtle")
        layout.addWidget(self.playback_hint_label)
        self.playback_timer = QTimer(self)
        self.playback_timer.setInterval(50)
        self.playback_timer.timeout.connect(self._tick_playback)
        self.canvas.clicked.connect(self._seek_to)
        self._update_controls()

    def _view_range(self) -> tuple[float, float]:
        visible = min(self.duration, max(0.25, self._viewport_seconds)) if self.duration > 0 else self._viewport_seconds
        start = self.position - visible / 2.0 if self._follow_playhead else self._view_start
        start = max(0.0, min(start, max(0.0, self.duration - visible)))
        return start, max(visible, 0.001)

    def _sync_pan_slider(self) -> None:
        if not hasattr(self, "pan_slider"):
            return
        start, visible = self._view_range()
        maximum = max(0.0, self.duration - visible)
        value = round(start / maximum * 10000) if maximum > 0 else 0
        self.pan_slider.blockSignals(True)
        self.pan_slider.setValue(max(0, min(10000, value)))
        self.pan_slider.blockSignals(False)

    def _on_zoom_changed(self, value: int) -> None:
        old_start, old_visible = self._view_range()
        center = old_start + old_visible / 2.0
        self._viewport_seconds = max(1.0, float(value))
        self.zoom_label.setText(f"{self._viewport_seconds:.1f}秒")
        self._follow_playhead = False
        self._view_start = center - self._viewport_seconds / 2.0
        self._sync_pan_slider()
        self.canvas.update()

    def _on_pan_changed(self, value: int) -> None:
        if self.duration <= 0:
            return
        visible = min(self.duration, max(0.25, self._viewport_seconds))
        maximum = max(0.0, self.duration - visible)
        self._view_start = maximum * max(0, min(10000, int(value))) / 10000.0
        self._follow_playhead = False
        self.canvas.update()

    def _update_controls(self) -> None:
        enabled = self.audio is not None and self.duration > 0
        self.playback_play_button.setEnabled(enabled and not self._playing)
        self.playback_pause_button.setEnabled(enabled and self._playing)
        self.playback_stop_button.setEnabled(enabled and (self._playing or self.position > 0))
        self.seek_slider.setEnabled(enabled)
        self.zoom_slider.setEnabled(enabled)
        self.pan_slider.setEnabled(enabled and self.duration > self._viewport_seconds)
        self.playback_time_label.setText(f"{format_seconds(self.position)} / {format_seconds(self.duration)}")

    def _set_position(self, position: float) -> None:
        self.position = max(0.0, min(float(position), self.duration))
        self.seek_slider.blockSignals(True)
        self.seek_slider.setValue(round(self.position / max(self.duration, 1e-9) * 10000))
        self.seek_slider.blockSignals(False)
        self._update_controls()
        self._sync_pan_slider()
        self.canvas.update()

    def _slider_pressed(self) -> None:
        self._slider_dragging = True

    def _slider_moved(self, value: int) -> None:
        if self.duration > 0:
            self._set_position(self.duration * value / 10000.0)

    def _slider_released(self) -> None:
        self._slider_dragging = False
        self._seek_to(self.duration * self.seek_slider.value() / 10000.0)

    def set_source_path(self, path: str | Path) -> None:
        self.stop_playback()
        self.source_path = Path(path).resolve()
        self.audio = None
        self.duration = 0.0
        self.position = 0.0
        self.rows = []
        self.selected_id = None
        self.waveform_points = []
        self._view_start = 0.0
        self._follow_playhead = True
        self.playback_hint_label.setText("波形を読み込み中…")
        self._update_controls()
        self.canvas.update()
        worker = PreviewLoadWorker(self.source_path, self)
        worker.loaded.connect(self._source_loaded)
        worker.failed.connect(self._source_failed)
        self._preview_worker = worker
        worker.start()

    def _source_loaded(self, path: str, audio, points) -> None:
        if self.source_path and Path(path).resolve() != self.source_path:
            return
        self.audio = audio
        self.duration = audio.duration
        self.waveform_points = list(points)
        self.playback_hint_label.setText("解析後にオンセット・分類マーカーを表示")
        self._set_position(0.0)
        self.canvas.update()

    def _source_failed(self, path: str, message: str) -> None:
        if self.source_path and Path(path).resolve() != self.source_path:
            return
        self.playback_hint_label.setText("WAVを読み込めませんでした")
        self.playback_status.emit(f"波形読み込みエラー: {message}")
        self._update_controls()
        self.canvas.update()

    def set_source_audio(self, path: str | Path, audio) -> None:
        """Set a preloaded source for offscreen tests and callers with cached audio."""
        self.stop_playback()
        self.source_path = Path(path).resolve()
        self.audio = audio
        self.duration = audio.duration
        self.position = 0.0
        self.waveform_points = _waveform_envelope(audio)
        self._view_start = 0.0
        self._follow_playhead = True
        self.playback_hint_label.setText("解析後にオンセット・分類マーカーを表示")
        self._update_controls()
        self.canvas.update()

    def set_result(self, result: AnalysisResult | None) -> None:
        if result is None:
            self.rows, self.selected_id = [], None
            if self.audio is None:
                self.duration = 0.0
            else:
                self.playback_hint_label.setText("解析後にオンセット・分類マーカーを表示")
        else:
            self.duration = self.audio.duration if self.audio is not None else result.duration
            self.rows = classify_hits(result)
            self.sample_rate = result.sample_rate
            self.bpm = result.settings.get("bpm")
            self.offset = float(result.settings.get("offset", 0.0) or 0.0)
            self.subdivision = int(result.settings.get("subdivision", 16) or 16)
            self._follow_playhead = True
            self.selected_id = self.rows[0]["id"] if self.rows else None
            self.playback_hint_label.setText("色と文字で分類を表示 · 波形クリックでシーク")
        self._set_position(self.position)
        self.canvas.update()

    def set_selected(self, hit_id: int | None) -> None:
        self.selected_id = hit_id
        self.canvas.update()

    def _native_stop(self) -> None:
        if sys.platform != "win32":
            return
        try:
            import winsound

            winsound.PlaySound(None, winsound.SND_PURGE)
        except Exception:
            pass

    def _cleanup_segment(self) -> None:
        if self._segment_path:
            self._segment_path.unlink(missing_ok=True)
            self._segment_path = None

    def _start_native(self, path: Path, start: float) -> None:
        if sys.platform != "win32":
            self.playback_hint_label.setText("この環境では元WAVの再生に対応していません")
            return
        try:
            import winsound

            winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
        except Exception as exc:
            self.playback_status.emit(f"再生エラー: {exc}")
            return
        self._playback_offset = start
        self._playback_started_at = time.monotonic()
        self._playing = True
        self.playback_timer.start()
        self.playback_hint_label.setText("元WAVを再生中 · クリックまたはスライダーでシーク")
        self._update_controls()

    def _play_from(self, start: float) -> None:
        if not self.source_path or self.duration <= 0:
            return
        start = max(0.0, min(float(start), self.duration))
        if start >= self.duration - 0.01:
            start = 0.0
        self._native_stop()
        self._cleanup_segment()
        if start <= 0.001:
            self._start_native(self.source_path, 0.0)
            return
        self._pending_start = start
        worker = PlaybackSegmentWorker(self.source_path, start, self)
        worker.ready.connect(self._segment_ready)
        worker.failed.connect(lambda message: self.playback_status.emit(f"再生準備エラー: {message}"))
        self._playback_worker = worker
        self.playback_hint_label.setText("再生位置を準備中…")
        worker.start()

    def _segment_ready(self, path: str, start: float) -> None:
        if self._pending_start is None or abs(self._pending_start - start) > 1e-6:
            Path(path).unlink(missing_ok=True)
            return
        self._pending_start = None
        self._segment_path = Path(path)
        self._start_native(self._segment_path, start)

    def play_playback(self) -> None:
        if self._playing:
            return
        self._follow_playhead = True
        self._play_from(self.position)

    def pause_playback(self) -> None:
        if not self._playing:
            return
        self._tick_playback()
        self._native_stop()
        self._playing = False
        self.playback_timer.stop()
        self._cleanup_segment()
        self.playback_hint_label.setText("一時停止中 · 再生で続きから再開")
        self._update_controls()

    def stop_playback(self) -> None:
        if self._playback_worker and self._playback_worker.isRunning():
            self._playback_worker.requestInterruption()
        self._pending_start = None
        self._native_stop()
        self._playing = False
        self.playback_timer.stop()
        self._cleanup_segment()
        self._set_position(0.0)
        if self.audio is not None:
            self.playback_hint_label.setText("解析後にオンセット・分類マーカーを表示")
        self._update_controls()

    def _tick_playback(self) -> None:
        if not self._playing:
            return
        current = self._playback_offset + time.monotonic() - self._playback_started_at
        if current >= self.duration:
            self._native_stop()
            self._playing = False
            self.playback_timer.stop()
            self._cleanup_segment()
            self._set_position(self.duration)
            self.playback_hint_label.setText("再生完了")
            return
        self._set_position(current)

    def _seek_to(self, position: float) -> None:
        if self.duration <= 0:
            return
        self._set_position(position)
        if self._playing:
            self._playing = False
            self.playback_timer.stop()
            self._play_from(self.position)

    def shutdown(self) -> None:
        self.stop_playback()
        if self._preview_worker and self._preview_worker.isRunning():
            self._preview_worker.requestInterruption()
            self._preview_worker.wait(3000)
        if self._playback_worker and self._playback_worker.isRunning():
            self._playback_worker.requestInterruption()
            self._playback_worker.wait(3000)


class MetricCard(QFrame):
    def __init__(self, caption: str, accent: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setMinimumHeight(84)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        self.value = QLabel("—")
        self.value.setObjectName("Value")
        self.value.setStyleSheet(f"color:{accent};")
        self.caption = QLabel(caption.upper())
        self.caption.setObjectName("MetricCaption")
        layout.addWidget(self.value)
        layout.addWidget(self.caption)


class MainWindow(QMainWindow):
    def __init__(self, initial_path: str | None = None):
        super().__init__()
        self.setWindowTitle("ステムリユース · BMSステム再利用解析")
        self.setMinimumSize(1120, 720)
        self.resize(1440, 900)
        self.setAcceptDrops(True)
        self.settings_store = QSettings("StemReuse", "BMSReuseAnalyzer")
        self.worker: AnalysisWorker | None = None
        self.batch_worker: BatchWorker | None = None
        self.result: AnalysisResult | None = None
        self.rows: list[dict] = []
        self.exported: dict[str, object] = {}
        self._last_progress_message = ""
        self._processing_stage = "待機中"
        self._analysis_started_at: float | None = None
        self._cluster_base_thresholds = (0.95, 0.94)
        self.processing_timer = QTimer(self)
        self.processing_timer.setInterval(150)
        self.processing_timer.timeout.connect(self._refresh_processing_status)

        self._build_ui()
        self._apply_theme(self.settings_store.value("theme", "Dark"))
        geometry = self.settings_store.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        if initial_path:
            self.set_input_path(initial_path)
        self._set_running(False)
        self._on_bpm_changed()
        self._log("準備完了。WAVステムをドロップするか、Ctrl+Oで選択してください。")

    def _build_ui(self) -> None:
        root = QWidget()
        root_layout = QVBoxLayout(root)
        root_layout.setContentsMargins(18, 15, 18, 12)
        root_layout.setSpacing(12)
        self.setCentralWidget(root)

        header = QHBoxLayout()
        brand_box = QVBoxLayout()
        brand_box.setSpacing(0)
        brand = QLabel("ステムリユース")
        brand.setObjectName("Brand")
        kicker = QLabel("BMS音声解析")
        kicker.setObjectName("Kicker")
        brand_box.addWidget(brand)
        brand_box.addWidget(kicker)
        header.addLayout(brand_box)
        header.addStretch(1)
        self.theme_combo = QComboBox()
        self.theme_combo.addItem("ダーク", "Dark")
        self.theme_combo.addItem("ライト", "Light")
        self.theme_combo.setToolTip("画面テーマを切り替え")
        self.theme_combo.currentTextChanged.connect(lambda text: self._apply_theme({"ダーク": "Dark", "ライト": "Light"}.get(text, "Dark")))
        header.addWidget(QLabel("テーマ"))
        header.addWidget(self.theme_combo)
        open_button = QPushButton("WAVを開く")
        open_button.setToolTip("解析するPCM WAVステムを選択（Ctrl+O）")
        open_button.clicked.connect(self._browse_input)
        header.addWidget(open_button)
        preset_load_button = QPushButton("設定読込")
        preset_load_button.clicked.connect(self._load_preset)
        header.addWidget(preset_load_button)
        preset_save_button = QPushButton("設定保存")
        preset_save_button.clicked.connect(self._save_preset)
        header.addWidget(preset_save_button)
        batch_button = QPushButton("フォルダ一括")
        batch_button.setToolTip("フォルダ内のWAVをまとめて解析")
        batch_button.clicked.connect(self._browse_batch)
        header.addWidget(batch_button)
        root_layout.addLayout(header)

        title = QHBoxLayout()
        title_label = QLabel("最小限のキー音セットを見つける")
        title_label.setObjectName("Title")
        title.addWidget(title_label)
        title.addStretch(1)
        self.status_label = QLabel("準備完了")
        self.status_label.setObjectName("Status")
        title.addWidget(self.status_label, alignment=Qt.AlignTop)
        root_layout.addLayout(title)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        root_layout.addWidget(splitter, 1)

        left = QFrame()
        left.setObjectName("Panel")
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(12, 12, 12, 12)
        left_layout.setSpacing(10)
        self.drop_zone = DropZone()
        self.drop_zone.path_dropped.connect(self.set_input_path)
        self.drop_zone.browse_requested.connect(self._browse_input)
        left_layout.addWidget(self.drop_zone)
        self.input_edit = QLineEdit()
        self.input_edit.setPlaceholderText("入力WAVのパス")
        self.input_edit.setReadOnly(True)
        self.input_edit.setToolTip("解析する入力WAVファイル")
        left_layout.addWidget(self.input_edit)

        settings_box = QGroupBox("解析設定")
        settings_form = QFormLayout(settings_box)
        settings_form.setContentsMargins(0, 4, 0, 0)
        settings_form.setHorizontalSpacing(12)
        settings_form.setVerticalSpacing(7)
        settings_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)

        def add_setting_row(caption: str, field: QWidget) -> None:
            settings_form.addRow(caption, field)
            label = settings_form.labelForField(field)
            if label:
                label.setFixedWidth(126)
                label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)

        self.instrument_combo = QComboBox()
        for key, label in INSTRUMENT_LABELS.items():
            self.instrument_combo.addItem(label, key)
        self.threshold_spin = self._double_spin(0.95, 0.0, 1.0, 0.001, 3)
        self.spectral_spin = self._double_spin(0.94, 0.0, 1.0, 0.001, 3)
        self.onset_spin = self._double_spin(0.35, 0.0, 1.0, 0.01, 2)
        self.pre_roll_spin = self._double_spin(5.0, 0.0, 1000.0, 1.0, 0, " ms")
        self.window_spin = self._double_spin(800.0, 10.0, 10000.0, 10.0, 0, " ms")
        self.bpm_spin = self._double_spin(0.0, 0.0, 400.0, 0.5, 1, " BPM")
        self.bpm_spin.setSpecialValueText("未入力")
        self.bpm_spin.setToolTip("20〜400の範囲で入力（必須）")
        self.beat_division_combo = QComboBox()
        for denominator in (4, 8, 16, 32):
            self.beat_division_combo.addItem(f"1/{denominator}", denominator)
        self.beat_division_combo.setCurrentIndex(2)
        self.beat_division_combo.setToolTip("BPMから最小間隔を計算する拍の分割")
        self.margin_spin = self._double_spin(90.0, 80.0, 100.0, 1.0, 0, " %")
        self.margin_spin.setToolTip("BPMから計算した間隔に適用する余裕")
        self.bpm_error_label = QLabel()
        self.bpm_error_label.setObjectName("BpmError")
        self.bpm_error_label.setWordWrap(True)
        self.bpm_error_label.setStyleSheet("color:#fb7185;font-size:9pt;")
        self.bpm_spin.valueChanged.connect(self._on_bpm_changed)
        self.margin_spin.valueChanged.connect(self._on_bpm_changed)
        self.beat_division_combo.currentIndexChanged.connect(self._on_bpm_changed)
        self.fade_in_spin = self._double_spin(2.0, 0.0, 1000.0, 0.5, 1, " ms")
        self.fade_out_spin = self._double_spin(2.0, 0.0, 1000.0, 0.5, 1, " ms")
        self.fade_in_spin.setToolTip("書き出す代表WAVの先頭フェード（0で無効）")
        self.fade_out_spin.setToolTip("書き出す代表WAVの末尾フェード（0で無効）")
        self.offset_spin = self._double_spin(0.0, -60.0, 60.0, 0.001, 3, " s")
        self.alignment_spin = self._double_spin(20.0, 0.0, 100.0, 0.5, 1, " ms")
        self.subdivision_spin = QSpinBox()
        self.subdivision_spin.setRange(1, 128)
        self.subdivision_spin.setValue(16)
        self.fast_compare_check = QCheckBox("高速比較（代表順が変わる場合があります）")
        self.fast_compare_check.setAccessibleName("高速比較（代表順が変わる場合があります）")
        self.fast_compare_check.setToolTip("特徴量に近い代表から比較します。結果の代表順が変わる可能性があります。")
        self.bms_channel_combo = QComboBox()
        self.bms_channel_combo.addItem("BGM 01（互換）", "01")
        self.bms_channel_combo.addItem("1Pキー 11", "11")
        self.bms_channel_combo.addItem("1Pキー 12", "12")
        add_setting_row("楽器", self.instrument_combo)
        add_setting_row("同一判定しきい値", self.threshold_spin)
        add_setting_row("スペクトルしきい値", self.spectral_spin)
        add_setting_row("オンセットしきい値", self.onset_spin)
        add_setting_row("プリロール", self.pre_roll_spin)
        add_setting_row("ウィンドウ長", self.window_spin)
        add_setting_row("BPM（必須）", self.bpm_spin)
        add_setting_row("最小間隔（拍）", self.beat_division_combo)
        add_setting_row("マージン(%)", self.margin_spin)
        add_setting_row("", self.bpm_error_label)
        add_setting_row("フェードイン(ms)", self.fade_in_spin)
        add_setting_row("フェードアウト(ms)", self.fade_out_spin)
        add_setting_row("グリッドオフセット", self.offset_spin)
        add_setting_row("分割数", self.subdivision_spin)
        add_setting_row("最大アライメント", self.alignment_spin)
        add_setting_row("比較モード", self.fast_compare_check)
        add_setting_row("BMSチャンネル", self.bms_channel_combo)
        left_layout.addWidget(settings_box)

        self.cluster_box = QGroupBox("使い回し度")
        cluster_form = QFormLayout(self.cluster_box)
        cluster_form.setContentsMargins(0, 4, 0, 0)
        cluster_form.setHorizontalSpacing(12)
        cluster_form.setVerticalSpacing(7)
        cluster_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.cluster_slider = QSlider(Qt.Horizontal)
        self.cluster_slider.setRange(70, 100)
        self.cluster_slider.setValue(95)
        self.cluster_slider.setAccessibleName("使い回し度")
        self.cluster_slider.setToolTip("解析後の保存済み比較を再分類。左は厳格、右は積極的")
        self.cluster_slider.valueChanged.connect(self._update_cluster_threshold_label)
        cluster_form.addRow("類似度基準", self.cluster_slider)
        self.cluster_threshold_label = QLabel("95%（波形0.950・スペクトル0.940）")
        self.cluster_threshold_label.setObjectName("Subtle")
        cluster_form.addRow("実効値", self.cluster_threshold_label)
        self.cluster_hint_label = QLabel("解析後に再解析なしで適用できます")
        self.cluster_hint_label.setObjectName("Subtle")
        self.cluster_hint_label.setWordWrap(True)
        cluster_form.addRow("", self.cluster_hint_label)
        cluster_actions = QHBoxLayout()
        self.cluster_apply_button = QPushButton("適用")
        self.cluster_apply_button.setObjectName("Primary")
        self.cluster_apply_button.setToolTip("保存済みの比較結果でクラスタを作り直します")
        self.cluster_apply_button.clicked.connect(self._apply_cluster_threshold)
        self.cluster_reset_button = QPushButton("初期値に戻す")
        self.cluster_reset_button.setToolTip("解析開始時の類似度基準に戻して適用")
        self.cluster_reset_button.clicked.connect(self._reset_cluster_threshold)
        cluster_actions.addWidget(self.cluster_apply_button)
        cluster_actions.addWidget(self.cluster_reset_button)
        cluster_form.addRow("", cluster_actions)
        self.cluster_box.setEnabled(False)
        left_layout.addWidget(self.cluster_box)

        output_box = QGroupBox("出力")
        output_layout = QVBoxLayout(output_box)
        output_layout.setContentsMargins(0, 4, 0, 0)
        output_layout.setSpacing(8)
        self.json_edit = QLineEdit()
        self.samples_edit = QLineEdit()
        self.csv_edit = QLineEdit()
        self.bms_edit = QLineEdit()
        self.bmson_edit = QLineEdit()
        self.csv_check = QCheckBox("イベントCSVを書き出す")
        self.csv_check.setChecked(True)
        self.samples_check = QCheckBox("代表WAVを書き出す")
        self.samples_check.setChecked(True)
        output_layout.addWidget(self._path_row("JSON", self.json_edit, self._browse_json))
        output_layout.addWidget(self._path_row("サンプル", self.samples_edit, self._browse_samples))
        output_layout.addWidget(self._path_row("CSV", self.csv_edit, self._browse_csv))
        output_layout.addWidget(self._path_row("BMS", self.bms_edit, self._browse_bms))
        output_layout.addWidget(self._path_row("BMSON", self.bmson_edit, self._browse_bmson))
        output_layout.addWidget(self.samples_check)
        output_layout.addWidget(self.csv_check)
        left_layout.addWidget(output_box)

        action_row = QHBoxLayout()
        self.analyze_button = QPushButton("解析を開始")
        self.analyze_button.setObjectName("Primary")
        self.analyze_button.setMinimumHeight(38)
        self.analyze_button.setToolTip("解析を開始（Ctrl+Enter）")
        self.analyze_button.clicked.connect(self.start_analysis)
        self.cancel_button = QPushButton("キャンセル")
        self.cancel_button.setObjectName("Danger")
        self.cancel_button.setMinimumHeight(38)
        self.cancel_button.clicked.connect(self.cancel_analysis)
        action_row.addWidget(self.analyze_button, 2)
        action_row.addWidget(self.cancel_button, 1)
        left_layout.addLayout(action_row)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        left_layout.addWidget(self.progress_bar)
        self.processing_status_label = QLabel("待機中")
        self.processing_status_label.setObjectName("Status")
        self.processing_status_label.setWordWrap(True)
        left_layout.addWidget(self.processing_status_label)

        log_group = QGroupBox("処理ログ")
        log_layout = QVBoxLayout(log_group)
        self.log_view = QPlainTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(600)
        self.log_view.setMinimumHeight(110)
        log_layout.addWidget(self.log_view)
        left_layout.addWidget(log_group)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFrameShape(QFrame.NoFrame)
        left_scroll.setWidget(left)
        splitter.addWidget(left_scroll)

        right = QFrame()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(10)
        self.waveform = WaveformView()
        self.waveform.setObjectName("WaveformView")
        self.waveform.hit_selected.connect(self._select_hit)
        self.waveform.playback_status.connect(self._log)
        right_layout.addWidget(self.waveform)

        metrics = QGridLayout()
        metrics.setSpacing(8)
        self.required_card = MetricCard("必要サンプル数", "#67e8f9")
        self.hits_card = MetricCard("検出ヒット数", "#60a5fa")
        self.reuse_card = MetricCard("再利用率", "#34d399")
        self.review_card = MetricCard("要確認", "#fbbf24")
        metrics.addWidget(self.required_card, 0, 0)
        metrics.addWidget(self.hits_card, 0, 1)
        metrics.addWidget(self.reuse_card, 0, 2)
        metrics.addWidget(self.review_card, 0, 3)
        right_layout.addLayout(metrics)

        review_splitter = QSplitter(Qt.Vertical)
        review_splitter.setChildrenCollapsible(False)
        table_panel = QFrame()
        table_layout = QVBoxLayout(table_panel)
        table_layout.setContentsMargins(0, 0, 0, 0)
        filter_row = QHBoxLayout()
        filter_row.addWidget(QLabel("ヒット一覧"))
        self.filter_combo = QComboBox()
        for key, label in FILTER_LABELS.items():
            self.filter_combo.addItem(label, key)
        self.filter_combo.currentIndexChanged.connect(self._refresh_table)
        filter_row.addWidget(self.filter_combo)
        self.next_review_button = QPushButton("次の要確認")
        self.next_review_button.setToolTip("判定保留・音の重なりを順番に確認")
        self.next_review_button.clicked.connect(self._select_next_review)
        filter_row.addWidget(self.next_review_button)
        self.search_edit = QLineEdit()
        self.search_edit.setPlaceholderText("ヒット番号または時刻で絞り込み…")
        self.search_edit.textChanged.connect(self._refresh_table)
        filter_row.addWidget(self.search_edit, 1)
        self.open_folder_button = QPushButton("サンプルフォルダを開く")
        self.open_folder_button.clicked.connect(self._open_samples_folder)
        filter_row.addWidget(self.open_folder_button)
        table_layout.addLayout(filter_row)
        self.hit_table = QTableWidget(0, 7)
        self.hit_table.setHorizontalHeaderLabels(["番号", "時刻", "分類", "信頼度", "音量差", "サンプル", "注意"])
        self.hit_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.hit_table.setSelectionMode(QTableWidget.SingleSelection)
        self.hit_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.hit_table.setAlternatingRowColors(True)
        self.hit_table.setSortingEnabled(True)
        self.hit_table.verticalHeader().setVisible(False)
        self.hit_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
        self.hit_table.horizontalHeader().setSectionResizeMode(5, QHeaderView.Stretch)
        self.hit_table.currentCellChanged.connect(self._table_selection_changed)
        table_layout.addWidget(self.hit_table, 1)
        review_splitter.addWidget(table_panel)

        detail_panel = QFrame()
        detail_panel.setObjectName("Panel")
        detail_layout = QHBoxLayout(detail_panel)
        detail_layout.setContentsMargins(12, 9, 12, 9)
        detail_box = QVBoxLayout()
        self.detail_title = QLabel("ヒットを選択してください")
        self.detail_title.setObjectName("DropTitle")
        self.detail_metrics = QLabel("—")
        self.detail_metrics.setObjectName("Subtle")
        self.detail_metrics.setWordWrap(True)
        detail_box.addWidget(self.detail_title)
        detail_box.addWidget(self.detail_metrics)
        detail_box.addStretch(1)
        detail_layout.addLayout(detail_box, 2)
        sample_box = QVBoxLayout()
        self.sample_list = QListWidget()
        self.sample_list.setMaximumHeight(82)
        sample_box.addWidget(QLabel("代表サンプル"))
        sample_box.addWidget(self.sample_list)
        button_row = QHBoxLayout()
        self.play_button = QPushButton("選択音を再生")
        self.play_button.clicked.connect(self._play_selected)
        self.export_folder_button = QPushButton("フォルダを表示")
        self.export_folder_button.clicked.connect(self._open_samples_folder)
        button_row.addWidget(self.play_button)
        button_row.addWidget(self.export_folder_button)
        sample_box.addLayout(button_row)
        detail_layout.addLayout(sample_box, 1)
        review_splitter.addWidget(detail_panel)
        review_splitter.setStretchFactor(0, 4)
        review_splitter.setStretchFactor(1, 1)
        right_layout.addWidget(review_splitter, 1)
        splitter.addWidget(right)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 1000])

        status = self.statusBar()
        status.showMessage("準備完了")

        self.shortcut_open = QShortcut(QKeySequence("Ctrl+O"), self)
        self.shortcut_open.activated.connect(self._browse_input)
        self.shortcut_start = QShortcut(QKeySequence("Ctrl+Return"), self)
        self.shortcut_start.activated.connect(self.start_analysis)
        self.shortcut_cancel = QShortcut(QKeySequence("Escape"), self)
        self.shortcut_cancel.activated.connect(self.cancel_analysis)
        self.shortcut_play = QShortcut(QKeySequence("Space"), self)
        self.shortcut_play.activated.connect(self._play_selected)
        self.shortcut_same = QShortcut(QKeySequence("S"), self)
        self.shortcut_same.activated.connect(lambda: self._apply_review("S"))
        self.shortcut_gain = QShortcut(QKeySequence("G"), self)
        self.shortcut_gain.activated.connect(lambda: self._apply_review("G"))
        self.shortcut_different = QShortcut(QKeySequence("D"), self)
        self.shortcut_different.activated.connect(lambda: self._apply_review("D"))
        self.shortcut_ignore = QShortcut(QKeySequence("I"), self)
        self.shortcut_ignore.activated.connect(lambda: self._apply_review("I"))

    @staticmethod
    def _double_spin(value, minimum, maximum, step, decimals, suffix: str = "") -> QDoubleSpinBox:
        spin = QDoubleSpinBox()
        spin.setRange(minimum, maximum)
        spin.setSingleStep(step)
        spin.setDecimals(decimals)
        spin.setValue(value)
        if suffix:
            spin.setSuffix(suffix)
        return spin

    @staticmethod
    def _path_row(caption: str, edit: QLineEdit, callback) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        label = QLabel(caption)
        label.setFixedWidth(72)
        label.setAlignment(Qt.AlignLeft | Qt.AlignVCenter)
        layout.addWidget(label)
        layout.addWidget(edit, 1)
        button = QPushButton("参照…")
        button.setMinimumWidth(72)
        button.setAccessibleName(f"{caption}の保存先を参照")
        button.setToolTip(f"{caption}の保存先を選択")
        button.clicked.connect(callback)
        layout.addWidget(button)
        return widget

    def _apply_theme(self, theme: str) -> None:
        theme = theme if theme in {"Dark", "Light"} else "Dark"
        if hasattr(self, "theme_combo") and self.theme_combo.currentData() != theme:
            self.theme_combo.blockSignals(True)
            index = self.theme_combo.findData(theme)
            if index >= 0:
                self.theme_combo.setCurrentIndex(index)
            self.theme_combo.blockSignals(False)
        QApplication.instance().setStyleSheet(DARK_STYLE if theme == "Dark" else LIGHT_STYLE)
        self.settings_store.setValue("theme", theme)

    def _log(self, message: str) -> None:
        self.log_view.appendPlainText(message)
        self.statusBar().showMessage(message, 5000)

    def set_input_path(self, raw_path: str) -> None:
        path = Path(raw_path).expanduser()
        if path.suffix.casefold() != ".wav":
            self._show_error("対応していない入力形式", "PCM WAVファイルを選択してください。")
            return
        if not path.exists() or not path.is_file():
            self._show_error("入力ファイルが見つかりません", str(path))
            return
        path = path.resolve()
        self.input_edit.setText(str(path))
        self.drop_zone.title.setText(path.name)
        self.drop_zone.hint.setText("PCM WAV準備完了 · Ctrl+Enterで解析")
        self.result = None
        self.rows = []
        self.waveform.set_source_path(path)
        if hasattr(self, "hit_table"):
            self.hit_table.setRowCount(0)
        if hasattr(self, "sample_list"):
            self.sample_list.clear()
        self.json_edit.setText(str(path.with_suffix(".bra.json")))
        self.samples_edit.setText(str(path.parent / f"{path.stem}_keysounds"))
        self.csv_edit.setText(str(path.with_suffix(".csv")))
        self.bms_edit.setText(str(path.with_suffix(".bms")))
        self.bmson_edit.setText(str(path.parent / f"{path.stem}_keysounds" / f"{path.stem}.bmson"))
        self._log(f"入力を選択しました: {path}")
        self._update_start_enabled()

    def _browse_input(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "WAVステムを選択", str(Path(self.input_edit.text()).parent if self.input_edit.text() else Path.home()), "PCM WAV (*.wav)")
        if path:
            self.set_input_path(path)

    def _browse_json(self) -> None:
        current = self.json_edit.text() or str(Path.home() / "analysis.bra.json")
        path, _ = QFileDialog.getSaveFileName(self, "解析JSONの保存先", current, "BMS解析 (*.bra.json);;JSON (*.json)")
        if path:
            self.json_edit.setText(path)

    def _browse_csv(self) -> None:
        current = self.csv_edit.text() or str(Path.home() / "events.csv")
        path, _ = QFileDialog.getSaveFileName(self, "イベントCSVの保存先", current, "CSV (*.csv)")
        if path:
            self.csv_edit.setText(path)

    def _browse_samples(self) -> None:
        current = self.samples_edit.text() or str(Path.home())
        path = QFileDialog.getExistingDirectory(self, "代表WAVフォルダ", current)
        if path:
            self.samples_edit.setText(path)

    def _browse_bms(self) -> None:
        current = self.bms_edit.text() or str(Path.home() / "chart.bms")
        path, _ = QFileDialog.getSaveFileName(self, "BMSの保存先", current, "BMS (*.bms)")
        if path:
            self.bms_edit.setText(path)

    def _browse_bmson(self) -> None:
        current = self.bmson_edit.text() or str(Path.home() / "chart.bmson")
        path, _ = QFileDialog.getSaveFileName(self, "BMSONの保存先", current, "BMSON (*.bmson);;JSON (*.json)")
        if path:
            self.bmson_edit.setText(path)

    def _browse_batch(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        if not self._bpm_is_valid():
            self._show_error("BPMが必要です", self._bpm_error_text())
            return
        folder = QFileDialog.getExistingDirectory(self, "一括解析するフォルダ", str(Path.home()))
        if not folder:
            return
        output = QFileDialog.getExistingDirectory(self, "一括出力フォルダ", str(Path(folder) / "bms-reuse-batch"))
        if not output:
            return
        try:
            settings = self._settings()
        except ValueError as exc:
            self._show_error("解析設定が正しくありません", str(exc))
            return
        self.batch_worker = BatchWorker(Path(folder), Path(output), settings, self)
        self.batch_worker.progress.connect(lambda percent, message: self._on_progress(percent, message))
        self.batch_worker.result_ready.connect(self._on_batch_result)
        self.batch_worker.failed.connect(self._on_batch_failed)
        self._processing_stage = "一括解析中"
        self.status_label.setText("一括解析中…")
        self._log(f"一括解析を開始: {folder}")
        self.batch_worker.start()

    @Slot(object)
    def _on_batch_result(self, manifest: dict) -> None:
        self.status_label.setText("一括解析完了")
        self._processing_stage = "一括解析完了"
        self._log(f"一括解析完了: {manifest.get('success', 0)}/{manifest.get('count', 0)}件 · manifest.json")

    @Slot(str)
    def _on_batch_failed(self, details: str) -> None:
        self.status_label.setText("一括解析エラー")
        self._processing_stage = "一括解析エラー"
        self._log(f"一括解析エラー: {details}")

    def _preset_settings(self) -> dict:
        return self._settings()

    def _save_preset(self) -> None:
        try:
            settings = self._preset_settings()
        except ValueError as exc:
            self._show_error("設定を保存できません", str(exc))
            return
        path, _ = QFileDialog.getSaveFileName(self, "解析プリセットを保存", str(Path.home() / "bms-reuse.preset.json"), "JSON (*.json)")
        if path:
            save_preset(path, settings)
            self._log(f"プリセットを保存しました: {path}")

    def _load_preset(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "解析プリセットを開く", str(Path.home()), "JSON (*.json)")
        if not path:
            return
        try:
            values = load_preset(path)
            widgets = {
                "threshold": self.threshold_spin,
                "spectral_threshold": self.spectral_spin,
                "onset_threshold": self.onset_spin,
                "pre_roll_ms": self.pre_roll_spin,
                "window_ms": self.window_spin,
                "bpm": self.bpm_spin,
                "margin_percent": self.margin_spin,
                "fade_in_ms": self.fade_in_spin,
                "fade_out_ms": self.fade_out_spin,
                "offset": self.offset_spin,
                "max_alignment_ms": self.alignment_spin,
                "subdivision": self.subdivision_spin,
            }
            for key, widget in widgets.items():
                if key in values:
                    widget.setValue(float(values[key]))
            instrument = values.get("instrument")
            if instrument:
                index = self.instrument_combo.findData(instrument)
                if index >= 0:
                    self.instrument_combo.setCurrentIndex(index)
            self.fast_compare_check.setChecked(bool(values.get("fast_compare", False)))
            if "bms_channel" in values:
                index = self.bms_channel_combo.findData(values["bms_channel"])
                if index >= 0:
                    self.bms_channel_combo.setCurrentIndex(index)
            if "beat_division" in values:
                index = self.beat_division_combo.findData(int(values["beat_division"]))
                if index >= 0:
                    self.beat_division_combo.setCurrentIndex(index)
            self._log(f"プリセットを読み込みました: {path}")
        except (OSError, ValueError, TypeError) as exc:
            self._show_error("プリセットを読み込めません", str(exc))

    @staticmethod
    def _same_path(left: str | Path, right: str | Path) -> bool:
        return str(Path(left).resolve()).casefold() == str(Path(right).resolve()).casefold()

    def _bpm_error_text(self) -> str:
        bpm = self.bpm_spin.value()
        if bpm <= 0:
            return "BPMを入力してください（20〜400）。"
        if not 20.0 <= bpm <= 400.0:
            return "BPMは20〜400の範囲で入力してください。"
        return ""

    def _bpm_is_valid(self) -> bool:
        return not self._bpm_error_text()

    def _on_bpm_changed(self, _value=None) -> None:
        message = self._bpm_error_text()
        self.bpm_error_label.setText(message)
        self._update_start_enabled()

    def _update_start_enabled(self) -> None:
        if not hasattr(self, "analyze_button"):
            return
        running = bool(self.worker and self.worker.isRunning())
        input_text = self.input_edit.text().strip()
        input_path = Path(input_text) if input_text else None
        input_valid = bool(input_path and input_path.is_file())
        self.analyze_button.setEnabled(not running and input_valid and self._bpm_is_valid())

    def _settings(self) -> dict:
        if not self._bpm_is_valid():
            raise ValueError(self._bpm_error_text())
        bpm = self.bpm_spin.value()
        denominator = int(self.beat_division_combo.currentData())
        margin = self.margin_spin.value()
        min_interval_sec = (60.0 / bpm) / denominator * (margin / 100.0)
        return {
            "instrument": self.instrument_combo.currentData() or "kick",
            "threshold": self.threshold_spin.value(),
            "spectral_threshold": self.spectral_spin.value(),
            "onset_threshold": self.onset_spin.value(),
            "min_separation_ms": min_interval_sec * 1000.0,
            "min_interval_sec": min_interval_sec,
            "beat_division": denominator,
            "margin_percent": margin,
            "margin": margin,
            "pre_roll_ms": self.pre_roll_spin.value(),
            "window_ms": self.window_spin.value(),
            "max_alignment_ms": self.alignment_spin.value(),
            "bpm": bpm,
            "fade_in_ms": self.fade_in_spin.value(),
            "fade_out_ms": self.fade_out_spin.value(),
            "offset": self.offset_spin.value(),
            "subdivision": self.subdivision_spin.value(),
            "fast_compare": self.fast_compare_check.isChecked(),
            "bms_channel": self.bms_channel_combo.currentData() or "01",
        }

    def _outputs(self) -> dict:
        input_path = Path(self.input_edit.text())
        output: dict[str, str | None] = {
            "json": self.json_edit.text().strip() or None,
            "samples": None,
            "csv": None,
            "bms": self.bms_edit.text().strip() or None,
            "bmson": self.bmson_edit.text().strip() or None,
        }
        if self.samples_check.isChecked() and self.samples_edit.text().strip():
            output["samples"] = self.samples_edit.text().strip()
        if self.csv_check.isChecked() and self.csv_edit.text().strip():
            output["csv"] = self.csv_edit.text().strip()
        for key, target in output.items():
            if target and key != "samples" and self._same_path(target, input_path):
                label = {"json": "JSON", "csv": "CSV"}.get(key, key)
                raise ValueError(f"{label}出力が入力WAVを上書きします")
        sample_dir = output.get("samples")
        if sample_dir:
            input_stem = input_path.stem.casefold()
            if Path(sample_dir).resolve() == input_path.parent.resolve() and input_stem.startswith("sample_") and input_stem[7:].isdigit():
                raise ValueError("サンプル出力が入力WAVを上書きします")
        return output

    def _refresh_processing_status(self) -> None:
        if not hasattr(self, "processing_status_label") or self._analysis_started_at is None:
            return
        elapsed = max(0.0, time.monotonic() - self._analysis_started_at)
        self.processing_status_label.setText(
            f"処理中…  経過 {elapsed:.1f}秒  ·  キャンセル可能\n{self._processing_stage}"
        )

    def _set_running(self, running: bool) -> None:
        self.cancel_button.setEnabled(running)
        self.drop_zone.setEnabled(not running)
        self.input_edit.setEnabled(not running)
        self._update_cluster_controls(running)
        if running:
            self.progress_bar.setValue(0)
            self.status_label.setText("解析中…")
            self._processing_stage = "準備中…"
            self._analysis_started_at = time.monotonic()
            self.processing_timer.start()
            self._refresh_processing_status()
        else:
            self.processing_timer.stop()
            self._analysis_started_at = None
            self.status_label.setText("準備完了" if self.result is None else "解析結果を表示中")
            self.processing_status_label.setText("待機中" if self.result is None else "解析結果を表示中")
            self._update_start_enabled()

    def _update_cluster_controls(self, running: bool | None = None) -> None:
        if not hasattr(self, "cluster_box"):
            return
        if running is None:
            running = bool(self.worker and self.worker.isRunning())
        self.cluster_box.setEnabled(bool(self.result) and not running)

    def _update_cluster_threshold_label(self, value: int) -> None:
        base_waveform, base_spectral = self._cluster_base_thresholds
        waveform = max(0.0, min(1.0, int(value) / 100.0))
        spectral = max(0.0, min(1.0, base_spectral + waveform - base_waveform))
        self.cluster_threshold_label.setText(f"{int(value)}%（波形{waveform:.3f}・スペクトル{spectral:.3f}）")

    def _cluster_threshold_values(self) -> tuple[float, float]:
        base_waveform, base_spectral = self._cluster_base_thresholds
        waveform = int(self.cluster_slider.value()) / 100.0
        spectral = max(0.0, min(1.0, base_spectral + waveform - base_waveform))
        return waveform, spectral

    def _apply_cluster_threshold(self) -> None:
        if not self.result or (self.worker and self.worker.isRunning()):
            return
        started = time.perf_counter()
        waveform, spectral = self._cluster_threshold_values()
        try:
            recluster_result(
                self.result,
                threshold=waveform,
                spectral_threshold=spectral,
                reexport=True,
            )
            profile_name = self.result.settings.get("recluster_profile", "custom")
            self.result.settings.setdefault("timings", {})["recluster_seconds"] = round(time.perf_counter() - started, 6)
            self.rows = classify_hits(self.result)
            self.waveform.set_result(self.result)
            summary = self.result.summary
            self.required_card.value.setText(str(summary["required_samples"]))
            self.hits_card.value.setText(str(summary["detected_hits"]))
            self.reuse_card.value.setText(f"{summary['reuse_ratio']:.1f}%")
            self.review_card.value.setText(str(sum(row["classification"] in {"UNSURE", "OVERLAP"} for row in self.rows)))
            self._refresh_table()
            self.sample_list.clear()
            for cluster in self.result.plan.clusters:
                self.sample_list.addItem(QListWidgetItem(f"sample_{cluster.id:03d}"))
            self._save_review_state()
            self._log(
                f"使い回し度を{int(self.cluster_slider.value())}%で適用: "
                f"{len(self.result.plan.clusters)}クラスタ · 判定{profile_name}"
            )
        except (OSError, ValueError, RuntimeError, TypeError) as exc:
            self._show_error("クラスタ設定を適用できません", str(exc))

    def _reset_cluster_threshold(self) -> None:
        waveform, _spectral = self._cluster_base_thresholds
        self.cluster_slider.setValue(round(waveform * 100.0))
        self._apply_cluster_threshold()

    def start_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        input_text = self.input_edit.text().strip()
        if not input_text:
            self._show_error("入力が必要です", "WAVステムをドロップするか、Ctrl+Oで選択してください。")
            return
        if not self._bpm_is_valid():
            self._on_bpm_changed()
            self._show_error("BPMが必要です", self._bpm_error_text())
            return
        input_path = Path(input_text)
        if not input_path.exists():
            self._show_error("入力ファイルが見つかりません", str(input_path))
            return
        try:
            outputs = self._outputs()
        except ValueError as exc:
            self._show_error("出力設定が正しくありません", str(exc))
            return
        self.result = None
        self.rows = []
        self.waveform.set_result(None)
        self.hit_table.setRowCount(0)
        self.sample_list.clear()
        self._set_running(True)
        self._log("解析を開始します…")
        try:
            settings = self._settings()
        except ValueError as exc:
            self._show_error("解析設定が正しくありません", str(exc))
            self._set_running(False)
            return
        self.worker = AnalysisWorker(input_path, settings, outputs, self)
        self.worker.progress.connect(self._on_progress)
        self.worker.result_ready.connect(self._on_result)
        self.worker.failed.connect(self._on_failed)
        self.worker.canceled.connect(self._on_canceled)
        self.worker.finished.connect(lambda: self._set_running(False))
        self.worker.start()

    def cancel_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.status_label.setText("キャンセル中…")
            self._processing_stage = "キャンセル処理中…"
            self.cancel_button.setEnabled(False)
            self._refresh_processing_status()
            self._log("キャンセルを要求しました…")

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        self._processing_stage = message
        self._refresh_processing_status()
        if message != self._last_progress_message:
            self._log(f"{percent:3d}%  {message}")
            self._last_progress_message = message

    @Slot(object, object)
    def _on_result(self, result: AnalysisResult, exported: dict) -> None:
        self.result, self.exported = result, exported
        self._cluster_base_thresholds = (
            float(result.settings.get("threshold", 0.95)),
            float(result.settings.get("spectral_threshold", 0.94)),
        )
        self.cluster_slider.blockSignals(True)
        self.cluster_slider.setValue(max(70, min(100, round(self._cluster_base_thresholds[0] * 100.0))))
        self.cluster_slider.blockSignals(False)
        self._update_cluster_threshold_label(self.cluster_slider.value())
        self._update_cluster_controls(False)
        self.rows = classify_hits(result)
        self.waveform.set_result(result)
        summary = result.summary
        self.required_card.value.setText(str(summary["required_samples"]))
        self.hits_card.value.setText(str(summary["detected_hits"]))
        self.reuse_card.value.setText(f"{summary['reuse_ratio']:.1f}%")
        review = sum(row["classification"] in {"UNSURE", "OVERLAP"} for row in self.rows)
        self.review_card.value.setText(str(review))
        self._refresh_table()
        self.sample_list.clear()
        sample_paths = exported.get("samples", []) if isinstance(exported, dict) else []
        sample_paths = sample_paths if isinstance(sample_paths, list) else []
        if sample_paths:
            for path in sample_paths:
                self.sample_list.addItem(QListWidgetItem(Path(path).name))
        else:
            for cluster in result.plan.clusters:
                self.sample_list.addItem(QListWidgetItem(f"sample_{cluster.id:03d}"))
        self._log(
            f"解析完了: ヒット{summary['detected_hits']}件 · 必要サンプル{summary['required_samples']}個 · "
            f"再利用率{summary['reuse_ratio']:.1f}%"
        )
        mode_label = "高速" if summary.get("compare_mode") == "fast" else "通常"
        self._log(
            f"比較: {summary.get('comparisons', len(result.comparisons))}件 · "
            f"キャッシュ再利用: {summary.get('comparison_cache_hits', 0)}件 · モード: {mode_label}"
        )
        validation = exported.get("validation", {}) if isinstance(exported, dict) else {}
        self._log(f"出力チェック: {'OK' if validation.get('ok', True) else '要確認'}")
        timings = summary.get("timings", {})
        if timings:
            self._log(
                "処理時間: 読込{load:.2f}秒 · オンセット{onset:.2f}秒 · ヒット切り出し{hit:.2f}秒 · "
                "比較{compare:.2f}秒 · 合計{total:.2f}秒".format(
                    load=timings.get("load_seconds", 0.0),
                    onset=timings.get("onset_seconds", 0.0),
                    hit=timings.get("hit_seconds", 0.0),
                    compare=timings.get("compare_seconds", 0.0),
                    total=timings.get("total_seconds", 0.0),
                )
            )
        profile = result.settings.get("similarity_profile", {})
        profile_name = "波形・スペクトル優先" if profile.get("name") == "waveform_spectral_v2" else profile.get("name", "類似度優先")
        self._log(
            "類似度判定: {name} · 波形≥{waveform:.3f} · スペクトル≥{spectral:.3f} · "
            "位置合わせ±{alignment:.1f}ms · 重なり警告{warnings}件".format(
                name=profile_name,
                waveform=float(profile.get("waveform_threshold", result.settings.get("threshold", 0.95))),
                spectral=float(profile.get("spectral_threshold", result.settings.get("spectral_threshold", 0.94))),
                alignment=float(profile.get("alignment_ms", result.settings.get("max_alignment_ms", 20.0))),
                warnings=summary.get("overlap_warnings", 0),
            )
        )
        self._processing_stage = "解析完了"
        self.status_label.setText("解析結果を表示中")

    @Slot(str)
    def _on_failed(self, details: str) -> None:
        if "Could not read WAV" in details:
            message = "WAVファイルを読み込めませんでした。PCM WAVか確認してください。"
        elif "must not overwrite" in details or "上書き" in details:
            message = "入力WAVを上書きする出力先は指定できません。"
        elif "PermissionError" in details:
            message = "ファイルへのアクセス権がありません。"
        else:
            message = "解析中にエラーが発生しました。入力と出力先を確認してください。"
        self._processing_stage = "解析エラー"
        self._log(f"解析エラー: {message}")
        self._show_error("解析に失敗しました", message, f"技術詳細（開発者向け）:\n{details}")

    @Slot()
    def _on_canceled(self) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("キャンセル済み")
        self._processing_stage = "キャンセル済み"
        self._log("解析をキャンセルしました。")

    def _refresh_table(self) -> None:
        if not hasattr(self, "hit_table"):
            return
        selected = self._selected_row_id()
        wanted = self.filter_combo.currentData() if hasattr(self, "filter_combo") else "All"
        query = self.search_edit.text().strip().casefold() if hasattr(self, "search_edit") else ""
        self.hit_table.setSortingEnabled(False)
        self.hit_table.setRowCount(0)
        for row in self.rows:
            if wanted == "REVIEW" and row["classification"] not in {"UNSURE", "OVERLAP"}:
                continue
            if wanted not in {"All", "REVIEW"} and row["classification"] != wanted:
                continue
            haystack = f"{row['id']} {row['time']:.3f} {row['sample_id']}".casefold()
            if query and query not in haystack:
                continue
            index = self.hit_table.rowCount()
            self.hit_table.insertRow(index)
            values = [
                str(row["id"]),
                format_seconds(row["time"]),
                CLASS_LABELS.get(row["classification"], row["classification"]),
                f"{row['confidence']:.1f}%",
                f"{row['gain_db']:+.2f} dB",
                row["sample_id"],
                "音の重なり" if row["overlap"] else "",
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value)
                if column == 0:
                    item.setData(Qt.UserRole, row["id"])
                if column == 2:
                    item.setForeground(QColor(CLASS_COLORS.get(row["classification"], "#94a3b8")))
                self.hit_table.setItem(index, column, item)
        self.hit_table.setSortingEnabled(True)
        if self.hit_table.rowCount():
            desired = next((i for i in range(self.hit_table.rowCount()) if self.hit_table.item(i, 0).data(Qt.UserRole) == selected), 0)
            self.hit_table.selectRow(desired)
            self._table_selection_changed(desired, 0, -1, -1)

    def _select_next_review(self) -> None:
        review_ids = [row["id"] for row in self.rows if row["classification"] in {"UNSURE", "OVERLAP"}]
        if not review_ids:
            self._log("要確認のヒットはありません")
            return
        current = self._selected_row_id()
        target = next((hit_id for hit_id in review_ids if current is None or hit_id > current), review_ids[0])
        for index in range(self.hit_table.rowCount()):
            item = self.hit_table.item(index, 0)
            if item and item.data(Qt.UserRole) == target:
                self.hit_table.selectRow(index)
                self._table_selection_changed(index, 0, -1, -1)
                return

    def _apply_review(self, key: str) -> None:
        hit_id = self._selected_row_id()
        if hit_id is None or not self.result:
            return
        overrides = self.result.settings.setdefault("review_overrides", {})
        overrides[str(hit_id)] = key
        row = next((item for item in self.rows if item["id"] == hit_id), None)
        if key == "D":
            old_cluster = next((cluster for cluster in self.result.plan.clusters if hit_id in cluster.hit_ids), None)
            if old_cluster and len(old_cluster.hit_ids) > 1:
                old_cluster.hit_ids.remove(hit_id)
                new_id = max((cluster.id for cluster in self.result.plan.clusters), default=0) + 1
                self.result.plan.clusters.append(Cluster(new_id, hit_id, [hit_id]))
                for event in self.result.plan.events:
                    if int(event.get("hit", -1)) == hit_id:
                        event["sample_id"] = f"sample_{new_id:03d}"
                self.result.settings.setdefault("review_targets", {})[str(hit_id)] = new_id
        if row:
            row["review_override"] = key
            row["classification"] = "IGNORED" if key == "I" else {"S": "SAME", "G": "GAIN_VARIANT", "D": "DIFFERENT"}[key]
        # Keep an explicit target for downstream exporters and future cluster UI.
        if key in {"S", "G"} and row and row.get("cluster_id"):
            self.result.settings.setdefault("review_targets", {})[str(hit_id)] = row["cluster_id"]
        if key == "I":
            exclude_hit(self.result, hit_id)
        else:
            refresh_reproducibility(self.result)
        self.rows = classify_hits(self.result)
        summary = self.result.summary
        self.required_card.value.setText(str(summary["required_samples"]))
        self.hits_card.value.setText(str(summary["detected_hits"]))
        self.reuse_card.value.setText(f"{summary['reuse_ratio']:.1f}%")
        self.review_card.value.setText(str(sum(item["classification"] in {"UNSURE", "OVERLAP"} for item in self.rows)))
        self._refresh_table()
        self._save_review_state()
        self._log(f"ヒット{hit_id:03d}を{key}で確定しました")

    def _save_review_state(self) -> None:
        if not self.result:
            return
        json_path = self.json_edit.text().strip()
        if not json_path and not isinstance(self.exported, dict):
            return
        try:
            excluded = {int(value) for value in self.result.settings.get("excluded_hits", [])}
            outputs = self.exported if isinstance(self.exported, dict) else {}
            sample_paths = outputs.get("samples", [])
            if isinstance(sample_paths, list) and sample_paths:
                old_sample_paths = {Path(path).resolve() for path in sample_paths}
                sample_dir = Path(sample_paths[0]).parent
                audio = load_audio(self.result.source)
                outputs["samples"] = [str(path) for path in write_hit_wavs(
                    sample_dir,
                    audio,
                    self.result.hits,
                    self.result.plan,
                    fade_in_ms=float(self.result.settings.get("fade_in_ms", 0.0)),
                    fade_out_ms=float(self.result.settings.get("fade_out_ms", 0.0)),
                )]
                new_sample_paths = {Path(path).resolve() for path in outputs["samples"]}
                for stale in old_sample_paths - new_sample_paths:
                    if stale.name.startswith("sample_") and stale.suffix.casefold() == ".wav":
                        stale.unlink(missing_ok=True)
            if outputs.get("csv"):
                write_hits_csv(outputs["csv"], self.result.hits, self.result.plan.events, excluded_hits=excluded)
            if outputs.get("bms"):
                write_bms(
                    outputs["bms"],
                    self.result.plan,
                    bpm=self.result.settings.get("bpm"),
                    offset=float(self.result.settings.get("offset", 0.0)),
                    subdivision=int(self.result.settings.get("subdivision", 16)),
                    channel=str(self.result.settings.get("bms_channel", "01")),
                    wav_prefix=relative_sample_prefix(outputs["bms"], Path(outputs["samples"][0]).parent if isinstance(outputs.get("samples"), list) and outputs["samples"] else None),
                    excluded_hits=excluded,
                )
            if outputs.get("bmson"):
                write_bmson(outputs["bmson"], self.result.plan, bpm=self.result.settings.get("bpm"), offset=float(self.result.settings.get("offset", 0.0)), excluded_hits=excluded)
            outputs["validation"] = validate_exports(self.result, outputs)
            self.result.settings["validation"] = outputs["validation"]
            self.result.settings["exports"] = {key: value for key, value in outputs.items() if key != "validation"}
            if json_path:
                write_json(json_path, self.result.to_dict())
        except OSError as exc:
            self._log(f"レビュー保存エラー: {exc}")

    def _selected_row_id(self) -> int | None:
        selected = self.hit_table.selectedItems() if hasattr(self, "hit_table") else []
        if selected:
            return selected[0].tableWidget().item(selected[0].row(), 0).data(Qt.UserRole)
        return None

    def _table_selection_changed(self, current_row: int, current_column: int, previous_row: int, previous_column: int) -> None:
        if current_row < 0 or current_row >= self.hit_table.rowCount():
            return
        item = self.hit_table.item(current_row, 0)
        if not item:
            return
        self._select_hit(item.data(Qt.UserRole))

    def _select_hit(self, hit_id: int) -> None:
        row = next((row for row in self.rows if row["id"] == hit_id), None)
        if not row:
            return
        self.waveform.set_selected(hit_id)
        label = CLASS_LABELS.get(row["classification"], row["classification"])
        self.detail_title.setText(f"ヒット {row['id']:03d}  ·  {label}")
        self.detail_title.setStyleSheet(f"color:{CLASS_COLORS.get(row['classification'], '#e5e7eb')};font-size:12pt;font-weight:700;")
        profile = self.result.settings.get("similarity_profile", {}) if self.result else {}
        profile_name = "波形・スペクトル優先" if profile.get("name") == "waveform_spectral_v2" else profile.get("name", "類似度優先")
        warning = "  ·  警告: 音の重なり" if row["overlap"] else ""
        self.detail_metrics.setText(
            f"時刻 {format_seconds(row['time'])}  ·  {row['sample_id']}  ·  信頼度 {row['confidence']:.1f}%  ·  音量差 {row['gain_db']:+.2f} dB{warning}\n"
            f"正規化波形 {row['waveform'] * 100:.2f}%   生波形 {row['raw'] * 100:.2f}%   スペクトル {row['spectral'] * 100:.2f}%   "
            f"アタック {row['attack'] * 100:.2f}%   ボディ {row['body'] * 100:.2f}%   テール {row['tail'] * 100:.2f}%\n"
            f"判定プロファイル {profile_name}  ·  "
            f"波形基準 {float(profile.get('waveform_threshold', 0.95)):.3f}  ·  "
            f"スペクトル基準 {float(profile.get('spectral_threshold', 0.94)):.3f}  ·  "
            f"位置合わせ ±{float(profile.get('alignment_ms', 20.0)):.1f}ms"
        )

    def _sample_path_for_selected(self) -> Path | None:
        hit_id = self._selected_row_id()
        row = next((row for row in self.rows if row["id"] == hit_id), None)
        if not row or not self.result or row["sample_id"] == "—":
            return None
        exported = self.exported.get("samples", []) if isinstance(self.exported, dict) else []
        for path in exported if isinstance(exported, list) else []:
            if Path(path).stem.casefold() == row["sample_id"].casefold():
                return Path(path)
        target = self.samples_edit.text().strip()
        return Path(target) / f"{row['sample_id']}.wav" if target else None

    def _play_selected(self) -> None:
        path = self._sample_path_for_selected()
        if not path or not path.exists():
            self._log("このヒットの代表WAVはまだありません。")
            return
        try:
            if sys.platform == "win32":
                import winsound

                winsound.PlaySound(str(path), winsound.SND_FILENAME | winsound.SND_ASYNC)
            else:
                from PySide6.QtCore import QProcess

                QProcess.startDetached("xdg-open", [str(path)])
            self._log(f"再生中: {path.name}")
        except Exception as exc:
            self._show_error("再生に失敗しました", str(exc))

    def _open_samples_folder(self) -> None:
        target = self.samples_edit.text().strip()
        if not target:
            return
        path = Path(target)
        path.mkdir(parents=True, exist_ok=True)
        try:
            if sys.platform == "win32":
                os.startfile(str(path))
            else:
                from PySide6.QtCore import QProcess

                QProcess.startDetached("xdg-open", [str(path)])
        except Exception as exc:
            self._show_error("フォルダを開けませんでした", str(exc))

    def _show_error(self, title: str, message: str, details: str | None = None) -> None:
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Critical)
        box.setWindowTitle(title)
        box.setText(message)
        if details:
            box.setDetailedText(details)
        box.exec()

    def dragEnterEvent(self, event) -> None:
        if event.mimeData().hasUrls():
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event) -> None:
        urls = event.mimeData().urls()
        if urls:
            self.set_input_path(urls[0].toLocalFile())
            event.acceptProposedAction()

    def closeEvent(self, event) -> None:
        self.waveform.shutdown()
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
        if self.batch_worker and self.batch_worker.isRunning():
            self.batch_worker.requestInterruption()
            self.batch_worker.wait(3000)
        self.settings_store.setValue("geometry", self.saveGeometry())
        event.accept()


def create_app(argv: list[str] | None = None) -> QApplication:
    os.environ.setdefault("QT_ENABLE_HIGHDPI_SCALING", "1")
    app = QApplication(argv or sys.argv)
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(APP_VERSION)
    app.setOrganizationName("StemReuse")
    app.setStyle("Fusion")
    font_family = "Yu Gothic UI"
    for font_path in (r"C:\Windows\Fonts\YuGothM.ttc", r"C:\Windows\Fonts\meiryo.ttc", r"C:\Windows\Fonts\NotoSansJP-VF.ttf"):
        if Path(font_path).exists():
            font_id = QFontDatabase.addApplicationFont(font_path)
            if font_id >= 0 and QFontDatabase.applicationFontFamilies(font_id):
                font_family = QFontDatabase.applicationFontFamilies(font_id)[0]
                break
    font = QFont(font_family, 10)
    font.setStyleHint(QFont.SansSerif)
    app.setFont(font)
    return app


def main(argv: list[str] | None = None) -> int:
    parser = __import__("argparse").ArgumentParser(prog="bms-reuse-gui", description="StemReuseデスクトップGUI")
    parser.add_argument("input", nargs="?", help="起動時に開くWAV（任意）")
    args = parser.parse_args(argv)
    app = create_app()
    window = MainWindow(args.input)
    window.show()
    return app.exec()


__all__ = ["AnalysisWorker", "MainWindow", "classify_hits", "create_app", "format_seconds", "main"]
