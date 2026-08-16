"""Qt desktop interface for the BMS stem reuse analyzer.

The GUI is intentionally a client of :func:`analyze_file`; it owns only
presentation, export orchestration, and the cancellation boundary.
"""

from __future__ import annotations

import os
import sys
import threading
import traceback
from pathlib import Path

try:
    from PySide6.QtCore import QEvent, QObject, QPoint, QRect, QSettings, QSize, QThread, Qt, Signal, Slot
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
        QTableWidget,
        QTableWidgetItem,
        QVBoxLayout,
        QWidget,
    )
except ImportError as exc:  # pragma: no cover - exercised by the CLI-only install
    raise RuntimeError("GUI requires PySide6. Install with: pip install .[gui]") from exc

from .application import AnalysisCancelled, AnalysisResult, analyze_file
from .audio.loader import load_audio
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.wav_exporter import write_hit_wavs


APP_NAME = "StemReuse"
APP_VERSION = "0.2.0"

CLASS_COLORS = {
    "BASE": "#60a5fa",
    "SAME": "#34d399",
    "GAIN_VARIANT": "#fbbf24",
    "DIFFERENT": "#fb7185",
    "UNSURE": "#c084fc",
    "OVERLAP": "#f97316",
}

CLASS_LABELS = {
    "BASE": "基準サンプル",
    "SAME": "同一",
    "GAIN_VARIANT": "音量違い",
    "DIFFERENT": "別音",
    "UNSURE": "判定保留",
    "OVERLAP": "音の重なり",
}

FILTER_LABELS = {
    "All": "すべて",
    "BASE": "基準サンプル",
    "SAME": "同一",
    "GAIN_VARIANT": "音量違い",
    "DIFFERENT": "別音",
    "UNSURE": "判定保留",
    "OVERLAP": "音の重なり",
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
    if message.startswith("Extracting ") and message.endswith(" hits"):
        return f"{message[11:-5]}個のヒットを切り出し中"
    if message == "Extracting features":
        return "特徴量を抽出中"
    if message == "Comparing and clustering hits":
        return "比較・クラスタリング中"
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
QMainWindow, QWidget { background: #0b1020; color: #e5e7eb; font-family: "Yu Gothic UI", "Meiryo UI", "Segoe UI"; font-size: 10pt; }
QFrame#Panel, QFrame#DropZone, QGroupBox { background: #111827; border: 1px solid #26344a; border-radius: 10px; }
QFrame#DropZone { border: 1px dashed #3b82f6; background: #0e1729; }
QFrame#DropZone[dragActive="true"] { background: #102442; border: 1px solid #38bdf8; }
QLabel#Brand { color: #67e8f9; font-size: 17pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Kicker { color: #7dd3fc; font-size: 8pt; font-weight: 700; letter-spacing: 1px; }
QLabel#Title { color: #f8fafc; font-size: 20pt; font-weight: 700; }
QLabel#Subtle, QLabel#Status { color: #94a3b8; }
QLabel#Value { color: #f8fafc; font-size: 17pt; font-weight: 700; }
QLabel#MetricCaption { color: #94a3b8; font-size: 8pt; font-weight: 600; text-transform: uppercase; }
QLabel#DropTitle { color: #dbeafe; font-size: 12pt; font-weight: 700; }
QLabel#DropHint { color: #93c5fd; }
QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox { background: #0b1220; border: 1px solid #334155; border-radius: 6px; padding: 6px 8px; color: #e5e7eb; min-height: 20px; }
QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus { border: 1px solid #38bdf8; }
QComboBox QAbstractItemView { background: #111827; color: #e5e7eb; selection-background-color: #1d4ed8; }
QPushButton { background: #1e293b; color: #e2e8f0; border: 1px solid #334155; border-radius: 6px; padding: 7px 12px; min-height: 20px; }
QPushButton:hover { background: #26354a; border-color: #64748b; }
QPushButton:pressed { background: #334155; }
QPushButton:disabled { color: #64748b; background: #111827; }
QPushButton#Primary { background: #0ea5e9; border-color: #38bdf8; color: #042f49; font-weight: 700; }
QPushButton#Primary:hover { background: #38bdf8; }
QPushButton#Danger { background: #3f1d2b; border-color: #be123c; color: #fecdd3; }
QCheckBox { spacing: 7px; color: #cbd5e1; }
QCheckBox::indicator { width: 15px; height: 15px; }
QCheckBox::indicator:unchecked { background: #0b1220; border: 1px solid #475569; border-radius: 3px; }
QCheckBox::indicator:checked { background: #0ea5e9; border: 1px solid #38bdf8; border-radius: 3px; }
QGroupBox { margin-top: 10px; padding: 14px 10px 10px 10px; font-weight: 700; color: #bae6fd; }
QGroupBox::title { subcontrol-origin: margin; left: 12px; padding: 0 5px; }
QTableWidget { background: #0d1525; alternate-background-color: #111c30; border: 1px solid #26344a; border-radius: 8px; gridline-color: #1e293b; selection-background-color: #164e63; selection-color: #f8fafc; }
QHeaderView::section { background: #172235; color: #93c5fd; border: none; border-right: 1px solid #26344a; border-bottom: 1px solid #26344a; padding: 7px; font-weight: 700; }
QListWidget { background: #0d1525; border: 1px solid #26344a; border-radius: 7px; }
QListWidget::item { padding: 6px; }
QListWidget::item:selected { background: #164e63; }
QPlainTextEdit { background: #08101c; border: 1px solid #26344a; border-radius: 7px; color: #94a3b8; font-family: "Cascadia Mono", Consolas; font-size: 9pt; }
QProgressBar { background: #0b1220; border: 1px solid #334155; border-radius: 5px; text-align: center; color: #dbeafe; height: 12px; }
QProgressBar::chunk { background: #0ea5e9; border-radius: 4px; }
QStatusBar { background: #08101c; color: #94a3b8; }
QSplitter::handle { background: #1e293b; }
QToolTip { background: #172235; color: #f8fafc; border: 1px solid #475569; padding: 5px; }
"""

LIGHT_STYLE = DARK_STYLE.replace("#0b1020", "#f5f7fb").replace("#111827", "#ffffff").replace("#0e1729", "#f0f9ff").replace("#0d1525", "#ffffff").replace("#0b1220", "#f8fafc").replace("#08101c", "#f1f5f9").replace("#e5e7eb", "#1e293b").replace("#f8fafc", "#0f172a").replace("#cbd5e1", "#334155").replace("#e2e8f0", "#1e293b").replace("#94a3b8", "#64748b")


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
    rows: list[dict] = []
    for hit in result.hits:
        report = report_by_hit.get(hit.id)
        classification = report.classification if report else "BASE"
        rows.append(
            {
                "id": hit.id,
                "time": hit.time,
                "classification": classification,
                "confidence": report.confidence if report else 100.0,
                "gain_db": report.gain_db if report else 0.0,
                "spectral": report.spectral_similarity if report else 1.0,
                "waveform": report.gain_normalized_similarity if report else 1.0,
                "attack": report.attack_similarity if report else 1.0,
                "body": report.body_similarity if report else 1.0,
                "tail": report.tail_similarity if report else 1.0,
                "overlap": bool(hit.overlap_warning or (report and report.overlap_warning)),
                "sample_id": sample_by_hit.get(hit.id, "—"),
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

    def cancel(self) -> None:
        self.cancel_event.set()

    def _on_progress(self, percent: int, message: str) -> None:
        self.progress.emit(max(0, min(100, percent)), localize_progress(message))

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
                exported["samples"] = [str(path) for path in write_hit_wavs(export_dir, audio, result.hits, result.plan)]
            csv_path = self.outputs.get("csv")
            if csv_path:
                self._on_progress(98, "Writing event CSV")
                exported["csv"] = str(write_hits_csv(csv_path, result.hits, result.plan.events))
            self._on_progress(100, "Analysis complete")
            self.result_ready.emit(result, exported)
        except AnalysisCancelled:
            self.canceled.emit()
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


class WaveformView(QWidget):
    hit_selected = Signal(int)

    def __init__(self, parent: QWidget | None = None):
        super().__init__(parent)
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.duration = 0.0
        self.rows: list[dict] = []
        self.selected_id: int | None = None
        self.setToolTip("ヒットマーカーをクリックして詳細を確認")

    def set_result(self, result: AnalysisResult | None) -> None:
        if result is None:
            self.duration, self.rows, self.selected_id = 0.0, [], None
        else:
            self.duration = result.duration
            self.rows = classify_hits(result)
            self.selected_id = self.rows[0]["id"] if self.rows else None
        self.update()

    def set_selected(self, hit_id: int | None) -> None:
        self.selected_id = hit_id
        self.update()

    def mousePressEvent(self, event) -> None:
        if not self.rows or self.duration <= 0:
            return
        x = max(0, min(self.width() - 1, event.position().x()))
        target = min(self.rows, key=lambda row: abs(row["time"] / self.duration * self.width() - x))
        self.selected_id = target["id"]
        self.hit_selected.emit(target["id"])
        self.update()

    def paintEvent(self, event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        rect = self.rect().adjusted(12, 10, -12, -24)
        painter.fillRect(self.rect(), QColor("#0a1220"))
        painter.setPen(QPen(QColor("#1e3a5f"), 1))
        for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
            x = rect.left() + round(rect.width() * fraction)
            painter.drawLine(x, rect.top(), x, rect.bottom())
            painter.setPen(QColor("#64748b"))
            painter.drawText(x + 3, self.height() - 7, format_seconds(self.duration * fraction))
            painter.setPen(QPen(QColor("#1e3a5f"), 1))
        painter.setPen(QPen(QColor("#27435e"), 1))
        painter.drawLine(rect.left(), rect.center().y(), rect.right(), rect.center().y())
        if not self.rows:
            painter.setPen(QColor("#64748b"))
            painter.drawText(rect, Qt.AlignCenter, "解析タイムライン · WAVをドロップして開始")
            return
        for row in self.rows:
            x = rect.left() + round(rect.width() * row["time"] / max(self.duration, 1e-9))
            color = QColor(CLASS_COLORS.get(row["classification"], "#94a3b8"))
            amplitude = max(0.12, min(1.0, (row.get("confidence", 50.0) or 50.0) / 100.0))
            height = max(5, round(rect.height() * 0.42 * amplitude))
            painter.setPen(QPen(color, 2 if row["id"] == self.selected_id else 1))
            painter.drawLine(x, rect.center().y() - height, x, rect.center().y() + height)
            if row["id"] == self.selected_id:
                painter.setPen(QPen(QColor("#f8fafc"), 1))
                painter.drawEllipse(QPoint(x, rect.center().y() - height - 4), 3, 3)


class MetricCard(QFrame):
    def __init__(self, caption: str, accent: str, parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName("Panel")
        self.setMinimumHeight(78)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(13, 10, 13, 10)
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
        self.result: AnalysisResult | None = None
        self.rows: list[dict] = []
        self.exported: dict[str, object] = {}
        self._last_progress_message = ""

        self._build_ui()
        self._apply_theme(self.settings_store.value("theme", "Dark"))
        geometry = self.settings_store.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        if initial_path:
            self.set_input_path(initial_path)
        self._set_running(False)
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
        root_layout.addLayout(header)

        title = QHBoxLayout()
        title_labels = QVBoxLayout()
        title_label = QLabel("最小限のキー音セットを見つける")
        title_label.setObjectName("Title")
        subtitle = QLabel("ヒットを検出・音色を比較し、BMS用の代表サンプルを書き出します。")
        subtitle.setObjectName("Subtle")
        title_labels.addWidget(title_label)
        title_labels.addWidget(subtitle)
        title.addLayout(title_labels)
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
        settings_form.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        self.instrument_combo = QComboBox()
        for key, label in INSTRUMENT_LABELS.items():
            self.instrument_combo.addItem(label, key)
        self.threshold_spin = self._double_spin(0.995, 0.0, 1.0, 0.001, 3)
        self.spectral_spin = self._double_spin(0.92, 0.0, 1.0, 0.001, 3)
        self.onset_spin = self._double_spin(0.35, 0.0, 1.0, 0.01, 2)
        self.separation_spin = self._double_spin(50.0, 1.0, 5000.0, 1.0, 0, " ms")
        self.pre_roll_spin = self._double_spin(5.0, 0.0, 1000.0, 1.0, 0, " ms")
        self.window_spin = self._double_spin(800.0, 10.0, 10000.0, 10.0, 0, " ms")
        self.bpm_spin = self._double_spin(0.0, 0.0, 999.0, 0.5, 1, " BPM")
        self.offset_spin = self._double_spin(0.0, -60.0, 60.0, 0.001, 3, " s")
        self.alignment_spin = self._double_spin(5.0, 0.0, 100.0, 0.5, 1, " ms")
        self.subdivision_spin = QSpinBox()
        self.subdivision_spin.setRange(1, 128)
        self.subdivision_spin.setValue(16)
        settings_form.addRow("楽器", self.instrument_combo)
        settings_form.addRow("同一判定しきい値", self.threshold_spin)
        settings_form.addRow("スペクトルしきい値", self.spectral_spin)
        settings_form.addRow("オンセットしきい値", self.onset_spin)
        settings_form.addRow("最小間隔", self.separation_spin)
        settings_form.addRow("プリロール", self.pre_roll_spin)
        settings_form.addRow("ウィンドウ長", self.window_spin)
        settings_form.addRow("BPM（任意）", self.bpm_spin)
        settings_form.addRow("グリッドオフセット", self.offset_spin)
        settings_form.addRow("分割数", self.subdivision_spin)
        settings_form.addRow("最大アライメント", self.alignment_spin)
        left_layout.addWidget(settings_box)

        output_box = QGroupBox("出力")
        output_layout = QVBoxLayout(output_box)
        self.json_edit = QLineEdit()
        self.samples_edit = QLineEdit()
        self.csv_edit = QLineEdit()
        self.csv_check = QCheckBox("イベントCSVを書き出す")
        self.csv_check.setChecked(True)
        self.samples_check = QCheckBox("代表WAVを書き出す")
        self.samples_check.setChecked(True)
        output_layout.addWidget(self._path_row("JSON", self.json_edit, self._browse_json))
        output_layout.addWidget(self._path_row("サンプル", self.samples_edit, self._browse_samples))
        output_layout.addWidget(self._path_row("CSV", self.csv_edit, self._browse_csv))
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
        self.waveform.setObjectName("Panel")
        self.waveform.hit_selected.connect(self._select_hit)
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
        layout.setSpacing(5)
        label = QLabel(caption)
        label.setMinimumWidth(52)
        layout.addWidget(label)
        layout.addWidget(edit, 1)
        button = QPushButton("…")
        button.setFixedWidth(30)
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
        self.json_edit.setText(str(path.with_suffix(".bra.json")))
        self.samples_edit.setText(str(path.parent / f"{path.stem}_keysounds"))
        self.csv_edit.setText(str(path.with_suffix(".csv")))
        self._log(f"入力を選択しました: {path}")

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

    @staticmethod
    def _same_path(left: str | Path, right: str | Path) -> bool:
        return str(Path(left).resolve()).casefold() == str(Path(right).resolve()).casefold()

    def _settings(self) -> dict:
        bpm = self.bpm_spin.value() or None
        return {
            "instrument": self.instrument_combo.currentData() or "kick",
            "threshold": self.threshold_spin.value(),
            "spectral_threshold": self.spectral_spin.value(),
            "onset_threshold": self.onset_spin.value(),
            "min_separation_ms": self.separation_spin.value(),
            "pre_roll_ms": self.pre_roll_spin.value(),
            "window_ms": self.window_spin.value(),
            "max_alignment_ms": self.alignment_spin.value(),
            "bpm": bpm,
            "offset": self.offset_spin.value(),
            "subdivision": self.subdivision_spin.value(),
        }

    def _outputs(self) -> dict:
        input_path = Path(self.input_edit.text())
        output: dict[str, str | None] = {"json": self.json_edit.text().strip() or None, "samples": None, "csv": None}
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

    def _set_running(self, running: bool) -> None:
        self.analyze_button.setEnabled(not running)
        self.cancel_button.setEnabled(running)
        self.drop_zone.setEnabled(not running)
        self.input_edit.setEnabled(not running)
        if running:
            self.progress_bar.setValue(0)
            self.status_label.setText("解析中…")
        else:
            self.status_label.setText("準備完了" if self.result is None else "解析結果を表示中")

    def start_analysis(self) -> None:
        if self.worker and self.worker.isRunning():
            return
        input_text = self.input_edit.text().strip()
        if not input_text:
            self._show_error("入力が必要です", "WAVステムをドロップするか、Ctrl+Oで選択してください。")
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
        self.worker = AnalysisWorker(input_path, self._settings(), outputs, self)
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
            self._log("キャンセルを要求しました…")

    @Slot(int, str)
    def _on_progress(self, percent: int, message: str) -> None:
        self.progress_bar.setValue(percent)
        self.status_label.setText(message)
        if message != self._last_progress_message:
            self._log(f"{percent:3d}%  {message}")
            self._last_progress_message = message

    @Slot(object, object)
    def _on_result(self, result: AnalysisResult, exported: dict) -> None:
        self.result, self.exported = result, exported
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
        self._log(f"解析エラー: {message}")
        self._show_error("解析に失敗しました", message, f"技術詳細（開発者向け）:\n{details}")

    @Slot()
    def _on_canceled(self) -> None:
        self.progress_bar.setValue(0)
        self.status_label.setText("キャンセル済み")
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
            if wanted != "All" and row["classification"] != wanted:
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
        self.detail_metrics.setText(
            f"時刻 {format_seconds(row['time'])}  ·  {row['sample_id']}  ·  信頼度 {row['confidence']:.1f}%  ·  音量差 {row['gain_db']:+.2f} dB\n"
            f"波形 {row['waveform'] * 100:.2f}%   スペクトル {row['spectral'] * 100:.2f}%   "
            f"アタック {row['attack'] * 100:.2f}%   ボディ {row['body'] * 100:.2f}%   テール {row['tail'] * 100:.2f}%"
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
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self.worker.wait(3000)
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
