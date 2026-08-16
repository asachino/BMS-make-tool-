import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from bms_reuse.application import analyze_file
from bms_reuse.export.wav_exporter import write_wav


@unittest.skipUnless(importlib.util.find_spec("PySide6"), "PySide6 is optional")
class GuiSupportTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from bms_reuse.gui import create_app

        cls.app = create_app([])

    def test_rows_and_time_format_are_review_ready(self):
        from bms_reuse.gui import classify_hits, format_seconds

        self.assertEqual(format_seconds(65.25), "01:05.25")
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "gui.wav"
            signal = [0.0] * 2400
            signal[100] = 1.0
            signal[1200] = 0.8
            write_wav(path, signal, 48000)
            result = analyze_file(path)
            rows = classify_hits(result)
            self.assertEqual(len(rows), result.summary["detected_hits"])
            self.assertEqual(rows[0]["classification"], "BASE")
            self.assertIn("sample_", rows[0]["sample_id"])

    def test_window_constructs_without_a_display(self):
        from bms_reuse.gui import CLASS_LABELS, MainWindow, localize_progress

        window = MainWindow()
        self.assertEqual(window.windowTitle(), "ステムリユース · BMSステム再利用解析")
        self.assertEqual(window.open_folder_button.text(), "サンプルフォルダを開く")
        self.assertEqual(window.hit_table.horizontalHeaderItem(2).text(), "分類")
        self.assertEqual(window.theme_combo.currentText(), "ダーク")
        self.assertEqual(window.theme_combo.currentData(), "Dark")
        self.assertEqual(window.instrument_combo.currentData(), "kick")
        self.assertEqual(window.filter_combo.itemData(2), "SAME")
        self.assertEqual(CLASS_LABELS["SAME"], "同一")
        self.assertEqual(CLASS_LABELS["GAIN_VARIANT"], "音量違い")
        self.assertEqual(CLASS_LABELS["DIFFERENT"], "別音")
        self.assertEqual(CLASS_LABELS["UNSURE"], "判定保留")
        self.assertEqual(CLASS_LABELS["OVERLAP"], "音の重なり")
        self.assertEqual(localize_progress("Analysis complete"), "解析完了")
        self.assertEqual(localize_progress("Extracting 12 hits"), "12個のヒットを切り出し中")
        visible_text = " ".join(
            [
                window.windowTitle(),
                window.drop_zone.title.text(),
                window.drop_zone.hint.text(),
                window.analyze_button.text(),
                window.cancel_button.text(),
                window.status_label.text(),
                window.required_card.caption.text(),
                window.hits_card.caption.text(),
                window.reuse_card.caption.text(),
                window.review_card.caption.text(),
            ]
        )
        for forbidden in ("Open WAV", "Analyze", "Cancel", "Ready", "Analysis", "Required", "Detected", "Review"):
            self.assertNotIn(forbidden, visible_text)
        self.assertFalse(window.cancel_button.isEnabled())
        window.close()


if __name__ == "__main__":
    unittest.main()
