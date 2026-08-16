import importlib.util
import os
import tempfile
import unittest
from pathlib import Path

from bms_reuse.application import analyze_file
from bms_reuse.audio.loader import load_audio
from bms_reuse.clustering.reuse_plan import Cluster, ReusePlan
from bms_reuse.extraction.hit_extractor import Hit
from bms_reuse.export.wav_exporter import write_wav
from bms_reuse.export.wav_exporter import write_hit_wavs


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
        from PySide6.QtWidgets import QLabel

        window = MainWindow()
        self.assertEqual(window.windowTitle(), "ステムリユース · BMSステム再利用解析")
        self.assertNotIn("ヒットを検出・音色を比較", " ".join(label.text() for label in window.findChildren(QLabel)))
        self.assertEqual(window.open_folder_button.text(), "サンプルフォルダを開く")
        self.assertEqual(window.hit_table.horizontalHeaderItem(2).text(), "分類")
        self.assertEqual(window.theme_combo.currentText(), "ダーク")
        self.assertEqual(window.theme_combo.currentData(), "Dark")
        self.assertEqual(window.instrument_combo.currentData(), "kick")
        self.assertEqual([window.beat_division_combo.itemText(i) for i in range(window.beat_division_combo.count())], ["1/4", "1/8", "1/16", "1/32"])
        self.assertEqual(window.margin_spin.value(), 90.0)
        self.assertEqual(window.fade_in_spin.value(), 2.0)
        self.assertEqual(window.fade_out_spin.value(), 2.0)
        self.assertEqual(window.filter_combo.itemData(2), "SAME")
        self.assertEqual(CLASS_LABELS["SAME"], "同一")
        self.assertEqual(CLASS_LABELS["GAIN_VARIANT"], "音量違い")
        self.assertEqual(CLASS_LABELS["DIFFERENT"], "別音")
        self.assertEqual(CLASS_LABELS["UNSURE"], "判定保留")
        self.assertEqual(CLASS_LABELS["OVERLAP"], "音の重なり")
        self.assertEqual(localize_progress("Analysis complete"), "解析完了")
        self.assertEqual(localize_progress("Extracting 12 hits"), "12個のヒットを切り出し中")
        self.assertEqual(localize_progress("Extracting hits 3/12"), "ヒットを切り出し中 3/12")
        self.assertEqual(
            localize_progress("Comparing and clustering hits 3/12 (8 comparisons)"),
            "比較・クラスタリング中 3/12 (8件比較)",
        )
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
        window._set_running(True)
        self.assertIn("処理中…", window.processing_status_label.text())
        self.assertIn("キャンセル可能", window.processing_status_label.text())
        window._on_progress(57, "比較・クラスタリング中 3/12 (8件比較)")
        self.assertIn("3/12", window.processing_status_label.text())
        window._set_running(False)
        window.close()

    def test_bpm_is_required_and_interval_formula_drives_settings(self):
        from bms_reuse.gui import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "日本語ステム.wav"
            write_wav(path, [0.0] * 100, 1000)
            window = MainWindow()
            window.set_input_path(str(path))
            self.assertFalse(window.analyze_button.isEnabled())
            self.assertIn("BPM", window.bpm_error_label.text())

            window.bpm_spin.setValue(174.0)
            self.assertTrue(window.analyze_button.isEnabled())
            settings = window._settings()
            expected = (60.0 / 174.0) / 16.0 * 0.90
            self.assertAlmostEqual(settings["min_interval_sec"], expected)
            self.assertAlmostEqual(settings["min_separation_ms"], expected * 1000.0)
            self.assertEqual(settings["bpm"], 174.0)
            self.assertEqual(settings["beat_division"], 16)
            self.assertEqual(settings["margin"], 90.0)
            self.assertEqual(settings["fade_in_ms"], 2.0)
            self.assertEqual(settings["fade_out_ms"], 2.0)

            window.beat_division_combo.setCurrentIndex(0)
            window.margin_spin.setValue(80.0)
            expected = (60.0 / 174.0) / 4.0 * 0.80
            self.assertAlmostEqual(window._settings()["min_interval_sec"], expected)
            window.bpm_spin.setValue(19.0)
            self.assertFalse(window.analyze_button.isEnabled())
            self.assertIn("20", window.bpm_error_label.text())
            window.close()

    def test_analysis_settings_and_export_fades_are_recorded(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "設定.wav"
            write_wav(source, [], 1000)
            result = analyze_file(
                source,
                bpm=120.0,
                beat_division=8,
                margin=80.0,
                min_interval_sec=(60.0 / 120.0) / 8.0 * 0.80,
                fade_in_ms=2.0,
                fade_out_ms=3.0,
            )
            settings = result.to_dict()["settings"]
            self.assertIn("timings", settings)
            self.assertIn("compare_seconds", settings["timings"])
            self.assertIn("total_seconds", result.summary["timings"])
            self.assertEqual(settings["bpm"], 120.0)
            self.assertEqual(settings["beat_division"], 8)
            self.assertEqual(settings["margin"], 80.0)
            self.assertAlmostEqual(settings["min_interval_sec"], 0.05)
            self.assertEqual(settings["fade_in_ms"], 2.0)
            self.assertEqual(settings["fade_out_ms"], 3.0)

            audio = load_audio(write_wav(Path(directory) / "tone.wav", [[1.0]] * 10, 1000))
            hit = Hit(1, 0, 0.0, audio.samples, 0, 10)
            plan = ReusePlan([Cluster(1, 1, [1])], [])
            paths = write_hit_wavs(Path(directory) / "samples", audio, [hit], plan, fade_in_ms=2.0, fade_out_ms=2.0)
            exported = load_audio(paths[0])
            values = exported.samples[:, 0].tolist() if hasattr(exported.samples, "tolist") else [row[0] for row in exported.samples]
            self.assertAlmostEqual(values[0], 0.0, places=4)
            self.assertAlmostEqual(values[1], 1.0, places=2)
            self.assertAlmostEqual(values[-2], 1.0, places=2)
            self.assertAlmostEqual(values[-1], 0.0, places=4)


if __name__ == "__main__":
    unittest.main()
