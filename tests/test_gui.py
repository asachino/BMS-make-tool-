import importlib.util
import json
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from bms_reuse.application import AnalysisResult, analyze_file
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
        from bms_reuse.gui import MainWindow, classify_hits, format_seconds

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
            window = MainWindow()
            window.result = result
            window.rows = rows
            window._select_hit(rows[0]["id"])
            self.assertIn("正規化波形", window.detail_metrics.text())
            self.assertIn("判定プロファイル 波形・スペクトル優先", window.detail_metrics.text())
            window.close()

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
        self.assertEqual(window.threshold_spin.value(), 0.95)
        self.assertEqual(window.spectral_spin.value(), 0.94)
        self.assertEqual(window.alignment_spin.value(), 20.0)
        self.assertFalse(window.fast_compare_check.isChecked())
        self.assertEqual(window.fast_compare_check.text(), "高速比較")
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
            localize_progress("Comparing and clustering hits 3/12 (8 comparisons, 2 cache hits)"),
            "比較・クラスタリング中 3/12 (8件比較, 2件キャッシュ再利用)",
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
            window.fast_compare_check.setChecked(True)
            self.assertTrue(window._settings()["fast_compare"])
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
            self.assertEqual(settings["max_alignment_ms"], 20.0)
            self.assertEqual(settings["bpm_snap_tolerance_ms"], 5.0)
            self.assertEqual(settings["compare_mode"], "normal")
            self.assertFalse(settings["fast_compare"])
            self.assertEqual(settings["similarity_profile"]["name"], "waveform_spectral_v2")
            self.assertEqual(settings["similarity_profile"]["waveform_threshold"], 0.95)
            self.assertEqual(settings["similarity_profile"]["spectral_threshold"], 0.94)
            self.assertEqual(settings["similarity_profile"]["alignment_ms"], 20.0)

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

    def test_waveform_preview_controls_and_cluster_markers(self):
        from bms_reuse.gui import WaveformView, classify_hits

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "元音声.wav"
            signal = [0.0] * 1200
            signal[100] = 0.8
            signal[700] = 0.6
            write_wav(source, signal, 1000)
            audio = load_audio(source)
            view = WaveformView()
            view.set_source_audio(source, audio)
            self.assertAlmostEqual(view.duration, 1.2)
            self.assertTrue(view.waveform_points)
            self.assertEqual(view.playback_play_button.text(), "再生")
            self.assertEqual(view.playback_pause_button.text(), "一時停止")
            self.assertEqual(view.playback_stop_button.text(), "停止")
            self.assertEqual(view.seek_slider.accessibleName(), "元WAV再生位置")
            self.assertGreaterEqual(view.minimumHeight(), 300)
            self.assertGreaterEqual(view.canvas.minimumHeight(), 220)
            self.assertEqual(view.marker_labels_check.text(), "分類ラベル")
            view.marker_labels_check.setChecked(False)
            self.assertFalse(view.marker_labels_check.isChecked())
            view._set_position(0.7)
            self.assertAlmostEqual(view.position, 0.7)
            self.assertGreater(view.seek_slider.value(), 0)
            view.resize(900, 230)
            self.assertFalse(view.canvas.grab().isNull())
            result = analyze_file(source, min_separation_ms=300.0, window_ms=200.0)
            view.set_result(result)
            rows = classify_hits(result)
            self.assertTrue(rows)
            self.assertEqual(view.rows[0]["cluster_id"], rows[0]["cluster_id"])
            self.assertIn("色と文字", view.playback_hint_label.text())
            self.assertFalse(view.playback_pause_button.isEnabled())
            view.shutdown()
            view.close()

    def test_waveform_stereo_zoom_and_pan_use_source_samples(self):
        from bms_reuse.gui import WaveformView

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "stereo.wav"
            samples = [[0.8 if index % 2 == 0 else -0.4, -0.7 if index % 3 == 0 else 0.2] for index in range(6000)]
            write_wav(source, samples, 1000, channels=2)
            view = WaveformView()
            view.set_source_audio(source, load_audio(source))
            self.assertTrue(view.waveform_points)
            self.assertTrue(len(view.waveform_points[0]) >= 4)
            view.zoom_slider.setValue(2)
            self.assertEqual(view.zoom_slider.value(), 2)
            self.assertFalse(view._follow_playhead)
            view.pan_slider.setValue(5000)
            self.assertFalse(view._follow_playhead)
            view.resize(900, 230)
            self.assertFalse(view.canvas.grab().isNull())
            view.shutdown()
            view.close()

    def test_cluster_reuse_setting_reclusters_without_reanalysis(self):
        from bms_reuse.gui import MainWindow
        from bms_reuse.similarity.score import SimilarityReport

        hits = [Hit(index, index, float(index), [0.0], index, index + 1) for index in range(3)]
        reports = [
            SimilarityReport(0, 1, 0.90, 0.90, 0.0, 0.90, 0.90, 0.90, 0.90, 0),
            SimilarityReport(0, 2, 0.99, 0.99, 0.0, 0.99, 0.99, 0.99, 0.99, 0),
        ]
        plan = ReusePlan(
            [Cluster(1, 0, [0, 1]), Cluster(2, 2, [2])],
            [
                {"hit": 0, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                {"hit": 1, "time": 1.0, "sample_id": "sample_001", "gain_db": 0.0},
                {"hit": 2, "time": 2.0, "sample_id": "sample_002", "gain_db": 0.0},
            ],
        )
        result = AnalysisResult(
            "stem.wav",
            1000,
            3.0,
            hits,
            reports,
            plan,
            {"threshold": 0.95, "spectral_threshold": 0.94, "review_overrides": {}, "review_targets": {}, "excluded_hits": []},
            "source-hash",
        )
        window = MainWindow()
        self.assertFalse(window.cluster_box.isEnabled())
        window._on_result(result, {})
        self.assertTrue(window.cluster_box.isEnabled())
        window.cluster_slider.setValue(90)
        window._apply_cluster_threshold()
        self.assertEqual([cluster.hit_ids for cluster in result.plan.clusters], [[0, 1, 2]])
        self.assertEqual({event["sample_id"] for event in result.plan.events}, {"sample_001"})
        self.assertEqual(result.settings["recluster_profile"], "custom")
        window.close()

    def test_representative_wav_count_matches_clusters_and_gain_variants(self):
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "source.wav"
            write_wav(source, [[0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], 1000)
            audio = load_audio(source)
            hits = [
                Hit(1, 0, 0.0, audio.samples[0:2], 0, 2),
                Hit(2, 2, 0.002, audio.samples[2:4], 2, 4),
                Hit(3, 4, 0.004, audio.samples[4:6], 4, 6),
            ]
            plan = ReusePlan(
                [Cluster(1, 1, [1, 2]), Cluster(2, 3, [3])],
                [
                    {"hit": 1, "sample_id": "sample_001", "gain_db": 0.0},
                    {"hit": 2, "sample_id": "sample_001", "gain_db": -6.0},
                    {"hit": 3, "sample_id": "sample_002", "gain_db": 0.0},
                ],
            )
            output = Path(directory) / "representatives"
            paths = write_hit_wavs(output, audio, hits, plan)
            self.assertEqual(len(paths), len(plan.clusters))
            self.assertEqual(len({path.name for path in paths}), len(plan.clusters))
            self.assertEqual(len(list(output.glob("sample_*.wav"))), len(plan.clusters))
            self.assertEqual([path.stem for path in paths], ["sample_001", "sample_002"])

    def test_cut_plan_and_sound_review_controls(self):
        from bms_reuse.gui import CUT_MODE_LABELS, MainWindow

        window = MainWindow()
        self.assertEqual(
            {window.cut_mode_combo.itemData(index) for index in range(window.cut_mode_combo.count())},
            set(CUT_MODE_LABELS),
        )
        window.bpm_spin.setValue(120.0)
        window.cut_mode_combo.setCurrentIndex(window.cut_mode_combo.findData("manual"))
        window.user_boundary_edit.setText("0.500, 1.000, 0.500, -1")
        settings = window._settings()
        self.assertEqual(settings["cut_plan"]["mode"], "manual")
        self.assertEqual(settings["cut_plan"]["points"], [0.5, 1.0])

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "音種.wav"
            signal = [0.0] * 1600
            signal[100] = 1.0
            signal[900] = 0.8
            write_wav(source, signal, 1000)
            result = analyze_file(source, min_separation_ms=300.0, window_ms=200.0)
            window._on_result(result, {})
            self.assertTrue(window.rows)
            hit_id = window.rows[0]["id"]
            window._select_hit(hit_id)
            snare_index = window.hit_type_combo.findData("snare")
            window.hit_type_combo.setCurrentIndex(snare_index)
            window._apply_hit_type()
            self.assertEqual(result.settings["hit_type_overrides"][str(hit_id)], "snare")
            self.assertEqual(next(row for row in window.rows if row["id"] == hit_id)["sound_type"], "snare")
            window._set_cut_override("excluded")
            self.assertNotIn(hit_id, result.settings["active_hit_ids"])
            self.assertEqual(next(row for row in window.rows if row["id"] == hit_id)["cut_state"], "除外")
            window._set_cut_override("accepted")
            self.assertIn(hit_id, result.settings["active_hit_ids"])
        window.close()

    def test_saved_json_restores_analysis_controls_and_pattern_units(self):
        from bms_reuse.gui import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "pattern.wav"
            write_wav(source, [0.0] * 1200, 1000)
            result = analyze_file(
                source,
                instrument="snare",
                bpm=120.0,
                beat_division=8,
                margin=88.0,
                min_interval_sec=(60.0 / 120.0) / 8.0 * 0.88,
                cut_plan={"mode": "pattern", "pattern": [0.25, 0.375]},
            )
            result.settings.update({"time_signature": "3/4", "swing_percent": 62.0})
            json_path = directory / "pattern.bra.json"
            json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
            window = MainWindow()
            with patch("bms_reuse.gui.QFileDialog.getOpenFileName", return_value=(str(json_path), "JSON")):
                window._open_analysis_json()
            self.assertIsNotNone(window.result)
            self.assertEqual(window.instrument_combo.currentData(), "snare")
            self.assertEqual(window.bpm_spin.value(), 120.0)
            self.assertEqual(window.beat_division_combo.currentData(), 8)
            self.assertEqual(window.margin_spin.value(), 88.0)
            self.assertEqual(window.cut_mode_combo.currentData(), "pattern")
            self.assertEqual(window.time_signature_combo.currentData(), "3/4")
            self.assertEqual(window.swing_spin.value(), 62.0)
            self.assertEqual(
                [float(value) for value in window.user_boundary_edit.text().split(", ")],
                [0.25, 0.375],
            )
            window.close()

    def test_smart_end_controls_restore_from_preset_and_json(self):
        from bms_reuse.gui import MainWindow
        from bms_reuse.project.presets import save_preset

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            preset_path = directory / "smart-end.preset.json"
            save_preset(
                preset_path,
                {
                    "instrument": "snare",
                    "smart_end": False,
                    "smart_end_apply_to_explicit": True,
                    "smart_end_settings": {
                        "silence_rms_db": -50.0,
                        "min_tail_ms": 35.0,
                        "max_tail_ms": 900.0,
                        "next_attack_margin_ms": 3.0,
                        "silence_ms": 77.0,
                        "safety_margin_ms": 9.0,
                        "zero_crossing_ms": 4.0,
                        "silence_peak_db": -28.0,
                    },
                    "smart_end_advanced": True,
                },
            )
            window = MainWindow()
            with patch("bms_reuse.gui.QFileDialog.getOpenFileName", return_value=(str(preset_path), "JSON")):
                window._load_preset()
            self.assertEqual(window.terminal_mode_combo.currentData(), "fixed")
            self.assertEqual(window.instrument_combo.currentData(), "snare")
            self.assertEqual(window.smart_end_silence_spin.value(), -50.0)
            self.assertEqual(window.smart_end_min_spin.value(), 35.0)
            self.assertEqual(window.smart_end_max_spin.value(), 900.0)
            self.assertEqual(window.smart_end_next_attack_spin.value(), 3.0)
            self.assertEqual(window.smart_end_silence_ms_spin.value(), 77.0)
            self.assertEqual(window.smart_end_safety_margin_spin.value(), 9.0)
            self.assertEqual(window.smart_end_zero_crossing_spin.value(), 4.0)
            self.assertEqual(window.smart_end_silence_peak_spin.value(), -28.0)
            self.assertTrue(window.smart_end_apply_explicit_check.isChecked())
            self.assertTrue(window.smart_end_advanced_check.isChecked())
            self.assertFalse(window.smart_end_box.isVisible())
            window.close()

            source = directory / "smart-end.wav"
            signal = [0.0] * 600
            for offset in range(120):
                signal[100 + offset] = 0.8 * (1.0 - offset / 140.0)
            write_wav(source, signal, 1000)
            result = analyze_file(
                source,
                instrument="snare",
                onset_threshold=0.1,
                min_separation_ms=30.0,
                bpm=120.0,
                smart_end=True,
                smart_end_settings={
                    "silence_rms_db": -48.0,
                    "min_tail_ms": 32.0,
                    "max_tail_ms": 850.0,
                    "next_attack_margin_ms": 4.0,
                    "silence_ms": 77.0,
                    "safety_margin_ms": 9.0,
                    "zero_crossing_ms": 4.0,
                    "silence_peak_db": -28.0,
                },
            )
            result.settings["smart_end_advanced"] = True
            json_path = directory / "smart-end.bra.json"
            json_path.write_text(json.dumps(result.to_dict(), ensure_ascii=False), encoding="utf-8")
            window = MainWindow()
            with patch("bms_reuse.gui.QFileDialog.getOpenFileName", return_value=(str(json_path), "JSON")):
                window._open_analysis_json()
            self.assertEqual(window.terminal_mode_combo.currentData(), "smart")
            self.assertEqual(window.instrument_combo.currentData(), "snare")
            self.assertEqual(window.smart_end_silence_spin.value(), -48.0)
            self.assertEqual(window.smart_end_min_spin.value(), 32.0)
            self.assertEqual(window.smart_end_max_spin.value(), 850.0)
            self.assertEqual(window.smart_end_next_attack_spin.value(), 4.0)
            self.assertEqual(window.smart_end_silence_ms_spin.value(), 77.0)
            self.assertEqual(window.smart_end_safety_margin_spin.value(), 9.0)
            self.assertEqual(window.smart_end_zero_crossing_spin.value(), 4.0)
            self.assertEqual(window.smart_end_silence_peak_spin.value(), -28.0)
            self.assertTrue(window.smart_end_advanced_check.isChecked())
            restored = window._settings()["smart_end_settings"]
            self.assertEqual(restored["silence_ms"], 77.0)
            self.assertEqual(restored["safety_margin_ms"], 9.0)
            self.assertEqual(restored["zero_crossing_ms"], 4.0)
            self.assertEqual(restored["silence_peak_db"], -28.0)
            window.close()

    def test_endpoint_detail_and_manual_fix_use_existing_preview_path(self):
        from bms_reuse.gui import MainWindow

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = write_wav(directory / "endpoint.wav", [[0.2]] * 100, 1000)
            audio = load_audio(source)
            hit = Hit(
                1, 0, 0.0, audio.samples, 0, 40,
                end_reason="silence", end_confidence=0.86,
                end_warnings=["TAIL_CUT"],
                effective_settings={"hard_end_sample": 80, "smart_end_requested": True, "smart_end_applied": True},
            )
            plan = ReusePlan(
                [Cluster(1, 1, [1])],
                [{"hit": 1, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0}],
            )
            samples_dir = directory / "samples"
            sample_paths = write_hit_wavs(samples_dir, audio, [hit], plan)
            exports = {
                "samples_dir": str(samples_dir),
                "samples": [str(path) for path in sample_paths],
            }
            result = AnalysisResult(
                str(source), 1000, 0.1, [hit], [], plan,
                {
                    "instrument": "kick",
                    "smart_end": True,
                    "smart_end_settings": {"silence_rms_db": -42.0, "min_tail_ms": 18.0, "max_tail_ms": 620.0},
                    "active_hit_ids": [1],
                    "review_overrides": {},
                    "review_targets": {},
                    "excluded_hits": [],
                    "exports": exports,
                    "bpm": 120.0,
                    "offset": 0.0,
                    "subdivision": 16,
                    "bms_channel": "01",
                },
                "endpoint-hash",
            )
            window = MainWindow()
            window._on_result(result, dict(exports))
            window._select_hit(1)
            self.assertIn("終端", window.detail_metrics.text())
            self.assertIn("切りすぎ注意", window.detail_metrics.text())
            self.assertTrue(window.fixed_length_preview_button.isEnabled())
            self.assertTrue(window.smart_end_preview_button.isEnabled())
            with patch.object(window.waveform, "play_range") as play_range:
                window._preview_endpoint_variant("fixed")
                play_range.assert_called_once_with(0.0, 0.08)
                play_range.reset_mock()
                window._preview_endpoint_variant("smart")
                play_range.assert_called_once_with(0.0, 0.04)
            window.terminal_duration_spin.setValue(20.0)
            window._apply_terminal_fix()
            self.assertEqual(hit.source_end, 20)
            self.assertEqual(hit.end_reason, "manual")
            self.assertEqual(result.settings["terminal_overrides"]["1"]["source_end"], 20)
            self.assertIn("手動境界", window.detail_metrics.text())
            self.assertAlmostEqual(window.rows[0]["end"], 0.02)
            self.assertEqual(load_audio(Path(window.exported["samples"][0])).frame_count, 20, result.settings.get("validation"))
            self.assertTrue(result.settings["validation"].get("ok"))
            self.assertTrue(window.fixed_length_preview_button.isEnabled())
            self.assertFalse(window.smart_end_preview_button.isEnabled())
            with patch.object(window.waveform, "play_range") as play_range:
                window._preview_selected_cut()
            play_range.assert_called_once_with(0.0, 0.02)
            window.close()

    def test_review_rebuild_syncs_plan_outputs_and_bmson_prefix(self):
        from bms_reuse.gui import MainWindow
        from bms_reuse.similarity.score import SimilarityReport

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "review.wav"
            write_wav(source, [[0.2], [0.3], [0.4], [0.5], [0.6], [0.7], [0.8], [0.9], [0.1], [0.2]], 1000)
            audio = load_audio(source)
            hits = [
                Hit(1, 0, 0.0, audio.samples[0:3], 0, 3),
                Hit(2, 3, 0.003, audio.samples[3:6], 3, 6),
                Hit(3, 6, 0.006, audio.samples[6:9], 6, 9),
            ]
            reports = [
                SimilarityReport(1, 2, 0.99, 0.99, 0.0, 0.99, 0.99, 0.99, 0.99, 0),
                SimilarityReport(1, 3, 0.10, 0.10, 0.0, 0.10, 0.10, 0.10, 0.10, 0),
            ]
            plan = ReusePlan(
                [Cluster(1, 1, [1, 2]), Cluster(2, 3, [3])],
                [
                    {"hit": 1, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                    {"hit": 2, "time": 0.003, "sample_id": "sample_001", "gain_db": 0.0},
                    {"hit": 3, "time": 0.006, "sample_id": "sample_002", "gain_db": 0.0},
                ],
            )
            samples_dir = directory / "samples"
            exports = {
                "json": str(directory / "review.bra.json"),
                "samples_dir": str(samples_dir),
                "samples": [],
                "csv": str(directory / "review.csv"),
                "bms": str(directory / "review.bms"),
                "bmson": str(directory / "review.bmson"),
            }
            settings = {
                "threshold": 0.95,
                "spectral_threshold": 0.94,
                "max_alignment_ms": 20.0,
                "bpm": 120.0,
                "offset": 0.0,
                "subdivision": 16,
                "bms_channel": "01",
                "instrument": "kick",
                "review_overrides": {},
                "review_targets": {},
                "excluded_hits": [],
                "active_hit_ids": [1, 2, 3],
                "exports": exports,
                "fade_in_ms": 0.0,
                "fade_out_ms": 0.0,
                "similarity_profile": {"name": "waveform_spectral_v2"},
                "recluster_profile": "balanced",
                "recluster_thresholds": {"waveform": 0.95, "spectral": 0.94, "gain_tolerance_db": 0.25},
            }
            result = AnalysisResult(str(source), 1000, 0.01, hits, reports, plan, settings, "hash")
            window = MainWindow()
            window._on_result(result, dict(exports))
            window._select_hit(2)
            window.review_target_combo.setCurrentIndex(window.review_target_combo.findData(2))
            window._apply_review("S")
            self.assertEqual(result.settings["review_targets"]["2"], 2)
            self.assertIn(2, next(cluster for cluster in result.plan.clusters if cluster.id == 2).hit_ids)
            window._select_hit(2)
            window.review_target_combo.setCurrentIndex(window.review_target_combo.findData(1))
            window._apply_review("G")
            self.assertEqual(result.settings["review_targets"]["2"], 1)
            window._select_hit(2)
            window._apply_review("D")
            self.assertTrue(any(cluster.representative_hit == 2 and cluster.hit_ids == [2] for cluster in result.plan.clusters))
            window._select_hit(2)
            window._apply_review("I")
            self.assertNotIn(2, result.settings["active_hit_ids"])
            self.assertNotIn(2, {int(event["hit"]) for event in result.plan.events})
            window._select_hit(2)
            window._apply_review("I")
            self.assertIn(2, result.settings["active_hit_ids"])
            self.assertIn(2, {int(event["hit"]) for event in result.plan.events})
            window._select_hit(1)
            window.hit_type_combo.setCurrentIndex(window.hit_type_combo.findData("snare"))
            window._apply_hit_type()
            self.assertEqual(result.hits[0].instrument, "snare")
            self.assertTrue(Path(exports["json"]).exists())
            self.assertTrue(Path(exports["bms"]).exists())
            self.assertTrue(Path(exports["bmson"]).exists())
            self.assertEqual(result.settings["validation"].get("ok"), True)
            bmson = json.loads(Path(exports["bmson"]).read_text(encoding="utf-8"))
            self.assertTrue(all(Path(sample["name"]).name.startswith("sample_") for sample in bmson["sound_samples"]))
            self.assertTrue(all("samples/" in sample["name"] for sample in bmson["sound_samples"]))
            window.close()

    def test_ignore_then_accept_cut_keeps_review_exclusion_synced(self):
        from bms_reuse.gui import MainWindow
        from bms_reuse.similarity.score import SimilarityReport

        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            source = directory / "ignore-then-accept.wav"
            write_wav(source, [[0.2], [0.3], [0.4], [0.5], [0.6], [0.7]], 1000)
            audio = load_audio(source)
            hits = [
                Hit(1, 0, 0.0, audio.samples[0:3], 0, 3),
                Hit(2, 3, 0.003, audio.samples[3:6], 3, 6),
            ]
            reports = [SimilarityReport(1, 2, 0.99, 0.99, 0.0, 0.99, 0.99, 0.99, 0.99, 0)]
            plan = ReusePlan(
                [Cluster(1, 1, [1, 2])],
                [
                    {"hit": 1, "time": 0.0, "sample_id": "sample_001", "gain_db": 0.0},
                    {"hit": 2, "time": 0.003, "sample_id": "sample_001", "gain_db": 0.0},
                ],
            )
            samples_dir = directory / "samples"
            exports = {
                "json": str(directory / "ignore-then-accept.bra.json"),
                "samples_dir": str(samples_dir),
                "samples": [],
                "csv": str(directory / "ignore-then-accept.csv"),
                "bms": str(directory / "ignore-then-accept.bms"),
                "bmson": str(directory / "ignore-then-accept.bmson"),
            }
            result = AnalysisResult(
                str(source),
                1000,
                0.006,
                hits,
                reports,
                plan,
                {
                    "threshold": 0.95,
                    "spectral_threshold": 0.94,
                    "review_overrides": {},
                    "review_targets": {},
                    "excluded_hits": [],
                    "active_hit_ids": [1, 2],
                    "exports": exports,
                    "bpm": 120.0,
                    "offset": 0.0,
                    "subdivision": 16,
                    "bms_channel": "01",
                },
                "ignore-then-accept-hash",
            )
            window = MainWindow()
            window._on_result(result, dict(exports))

            window._select_hit(2)
            window._apply_review("I")
            window._select_hit(2)
            window._set_cut_override("accepted")

            self.assertEqual(result.settings["review_overrides"]["2"], "I")
            self.assertEqual(result.settings["cut_overrides"]["2"], "accepted")
            self.assertNotIn(2, result.settings["active_hit_ids"])
            self.assertNotIn(2, {int(event["hit"]) for event in result.plan.events})
            self.assertTrue(all(2 not in cluster.hit_ids for cluster in result.plan.clusters))
            synced_exports = result.settings["exports"]
            self.assertEqual(len(synced_exports["samples"]), len(result.plan.clusters))
            self.assertTrue(all(Path(path).exists() for path in synced_exports["samples"]))
            self.assertTrue(result.settings["validation"].get("ok"), result.settings["validation"])
            window.close()


if __name__ == "__main__":
    unittest.main()
