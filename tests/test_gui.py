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
        from bms_reuse.gui import MainWindow

        window = MainWindow()
        self.assertEqual(window.windowTitle(), "StemReuse · BMS Stem Reuse Analyzer")
        self.assertFalse(window.cancel_button.isEnabled())
        window.close()


if __name__ == "__main__":
    unittest.main()
