"""Command line interface for the BMS stem reuse MVP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .application import analyze_file, record_output_timing
from .batch import run_batch
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.wav_exporter import write_hit_wavs
from .export.bms_exporter import relative_sample_prefix, write_bms
from .export.bmson_exporter import write_bmson
from .export.quality import validate_exports
from .project.presets import load_preset, save_preset
from .audio.loader import load_audio


def _same_path(left: Path, right: Path) -> bool:
    return str(left.resolve()).casefold() == str(right.resolve()).casefold()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="bms-reuse", description="Find reusable hits in a BMS instrument stem")
    commands = parser.add_subparsers(dest="command", required=True)
    analyze = commands.add_parser("analyze", help="analyze a PCM WAV stem")
    analyze.add_argument("input", type=Path)
    analyze.add_argument("--output", type=Path, help="analysis JSON path (default: <input>.bra.json)")
    analyze.add_argument("--export-dir", type=Path, help="write representative WAVs to this directory")
    analyze.add_argument("--csv", type=Path, help="also write a UTF-8 CSV event map")
    analyze.add_argument("--instrument", default="kick")
    analyze.add_argument("--threshold", type=float, default=0.95, help="gain-normalized waveform similarity threshold")
    analyze.add_argument("--spectral-threshold", type=float, default=0.94)
    analyze.add_argument("--onset-threshold", type=float, default=0.35)
    analyze.add_argument("--min-separation-ms", type=float, default=50.0)
    analyze.add_argument("--pre-roll-ms", type=float, default=5.0)
    analyze.add_argument("--window-ms", type=float, default=800.0)
    analyze.add_argument("--max-alignment-ms", type=float, default=20.0)
    analyze.add_argument("--bpm", type=float)
    analyze.add_argument("--offset", type=float, default=0.0)
    analyze.add_argument("--subdivision", type=int, default=16)
    analyze.add_argument("--beat-division", type=int, help="旧名。--subdivisionへ互換変換")
    analyze.add_argument("--fast-compare", action="store_true", help="rank representatives by shape features (ordering may change)")
    analyze.add_argument("--bms", type=Path, help="write an actual BMS event map (requires --bpm)")
    analyze.add_argument("--bmson", type=Path, help="write a BMSON 1.0 event map (requires --bpm)")
    analyze.add_argument("--bms-channel", default="01", help="BMS event channel (default: BGM 01; 11+ for key-sounds)")
    analyze.add_argument("--fade-in-ms", type=float, default=0.0)
    analyze.add_argument("--fade-out-ms", type=float, default=0.0)
    analyze.add_argument("--preset-in", type=Path, help="load analysis settings from JSON preset")
    analyze.add_argument("--preset-out", type=Path, help="save effective analysis settings as JSON preset")
    analyze.add_argument("--full-json", action="store_true", help="print the complete JSON result")
    batch = commands.add_parser("batch", help="analyze every WAV in a folder")
    batch.add_argument("folder", type=Path)
    batch.add_argument("--output-dir", type=Path)
    batch.add_argument("--recursive", action="store_true")
    batch.add_argument("--instrument", default="kick")
    batch.add_argument("--bpm", type=float)
    batch.add_argument("--offset", type=float, default=0.0)
    batch.add_argument("--subdivision", type=int, default=16)
    batch.add_argument("--beat-division", type=int, help="旧名。--subdivisionへ互換変換")
    batch.add_argument("--fast-compare", action="store_true")
    batch.add_argument("--bms-channel", default="01", help="BMS event channel (default: BGM 01; 11+ for key-sounds)")
    batch.add_argument("--fade-in-ms", type=float, default=0.0)
    batch.add_argument("--fade-out-ms", type=float, default=0.0)
    batch.add_argument("--bms", action="store_true", help="write BMS per input (requires --bpm)")
    batch.add_argument("--bmson", action="store_true", help="write BMSON per input (requires --bpm)")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "batch":
        try:
            manifest = run_batch(
                args.folder,
                output_dir=args.output_dir,
                recursive=args.recursive,
                instrument=args.instrument,
                bpm=args.bpm,
                offset=args.offset,
                subdivision=args.beat_division if args.beat_division is not None else args.subdivision,
                fast_compare=args.fast_compare,
                bms_channel=args.bms_channel,
                fade_in_ms=args.fade_in_ms,
                fade_out_ms=args.fade_out_ms,
                export_bms=args.bms,
                export_bmson=args.bmson,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command != "analyze":
        return 2
    try:
        output = args.output or args.input.with_suffix(".bra.json")
        if _same_path(output, args.input):
            raise ValueError("output must not overwrite the input WAV")
        if args.csv and _same_path(args.csv, args.input):
            raise ValueError("csv output must not overwrite the input WAV")
        settings = {
            "instrument": args.instrument,
            "threshold": args.threshold,
            "spectral_threshold": args.spectral_threshold,
            "onset_threshold": args.onset_threshold,
            "min_separation_ms": args.min_separation_ms,
            "pre_roll_ms": args.pre_roll_ms,
            "window_ms": args.window_ms,
            "max_alignment_ms": args.max_alignment_ms,
            "bpm": args.bpm,
            "offset": args.offset,
            "subdivision": args.beat_division if args.beat_division is not None else args.subdivision,
            "beat_division": args.beat_division,
            "fade_in_ms": args.fade_in_ms,
            "fade_out_ms": args.fade_out_ms,
            "fast_compare": args.fast_compare,
            "bms_channel": args.bms_channel,
        }
        if args.preset_in:
            preset = load_preset(args.preset_in)
            defaults = {
                "instrument": "kick", "threshold": 0.95, "spectral_threshold": 0.94,
                "onset_threshold": 0.35, "min_separation_ms": 50.0, "pre_roll_ms": 5.0,
                "window_ms": 800.0, "max_alignment_ms": 20.0, "bpm": None,
                "offset": 0.0, "subdivision": 16, "beat_division": None, "fade_in_ms": 0.0,
                "fade_out_ms": 0.0, "fast_compare": False, "bms_channel": "01",
            }
            cli_values = dict(settings)
            settings = dict(preset)
            if "subdivision" not in settings and "beat_division" in settings:
                settings["subdivision"] = settings["beat_division"]
            settings.update({
                key: value
                for key, value in cli_values.items()
                if key in defaults and value != defaults[key]
            })
        if args.preset_out:
            save_preset(args.preset_out, settings)
        result = analyze_file(
            args.input,
            **settings,
        )
        output_started = time.perf_counter()
        data = result.to_dict()
        if args.export_dir:
            for cluster in result.plan.clusters:
                sample_path = args.export_dir / f"sample_{cluster.id:03d}.wav"
                if _same_path(sample_path, args.input):
                    raise ValueError("export-dir would overwrite the input WAV")
        write_json(output, data)
        if args.export_dir:
            audio = load_audio(args.input)
            exported_samples = write_hit_wavs(args.export_dir, audio, result.hits, result.plan)
        else:
            exported_samples = []
        if args.csv:
            write_hits_csv(args.csv, result.hits, result.plan.events)
        if args.bms:
            write_bms(
                args.bms,
                result.plan,
                bpm=settings.get("bpm"),
                offset=settings.get("offset", 0.0),
                subdivision=settings.get("subdivision", 16),
                channel=str(settings.get("bms_channel", args.bms_channel)),
                wav_prefix=relative_sample_prefix(args.bms, args.export_dir),
            )
        if args.bmson:
            write_bmson(args.bmson, result.plan, bpm=settings.get("bpm"), offset=settings.get("offset", 0.0))
        exported = {
            "json": str(output),
            "samples": [str(path) for path in exported_samples],
            "samples_dir": str(args.export_dir) if args.export_dir else None,
            "csv": str(args.csv) if args.csv else None,
            "bms": str(args.bms) if args.bms else None,
            "bmson": str(args.bmson) if args.bmson else None,
        }
        record_output_timing(result, time.perf_counter() - output_started)
        data = result.to_dict()
        data["exports"] = exported
        data["validation"] = validate_exports(result, exported)
        write_json(output, data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data if args.full_json else result.summary, ensure_ascii=False, indent=2))
    return 0
