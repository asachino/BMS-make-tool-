"""Command line interface for the BMS stem reuse MVP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from inspect import signature
from pathlib import Path
from typing import Mapping

from .application import (
    analysis_result_from_dict,
    analyze_file,
    recluster_result,
    record_output_timing,
    relative_sample_prefix_for_export,
)
from .batch import run_batch
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.wav_exporter import write_hit_wavs
from .export.bms_exporter import write_bms
from .export.bmson_exporter import write_bmson
from .export.quality import validate_exports
from .project.presets import load_preset, save_preset
from .audio.loader import load_audio


_ANALYZE_SETTING_KEYS = frozenset(signature(analyze_file).parameters) - {"path"}
_SMART_END_ADVANCED_KEYS = frozenset({
    "enabled", "apply_to_explicit", "min_tail_ms", "max_tail_ms", "silence_ms",
    "silence_rms_db", "silence_peak_db", "frame_ms", "zero_crossing_ms",
    "safety_margin_ms", "next_attack_margin_ms", "attack_window_ms",
    "tail_min_ms", "tail_max_ms", "max_duration_ms",
})


def _preset_analysis_settings(values: Mapping[str, object]) -> dict:
    """Convert GUI-compatible preset data to safe ``analyze_file`` kwargs.

    Presets are also consumed by the GUI, so they may contain display-only
    state such as ``smart_end_advanced``.  Keep its actual endpoint values
    when it is a mapping, but never pass the GUI container or unknown metadata
    to the backend.
    """
    settings = dict(values) if isinstance(values, Mapping) else {}
    advanced = settings.pop("smart_end_advanced", None)
    advanced_values: dict = {}
    if isinstance(advanced, Mapping):
        nested = advanced.get("settings")
        if isinstance(nested, Mapping):
            advanced_values.update(nested)
        nested = advanced.get("smart_end_settings")
        if isinstance(nested, Mapping):
            advanced_values.update(nested)
        advanced_values.update({
            key: value
            for key, value in advanced.items()
            if key in _SMART_END_ADVANCED_KEYS
        })
    if advanced_values:
        smart_settings = dict(settings.get("smart_end_settings") or {})
        smart_settings.update({
            key: value
            for key, value in advanced_values.items()
            if key in _SMART_END_ADVANCED_KEYS
            and key not in {"enabled", "apply_to_explicit"}
        })
        settings["smart_end_settings"] = smart_settings
        if "enabled" in advanced_values and "smart_end" not in settings:
            settings["smart_end"] = advanced_values["enabled"]
        if "apply_to_explicit" in advanced_values and "smart_end_apply_to_explicit" not in settings:
            settings["smart_end_apply_to_explicit"] = advanced_values["apply_to_explicit"]
    return {
        key: value
        for key, value in settings.items()
        if key in _ANALYZE_SETTING_KEYS
    }


def _same_path(left: Path, right: Path) -> bool:
    return str(left.resolve()).casefold() == str(right.resolve()).casefold()


def _number_list(value):
    if not value:
        return None
    try:
        return [float(item.strip()) for item in str(value).split(",") if item.strip()]
    except ValueError as exc:
        raise ValueError("loop points/pattern must be comma-separated numbers") from exc


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
    analyze.add_argument("--smart-end", action=argparse.BooleanOptionalAction, default=True, help="decay-aware safe endpoint detection (default: on)")
    analyze.add_argument("--smart-end-apply-explicit", action="store_true", help="allow smart endpoint to shorten manual/pattern boundaries")
    analyze.add_argument("--smart-end-min-tail-ms", type=float)
    analyze.add_argument("--smart-end-max-tail-ms", type=float)
    analyze.add_argument("--smart-end-silence-ms", type=float)
    analyze.add_argument("--smart-end-safety-margin-ms", type=float)
    analyze.add_argument("--smart-end-zero-crossing-ms", type=float)
    analyze.add_argument("--smart-end-next-attack-margin-ms", type=float)
    analyze.add_argument("--loop-rule", choices=("off", "seconds", "beats", "bars", "points", "grid"), default="off")
    analyze.add_argument("--loop-seconds", type=float)
    analyze.add_argument("--loop-beats", type=float)
    analyze.add_argument("--loop-bars", type=float)
    analyze.add_argument("--loop-start-sec", type=float, default=0.0)
    analyze.add_argument("--loop-points", help="manual cut points in seconds, comma separated")
    analyze.add_argument("--loop-pattern", help="repeating cut intervals in seconds, comma separated")
    analyze.add_argument("--cut-plan", choices=("auto", "grid", "manual", "pattern"), default=None)
    analyze.add_argument("--no-automation-detection", action="store_true")
    analyze.add_argument("--automation-volume-threshold-db", type=float, default=3.0)
    analyze.add_argument("--automation-timbre-threshold", type=float, default=0.18)
    analyze.add_argument("--automation-pan-threshold-db", type=float, default=3.0)
    analyze.add_argument("--automation-chop-floor", type=float, default=0.08)
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
    batch.add_argument("--smart-end", action=argparse.BooleanOptionalAction, default=True)
    batch.add_argument("--smart-end-apply-explicit", action="store_true")
    batch.add_argument("--smart-end-min-tail-ms", type=float)
    batch.add_argument("--smart-end-max-tail-ms", type=float)
    batch.add_argument("--smart-end-silence-ms", type=float)
    batch.add_argument("--smart-end-safety-margin-ms", type=float)
    batch.add_argument("--smart-end-zero-crossing-ms", type=float)
    batch.add_argument("--smart-end-next-attack-margin-ms", type=float)
    batch.add_argument("--loop-rule", choices=("off", "seconds", "beats", "bars", "points", "grid"), default="off")
    batch.add_argument("--loop-seconds", type=float)
    batch.add_argument("--loop-beats", type=float)
    batch.add_argument("--loop-bars", type=float)
    batch.add_argument("--loop-start-sec", type=float, default=0.0)
    batch.add_argument("--loop-points")
    batch.add_argument("--loop-pattern")
    batch.add_argument("--cut-plan", choices=("auto", "grid", "manual", "pattern"), default=None)
    batch.add_argument("--no-automation-detection", action="store_true")
    batch.add_argument("--automation-volume-threshold-db", type=float, default=3.0)
    batch.add_argument("--automation-timbre-threshold", type=float, default=0.18)
    batch.add_argument("--automation-pan-threshold-db", type=float, default=3.0)
    batch.add_argument("--automation-chop-floor", type=float, default=0.08)
    batch.add_argument("--bms", action="store_true", help="write BMS per input (requires --bpm)")
    batch.add_argument("--bmson", action="store_true", help="write BMSON per input (requires --bpm)")
    recluster = commands.add_parser("recluster", help="reuse saved comparisons to change clustering without re-analysis")
    recluster.add_argument("input", type=Path, help="analysis JSON from analyze")
    recluster.add_argument("--output", type=Path, help="output JSON (default: <input>.recluster.bra.json)")
    recluster.add_argument("--reuse-level", default="balanced", help="strict, balanced, aggressive, or a numeric threshold")
    recluster.add_argument("--threshold", type=float, help="continuous waveform threshold")
    recluster.add_argument("--spectral-threshold", type=float)
    recluster.add_argument("--no-reexport", action="store_true", help="update JSON only; do not rewrite existing exports")
    recluster.add_argument("--full-json", action="store_true", help="print the complete JSON result")
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
                smart_end=args.smart_end,
                smart_end_apply_to_explicit=args.smart_end_apply_explicit,
                smart_end_settings={
                    key: value for key, value in {
                        "min_tail_ms": args.smart_end_min_tail_ms,
                        "max_tail_ms": args.smart_end_max_tail_ms,
                        "silence_ms": args.smart_end_silence_ms,
                        "safety_margin_ms": args.smart_end_safety_margin_ms,
                        "zero_crossing_ms": args.smart_end_zero_crossing_ms,
                        "next_attack_margin_ms": args.smart_end_next_attack_margin_ms,
                    }.items() if value is not None
                },
                loop_rule=args.loop_rule,
                loop_seconds=args.loop_seconds,
                loop_beats=args.loop_beats,
                loop_bars=args.loop_bars,
                loop_start_sec=args.loop_start_sec,
                loop_points=_number_list(args.loop_points),
                loop_pattern=_number_list(args.loop_pattern),
                cut_plan=args.cut_plan,
                automation_detection=not args.no_automation_detection,
                automation_volume_threshold_db=args.automation_volume_threshold_db,
                automation_timbre_threshold=args.automation_timbre_threshold,
                automation_pan_threshold_db=args.automation_pan_threshold_db,
                automation_chop_floor=args.automation_chop_floor,
                export_bms=args.bms,
                export_bmson=args.bmson,
            )
        except (OSError, ValueError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(manifest, ensure_ascii=False, indent=2))
        return 0
    if args.command == "recluster":
        try:
            output = args.output or args.input.with_suffix(".recluster.bra.json")
            if _same_path(output, args.input):
                raise ValueError("output must not overwrite the input JSON")
            data = json.loads(args.input.read_text(encoding="utf-8"))
            result = analysis_result_from_dict(data)
            if isinstance(result.settings.get("exports"), dict):
                result.settings["exports"]["json"] = str(output)
            # Create the destination before export validation checks its
            # presence; the final schema is written again below.
            write_json(output, result.to_dict())
            recluster_result(
                result,
                reuse_level=args.reuse_level,
                threshold=args.threshold,
                spectral_threshold=args.spectral_threshold,
                reexport=not args.no_reexport,
            )
            output_data = result.to_dict()
            write_json(output, output_data)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 1
        print(json.dumps(output_data if args.full_json else result.summary, ensure_ascii=False, indent=2))
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
            "smart_end": args.smart_end,
            "smart_end_apply_to_explicit": args.smart_end_apply_explicit,
            "smart_end_settings": {
                key: value for key, value in {
                    "min_tail_ms": args.smart_end_min_tail_ms,
                    "max_tail_ms": args.smart_end_max_tail_ms,
                    "silence_ms": args.smart_end_silence_ms,
                    "safety_margin_ms": args.smart_end_safety_margin_ms,
                    "zero_crossing_ms": args.smart_end_zero_crossing_ms,
                    "next_attack_margin_ms": args.smart_end_next_attack_margin_ms,
                }.items() if value is not None
            },
            "fast_compare": args.fast_compare,
            "bms_channel": args.bms_channel,
            "loop_rule": args.loop_rule,
            "loop_seconds": args.loop_seconds,
            "loop_beats": args.loop_beats,
            "loop_bars": args.loop_bars,
            "loop_start_sec": args.loop_start_sec,
            "loop_points": _number_list(args.loop_points),
            "loop_pattern": _number_list(args.loop_pattern),
            "cut_plan": args.cut_plan,
            "automation_detection": not args.no_automation_detection,
            "automation_volume_threshold_db": args.automation_volume_threshold_db,
            "automation_timbre_threshold": args.automation_timbre_threshold,
            "automation_pan_threshold_db": args.automation_pan_threshold_db,
            "automation_chop_floor": args.automation_chop_floor,
        }
        if args.preset_in:
            preset = _preset_analysis_settings(load_preset(args.preset_in))
            defaults = {
                "instrument": "kick", "threshold": 0.95, "spectral_threshold": 0.94,
                "onset_threshold": 0.35, "min_separation_ms": 50.0, "pre_roll_ms": 5.0,
                "window_ms": 800.0, "max_alignment_ms": 20.0, "bpm": None,
                "offset": 0.0, "subdivision": 16, "beat_division": None, "fade_in_ms": 0.0,
                "fade_out_ms": 0.0, "fast_compare": False, "bms_channel": "01",
                "smart_end": True, "smart_end_apply_to_explicit": False, "smart_end_settings": {},
                "loop_rule": "off", "loop_seconds": None, "loop_beats": None, "loop_bars": None,
                "loop_start_sec": 0.0, "loop_points": None, "loop_pattern": None, "cut_plan": None,
                "automation_detection": True, "automation_volume_threshold_db": 3.0,
                "automation_timbre_threshold": 0.18, "automation_pan_threshold_db": 3.0,
                "automation_chop_floor": 0.08,
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
        # Keep the backend boundary safe even when a future preset loader
        # preserves additional GUI metadata.
        settings = {
            key: value
            for key, value in settings.items()
            if key in _ANALYZE_SETTING_KEYS
        }
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
                wav_prefix=relative_sample_prefix_for_export(args.bms, args.export_dir),
            )
        if args.bmson:
            write_bmson(
                args.bmson,
                result.plan,
                bpm=settings.get("bpm"),
                offset=settings.get("offset", 0.0),
                wav_prefix=relative_sample_prefix_for_export(args.bmson, args.export_dir),
            )
        exported = {
            "json": str(output),
            "samples": [str(path) for path in exported_samples],
            "samples_dir": str(args.export_dir) if args.export_dir else None,
            "csv": str(args.csv) if args.csv else None,
            "bms": str(args.bms) if args.bms else None,
            "bmson": str(args.bmson) if args.bmson else None,
        }
        result.settings["exports"] = dict(exported)
        record_output_timing(result, time.perf_counter() - output_started)
        result.settings["validation"] = validate_exports(result, exported)
        data = result.to_dict()
        write_json(output, data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data if args.full_json else result.summary, ensure_ascii=False, indent=2))
    return 0
