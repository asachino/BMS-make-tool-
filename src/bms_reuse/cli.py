"""Command line interface for the BMS stem reuse MVP."""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

from .application import analyze_file, record_output_timing
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.wav_exporter import write_hit_wavs
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
    analyze.add_argument("--threshold", type=float, default=0.995, help="gain-normalized similarity threshold")
    analyze.add_argument("--spectral-threshold", type=float, default=0.92)
    analyze.add_argument("--onset-threshold", type=float, default=0.35)
    analyze.add_argument("--min-separation-ms", type=float, default=50.0)
    analyze.add_argument("--pre-roll-ms", type=float, default=5.0)
    analyze.add_argument("--window-ms", type=float, default=800.0)
    analyze.add_argument("--max-alignment-ms", type=float, default=5.0)
    analyze.add_argument("--bpm", type=float)
    analyze.add_argument("--offset", type=float, default=0.0)
    analyze.add_argument("--subdivision", type=int, default=16)
    analyze.add_argument("--fast-compare", action="store_true", help="rank representatives by shape features (ordering may change)")
    analyze.add_argument("--full-json", action="store_true", help="print the complete JSON result")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command != "analyze":
        return 2
    try:
        output = args.output or args.input.with_suffix(".bra.json")
        if _same_path(output, args.input):
            raise ValueError("output must not overwrite the input WAV")
        if args.csv and _same_path(args.csv, args.input):
            raise ValueError("csv output must not overwrite the input WAV")
        result = analyze_file(
            args.input,
            instrument=args.instrument,
            threshold=args.threshold,
            spectral_threshold=args.spectral_threshold,
            onset_threshold=args.onset_threshold,
            min_separation_ms=args.min_separation_ms,
            pre_roll_ms=args.pre_roll_ms,
            window_ms=args.window_ms,
            max_alignment_ms=args.max_alignment_ms,
            bpm=args.bpm,
            offset=args.offset,
            subdivision=args.subdivision,
            fast_compare=args.fast_compare,
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
            write_hit_wavs(args.export_dir, audio, result.hits, result.plan)
        if args.csv:
            write_hits_csv(args.csv, result.hits, result.plan.events)
        record_output_timing(result, time.perf_counter() - output_started)
        data = result.to_dict()
        write_json(output, data)
    except (OSError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 1
    print(json.dumps(data if args.full_json else result.summary, ensure_ascii=False, indent=2))
    return 0
