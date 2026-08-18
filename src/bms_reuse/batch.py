"""Folder-level analysis helper."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable

from .application import analyze_file
from .export.bms_exporter import relative_sample_prefix, write_bms
from .export.bmson_exporter import write_bmson
from .export.csv_exporter import write_hits_csv
from .export.json_exporter import write_json
from .export.quality import validate_exports
from .export.wav_exporter import write_hit_wavs
from .audio.loader import load_audio


def analyze_folder(
    folder: str | Path,
    *,
    recursive: bool = False,
    progress: Callable[[int, str], None] | None = None,
    **settings,
) -> list:
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"folder does not exist: {folder}")
    paths = sorted(folder.rglob("*.wav") if recursive else folder.glob("*.wav"), key=lambda p: (str(p).casefold(), str(p)))
    results = []
    for index, path in enumerate(paths):
        if progress:
            progress(round(index / max(1, len(paths)) * 100), f"{path.name} を解析中")
        results.append(analyze_file(path, **settings))
    if progress:
        progress(100, f"{len(results)}ファイル完了")
    return results


def run_batch(
    folder: str | Path,
    *,
    output_dir: str | Path | None = None,
    recursive: bool = False,
    export_samples: bool = True,
    export_bms: bool = False,
    export_bmson: bool = False,
    progress: Callable[[int, str], None] | None = None,
    **settings,
) -> dict:
    """Analyze every WAV, continue after failures, and write a manifest."""
    folder = Path(folder)
    if not folder.is_dir():
        raise ValueError(f"folder does not exist: {folder}")
    paths = sorted(folder.rglob("*.wav") if recursive else folder.glob("*.wav"), key=lambda p: (str(p).casefold(), str(p)))
    root = Path(output_dir) if output_dir else folder / "bms-reuse-batch"
    root.mkdir(parents=True, exist_ok=True)
    items: list[dict] = []
    for index, path in enumerate(paths):
        item = {"input": str(path), "ok": False}
        try:
            result = analyze_file(path, **settings)
            relative = path.relative_to(folder).with_suffix("")
            target = root.joinpath(*relative.parts)
            target.mkdir(parents=True, exist_ok=True)
            exported: dict[str, object] = {}
            json_path = target / f"{path.stem}.bra.json"
            exported["json"] = str(json_path)
            exported["csv"] = str(write_hits_csv(target / f"{path.stem}.csv", result.hits, result.plan.events))
            if export_samples:
                audio = load_audio(path)
                exported["samples"] = [str(p) for p in write_hit_wavs(target / "keysounds", audio, result.hits, result.plan, fade_in_ms=float(settings.get("fade_in_ms", 0.0)), fade_out_ms=float(settings.get("fade_out_ms", 0.0)))]
                exported["samples_dir"] = str(target / "keysounds")
            if export_bms:
                bms_path = target / f"{path.stem}.bms"
                exported["bms"] = str(write_bms(
                    bms_path,
                    result.plan,
                    bpm=settings.get("bpm"),
                    offset=float(settings.get("offset", 0.0)),
                    subdivision=int(settings.get("subdivision", 16)),
                    channel=str(settings.get("bms_channel", "01")),
                    wav_prefix=relative_sample_prefix(bms_path, target / "keysounds"),
                ))
            if export_bmson:
                bmson_path = target / "keysounds" / f"{path.stem}.bmson"
                exported["bmson"] = str(write_bmson(bmson_path, result.plan, bpm=settings.get("bpm"), offset=float(settings.get("offset", 0.0))))
            validation = validate_exports(result, exported)
            result.settings["validation"] = validation
            result.settings["exports"] = dict(exported)
            write_json(json_path, result.to_dict())
            item.update({"ok": bool(validation.get("ok", False)), "outputs": exported, "summary": result.summary, "validation": validation})
        except Exception as exc:  # batch intentionally continues after one bad input
            item["error"] = str(exc)
        items.append(item)
        if progress:
            progress(round((index + 1) / max(1, len(paths)) * 100), f"{path.name} 完了" if item["ok"] else f"{path.name} 失敗")
    manifest = {"version": 1, "folder": str(folder), "count": len(paths), "success": sum(bool(item["ok"]) for item in items), "items": items}
    (root / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest
