from __future__ import annotations

import csv
import json
from pathlib import Path


def write_hits_csv(path: str | Path, hits, events: list[dict], *, excluded_hits: set[int] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event_by_hit = {event["hit"]: event for event in events}
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        # Keep the original five columns first for spreadsheet compatibility;
        # endpoint provenance is appended so old consumers can ignore it.
        writer.writerow([
            "hit", "time", "sample_id", "gain_db", "overlap_warning",
            "source_start", "source_end", "end_reason", "end_confidence",
            "end_warnings", "warnings", "effective_settings",
        ])
        for hit in hits:
            if excluded_hits and hit.id in excluded_hits:
                continue
            event = event_by_hit[hit.id]
            writer.writerow([
                hit.id,
                hit.time,
                event["sample_id"],
                event["gain_db"],
                hit.overlap_warning,
                hit.source_start,
                hit.source_end,
                getattr(hit, "end_reason", "window"),
                getattr(hit, "end_confidence", 0.0),
                json.dumps(getattr(hit, "end_warnings", []) or [], ensure_ascii=False, separators=(",", ":")),
                json.dumps(getattr(hit, "end_warnings", []) or [], ensure_ascii=False, separators=(",", ":")),
                json.dumps(getattr(hit, "effective_settings", {}) or {}, ensure_ascii=False, sort_keys=True, separators=(",", ":")),
            ])
    return path
