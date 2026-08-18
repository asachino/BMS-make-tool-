from __future__ import annotations

import csv
from pathlib import Path


def write_hits_csv(path: str | Path, hits, events: list[dict], *, excluded_hits: set[int] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    event_by_hit = {event["hit"]: event for event in events}
    with path.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(["hit", "time", "sample_id", "gain_db", "overlap_warning"])
        for hit in hits:
            if excluded_hits and hit.id in excluded_hits:
                continue
            event = event_by_hit[hit.id]
            writer.writerow([hit.id, hit.time, event["sample_id"], event["gain_db"], hit.overlap_warning])
    return path
