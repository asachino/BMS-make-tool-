from __future__ import annotations

from dataclasses import dataclass, field
@dataclass
class Project:
    source: str
    settings: dict = field(default_factory=dict)
    hits: list[dict] = field(default_factory=list)
    comparisons: list[dict] = field(default_factory=list)
    clusters: list[dict] = field(default_factory=list)
    reuse_plan: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "source": self.source,
            "settings": self.settings,
            "hits": self.hits,
            "comparisons": self.comparisons,
            "clusters": self.clusters,
            "reuse_plan": self.reuse_plan,
        }
