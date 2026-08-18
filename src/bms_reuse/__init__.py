"""BMS stem reuse analysis without a mandatory third-party DSP stack."""

from .application import (
    AnalysisResult,
    analysis_result_from_dict,
    analyze_file,
    recluster_result,
    refresh_review_plan,
    relative_sample_prefix_for_export,
    set_review_state,
)
from .detection.loop_rules import build_cut_onsets
from .features.automation import detect_automation
from .features.percussion import normalize_instrument

__all__ = [
    "AnalysisResult", "analysis_result_from_dict", "analyze_file", "recluster_result",
    "refresh_review_plan", "relative_sample_prefix_for_export", "set_review_state",
    "build_cut_onsets", "detect_automation", "normalize_instrument",
]
__version__ = "0.1.0"
