"""Small cross-correlation alignment used before comparison."""

from __future__ import annotations

from .._numeric import dot, length, norm


def align_pair(reference, candidate, max_shift_samples: int = 0):
    """Return overlapping arrays and the candidate shift with maximum correlation."""
    n = min(length(reference), length(candidate))
    reference = reference[:n]
    candidate = candidate[:n]
    max_shift_samples = max(0, min(max_shift_samples, max(0, n - 1)))
    best_shift = 0
    best_score = float("-inf")
    for shift in range(-max_shift_samples, max_shift_samples + 1):
        if shift >= 0:
            left, right = reference[: n - shift], candidate[shift:n]
        else:
            left, right = reference[-shift:n], candidate[: n + shift]
        denominator = norm(left) * norm(right)
        score = abs(dot(left, right) / denominator) if denominator else float("-inf")
        if score > best_score:
            best_score, best_shift = score, shift
    if best_shift >= 0:
        return reference[: n - best_shift], candidate[best_shift:n], best_shift
    return reference[-best_shift:n], candidate[: n + best_shift], best_shift
