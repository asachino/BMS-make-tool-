"""Small cross-correlation alignment used before comparison."""

from __future__ import annotations

from .._numeric import dot, length, norm, np


def align_pair(reference, candidate, max_shift_samples: int = 0):
    """Return overlapping arrays and the candidate shift with maximum correlation."""
    n = min(length(reference), length(candidate))
    reference = reference[:n]
    candidate = candidate[:n]
    max_shift_samples = max(0, min(max_shift_samples, max(0, n - 1)))
    if np is not None and hasattr(reference, "shape") and hasattr(candidate, "shape"):
        if n == 0:
            return reference, candidate, 0
        shifts = np.arange(-max_shift_samples, max_shift_samples + 1)
        fft_size = 1 << (2 * n - 1).bit_length()
        correlation_full = np.fft.irfft(
            np.fft.rfft(candidate, fft_size) * np.fft.rfft(reference[::-1], fft_size),
            fft_size,
        )[: 2 * n - 1]
        correlation = correlation_full[n - 1 + shifts]
        reference_energy = np.concatenate(([0.0], np.cumsum(reference * reference)))
        candidate_energy = np.concatenate(([0.0], np.cumsum(candidate * candidate)))
        nonnegative = shifts >= 0
        left_energy = np.empty(shifts.shape, dtype=float)
        right_energy = np.empty(shifts.shape, dtype=float)
        left_energy[nonnegative] = reference_energy[n - shifts[nonnegative]]
        right_energy[nonnegative] = candidate_energy[n] - candidate_energy[shifts[nonnegative]]
        negative = ~nonnegative
        left_energy[negative] = reference_energy[n] - reference_energy[-shifts[negative]]
        right_energy[negative] = candidate_energy[n + shifts[negative]]
        denominator = np.sqrt(left_energy) * np.sqrt(right_energy)
        scores = np.full(shifts.shape, -np.inf, dtype=float)
        valid = denominator != 0
        scores[valid] = np.abs(correlation[valid] / denominator[valid])
        best_score = float(np.max(scores))
        if not np.isfinite(best_score):
            # The scalar implementation keeps shift zero when every overlap
            # has zero energy (all scores remain -inf).
            best_shift = 0
        else:
            # FFT round-off can make mathematically tied edge scores differ by
            # a few ulps; retain the original scan's first-win behavior.
            tied = np.flatnonzero(scores >= best_score - 1e-12)
            best_shift = int(shifts[int(tied[0] if len(tied) else np.argmax(scores))])
        if best_shift >= 0:
            return reference[: n - best_shift], candidate[best_shift:n], best_shift
        return reference[-best_shift:n], candidate[: n + best_shift], best_shift
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
