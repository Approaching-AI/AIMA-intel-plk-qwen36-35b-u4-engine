#!/usr/bin/env python3
"""Deterministic confidence bounds for iq36 performance promotion gates.

Promotion decisions operate on paired blocks or raw component samples.  A
repeat/confirm median spread is reported separately as an environment
diagnostic and is never a performance veto.
"""

from __future__ import annotations

import math
import random
import statistics
from collections.abc import Sequence
from typing import Any


DEFAULT_CONFIDENCE = 0.95
DEFAULT_BOOTSTRAP_RESAMPLES = 20_000
DEFAULT_BOOTSTRAP_SEED = 0x5136


def _positive_finite(values: Sequence[float], label: str) -> list[float]:
  samples = [float(value) for value in values]
  if not samples:
    raise ValueError(f"{label} must not be empty")
  if any(not math.isfinite(value) or value <= 0.0 for value in samples):
    raise ValueError(f"{label} must contain only positive finite values")
  return samples


def _nearest_rank(sorted_values: Sequence[float], probability: float) -> float:
  if not 0.0 <= probability <= 1.0:
    raise ValueError("probability must be in [0, 1]")
  rank = max(1, math.ceil(probability * len(sorted_values)))
  return float(sorted_values[min(rank - 1, len(sorted_values) - 1)])


def bootstrap_median_bound(
    values: Sequence[float], *, side: str, confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED) -> float:
  """Return a deterministic one-sided percentile-bootstrap median bound."""
  samples = _positive_finite(values, "values")
  if side not in {"lower", "upper"}:
    raise ValueError("side must be 'lower' or 'upper'")
  if not 0.5 < confidence < 1.0:
    raise ValueError("confidence must be in (0.5, 1.0)")
  if resamples < 1_000:
    raise ValueError("resamples must be at least 1000")
  rng = random.Random(seed)
  medians = sorted(
      statistics.median(rng.choices(samples, k=len(samples)))
      for _ in range(resamples))
  probability = 1.0 - confidence if side == "lower" else confidence
  return _nearest_rank(medians, probability)


def dispersion_diagnostic(values: Sequence[float]) -> dict[str, Any]:
  """Report robust dispersion without turning it into a promotion gate."""
  samples = _positive_finite(values, "values")
  median = statistics.median(samples)
  mad = statistics.median(abs(value - median) for value in samples)
  robust_cv = 1.4826 * mad / median
  if robust_cv <= 0.01:
    classification = "normal"
    action = "none"
  elif robust_cv <= 0.02:
    classification = "warning"
    action = "collect_more_samples"
  else:
    classification = "investigate"
    action = "inspect_power_thermal_and_background_telemetry"
  return {
      "metric": "1.4826_mad_over_median",
      "robust_cv": robust_cv,
      "classification": classification,
      "action": action,
      "promotion_gate": False,
  }


def latency_cap_inference(
    values: Sequence[float], *, cap: float, min_samples: int = 20,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
  """Test whether a component median-latency upper bound clears its cap."""
  samples = _positive_finite(values, "latency samples")
  if not math.isfinite(cap) or cap <= 0.0:
    raise ValueError("cap must be positive and finite")
  upper = bootstrap_median_bound(
      samples, side="upper", confidence=confidence, resamples=resamples,
      seed=seed)
  enough_samples = len(samples) >= min_samples
  return {
      "method": "one_sided_percentile_bootstrap_median",
      "confidence": confidence,
      "bootstrap_resamples": resamples,
      "bootstrap_seed": seed,
      "sample_count": len(samples),
      "minimum_sample_count": min_samples,
      "point_estimate_ms": statistics.median(samples),
      "upper_confidence_bound_ms": upper,
      "cap_ms": cap,
      "sample_count_pass": enough_samples,
      "rate_pass": enough_samples and upper <= cap,
      "dispersion": dispersion_diagnostic(samples),
  }


def paired_speedup_inference(
    native_values: Sequence[float], reference_values: Sequence[float], *,
    target_ratio: float, min_blocks: int = 8,
    confidence: float = DEFAULT_CONFIDENCE,
    resamples: int = DEFAULT_BOOTSTRAP_RESAMPLES,
    seed: int = DEFAULT_BOOTSTRAP_SEED) -> dict[str, Any]:
  """Test paired native/reference throughput blocks against a speedup target."""
  native = _positive_finite(native_values, "native block values")
  reference = _positive_finite(reference_values, "reference block values")
  if len(native) != len(reference):
    raise ValueError("native and reference block counts must match")
  if not math.isfinite(target_ratio) or target_ratio <= 0.0:
    raise ValueError("target_ratio must be positive and finite")
  ratios = [candidate / baseline for candidate, baseline in zip(native, reference)]
  lower = bootstrap_median_bound(
      ratios, side="lower", confidence=confidence, resamples=resamples,
      seed=seed)
  enough_blocks = len(ratios) >= min_blocks
  return {
      "method": "paired_one_sided_percentile_bootstrap_median_ratio",
      "schedule": "interleaved_abba_blocks",
      "confidence": confidence,
      "bootstrap_resamples": resamples,
      "bootstrap_seed": seed,
      "paired_block_count": len(ratios),
      "minimum_paired_block_count": min_blocks,
      "point_estimate_ratio": statistics.median(ratios),
      "lower_confidence_bound_ratio": lower,
      "target_ratio": target_ratio,
      "sample_count_pass": enough_blocks,
      "rate_pass": enough_blocks and lower >= target_ratio,
      "dispersion": dispersion_diagnostic(ratios),
  }
