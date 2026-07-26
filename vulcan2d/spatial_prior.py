"""CAFM-derived geometric priors for the VULCAN-2D coarse patch model.

The supplied two-column file contains projected conductive-spot area bins and
their counts.  It constrains lateral heterogeneity, but it does not identify a
filament diameter, defect chemistry, current density, or switching threshold.
Consequently this module only maps the measured *area distribution* to
normalized coarse weights.  Electrical disorder remains a separate model term.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import numpy as np


_ROW = re.compile(r"^\s*([0-9.eE+-]+)\s+(\d+)\s*$")


@dataclass(frozen=True)
class SpotAreaDistribution:
    """Binned projected conductive-spot areas and integer object counts."""

    area_m2: np.ndarray
    counts: np.ndarray

    @classmethod
    def from_text(cls, path):
        rows = []
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            match = _ROW.match(line)
            if match:
                rows.append((float(match.group(1)), int(match.group(2))))
        if not rows:
            raise ValueError(f"No numeric area/count rows found in {path}")
        area = np.asarray([row[0] for row in rows], dtype=float)
        counts = np.asarray([row[1] for row in rows], dtype=int)
        if np.any(area <= 0.0) or np.any(counts < 0) or np.sum(counts) <= 0:
            raise ValueError("Area bins must be positive and counts non-negative")
        order = np.argsort(area)
        return cls(area[order], counts[order])

    @property
    def spot_count(self):
        return int(np.sum(self.counts))

    @property
    def diameter_nm(self):
        return 2.0 * np.sqrt(self.area_m2 / np.pi) * 1e9

    def weighted_quantile(self, values, q):
        if not 0.0 <= q <= 1.0:
            raise ValueError("q must lie in [0, 1]")
        values = np.asarray(values, dtype=float)
        order = np.argsort(values)
        cumulative = np.cumsum(self.counts[order])
        index = int(np.searchsorted(cumulative, q * cumulative[-1], side="left"))
        return float(values[order][min(index, len(values) - 1)])

    def coarse_area_weights(self, k):
        """Return K size-stratified geometric weights with equal spot counts.

        The spots are sorted by projected area, split into K nearly equal-count
        strata, and each stratum receives its share of total projected area.
        This preserves the measured size heterogeneity without claiming that K
        equals the number of CAFM spots or that size fixes conductivity.
        """

        k = int(k)
        if k < 1 or k > self.spot_count:
            raise ValueError("k must be between 1 and the number of spots")
        observations = np.repeat(self.area_m2, self.counts)
        groups = np.array_split(np.sort(observations), k)
        weights = np.asarray([np.sum(group) for group in groups], dtype=float)
        return weights / np.sum(weights)

    def summary(self, k=10):
        total = self.spot_count
        mean_area = float(np.sum(self.area_m2 * self.counts) / total)
        mean_diameter = float(np.sum(self.diameter_nm * self.counts) / total)
        variance = float(
            np.sum(self.counts * (self.area_m2 - mean_area) ** 2) / total
        )
        weights = self.coarse_area_weights(k)
        return {
            "spot_count": total,
            "bins": int(len(self.area_m2)),
            "nonzero_bins": int(np.count_nonzero(self.counts)),
            "area_nm2": {
                "mean": mean_area * 1e18,
                "median": self.weighted_quantile(self.area_m2, 0.5) * 1e18,
                "q90": self.weighted_quantile(self.area_m2, 0.9) * 1e18,
                "q95": self.weighted_quantile(self.area_m2, 0.95) * 1e18,
                "coefficient_of_variation": float(np.sqrt(variance) / mean_area),
            },
            "equivalent_diameter_nm": {
                "mean_of_spot_diameters": mean_diameter,
                "diameter_of_mean_area": 2.0 * np.sqrt(mean_area / np.pi) * 1e9,
                "median": self.weighted_quantile(self.diameter_nm, 0.5),
                "q90": self.weighted_quantile(self.diameter_nm, 0.9),
                "q95": self.weighted_quantile(self.diameter_nm, 0.95),
            },
            "coarse_prior": {
                "k": k,
                "area_weights": [float(value) for value in weights],
                "effective_patch_count": float(1.0 / np.sum(weights ** 2)),
                "construction": "equal-count size strata weighted by projected area",
            },
            "claim_scope": (
                "geometric heterogeneity prior only; not a filament diameter, "
                "defect-species label, current density, or switching threshold"
            ),
        }
