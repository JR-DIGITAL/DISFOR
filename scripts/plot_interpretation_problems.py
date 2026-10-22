"""Plot NDMI time series of samples illustrating common interpretation problems.

Same layout as plot_example_timeseries.py, but instead of one chip per label
segment, N_CHIPS random chips spread over the time series are shown. The time
series is split into N_CHIPS equally long parts (by observation count) and one
chip is drawn from each part, preferring fully clear leaf-on chips.

Usage (from the repository root):
    python scripts/plot_interpretation_problems.py [sample_id ...]
"""

import sys

import numpy as np

from plot_example_timeseries import (
    FIGURE_HOME,
    LEAF_ON_MONTHS,
    first_valid_chip,
    main,
)

DEFAULT_SAMPLE_IDS = [3345, 1182, 2834]
N_CHIPS = 4
SEED = 42


def random_chips(sample_id, dates, labels, clear, starts, ends, tiff_path):
    """N_CHIPS random chips, one from each of N_CHIPS parts of the time series"""
    # Seeded per sample, so a row doesn't change when other samples are added
    rng = np.random.default_rng([SEED, sample_id])
    row_chips = []
    for part in np.array_split(np.arange(len(dates)), N_CHIPS):
        fully_clear = clear[part] == 100
        leaf_on = np.array([dates[i].month in LEAF_ON_MONTHS for i in part])
        group = np.where(fully_clear & leaf_on, 0, np.where(fully_clear, 1, 2))
        # Random order within the fully clear groups, clearest first otherwise
        order = np.lexsort((rng.random(len(part)), -clear[part], group))
        idx, img = first_valid_chip(sample_id, dates, part[order], tiff_path)
        row_chips.append((labels[idx], dates[idx], img))
    return row_chips


if __name__ == "__main__":
    sample_ids = [int(s) for s in sys.argv[1:]] or DEFAULT_SAMPLE_IDS
    fig = main(sample_ids, pick_chips=random_chips)
    fig.write_image(FIGURE_HOME / "interpretation_problems.pdf", scale=3)
    fig.write_image(FIGURE_HOME / "interpretation_problems.png", scale=3)
