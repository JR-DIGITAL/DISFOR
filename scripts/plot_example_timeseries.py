"""Plot NDMI time series of several samples with example image chips.

Each sample is one row: the NDMI time series on the left and one image chip
per label segment on the right. Segments are shown as shaded boxes, events
(see EVENT_LABELS) as dashed vertical bars.

Usage (from the repository root):
    python scripts/plot_example_timeseries.py [sample_id ...]
"""

import json
import sys
from pathlib import Path

import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import polars as pl
import tifffile
from plotly.subplots import make_subplots

import disfor
from utils import EVENT_LABELS

# Clear cut, thinning with canopy closure, bark beetle without salvage, gypsy moth,
# windthrow with salvage, wildfire
DEFAULT_SAMPLE_IDS = [2739, 3087, 862, 564, 3540, 172]
FIGURE_HOME = (
    Path.home()
    / "Documents"
    / "Projects"
    / "2026"
    / "DISFOR"
    / "DISFOR-Paper"
    / "figures"
)
DATA_DIR = Path(__file__).parent.parent / "data"

# Number of observations into a segment from which the example chip is taken
CHIP_OFFSET = 5
# Months from which example chips are preferably taken
LEAF_ON_MONTHS = range(5, 10)
# Row and column of the interpreted pixel within the chip
SAMPLE_PIXEL = 16
# Segment labels which end at their label's end date rather than lasting until
# the next label starts
END_AT_LABEL_END = {110}


def get_rgb_array(sample_id, timestamp, data_path):
    """RGB chip scaled to 0-1, or None if it contains invalid (0 or 65535) values"""
    raw = tifffile.imread(f"{data_path}/{sample_id}/{timestamp}.tif")[:, :, [2, 1, 0]]
    if np.isin(raw, [0, 65535]).any():
        return None
    gain = 5
    # Gamma < 1 brightens dark areas (forest) more than bright ones (clearings)
    gamma = 0.8
    return np.clip(raw / 10000 * gain, 0, 1) ** gamma


def load_sample(pixel_data, labels_df, sample_id):
    # Start date of the label each observation belongs to. Segments are split on
    # this rather than on label changes, so that consecutive labels of the same
    # class (e.g. sequential clear cuts) stay separate.
    label_starts = (
        labels_df.filter(sample_id=sample_id)
        .select(label_start=pl.col.start.dt.date(), label_end=pl.col.end.dt.date())
        .sort("label_start")
    )
    sample = (
        pixel_data.filter(pl.col.percent_clear_8x8 > 90, sample_id=sample_id)
        .sort("timestamps")
        .join_asof(label_starts, left_on="timestamps", right_on="label_start")
        .select(
            "timestamps",
            "percent_clear_32x32",
            # Observations before the first label belong to its segment
            pl.col.label_start.fill_null(strategy="backward"),
            pl.col.label_end.fill_null(strategy="backward"),
            label=pl.col.label.fill_null(110).cast(pl.Int32),
            NDMI=(pl.col.B08.cast(pl.Int32) - pl.col.B11) / (pl.col.B08 + pl.col.B11),
        )
        .sort("timestamps")
    )
    dates = sample["timestamps"].to_list()
    labels = sample["label"].to_numpy()
    ndmi = sample["NDMI"].to_numpy()
    clear = sample["percent_clear_32x32"].to_numpy()

    new_segment = (
        sample.select(
            pl.col.label_start.ne_missing(pl.col.label_start.shift())
            | pl.col.label.ne_missing(pl.col.label.shift())
        )
        .to_series()
        .to_numpy()
    )
    new_segment[0] = True
    segment_starts = np.where(new_segment)[0]
    segment_ends = np.append(segment_starts[1:] - 1, len(labels) - 1)

    # Cut off observations after the label's end date
    label_ends = sample["label_end"].to_list()
    for i, (start, end) in enumerate(zip(segment_starts, segment_ends)):
        if labels[start] in END_AT_LABEL_END and label_ends[start] is not None:
            while end > start and dates[end] > label_ends[start]:
                end -= 1
            segment_ends[i] = end
    return dates, labels, ndmi, clear, segment_starts, segment_ends


def chip_candidates(dates, clear, start, end):
    """Indices of the segment from start to end, ordered by chip preference.

    Prefers fully clear chips from leaf-on months, then fully clear chips from
    any month, then the clearest chips. Within each group, chips closer to
    CHIP_OFFSET observations into the segment come first.
    """
    idx = np.arange(start, end + 1)
    target = min(start + CHIP_OFFSET, end)
    fully_clear = clear[idx] == 100
    leaf_on = np.array([dates[i].month in LEAF_ON_MONTHS for i in idx])
    group = np.where(fully_clear & leaf_on, 0, np.where(fully_clear, 1, 2))
    order = np.lexsort((np.abs(idx - target), -clear[idx], group))
    return idx[order]


def first_valid_chip(sample_id, dates, candidates, tiff_path):
    """Index and image of the first candidate without invalid pixel values"""
    for idx in candidates:
        img = get_rgb_array(sample_id, f"{dates[idx]:%Y-%m-%d}", tiff_path)
        if img is not None:
            return idx, img
    return idx, None


def segment_chips(sample_id, dates, labels, clear, starts, ends, tiff_path):
    """One chip per label segment, as (label, date, image) tuples"""
    row_chips = []
    for start, end in zip(starts, ends):
        candidates = chip_candidates(dates, clear, start, end)
        idx, img = first_valid_chip(sample_id, dates, candidates, tiff_path)
        row_chips.append((labels[start], dates[idx], img))
    return row_chips


def main(sample_ids, pick_chips=segment_chips):
    """Build the figure. pick_chips returns the (label, date, image) chips of a row."""
    with open(DATA_DIR / "classes.json", "r") as fp:
        class_mapping = json.load(fp)
    class_names = {
        int(k): f"{v.split(' (')[0]} ({k})" for k, v in class_mapping.items()
    }

    pixel_data = pl.read_parquet(DATA_DIR / "pixel_data.parquet")
    labels_df = pl.read_parquet(DATA_DIR / "labels.parquet")
    samples = [
        load_sample(pixel_data, labels_df, sample_id) for sample_id in sample_ids
    ]
    # If the S2 sample data hasn't been downloaded before, this will take a while
    tiff_path = disfor.get("tiffs")

    # Consistent colors for each label across all rows
    all_labels = np.unique(np.concatenate([s[1] for s in samples]))
    palette = px.colors.qualitative.T10
    color_map = {lbl: palette[i % len(palette)] for i, lbl in enumerate(all_labels)}

    n_rows = len(samples)
    chips = [
        pick_chips(sample_id, dates, labels, clear, starts, ends, tiff_path)
        for sample_id, (dates, labels, _, clear, starts, ends) in zip(
            sample_ids, samples
        )
    ]
    n_chips = max(len(row_chips) for row_chips in chips)

    # Subplot titles in grid order, empty for the time series and missing chips
    titles = []
    for row_chips in chips:
        titles.append("")
        titles.extend(f"{lbl}<br>{date:%Y-%m-%d}" for lbl, date, _ in row_chips)
        titles.extend([""] * (n_chips - len(row_chips)))

    fig = make_subplots(
        rows=n_rows,
        cols=1 + n_chips,
        column_widths=[0.45] + [0.55 / n_chips] * n_chips,
        horizontal_spacing=0.01,
        vertical_spacing=0.06,
        subplot_titles=titles,
    )

    shown_labels = set()
    date_min = min(s[0][0] for s in samples)
    date_max = max(s[0][-1] for s in samples)

    for row, (sample_id, (dates, labels, ndmi, _, starts, ends)) in enumerate(
        zip(sample_ids, samples), start=1
    ):
        # --- Left: NDMI timeline, segments as shaded boxes, events as bars ---
        y_min, y_max = np.nanmin(ndmi), np.nanmax(ndmi)
        for start, end in zip(starts, ends):
            lbl = labels[start]
            if lbl in EVENT_LABELS:
                x = [dates[start]] * 2
                y = [y_min, y_max]
                style = dict(
                    mode="lines",
                    line=dict(color=color_map[lbl], width=2, dash="dash"),
                )
            else:
                x0, x1 = dates[start], dates[end]
                x = [x0, x1, x1, x0, x0]
                y = [y_min, y_min, y_max, y_max, y_min]
                style = dict(
                    mode="lines",
                    fill="toself",
                    fillcolor=color_map[lbl],
                    opacity=0.2,
                    line=dict(width=0),
                )
            fig.add_trace(
                go.Scatter(
                    x=x,
                    y=y,
                    name=class_names.get(lbl, str(lbl)),
                    legendgroup=str(lbl),
                    showlegend=lbl not in shown_labels,
                    hoverinfo="skip",
                    **style,
                ),
                row=row,
                col=1,
            )
            shown_labels.add(lbl)

        # NDMI points on top of the boxes and bars
        fig.add_trace(
            go.Scatter(
                x=dates,
                y=ndmi,
                mode="markers",
                name="NDMI",
                legendgroup="NDMI",
                showlegend=row == 1,
                marker=dict(color="grey", size=4),
            ),
            row=row,
            col=1,
        )
        fig.update_xaxes(
            range=[date_min, date_max], showticklabels=row == n_rows, row=row, col=1
        )
        fig.update_yaxes(title_text=f"Sample {sample_id}<br>NDMI", row=row, col=1)

        # --- Right: image chips ---
        for col, (_, _, img) in enumerate(chips[row - 1], start=2):
            if img is None:
                continue
            fig.add_trace(go.Image(z=(img * 255).astype(np.uint8)), row=row, col=col)
            # Mark the interpreted pixel
            fig.add_shape(
                type="rect",
                x0=SAMPLE_PIXEL - 0.5,
                x1=SAMPLE_PIXEL + 0.5,
                y0=SAMPLE_PIXEL - 0.5,
                y1=SAMPLE_PIXEL + 0.5,
                line=dict(color="yellow", width=1.5),
                row=row,
                col=col,
            )

        for col in range(2, 2 + n_chips):
            fig.update_xaxes(visible=False, row=row, col=col)
            fig.update_yaxes(visible=False, row=row, col=col)

    fig.update_layout(
        height=200 * n_rows + 80,
        width=1100,
        margin=dict(l=0, r=0, t=30, b=0),
        legend=dict(
            orientation="h",
            yanchor="top",
            y=-0.04,
            xanchor="center",
            x=0.5,
            title_text="Labels",
        ),
    )
    # Make subplot titles smaller
    fig.update_annotations(font_size=9)

    # Panel labels a), b), ... at the top left of each row
    for row in range(1, n_rows + 1):
        subplot = fig.get_subplot(row, 1)
        fig.add_annotation(
            text=f"<b>{chr(ord('a') + row - 1)})</b>",
            x=0,
            y=subplot.yaxis.domain[1],
            xref="paper",
            yref="paper",
            xanchor="left",
            yanchor="bottom",
            showarrow=False,
            font_size=14,
        )
    return fig


if __name__ == "__main__":
    sample_ids = [int(s) for s in sys.argv[1:]] or DEFAULT_SAMPLE_IDS
    fig = main(sample_ids)
    fig.write_image(FIGURE_HOME / "f03_multi.pdf", scale=3)
    fig.write_image(FIGURE_HOME / "f03_multi.png", scale=3)
