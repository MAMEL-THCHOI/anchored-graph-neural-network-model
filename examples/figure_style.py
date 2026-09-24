#!/usr/bin/env python3
"""Reusable Matplotlib style and component exporters for Figure_CTH26Aug."""

from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
from typing import Iterable

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.transforms import Bbox
from PIL import Image

INK = "#222222"
RED = "#B7221A"
DARK_RED = "#7F1712"
GRAY = "#6E6E6E"
MID_GRAY = "#A0A0A0"
LIGHT_GRAY = "#D7D7D7"
WHITE = "#FFFFFF"
PUBLICATION_DPI = 600
COMMON_BOX_SIDE_IN = 3.80


def configure_style() -> None:
    mpl.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "mathtext.fontset": "stix",
            "axes.labelsize": 11.5,
            "xtick.labelsize": 10.5,
            "ytick.labelsize": 10.5,
            "legend.fontsize": 10.0,
            "axes.linewidth": 0.9,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "xtick.major.width": 0.8,
            "ytick.major.width": 0.8,
            "xtick.major.size": 4.0,
            "ytick.major.size": 4.0,
            "axes.grid": False,
            "figure.facecolor": WHITE,
            "axes.facecolor": WHITE,
            "savefig.facecolor": WHITE,
            "savefig.dpi": PUBLICATION_DPI,
            "svg.fonttype": "none",
        }
    )


def style_axes(ax: plt.Axes, *, square: bool = False) -> None:
    ax.grid(False)
    ax.set_frame_on(True)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color(INK)
        spine.set_linewidth(0.9)
    ax.tick_params(direction="out", colors=INK, top=True, right=True)
    if square:
        ax.set_box_aspect(1)


def union_artist_bbox(fig: plt.Figure, artists: Iterable, pad_in: float = 0.06) -> Bbox:
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    boxes = []
    for artist in artists:
        box = artist.get_tightbbox(renderer) if hasattr(artist, "get_tightbbox") else None
        if box is None and hasattr(artist, "get_window_extent"):
            box = artist.get_window_extent(renderer)
        if box is not None:
            boxes.append(box)
    if not boxes:
        raise ValueError("No visible artist bounding boxes were found")
    box_px = Bbox.union(boxes)
    box_in = box_px.transformed(fig.dpi_scale_trans.inverted())
    return Bbox.from_extents(
        box_in.x0 - pad_in,
        box_in.y0 - pad_in,
        box_in.x1 + pad_in,
        box_in.y1 + pad_in,
    )


@contextmanager
def hidden_artists(artists: Iterable):
    states = [(artist, artist.get_visible()) for artist in artists]
    for artist, _ in states:
        artist.set_visible(False)
    try:
        yield
    finally:
        for artist, state in states:
            artist.set_visible(state)


def export_component(
    fig: plt.Figure,
    artists: Iterable,
    output_base: Path,
    *,
    hide: Iterable = (),
    pad_in: float = 0.06,
) -> tuple[Path, Path]:
    """Export a true-vector SVG and 600-dpi PNG around selected artists."""

    output_base = Path(output_base)
    output_base.parent.mkdir(parents=True, exist_ok=True)
    with hidden_artists(hide):
        bbox = union_artist_bbox(fig, artists, pad_in=pad_in)
        png = output_base.with_suffix(".png")
        svg = output_base.with_suffix(".svg")
        fig.savefig(png, dpi=PUBLICATION_DPI, bbox_inches=bbox, facecolor=WHITE)
        fig.savefig(svg, bbox_inches=bbox, facecolor=WHITE)
    with Image.open(png) as image:
        image.load()
        image.save(png, format="PNG", dpi=(PUBLICATION_DPI, PUBLICATION_DPI))
    return png, svg


def panel_letter_artists(fig: plt.Figure) -> list:
    labels = {f"({letter})" for letter in "abcdefghijklmnopqrstuvwxyz"}
    return [text for text in fig.findobj(mpl.text.Text) if text.get_text().strip() in labels]

