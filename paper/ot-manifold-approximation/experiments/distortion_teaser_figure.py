#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.5.1",
# ]
# ///
"""Introductory figure: the flat latent map is distorted, the surface is not.

The figure is analytic and depends on no trained model.  It reuses the
mixed-curvature decoder of Section~\\ref{sec:controlled-experiment}, whose
diagonal-Gaussian W2 equals the chord geometry of the graph (u, v, h(u,v)).
Two paths with *identical* planar length are selected programmatically as the
extreme cases of intrinsic length, so nothing is hand-picked:

    left  panel  the map you draw   -- flat, the two paths look the same
    right panel  what the decoder sees -- the same two paths, visibly unequal

Color encodes the local magnification sqrt(det G) = sqrt(1 + |grad h|^2), the
factor by which the pullback metric stretches a latent displacement.  It is a
magnitude, so the ramp is single-hue and light-to-dark.

Run with:
    uv run --python 3.12 distortion_teaser_figure.py
"""

from __future__ import annotations

import pathlib

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.colors import LinearSegmentedColormap

GRID_SIDE = 21
RUN_STEPS = 4

# Single-hue sequential ramp (light -> dark), for a magnitude with no polarity.
MAGNIFICATION_CMAP = LinearSegmentedColormap.from_list(
    "pullback_magnification",
    ["#eef3f8", "#c6d9ea", "#8fb4d4", "#4f83b3", "#245a8a", "#0f3557"],
)
PATH_STRETCHED = "#b3341f"
PATH_PRESERVED = "#1f6f4a"
INK_PRIMARY = "#1a1a1a"
INK_MUTED = "#6b6b6b"

TEXT = {
    "en": {
        "left": "The map you draw",
        "right": "The geometry the decoder sees",
        "left_note": "equal length on the plot",
        "cbar": r"local magnification  $\sqrt{\det G}$",
        "stretched": "stretched",
        "preserved": "nearly preserved",
        "caption_a": "A",
        "caption_b": "B",
    },
    "ja": {
        "left": "描いている地図",
        "right": "デコーダーが見ている幾何",
        "left_note": "プロット上では同じ長さ",
        "cbar": r"局所的な拡大率  $\sqrt{\det G}$",
        "stretched": "引き伸ばされる",
        "preserved": "ほぼ保たれる",
        "caption_a": "A",
        "caption_b": "B",
    },
}


def japanese_font_family() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Hiragino Sans",
        "Yu Gothic",
        "Noto Sans CJK JP",
        "IPAexGothic",
        "IPAGothic",
    ):
        if candidate in installed:
            return candidate
    return "sans-serif"


def height(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """The mixed-curvature height field of the controlled decoder."""

    first = 0.48 * np.exp(-6.4 * ((u + 0.42) ** 2 + 1.15 * (v - 0.24) ** 2))
    second = 0.38 * np.exp(-7.2 * (1.2 * (u - 0.38) ** 2 + (v + 0.30) ** 2))
    return first + second + 0.30 * u * v


def height_gradient(u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    first = 0.48 * np.exp(-6.4 * ((u + 0.42) ** 2 + 1.15 * (v - 0.24) ** 2))
    second = 0.38 * np.exp(-7.2 * (1.2 * (u - 0.38) ** 2 + (v + 0.30) ** 2))
    du = first * (-12.8 * (u + 0.42)) + second * (-17.28 * (u - 0.38)) + 0.30 * v
    dv = first * (-14.72 * (v - 0.24)) + second * (-14.4 * (v + 0.30)) + 0.30 * u
    return du, dv


def magnification(u: np.ndarray, v: np.ndarray) -> np.ndarray:
    """sqrt(det G) for the graph metric G = I + grad h grad h^T."""

    du, dv = height_gradient(u, v)
    return np.sqrt(1.0 + du**2 + dv**2)


def extreme_horizontal_runs() -> tuple[dict, dict]:
    """Two horizontal runs of identical planar length, extreme in true length.

    Selected over every admissible run so that neither path is hand-picked.
    """

    axis = np.linspace(-1.0, 1.0, GRID_SIDE)
    records = []
    for row in range(GRID_SIDE):
        for col in range(GRID_SIDE - RUN_STEPS):
            us = axis[col : col + RUN_STEPS + 1]
            vs = np.full_like(us, axis[row])
            hs = height(us, vs)
            planar = float(us[-1] - us[0])
            intrinsic = float(np.sum(np.hypot(np.diff(us), np.diff(hs))))
            records.append(
                {
                    "u": us,
                    "v": vs,
                    "h": hs,
                    "planar_length": planar,
                    "intrinsic_length": intrinsic,
                }
            )
    records.sort(key=lambda item: item["intrinsic_length"])
    return records[-1], records[0]


def draw(language: str, output_dir: pathlib.Path) -> None:
    text = TEXT[language]
    if language == "ja":
        plt.rcParams["font.family"] = japanese_font_family()
    else:
        plt.rcParams["font.family"] = "sans-serif"
    plt.rcParams["axes.unicode_minus"] = False

    axis = np.linspace(-1.0, 1.0, 241)
    uu, vv = np.meshgrid(axis, axis, indexing="xy")
    factor = magnification(uu, vv)
    stretched, preserved = extreme_horizontal_runs()

    figure = plt.figure(figsize=(10.6, 4.2))
    flat_axis = figure.add_axes([0.045, 0.215, 0.335, 0.665])
    surface_axis = figure.add_axes([0.415, 0.10, 0.575, 0.85], projection="3d")

    mesh = flat_axis.pcolormesh(
        uu,
        vv,
        factor,
        cmap=MAGNIFICATION_CMAP,
        shading="auto",
        rasterized=True,
    )
    for value in np.linspace(-1.0, 1.0, GRID_SIDE)[::2]:
        flat_axis.axhline(value, color="white", linewidth=0.3, alpha=0.35)
        flat_axis.axvline(value, color="white", linewidth=0.3, alpha=0.35)

    for run, color, label in (
        (stretched, PATH_STRETCHED, text["caption_a"]),
        (preserved, PATH_PRESERVED, text["caption_b"]),
    ):
        flat_axis.plot(
            run["u"], run["v"], color=color, linewidth=3.2, solid_capstyle="round", zorder=5
        )
        flat_axis.plot(
            run["u"][[0, -1]],
            run["v"][[0, -1]],
            "o",
            color=color,
            markersize=5.0,
            markeredgecolor="white",
            markeredgewidth=1.1,
            zorder=6,
        )
        flat_axis.annotate(
            label,
            xy=(float(run["u"].mean()), float(run["v"][0])),
            xytext=(0, 10),
            textcoords="offset points",
            ha="center",
            color=color,
            fontsize=12,
            fontweight="bold",
            zorder=7,
        )

    figure.text(
        0.2125, 0.935, text["left"], ha="center", va="center",
        fontsize=12.5, color=INK_PRIMARY,
    )
    flat_axis.set_aspect("equal")
    flat_axis.set_xticks([])
    flat_axis.set_yticks([])
    for spine in flat_axis.spines.values():
        spine.set_edgecolor("#cccccc")
    flat_axis.text(
        0.5,
        0.975,
        f"{text['left_note']}:  A = B = {stretched['planar_length']:.2f}",
        transform=flat_axis.transAxes,
        ha="center",
        va="top",
        fontsize=10,
        color=INK_PRIMARY,
        bbox={"facecolor": "white", "alpha": 0.82, "edgecolor": "none", "pad": 2.2},
    )

    colorbar_axis = figure.add_axes([0.085, 0.125, 0.255, 0.030])
    colorbar = figure.colorbar(mesh, cax=colorbar_axis, orientation="horizontal")
    colorbar.set_label(text["cbar"], fontsize=9.5, color=INK_MUTED, labelpad=2)
    colorbar.ax.tick_params(labelsize=8, colors=INK_MUTED, length=2, pad=1)
    colorbar.outline.set_edgecolor("#cccccc")

    surf_axis_grid = np.linspace(-1.0, 1.0, 121)
    su, sv = np.meshgrid(surf_axis_grid, surf_axis_grid, indexing="xy")
    sh = height(su, sv)
    normalized = (magnification(su, sv) - factor.min()) / (factor.max() - factor.min())
    surface_axis.plot_surface(
        su,
        sv,
        sh,
        facecolors=MAGNIFICATION_CMAP(normalized),
        rstride=1,
        cstride=1,
        linewidth=0,
        antialiased=True,
        shade=False,
    )
    surface_axis.plot_wireframe(
        su,
        sv,
        sh,
        rstride=10,
        cstride=10,
        color="#5b7fa0",
        linewidth=0.35,
        alpha=0.5,
    )
    for run, color, label, offset in (
        (stretched, PATH_STRETCHED, text["caption_a"], 0.20),
        (preserved, PATH_PRESERVED, text["caption_b"], 0.13),
    ):
        surface_axis.plot(
            run["u"],
            run["v"],
            run["h"] + 0.012,
            color=color,
            linewidth=3.6,
            solid_capstyle="round",
            zorder=10,
        )
        surface_axis.text(
            float(run["u"][0]) - 0.10,
            float(run["v"][0]),
            float(run["h"].max()) + offset,
            f"{label} = {run['intrinsic_length']:.2f}",
            color=color,
            fontsize=11.5,
            fontweight="bold",
            ha="center",
            zorder=11,
        )

    ratio = stretched["intrinsic_length"] / preserved["intrinsic_length"]
    figure.text(
        0.70, 0.935, text["right"], ha="center", va="center",
        fontsize=12.5, color=INK_PRIMARY,
    )
    surface_axis.set_xticks([])
    surface_axis.set_yticks([])
    surface_axis.set_zticks([])
    surface_axis.set_box_aspect((1.0, 1.0, 0.5))
    surface_axis.view_init(elev=30, azim=-60)
    surface_axis.grid(False)
    for pane_axis in (surface_axis.xaxis, surface_axis.yaxis, surface_axis.zaxis):
        pane_axis.pane.set_visible(False)
        pane_axis.line.set_color((1.0, 1.0, 1.0, 0.0))
    figure.text(
        0.70,
        0.055,
        f"A / B = {ratio:.2f}   ({text['stretched']} / {text['preserved']})",
        ha="center",
        va="center",
        fontsize=10.5,
        color=INK_MUTED,
    )

    suffix = "" if language == "en" else "_ja"
    for extension in ("png", "pdf"):
        figure.savefig(
            output_dir / f"distortion_teaser{suffix}.{extension}",
            dpi=220,
            bbox_inches="tight",
        )
    plt.close(figure)

    print(
        f"[{language}] planar A = planar B = {stretched['planar_length']:.4f}; "
        f"intrinsic A = {stretched['intrinsic_length']:.4f}, "
        f"B = {preserved['intrinsic_length']:.4f}, ratio = {ratio:.4f}"
    )


def main() -> None:
    output_dir = pathlib.Path(__file__).resolve().parent
    for language in ("en", "ja"):
        draw(language, output_dir)
    print(f"Wrote distortion teaser figures to {output_dir}")


if __name__ == "__main__":
    main()
