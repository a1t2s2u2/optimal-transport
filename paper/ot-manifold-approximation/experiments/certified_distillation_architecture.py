#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "matplotlib>=3.9",
# ]
# ///
"""Draw the paper's certified decoder-head distillation architecture.

The PDF outputs remain vector graphics; matching PNG files are generated only
for quick inspection.  English and Japanese versions share exactly the same
layout so that either can be included as a full-width paper figure.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import (
    Circle,
    FancyArrowPatch,
    FancyBboxPatch,
    Rectangle,
)

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent

INK = "#21313C"
MUTED = "#5E6B73"
LANE = "#F7F9FA"
TEACHER = "#24577A"
TEACHER_LIGHT = "#EAF2F7"
STUDENT = "#287A55"
STUDENT_LIGHT = "#E9F5EE"
OFFLINE = "#B86B25"
OFFLINE_LIGHT = "#FFF3E6"
LATENT = "#6C5A9E"
LATENT_LIGHT = "#F1EEF8"
FAIL = "#B6413A"
WHITE = "#FFFFFF"


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


def rounded_box(
    ax: Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    face: str,
    edge: str,
    linewidth: float = 1.35,
    radius: float = 0.12,
    zorder: float = 2.0,
) -> FancyBboxPatch:
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle=f"round,pad=0.02,rounding_size={radius}",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 1.55,
    style: str = "-|>",
    dashed: bool = False,
    connection: str = "arc3,rad=0",
    zorder: float = 3.0,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle=style,
        mutation_scale=11,
        linewidth=linewidth,
        linestyle=(0, (4, 3)) if dashed else "solid",
        color=color,
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def add_text(
    ax: Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 9.0,
    color: str = INK,
    weight: str = "normal",
    horizontal: str = "center",
    vertical: str = "center",
    linespacing: float = 1.18,
    zorder: float = 5.0,
) -> None:
    ax.text(
        x,
        y,
        value,
        fontsize=size,
        color=color,
        fontweight=weight,
        ha=horizontal,
        va=vertical,
        linespacing=linespacing,
        zorder=zorder,
    )


def draw_input(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.22, 1.18), face=WHITE, edge=INK)
    # A tiny vector "image" makes the input recognizable without embedding a
    # raster asset in the otherwise vector PDF.
    ax.add_patch(
        Rectangle(
            (x - 0.43, y - 0.20),
            0.42,
            0.53,
            facecolor="#172027",
            edgecolor="none",
            zorder=3,
        )
    )
    points = [
        (x - 0.34, y + 0.21),
        (x - 0.22, y + 0.27),
        (x - 0.11, y + 0.18),
        (x - 0.15, y + 0.06),
        (x - 0.31, y + 0.02),
        (x - 0.35, y - 0.10),
        (x - 0.20, y - 0.14),
        (x - 0.09, y - 0.07),
    ]
    xs, ys = zip(*points)
    ax.plot(xs, ys, color=WHITE, linewidth=2.0, solid_capstyle="round", zorder=4)
    add_text(ax, x + 0.22, y + 0.14, r"$x$", size=14, weight="bold")
    add_text(
        ax,
        x + 0.20,
        y - 0.16,
        "input" if language == "en" else "入力",
        size=7.8,
        color=MUTED,
    )


def draw_encoder(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.48, 1.18), face="#EDF2F5", edge="#566873")
    title = "encoder" if language == "en" else "エンコーダ"
    add_text(ax, x, y + 0.28, title, size=8.6, weight="bold")
    add_text(ax, x, y - 0.01, r"$q_\phi(z\mid x)$", size=10.4)
    add_text(
        ax,
        x,
        y - 0.31,
        "posterior code" if language == "en" else "事後分布のコード",
        size=7.0,
        color=MUTED,
    )


def draw_latent(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.30, 1.18), face=LATENT_LIGHT, edge=LATENT)
    for dx, dy, radius in (
        (-0.30, 0.13, 0.045),
        (-0.14, 0.28, 0.045),
        (0.03, 0.13, 0.045),
        (0.18, 0.27, 0.045),
        (0.31, 0.08, 0.045),
    ):
        ax.add_patch(
            Circle(
                (x + dx, y + dy), radius, facecolor=LATENT, edgecolor="none", zorder=4
            )
        )
    add_text(ax, x, y - 0.02, r"$z\in\mathbb{R}^{d}$", size=10.2, weight="bold")
    add_text(
        ax,
        x,
        y - 0.34,
        "latent code" if language == "en" else "潜在コード",
        size=7.2,
        color=MUTED,
    )


def draw_trunk(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.62, 1.18), face="#F0F4F6", edge="#526A78")
    for layer, layer_x in enumerate((x - 0.44, x - 0.14, x + 0.16)):
        count = (2, 3, 2)[layer]
        positions = [y + (index - (count - 1) / 2.0) * 0.22 for index in range(count)]
        if layer:
            previous_x = (x - 0.44, x - 0.14, x + 0.16)[layer - 1]
            previous_count = (2, 3, 2)[layer - 1]
            previous_positions = [
                y + (index - (previous_count - 1) / 2.0) * 0.22
                for index in range(previous_count)
            ]
            for py in previous_positions:
                for cy in positions:
                    ax.plot(
                        [previous_x, layer_x],
                        [py, cy],
                        color="#AAB7BE",
                        linewidth=0.55,
                        zorder=3,
                    )
        for node_y in positions:
            ax.add_patch(
                Circle(
                    (layer_x, node_y),
                    0.045,
                    facecolor="#526A78",
                    edgecolor="none",
                    zorder=4,
                )
            )
    add_text(ax, x + 0.47, y + 0.16, r"$h_\theta(z)$", size=9.7, weight="bold")
    add_text(
        ax,
        x + 0.47,
        y - 0.15,
        "shared\n+ frozen" if language == "en" else "共有\n+ 固定",
        size=6.8,
        color=MUTED,
    )


def draw_head(
    ax: Axes,
    x: float,
    y: float,
    *,
    teacher: bool,
    language: str,
) -> None:
    color = TEACHER if teacher else STUDENT
    face = TEACHER_LIGHT if teacher else STUDENT_LIGHT
    rounded_box(ax, (x, y), (1.85, 1.02), face=face, edge=color, linewidth=1.55)
    if teacher:
        title = r"dense head $W$" if language == "en" else r"教師 head $W$"
        formula = r"$W\,h(z)+b$"
        foot = r"$W\in\mathbb{R}^{N\times p}$"
    else:
        title = r"rank-$r$ head $U_r$" if language == "en" else r"学生 head $U_r$"
        formula = r"$U_r h(z)+b$"
        foot = r"$U_r=A_rB_r$"
    add_text(ax, x, y + 0.27, title, size=7.3, color=color, weight="bold")
    add_text(ax, x, y - 0.01, formula, size=9.2)
    add_text(ax, x, y - 0.30, foot, size=7.3, color=MUTED)


def draw_gaussian(
    ax: Axes,
    x: float,
    y: float,
    *,
    teacher: bool,
    language: str,
) -> None:
    color = TEACHER if teacher else STUDENT
    face = TEACHER_LIGHT if teacher else STUDENT_LIGHT
    rounded_box(ax, (x, y), (2.08, 1.02), face=face, edge=color, linewidth=1.35)
    label = r"teacher $K_z^T$" if teacher else r"student $K_z^S$"
    if language == "ja":
        label = r"教師 Gaussian $K_z^T$" if teacher else r"学生 Gaussian $K_z^S$"
    superscript = "T" if teacher else "S"
    add_text(ax, x, y + 0.29, label, size=7.1, color=color, weight="bold")
    add_text(
        ax,
        x - 0.10,
        y + 0.01,
        rf"$K_z^{superscript}=\mathcal{{N}}(\mu_{superscript},$",
        size=7.8,
    )
    add_text(
        ax, x - 0.10, y - 0.24, rf"$\mathrm{{diag}}\,\sigma_{superscript}^2)$", size=7.8
    )
    # Small density glyph.
    curve_x = [x + 0.71 + 0.07 * index for index in range(7)]
    curve_y = [
        y - 0.31 + 0.30 * math.exp(-0.5 * ((index - 3) / 1.25) ** 2)
        for index in range(7)
    ]
    ax.plot(curve_x, curve_y, color=color, linewidth=1.5, zorder=5)
    ax.plot(
        [x + 0.67, x + 1.18],
        [y - 0.31, y - 0.31],
        color=MUTED,
        linewidth=0.55,
        zorder=4,
    )


def draw_acceptance(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (4.42, 2.68), face=WHITE, edge=INK, linewidth=1.55)
    title = (
        "Pass only if both tolerances hold"
        if language == "en"
        else "2つの許容値による合否判定"
    )
    add_text(ax, x, y + 1.03, title, size=8.6, weight="bold")
    ax.plot([x - 1.95, x + 1.95], [y + 0.78, y + 0.78], color="#D9E0E4", linewidth=0.8)

    output_label = "output fidelity" if language == "en" else "出力の保存"
    geometry_label = "local geometry" if language == "en" else "局所幾何の保存"
    all_pair_label = "all intrinsic pairs" if language == "en" else "すべての内在距離"
    add_text(
        ax,
        x - 1.78,
        y + 0.49,
        output_label,
        size=7.5,
        color=TEACHER,
        weight="bold",
        horizontal="left",
    )
    add_text(
        ax,
        x - 1.78,
        y + 0.18,
        r"$E_0=\sup_z W_2(K_z^S,K_z^T)\leq\eta$",
        size=8.9,
        horizontal="left",
    )
    add_text(
        ax,
        x - 1.78,
        y - 0.17,
        geometry_label,
        size=7.5,
        color=STUDENT,
        weight="bold",
        horizontal="left",
    )
    add_text(
        ax,
        x - 1.78,
        y - 0.48,
        r"$D_{\mathrm{loc}}\leq\tau$",
        size=9.0,
        horizontal="left",
    )
    add_text(ax, x - 0.18, y - 0.48, r"$\Longrightarrow$", size=10.2)
    add_text(
        ax, x + 0.22, y - 0.31, all_pair_label, size=7.1, color=MUTED, horizontal="left"
    )
    add_text(
        ax,
        x + 0.22,
        y - 0.57,
        r"$D_{\mathrm{pair}}\leq\tau$",
        size=9.0,
        horizontal="left",
    )

    rounded_box(
        ax,
        (x, y - 1.01),
        (3.18, 0.40),
        face=STUDENT_LIGHT,
        edge=STUDENT,
        linewidth=1.1,
        radius=0.16,
        zorder=4,
    )
    decision = (
        "PASS only if both hold" if language == "en" else "両方を満たすときだけ合格"
    )
    add_text(ax, x, y - 1.01, decision, size=7.2, color=STUDENT, weight="bold")


def draw_calibration(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.82, 1.18), face=OFFLINE_LIGHT, edge=OFFLINE)
    title = r"calibration $z_i$" if language == "en" else r"較正コード $z_i$"
    add_text(ax, x, y + 0.28, title, size=7.9, color=OFFLINE, weight="bold")
    add_text(ax, x, y - 0.02, r"$\{z_i\}_{i=1}^{m}$", size=10.3)
    add_text(
        ax,
        x,
        y - 0.34,
        "no labels needed" if language == "en" else "ラベル不要",
        size=7.0,
        color=MUTED,
    )


def draw_jets(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.82, 1.18), face=OFFLINE_LIGHT, edge=OFFLINE)
    title = r"samples $(h_i,J_i)$" if language == "en" else r"値と微分 $(h_i,J_i)$"
    add_text(ax, x, y + 0.28, title, size=7.8, color=OFFLINE, weight="bold")
    add_text(ax, x, y - 0.02, r"$h_i=h(z_i)$", size=8.8)
    add_text(ax, x, y - 0.32, r"$J_i=J_h(z_i)$", size=8.8)


def draw_covariance(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (2.66, 1.38), face=OFFLINE_LIGHT, edge=OFFLINE)
    title = "covariance $C$" if language == "en" else "共分散 $C$"
    add_text(ax, x, y + 0.42, title, size=7.8, color=OFFLINE, weight="bold")
    add_text(
        ax,
        x,
        y + 0.03,
        r"$C=\frac{1}{m}\sum_i [h_i h_i^\top$",
        size=8.2,
    )
    add_text(
        ax,
        x,
        y - 0.25,
        r"$+\lambda J_iJ_i^\top]+\rho I$",
        size=8.2,
    )
    add_text(
        ax,
        x,
        y - 0.50,
        r"$C\succ0$",
        size=7.1,
        color=MUTED,
    )


def draw_svd(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(ax, (x, y), (1.78, 1.18), face=OFFLINE_LIGHT, edge=OFFLINE)
    title = "whiten + SVD" if language == "en" else "白色化 + SVD"
    add_text(ax, x, y + 0.29, title, size=7.9, color=OFFLINE, weight="bold")
    add_text(ax, x, y - 0.01, r"$B=WC^{1/2}$", size=9.1)
    add_text(ax, x, y - 0.32, r"$B\;\mapsto\;B_r$", size=8.7)


def draw_solution(ax: Axes, x: float, y: float, language: str) -> None:
    rounded_box(
        ax, (x, y), (1.95, 1.18), face=STUDENT_LIGHT, edge=STUDENT, linewidth=1.55
    )
    title = r"rank-$r$ head"
    add_text(ax, x, y + 0.29, title, size=7.8, color=STUDENT, weight="bold")
    add_text(ax, x, y - 0.02, r"$U_r=B_rC^{-1/2}$", size=9.4)
    add_text(
        ax,
        x,
        y - 0.34,
        "one SVD" if language == "en" else "SVDを1回だけ実行",
        size=7.0,
        color=MUTED,
    )


def build_figure(language: str) -> Figure:
    japanese = language == "ja"
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # The canvas is only moderately wider than a two-column paper figure.  It
    # leaves enough physical room for the equations while keeping labels
    # readable after inclusion at ``\textwidth``.
    figure, ax = plt.subplots(figsize=(12.0, 5.33))
    figure.patch.set_facecolor(WHITE)
    ax.set_xlim(0.0, 18.0)
    ax.set_ylim(0.0, 8.0)
    ax.set_aspect("equal")
    ax.axis("off")

    title = (
        "Certified low-rank distillation of a Gaussian decoder"
        if language == "en"
        else "Gaussianデコーダーの保証付き低ランク蒸留"
    )
    subtitle = (
        "solid arrows: inference and comparison     dashed arrow: one-shot offline compression"
        if language == "en"
        else "実線：推論と比較     破線：圧縮前に一度だけ行う計算"
    )
    add_text(ax, 9.0, 7.72, title, size=15.2, weight="bold")
    add_text(ax, 9.0, 7.37, subtitle, size=8.1, color=MUTED)

    upper = FancyBboxPatch(
        (0.20, 3.63),
        17.60,
        3.45,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor=LANE,
        edgecolor="#D9E1E5",
        linewidth=0.9,
        zorder=0,
    )
    lower = FancyBboxPatch(
        (0.20, 0.20),
        17.60,
        3.05,
        boxstyle="round,pad=0.02,rounding_size=0.14",
        facecolor="#FFFCF8",
        edgecolor="#E6D7C8",
        linewidth=0.9,
        zorder=0,
    )
    ax.add_patch(upper)
    ax.add_patch(lower)

    upper_label = (
        "A. Frozen teacher and compressed student"
        if language == "en"
        else "A. 固定した教師と圧縮した学生"
    )
    lower_label = (
        "B. Calibration and closed-form compression"
        if language == "en"
        else "B. 較正データから閉形式で圧縮"
    )
    add_text(
        ax,
        0.46,
        6.85,
        upper_label,
        size=8.0,
        color=MUTED,
        weight="bold",
        horizontal="left",
    )
    add_text(
        ax,
        0.46,
        3.01,
        lower_label,
        size=8.0,
        color=OFFLINE,
        weight="bold",
        horizontal="left",
    )

    main_y = 5.38
    teacher_y = 6.12
    student_y = 4.64
    input_x, encoder_x, latent_x, trunk_x = 0.95, 2.52, 4.10, 5.87
    head_x, gaussian_x, acceptance_x = 9.72, 11.93, 15.47

    draw_input(ax, input_x, main_y, language)
    draw_encoder(ax, encoder_x, main_y, language)
    draw_latent(ax, latent_x, main_y, language)
    draw_trunk(ax, trunk_x, main_y, language)
    draw_head(ax, head_x, teacher_y, teacher=True, language=language)
    draw_head(ax, head_x, student_y, teacher=False, language=language)
    draw_gaussian(ax, gaussian_x, teacher_y, teacher=True, language=language)
    draw_gaussian(ax, gaussian_x, student_y, teacher=False, language=language)
    draw_acceptance(ax, acceptance_x, main_y, language)

    arrow(ax, (1.58, main_y), (1.76, main_y))
    arrow(ax, (3.27, main_y), (3.44, main_y))
    arrow(ax, (4.76, main_y), (5.04, main_y))
    # The trunk is shared; a single branch point feeds both heads.
    ax.plot([6.70, 7.35], [main_y, main_y], color=INK, linewidth=1.55, zorder=3)
    ax.add_patch(
        Circle((7.35, main_y), 0.055, facecolor=INK, edgecolor="none", zorder=4)
    )
    arrow(
        ax,
        (7.35, main_y),
        (8.78, teacher_y),
        color=TEACHER,
        connection="arc3,rad=-0.08",
    )
    arrow(
        ax, (7.35, main_y), (8.78, student_y), color=STUDENT, connection="arc3,rad=0.08"
    )
    arrow(ax, (10.66, teacher_y), (10.88, teacher_y), color=TEACHER)
    arrow(ax, (10.66, student_y), (10.88, student_y), color=STUDENT)
    arrow(
        ax,
        (12.99, teacher_y),
        (13.25, 5.91),
        color=TEACHER,
        connection="arc3,rad=-0.08",
    )
    arrow(
        ax, (12.99, student_y), (13.25, 4.86), color=STUDENT, connection="arc3,rad=0.08"
    )

    calibration_y = 1.48
    calibration_x, jets_x, covariance_x = 1.34, 3.49, 5.95
    svd_x, solution_x = 8.25, 10.31
    draw_calibration(ax, calibration_x, calibration_y, language)
    draw_jets(ax, jets_x, calibration_y, language)
    draw_covariance(ax, covariance_x, calibration_y, language)
    draw_svd(ax, svd_x, calibration_y, language)
    draw_solution(ax, solution_x, calibration_y, language)
    arrow(ax, (2.27, calibration_y), (2.55, calibration_y), color=OFFLINE)
    arrow(ax, (4.42, calibration_y), (4.59, calibration_y), color=OFFLINE)
    arrow(ax, (7.30, calibration_y), (7.34, calibration_y), color=OFFLINE)
    arrow(ax, (9.16, calibration_y), (9.32, calibration_y), color=OFFLINE)

    # Install the analytic solution in the student head.  A right-angle path
    # keeps the arrow away from the Gaussian output and the main inference flow.
    ax.plot(
        [solution_x, solution_x, head_x],
        [2.09, 3.88, 3.88],
        color=STUDENT,
        linewidth=1.45,
        linestyle=(0, (4, 3)),
        zorder=2,
    )
    arrow(
        ax,
        (head_x, 3.88),
        (head_x, 4.11),
        color=STUDENT,
        linewidth=1.45,
        dashed=True,
    )
    install_label = "install $U_r$" if language == "en" else "$U_r$を設定"
    add_text(ax, 10.93, 3.56, install_label, size=7.2, color=STUDENT, weight="bold")

    explanation = (
        "The trunk is never retrained.\n"
        "Increase rank until output and geometry both pass."
        if language == "en"
        else "trunkは再学習しない。\n出力と幾何の両方が合格するまでrankを増やす。"
    )
    rounded_box(
        ax, (14.34, 1.48), (5.95, 1.18), face=WHITE, edge="#D5C2AF", linewidth=1.0
    )
    add_text(
        ax,
        11.62,
        1.75,
        "KEY IDEA" if language == "en" else "要点",
        size=7.2,
        color=OFFLINE,
        weight="bold",
        horizontal="left",
    )
    add_text(ax, 11.62, 1.39, explanation, size=8.0, color=INK, horizontal="left")

    return figure


def save(language: str) -> None:
    figure = build_figure(language)
    suffix = "_ja" if language == "ja" else ""
    base = HERE / f"certified_distillation_architecture{suffix}"
    figure.savefig(base.with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.04)
    figure.savefig(
        base.with_suffix(".png"), dpi=230, bbox_inches="tight", pad_inches=0.04
    )
    plt.close(figure)


def main() -> None:
    save("en")
    save("ja")
    print("wrote certified distillation architecture figures (PDF + PNG, EN + JA)")


if __name__ == "__main__":
    main()
