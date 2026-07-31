#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.15"
# dependencies = [
#   "matplotlib>=3.9",
# ]
# ///
"""Draw the compact decoder-distillation architecture used in the paper.

The figure is sized at the paper's final two-column width.  PDF outputs stay
vector graphics; PNG copies are generated only for visual inspection.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.figure import Figure
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent

INK = "#26343D"
MUTED = "#64717A"
LINE = "#9AA5AC"
NEUTRAL = "#F5F7F8"
TEACHER = "#2E6488"
TEACHER_FILL = "#EEF5F9"
STUDENT = "#2F7D59"
STUDENT_FILL = "#EEF7F1"
OFFLINE = "#A9682D"
OFFLINE_FILL = "#FFF7ED"
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


def add_text(
    ax: Axes,
    x: float,
    y: float,
    value: str,
    *,
    size: float = 7.2,
    color: str = INK,
    weight: str = "normal",
    horizontal: str = "center",
    vertical: str = "center",
    linespacing: float = 1.1,
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


def box(
    ax: Axes,
    center: tuple[float, float],
    size: tuple[float, float],
    *,
    face: str = NEUTRAL,
    edge: str = LINE,
    linewidth: float = 0.85,
) -> FancyBboxPatch:
    x, y = center
    width, height = size
    patch = FancyBboxPatch(
        (x - width / 2.0, y - height / 2.0),
        width,
        height,
        boxstyle="round,pad=0.02,rounding_size=0.055",
        facecolor=face,
        edgecolor=edge,
        linewidth=linewidth,
        zorder=2,
    )
    ax.add_patch(patch)
    return patch


def arrow(
    ax: Axes,
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    color: str = INK,
    linewidth: float = 0.9,
    dashed: bool = False,
    connection: str = "arc3,rad=0",
    zorder: float = 3.0,
) -> FancyArrowPatch:
    patch = FancyArrowPatch(
        start,
        end,
        arrowstyle="-|>",
        mutation_scale=7.5,
        linewidth=linewidth,
        linestyle=(0, (3.2, 2.2)) if dashed else "solid",
        color=color,
        connectionstyle=connection,
        shrinkA=0,
        shrinkB=0,
        zorder=zorder,
    )
    ax.add_patch(patch)
    return patch


def routed_arrow(
    ax: Axes,
    points: list[tuple[float, float]],
    *,
    color: str,
    dashed: bool = False,
    linewidth: float = 0.85,
) -> None:
    style = (0, (3.2, 2.2)) if dashed else "solid"
    xs, ys = zip(*points[:-1])
    ax.plot(
        xs,
        ys,
        color=color,
        linewidth=linewidth,
        linestyle=style,
        zorder=2.5,
    )
    arrow(
        ax,
        points[-2],
        points[-1],
        color=color,
        linewidth=linewidth,
        dashed=dashed,
    )


def draw_forward_path(ax: Axes, language: str) -> None:
    if language == "en":
        row_label = "FORWARD COMPARISON"
        latent_label = "latent code"
        trunk_title = "shared, frozen trunk"
        teacher_title = "teacher (frozen)"
        student_title = r"rank-$r$ student"
        compare_title = "compare"
        output_label = r"output: $E_0$"
        local_label = r"local geometry: $D_{\mathrm{loc}}$"
        accept_title = r"accept rank $r$"
        global_label = r"global $s,M_1,\ell_1$ bounds"
    else:
        row_label = "順伝播と比較"
        latent_label = "潜在コード"
        trunk_title = "共有・固定trunk"
        teacher_title = "教師（固定）"
        student_title = r"rank-$r$ 学生"
        compare_title = "比較"
        output_label = r"出力: $E_0$"
        local_label = r"局所幾何: $D_{\mathrm{loc}}$"
        accept_title = r"rank $r$ を採用"
        global_label = r"大域的な $s,M_1,\ell_1$ 境界"

    add_text(
        ax,
        0.12,
        5.18,
        row_label,
        size=6.2,
        color=MUTED,
        weight="bold",
        horizontal="left",
    )

    box(ax, (0.78, 3.78), (1.20, 0.88), face=WHITE, edge=LINE)
    add_text(ax, 0.78, 3.94, r"$z$", size=9.0, weight="bold")
    add_text(ax, 0.78, 3.61, latent_label, size=5.2, color=MUTED)

    box(ax, (3.02, 3.78), (2.35, 0.88), face=NEUTRAL, edge="#71818A")
    add_text(ax, 3.02, 3.97, trunk_title, size=6.3, weight="bold")
    add_text(ax, 3.02, 3.64, r"$h=h_\theta(z)$; shared $b$", size=6.4)

    box(
        ax,
        (6.72, 4.34),
        (3.05, 0.82),
        face=TEACHER_FILL,
        edge=TEACHER,
        linewidth=0.95,
    )
    add_text(ax, 5.42, 4.55, teacher_title, size=5.8, color=TEACHER, horizontal="left")
    add_text(ax, 6.72, 4.25, r"$F_T=Wh+b\;\mapsto\;K_z^T$", size=6.5)

    box(
        ax,
        (6.72, 3.22),
        (3.05, 0.82),
        face=STUDENT_FILL,
        edge=STUDENT,
        linewidth=0.95,
    )
    add_text(ax, 5.42, 3.43, student_title, size=5.8, color=STUDENT, horizontal="left")
    add_text(
        ax,
        6.72,
        3.13,
        r"$F_{S,r}=U_rh+b\;\mapsto\;K_z^{S,r}$",
        size=6.3,
    )

    box(ax, (10.42, 3.78), (2.45, 1.88), face=WHITE, edge=LINE)
    add_text(ax, 10.42, 4.36, compare_title, size=6.2, weight="bold")
    add_text(ax, 10.42, 3.88, output_label, size=6.4)
    add_text(ax, 10.42, 3.48, local_label, size=6.1)

    box(
        ax,
        (15.13, 3.78),
        (3.72, 1.88),
        face=STUDENT_FILL,
        edge=STUDENT,
        linewidth=0.95,
    )
    add_text(ax, 15.13, 4.40, accept_title, size=6.4, color=STUDENT, weight="bold")
    add_text(ax, 15.13, 4.00, r"$E_0\leq\eta$", size=7.3)
    add_text(
        ax,
        15.13,
        3.65,
        r"$\overline{\varepsilon}_r\leq\varepsilon_\tau(s,M_1)$",
        size=7.2,
    )
    add_text(
        ax,
        15.13,
        3.27,
        r"$\Longrightarrow\ D_{\mathrm{pair}}\leq\tau$",
        size=7.5,
        weight="bold",
    )
    add_text(ax, 15.13, 2.99, global_label, size=5.2, color=MUTED)

    arrow(ax, (1.38, 3.78), (1.84, 3.78))
    ax.plot([4.20, 4.65], [3.78, 3.78], color=INK, linewidth=0.9, zorder=3)
    ax.plot([4.65, 4.65], [3.22, 4.34], color=INK, linewidth=0.9, zorder=3)
    arrow(ax, (4.65, 4.34), (5.19, 4.34), color=TEACHER)
    arrow(ax, (4.65, 3.22), (5.19, 3.22), color=STUDENT)
    arrow(ax, (8.25, 4.34), (9.19, 4.08), color=TEACHER)
    arrow(ax, (8.25, 3.22), (9.19, 3.48), color=STUDENT)
    arrow(ax, (11.65, 3.78), (13.27, 3.78))


def draw_compression_path(ax: Axes, language: str) -> None:
    if language == "en":
        row_label = "ONE-SHOT COMPRESSION"
        code_title = "codes"
        jets_title = "calibration jets"
        matrix_title = "matrix $C$"
        svd_title = r"whitened rank-$r$ SVD"
        solution_title = "student head"
        install_label = r"install $U_r$"
        certify_label = "spectral certificate"
    else:
        row_label = "1回限りの圧縮"
        code_title = "較正点"
        jets_title = "値・Jacobian"
        matrix_title = "較正行列 $C$"
        svd_title = r"白色化 rank-$r$ SVD"
        solution_title = "学生head"
        install_label = r"$U_r$ を設定"
        certify_label = "スペクトル保証"

    add_text(
        ax,
        0.12,
        1.76,
        row_label,
        size=6.2,
        color=OFFLINE,
        weight="bold",
        horizontal="left",
    )

    box(ax, (0.88, 0.82), (1.42, 0.86), face=OFFLINE_FILL, edge=OFFLINE)
    add_text(ax, 0.88, 1.02, code_title, size=5.6, color=OFFLINE, weight="bold")
    add_text(ax, 0.88, 0.65, r"$\{z_i\}_{i=1}^m$", size=7.1)

    box(ax, (2.95, 0.82), (1.90, 1.06), face=OFFLINE_FILL, edge=OFFLINE)
    add_text(ax, 2.95, 1.02, jets_title, size=5.5, color=OFFLINE, weight="bold")
    add_text(ax, 2.95, 0.78, r"$h_i=h_\theta(z_i)$", size=5.9)
    add_text(ax, 2.95, 0.51, r"$J_i=J_{h_\theta}(z_i)$", size=5.9)

    box(ax, (5.61, 0.82), (2.88, 1.06), face=OFFLINE_FILL, edge=OFFLINE)
    add_text(ax, 5.61, 1.15, matrix_title, size=5.6, color=OFFLINE, weight="bold")
    add_text(ax, 5.61, 0.82, r"$C=m^{-1}\!\sum_i(h_ih_i^\top$", size=5.8)
    add_text(ax, 5.61, 0.51, r"$+\lambda J_iJ_i^\top)+\rho I\succ0$", size=5.8)

    box(ax, (8.92, 0.82), (2.55, 1.06), face=OFFLINE_FILL, edge=OFFLINE)
    add_text(ax, 8.92, 1.15, svd_title, size=5.2, color=OFFLINE, weight="bold")
    add_text(ax, 8.92, 0.82, r"$B=WC^{1/2}$", size=6.9)
    add_text(ax, 8.92, 0.51, r"$B_r=[B]_r$", size=6.8)

    box(
        ax,
        (12.08, 0.82),
        (2.63, 1.06),
        face=STUDENT_FILL,
        edge=STUDENT,
        linewidth=0.95,
    )
    add_text(
        ax,
        12.08,
        1.15,
        solution_title,
        size=5.7,
        color=STUDENT,
        weight="bold",
    )
    add_text(ax, 12.08, 0.82, r"$U_r=B_rC^{-1/2}$", size=6.5)
    add_text(
        ax,
        12.08,
        0.51,
        r"$\overline{\varepsilon}_r=\sigma_{r+1}(B)\ell_1$",
        size=6.2,
    )

    arrow(ax, (1.59, 0.82), (2.04, 0.82), color=OFFLINE)
    arrow(ax, (3.86, 0.82), (4.17, 0.82), color=OFFLINE)
    arrow(ax, (7.05, 0.82), (7.65, 0.82), color=OFFLINE)
    arrow(ax, (10.20, 0.82), (10.77, 0.82), color=OFFLINE)

    # The analytic head is installed without retraining the shared trunk.
    routed_arrow(
        ax,
        [(12.08, 1.35), (12.08, 2.04), (8.58, 2.04), (8.58, 3.22), (8.25, 3.22)],
        color=STUDENT,
        dashed=True,
    )
    add_text(ax, 10.45, 1.91, install_label, size=5.5, color=STUDENT, weight="bold")

    # The same singular spectrum also feeds the global safe-rank test.
    routed_arrow(
        ax,
        [(13.40, 0.82), (15.13, 0.82), (15.13, 2.80)],
        color=OFFLINE,
        linewidth=0.85,
    )
    add_text(
        ax,
        15.32,
        1.77,
        certify_label,
        size=5.2,
        color=OFFLINE,
        horizontal="left",
        vertical="center",
    )


def build_figure(language: str) -> Figure:
    if language == "ja":
        plt.rcParams["font.family"] = japanese_font_family()
    else:
        plt.rcParams["font.family"] = "DejaVu Sans"
    plt.rcParams["mathtext.fontset"] = "stix"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42

    # 7.05 in is the paper's two-column text width.  Generating at final size
    # keeps all labels at true publication font sizes.
    figure, ax = plt.subplots(figsize=(7.05, 2.15))
    figure.patch.set_facecolor(WHITE)
    figure.subplots_adjust(left=0.005, right=0.995, bottom=0.005, top=0.995)
    ax.set_xlim(0.0, 18.0)
    ax.set_ylim(0.15, 5.55)
    ax.set_aspect("equal")
    ax.axis("off")

    draw_forward_path(ax, language)
    draw_compression_path(ax, language)
    return figure


def save(language: str) -> None:
    figure = build_figure(language)
    suffix = "_ja" if language == "ja" else ""
    base = HERE / f"certified_distillation_architecture{suffix}"
    figure.savefig(base.with_suffix(".pdf"), facecolor=WHITE)
    figure.savefig(base.with_suffix(".png"), dpi=260, facecolor=WHITE)
    plt.close(figure)


def main() -> None:
    save("en")
    save("ja")
    print("wrote compact architecture figures (PDF + PNG, EN + JA)")


if __name__ == "__main__":
    main()
