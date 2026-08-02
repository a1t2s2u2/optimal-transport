#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
# ]
# ///
"""Exact low-rank compression frontier for a Gaussian image decoder."""

from __future__ import annotations

import csv
import math
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
IMAGE_SIDE = 28
OUTPUT_DIMENSION = IMAGE_SIDE**2
TEACHER_BLOCKS = 128
FEATURE_DIMENSION = 4 * TEACHER_BLOCKS
PRIMARY_TOLERANCE = 0.05
OUTPUT_TOLERANCE = 0.05
TOLERANCES = (0.10, 0.05, 0.02, 0.01)
DISPLAY_COORDINATE = (1.10, -0.70)
IMAGE_SCALE = 1.35


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


def spectral_energy(blocks: int) -> float:
    return float(sum(index ** -2 for index in range(1, blocks + 1)))


def exact_distortion(blocks: int) -> float:
    return 1.0 - math.sqrt(
        spectral_energy(blocks) / spectral_energy(TEACHER_BLOCKS)
    )


def exact_output_error(blocks: int) -> float:
    """Uniform W2 teacher--prefix error (the covariance is fixed)."""
    tail = sum(index ** -4 for index in range(blocks + 1, TEACHER_BLOCKS + 1))
    return math.sqrt(2.0 * tail)


def optimal_output_rms(rank: int) -> float:
    """Eckart--Young lower bound for every shared-trunk rank-r head."""
    singular_values = np.repeat(
        np.asarray([index ** -2 for index in range(1, TEACHER_BLOCKS + 1)]),
        4,
    )
    return math.sqrt(0.5 * float(np.square(singular_values[rank:]).sum()))


def minimum_blocks(tolerance: float) -> int:
    for blocks in range(1, TEACHER_BLOCKS + 1):
        if exact_distortion(blocks) <= tolerance:
            return blocks
    return TEACHER_BLOCKS


def head_parameters(rank: int) -> int:
    return rank * (OUTPUT_DIMENSION + FEATURE_DIMENSION) + OUTPUT_DIMENSION


def teacher_head_parameters() -> int:
    return OUTPUT_DIMENSION * FEATURE_DIMENSION + OUTPUT_DIMENSION


def dct_dictionary() -> np.ndarray:
    points = np.arange(IMAGE_SIDE, dtype=np.float64)
    basis = np.empty((IMAGE_SIDE, IMAGE_SIDE), dtype=np.float64)
    for frequency in range(IMAGE_SIDE):
        scale = math.sqrt(1.0 / IMAGE_SIDE) if frequency == 0 else math.sqrt(
            2.0 / IMAGE_SIDE
        )
        basis[:, frequency] = scale * np.cos(
            math.pi * (2.0 * points + 1.0) * frequency / (2.0 * IMAGE_SIDE)
        )
    frequencies = sorted(
        ((left, right) for left in range(IMAGE_SIDE) for right in range(IMAGE_SIDE)),
        key=lambda pair: (pair[0] + pair[1], max(pair), pair[0]),
    )[:FEATURE_DIMENSION]
    atoms = [
        np.outer(basis[:, left], basis[:, right]).reshape(-1)
        for left, right in frequencies
    ]
    dictionary = np.column_stack(atoms)
    gram_error = np.linalg.norm(
        dictionary.T @ dictionary - np.eye(FEATURE_DIMENSION), ord=2
    )
    if gram_error > 1.0e-10:
        raise RuntimeError(f"DCT dictionary is not orthonormal: {gram_error}")
    return dictionary


def feature_vector(x_value: float, y_value: float, blocks: int) -> np.ndarray:
    values: list[float] = []
    for index in range(1, blocks + 1):
        values.extend(
            [
                math.cos(index * x_value),
                math.sin(index * x_value),
                math.cos(index * y_value),
                math.sin(index * y_value),
            ]
        )
    return np.asarray(values, dtype=np.float64)


def coefficient_vector(blocks: int) -> np.ndarray:
    values: list[float] = []
    for index in range(1, blocks + 1):
        coefficient = index ** -2
        values.extend([coefficient] * 4)
    return np.asarray(values, dtype=np.float64)


def decoded_image(dictionary: np.ndarray, blocks: int) -> np.ndarray:
    features = feature_vector(*DISPLAY_COORDINATE, blocks)
    coefficients = coefficient_vector(blocks)
    centered = dictionary[:, : 4 * blocks] @ (coefficients * features)
    # IMAGE_SCALE changes display contrast only; all theoretical errors use the
    # unscaled Gaussian mean map in the paper.
    return (0.5 + IMAGE_SCALE * centered).reshape(IMAGE_SIDE, IMAGE_SIDE)


def rows() -> list[dict[str, float]]:
    teacher_parameters = teacher_head_parameters()
    values: list[dict[str, float]] = []
    total_energy = spectral_energy(TEACHER_BLOCKS)
    for blocks in range(1, TEACHER_BLOCKS + 1):
        rank = 4 * blocks
        distortion = exact_distortion(blocks)
        remaining = total_energy - spectral_energy(blocks)
        parameters = head_parameters(rank)
        values.append(
            {
                "retained_blocks": float(blocks),
                "student_rank": float(rank),
                "head_parameters": float(parameters),
                "head_compression": teacher_parameters / parameters,
                "exact_local_distortion": distortion,
                "exact_pair_distortion": distortion,
                "exact_uniform_output_error": exact_output_error(blocks),
                "optimal_rank_output_rms_lower_bound": optimal_output_rms(rank),
                "optimal_jacobian_loss": 2.0 * remaining,
                "pair_accuracy_at_5_percent": float(
                    distortion <= PRIMARY_TOLERANCE
                ),
                "simultaneous_pass_at_5_percent": float(
                    distortion <= PRIMARY_TOLERANCE
                    and exact_output_error(blocks) <= OUTPUT_TOLERANCE
                ),
            }
        )
    return values


def write_csv(values: list[dict[str, float]]) -> None:
    with (HERE / "lowrank_torus_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.DictWriter(handle, fieldnames=list(values[0]))
        writer.writeheader()
        writer.writerows(values)


def write_table() -> None:
    english = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\eta=\tau$ & Min. rank & Head params. & Compression & $E_0$ & $D_{\mathrm{pair}}$ \\",
        r"\midrule",
    ]
    japanese = [
        r"\begin{tabular}{rrrrrr}",
        r"\toprule",
        r"$\eta=\tau$ & 最小rank & headパラメータ & 圧縮率 & $E_0$ & $D_{\mathrm{pair}}$ \\",
        r"\midrule",
    ]
    teacher_parameters = teacher_head_parameters()
    for tolerance in TOLERANCES:
        blocks = next(
            block_count
            for block_count in range(1, TEACHER_BLOCKS + 1)
            if exact_distortion(block_count) <= tolerance
            and exact_output_error(block_count) <= tolerance
        )
        rank = 4 * blocks
        parameters = head_parameters(rank)
        row = (
            f"{100.0 * tolerance:.0f}\\% & {rank} & {parameters:,} & "
            f"{teacher_parameters / parameters:.2f}$\\times$ & "
            f"{100.0 * exact_output_error(blocks):.3f}\\% & "
            f"{100.0 * exact_distortion(blocks):.3f}\\% \\\\"
        )
        english.append(row)
        japanese.append(row)
    ending = [r"\bottomrule", r"\end{tabular}"]
    (HERE / "lowrank_torus_table.tex").write_text(
        "\n".join(english + ending) + "\n", encoding="utf-8"
    )
    (HERE / "lowrank_torus_table_ja.tex").write_text(
        "\n".join(japanese + ending) + "\n", encoding="utf-8"
    )


def save_figure(values: list[dict[str, float]], language: str) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    ranks = np.asarray([value["student_rank"] for value in values])
    distortion = 100.0 * np.asarray(
        [value["exact_local_distortion"] for value in values]
    )
    output_error = 100.0 * np.asarray(
        [value["exact_uniform_output_error"] for value in values]
    )
    threshold_blocks = minimum_blocks(PRIMARY_TOLERANCE)
    threshold_rank = 4 * threshold_blocks
    failed_rank = threshold_rank - 4

    dictionary = dct_dictionary()
    teacher_image = decoded_image(dictionary, TEACHER_BLOCKS)
    failed_image = decoded_image(dictionary, threshold_blocks - 1)
    passed_image = decoded_image(dictionary, threshold_blocks)

    figure = plt.figure(figsize=(12.4, 3.45))
    grid = figure.add_gridspec(1, 4, width_ratios=[1.75, 1.0, 1.0, 1.0])
    axis = figure.add_subplot(grid[0, 0])
    axis.loglog(
        ranks,
        distortion,
        color="#286f9e",
        lw=2.3,
        label="距離歪み" if japanese else "distance distortion",
    )
    axis.loglog(
        ranks,
        output_error,
        color="#8b5a9f",
        lw=2.0,
        ls="-.",
        label=r"出力誤差 $E_0$" if japanese else r"output error $E_0$",
    )
    axis.axhline(5.0, color="#333333", ls="--", lw=1.2)
    axis.axvline(threshold_rank, color="#298653", ls=":", lw=1.5)
    axis.scatter(
        [failed_rank, threshold_rank],
        [
            100.0 * exact_distortion(threshold_blocks - 1),
            100.0 * exact_distortion(threshold_blocks),
        ],
        c=["#c9342f", "#298653"],
        s=55,
        zorder=4,
    )
    axis.annotate(
        f"rank {failed_rank}: {100.0 * exact_distortion(threshold_blocks - 1):.3f}%",
        (failed_rank, 100.0 * exact_distortion(threshold_blocks - 1)),
        xytext=(-5, 22),
        textcoords="offset points",
        fontsize=8,
        ha="right",
        color="#a52a25",
    )
    axis.annotate(
        f"rank {threshold_rank}: {100.0 * exact_distortion(threshold_blocks):.3f}%",
        (threshold_rank, 100.0 * exact_distortion(threshold_blocks)),
        xytext=(8, -26),
        textcoords="offset points",
        fontsize=8,
        color="#226d43",
    )
    axis.set_xlabel("学生headのrank" if japanese else "student-head rank")
    axis.set_ylabel(
        "出力誤差・距離歪み（%）"
        if japanese
        else "output error and distance distortion (%)"
    )
    axis.set_title("5%境界はrank 20と24の間" if japanese else "the 5% boundary lies between ranks 20 and 24")
    axis.grid(alpha=0.18, which="both")
    axis.legend(frameon=False, fontsize=7, loc="lower left")

    image_specs = (
        (teacher_image, "教師 rank 512" if japanese else "teacher rank 512", None),
        (
            failed_image,
            f"学生 rank {failed_rank}\n不合格：D={100.0 * exact_distortion(threshold_blocks - 1):.3f}%, "
            f"E0={100.0 * exact_output_error(threshold_blocks - 1):.3f}%"
            if japanese
            else f"student rank {failed_rank}\nfail: D={100.0 * exact_distortion(threshold_blocks - 1):.3f}%, "
            f"E0={100.0 * exact_output_error(threshold_blocks - 1):.3f}%",
            "#c9342f",
        ),
        (
            passed_image,
            f"学生 rank {threshold_rank}\n合格：D={100.0 * exact_distortion(threshold_blocks):.3f}%, "
            f"E0={100.0 * exact_output_error(threshold_blocks):.3f}%"
            if japanese
            else f"student rank {threshold_rank}\npass: D={100.0 * exact_distortion(threshold_blocks):.3f}%, "
            f"E0={100.0 * exact_output_error(threshold_blocks):.3f}%",
            "#298653",
        ),
    )
    display_min = min(float(image.min()) for image, _, _ in image_specs)
    display_max = max(float(image.max()) for image, _, _ in image_specs)
    for column, (image, title, colour) in enumerate(image_specs, start=1):
        image_axis = figure.add_subplot(grid[0, column])
        image_axis.imshow(
            image,
            cmap="gray",
            vmin=display_min,
            vmax=display_max,
            interpolation="nearest",
        )
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        image_axis.set_title(title, fontsize=9, color=colour or "#202020")
        if colour is not None:
            for spine in image_axis.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(2.2)
            residual_axis = image_axis.inset_axes([0.04, 0.04, 0.38, 0.38])
            residual = np.abs(image - teacher_image)
            residual_axis.imshow(
                residual,
                cmap="magma",
                vmin=0.0,
                vmax=max(
                    float(np.abs(failed_image - teacher_image).max()),
                    1.0e-12,
                ),
                interpolation="nearest",
            )
            residual_axis.set_xticks([])
            residual_axis.set_yticks([])
            residual_axis.set_title(
                "差分" if japanese else "absolute error", fontsize=6, color="white"
            )

    figure.tight_layout()
    figure.savefig(
        HERE / f"lowrank_torus{suffix}.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def main() -> None:
    values = rows()
    write_csv(values)
    write_table()
    save_figure(values, "en")
    save_figure(values, "ja")
    blocks = minimum_blocks(PRIMARY_TOLERANCE)
    failed = blocks - 1
    teacher_parameters = teacher_head_parameters()
    passed_parameters = head_parameters(4 * blocks)
    print(
        f"rank {4 * failed}: {100.0 * exact_distortion(failed):.6f}% (fail)"
    )
    print(f"  output error: {100.0 * exact_output_error(failed):.6f}%")
    print(
        f"rank {4 * blocks}: {100.0 * exact_distortion(blocks):.6f}% (pass)"
    )
    print(f"  output error: {100.0 * exact_output_error(blocks):.6f}%")
    print(f"rank 23 optimal output RMS lower bound: {100.0 * optimal_output_rms(23):.6f}%")
    print(
        f"head parameters {teacher_parameters:,} -> {passed_parameters:,} "
        f"({teacher_parameters / passed_parameters:.3f}x)"
    )


if __name__ == "__main__":
    main()
