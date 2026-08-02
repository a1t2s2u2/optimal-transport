#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.5.1",
#   "POT==0.9.7.post1",
#   "scipy==1.18.0",
#   "torch==2.13.0",
#   "torchvision==0.28.0",
# ]
# ///
"""Correct a two-dimensional VAE chart with its local Wasserstein metric.

This is the paper's constructive experiment.  It deliberately separates four
objects that were conflated in the earlier visualization:

* actual MNIST posterior points provide the chart support;
* exact local pixel-W2 values provide the intrinsic ruler;
* an ambient invertible neural flow maps the chart into R3; and
* minimal bending selects one visible representative without claiming that it
  is a unique hidden ``MNIST shape``.

The learned surface is not restricted to ``(z1, z2, height)``.  It is the
restriction of a diffeomorphism of R3 to a lifted plane, so all three output
coordinates move while the continuous map remains injective.
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import asdict
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from ot_surface_data import load_or_build_mnist_actual_posterior_surface
from ot_surface_figure import OTSurfaceFigureData, save_bilingual_ot_surface_figures
from ot_surface_flow import SurfaceFlowConfig, evaluate_surface, fit_surface_flow

HERE = Path(__file__).resolve().parent
RESULTS_PATH = HERE / "ot_isometric_visualization_results.csv"
CONTROL_RESULTS_PATH = HERE / "ot_isometric_control_results.csv"
HISTORY_PATH = HERE / "ot_isometric_visualization_history.csv"
TABLE_PATH = HERE / "ot_isometric_visualization_table.tex"
TABLE_JA_PATH = HERE / "ot_isometric_visualization_table_ja.tex"
FIGURE_STEM = HERE / "ot_isometric_mnist"
CONTROL_FIGURE_STEM = HERE / "ot_isometric_control"


def face_edge_lengths(
    faces: np.ndarray, edges: np.ndarray, edge_lengths: np.ndarray
) -> np.ndarray:
    """Return per-face lengths in the order ``(l01,l12,l20)``."""

    lookup = {
        tuple(edge): float(length)
        for edge, length in zip(np.asarray(edges).tolist(), edge_lengths)
    }

    def length(first: int, second: int) -> float:
        return lookup[tuple(sorted((int(first), int(second))))]

    result = np.empty((len(faces), 3), dtype=np.float64)
    for index, (first, second, third) in enumerate(np.asarray(faces)):
        result[index] = (
            length(first, second),
            length(second, third),
            length(third, first),
        )
    return result


def pair_distances(coordinates: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    differences = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    return np.linalg.norm(differences, axis=1)


def optimal_global_scale(
    coordinates: np.ndarray, pairs: np.ndarray, target_distances: np.ndarray
) -> float:
    """Fit one positive scale under relative squared distance loss."""

    base = pair_distances(coordinates, pairs)
    ratios = base / np.maximum(target_distances, 1.0e-12)
    denominator = float(np.dot(ratios, ratios))
    if denominator <= 1.0e-12:
        raise ValueError("the baseline chart has no nonzero queried distance")
    return float(np.sum(ratios) / denominator)


def relative_rms(
    coordinates: np.ndarray, pairs: np.ndarray, target_distances: np.ndarray
) -> float:
    estimated = pair_distances(coordinates, pairs)
    relative = (estimated - target_distances) / np.maximum(target_distances, 1.0e-12)
    return float(np.sqrt(np.mean(relative * relative)))


def query_masks(
    edges: np.ndarray, query_pairs: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    edge_set = {tuple(edge) for edge in np.asarray(edges).tolist()}
    edge_mask = np.asarray(
        [tuple(pair) in edge_set for pair in np.asarray(query_pairs).tolist()],
        dtype=bool,
    )
    return edge_mask, ~edge_mask


def checkerboard_mesh(side: int = 15) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    horizontal, vertical = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack([horizontal.ravel(), vertical.ravel()])
    faces: list[tuple[int, int, int]] = []
    for row in range(side - 1):
        for column in range(side - 1):
            lower_left = row * side + column
            lower_right = lower_left + 1
            upper_left = lower_left + side
            upper_right = upper_left + 1
            if (row + column) % 2 == 0:
                faces.extend(
                    [
                        (lower_left, lower_right, upper_right),
                        (lower_left, upper_right, upper_left),
                    ]
                )
            else:
                faces.extend(
                    [
                        (lower_left, lower_right, upper_left),
                        (lower_right, upper_right, upper_left),
                    ]
                )
    return coordinates, np.asarray(faces, dtype=np.int64)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def controlled_problem() -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Return a warped chart and an exactly known Gaussian-W2 surface."""

    parameter, faces = checkerboard_mesh()
    horizontal = parameter[:, 0]
    vertical = parameter[:, 1]
    chart = np.column_stack(
        [
            horizontal + 0.28 * np.sin(1.25 * vertical) + 0.05 * horizontal * vertical,
            vertical + 0.20 * np.sin(1.10 * horizontal) - 0.04 * horizontal**2,
        ]
    )
    first = chart[faces[:, 1]] - chart[faces[:, 0]]
    second = chart[faces[:, 2]] - chart[faces[:, 0]]
    signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if np.any(signed_area <= 0.0):
        raise AssertionError("the controlled chart warp reversed a face")

    height = (
        0.46 * np.exp(-5.6 * ((horizontal + 0.38) ** 2 + 1.1 * (vertical - 0.24) ** 2))
        - 0.30
        * np.exp(-6.2 * (1.2 * (horizontal - 0.40) ** 2 + (vertical + 0.32) ** 2))
        + 0.24 * horizontal * vertical
    )
    graph = np.column_stack([horizontal, vertical, height])
    angle_y = math.radians(38.0)
    angle_x = math.radians(-24.0)
    rotation_y = np.asarray(
        [
            [math.cos(angle_y), 0.0, math.sin(angle_y)],
            [0.0, 1.0, 0.0],
            [-math.sin(angle_y), 0.0, math.cos(angle_y)],
        ]
    )
    rotation_x = np.asarray(
        [
            [1.0, 0.0, 0.0],
            [0.0, math.cos(angle_x), -math.sin(angle_x)],
            [0.0, math.sin(angle_x), math.cos(angle_x)],
        ]
    )
    truth = graph @ (rotation_x @ rotation_y).T
    return chart, faces, truth, height


def coordinate_face_lengths(coordinates: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.linalg.norm(coordinates[faces[:, 1]] - coordinates[faces[:, 0]], axis=1),
            np.linalg.norm(coordinates[faces[:, 2]] - coordinates[faces[:, 1]], axis=1),
            np.linalg.norm(coordinates[faces[:, 0]] - coordinates[faces[:, 2]], axis=1),
        ]
    )


def rigid_alignment(
    coordinates: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    centered = coordinates - np.mean(coordinates, axis=0)
    target = reference - np.mean(reference, axis=0)
    left, _, right = np.linalg.svd(centered.T @ target)
    rotation = left @ right
    if np.linalg.det(rotation) < 0.0:
        left[:, -1] *= -1.0
        rotation = left @ right
    aligned = centered @ rotation + np.mean(reference, axis=0)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return aligned, rmse


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


def add_surface(
    axis, coordinates: np.ndarray, faces: np.ndarray, values: np.ndarray
) -> None:
    normalization = mpl.colors.Normalize(
        vmin=float(np.min(values)), vmax=float(np.max(values))
    )
    face_values = np.mean(values[faces], axis=1)
    collection = Poly3DCollection(
        coordinates[faces],
        facecolors=mpl.colormaps["coolwarm"](normalization(face_values)),
        edgecolors=(0.08, 0.08, 0.08, 0.18),
        linewidths=0.22,
        alpha=0.96,
    )
    axis.add_collection3d(collection)
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    span = np.maximum(maximum - minimum, 0.08 * np.max(maximum - minimum))
    axis.set_xlim(minimum[0], maximum[0])
    axis.set_ylim(minimum[1], maximum[1])
    axis.set_zlim(minimum[2], maximum[2])
    axis.set_box_aspect(span, zoom=1.15)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(False)


def save_control_figure(
    chart: np.ndarray,
    faces: np.ndarray,
    truth: np.ndarray,
    corrected: np.ndarray,
    height: np.ndarray,
    metric_before: float,
    metric_after: float,
    aligned_rmse: float,
    japanese: bool,
) -> None:
    suffix = "_ja" if japanese else ""
    font = {"font.family": japanese_font_family()} if japanese else {}
    with mpl.rc_context(font):
        figure = plt.figure(figsize=(12.6, 4.65))
        grid = figure.add_gridspec(1, 3, wspace=0.05)
        flat_axis = figure.add_subplot(grid[0, 0])
        truth_axis = figure.add_subplot(grid[0, 1], projection="3d")
        output_axis = figure.add_subplot(grid[0, 2], projection="3d")
        flat_axis.tripcolor(
            chart[:, 0],
            chart[:, 1],
            faces,
            height,
            cmap="coolwarm",
            shading="gouraud",
        )
        flat_axis.triplot(
            chart[:, 0], chart[:, 1], faces, color=(0.1, 0.1, 0.1, 0.22), linewidth=0.25
        )
        flat_axis.set_aspect("equal", adjustable="box")
        flat_axis.set_xticks([])
        flat_axis.set_yticks([])
        flat_axis.set_title(
            "(a) 歪んだ2次元座標\n入力は座標と局所Gaussian W2のみ"
            if japanese
            else "(a) Distorted 2D coordinates\ninput: chart plus local Gaussian W2 only",
            fontsize=10,
        )
        add_surface(truth_axis, truth, faces, height)
        truth_axis.view_init(elev=27, azim=-60)
        truth_axis.set_title(
            "(b) 隠した正解曲面\n適合には使用しない"
            if japanese
            else "(b) Hidden reference surface\nnever used for fitting",
            fontsize=10,
        )
        add_surface(output_axis, corrected, faces, height)
        output_axis.view_init(elev=27, azim=-60)
        output_axis.set_title(
            "(c) OT計量から補正した曲面\n全x,y,zを学習"
            if japanese
            else "(c) Surface corrected from OT metric\nall x, y, z coordinates learned",
            fontsize=10,
        )
        figure.suptitle(
            "既知チェック：歪んだ地図のOT計量を、一定曲率を仮定せず曲面化"
            if japanese
            else "Known check: realize a warped map's OT metric without assuming constant curvature",
            fontsize=12.5,
            y=0.975,
        )
        figure.text(
            0.5,
            0.895,
            (
                f"計量偏差 {100.0 * metric_before:.1f}% → {100.0 * metric_after:.1f}%"
                if japanese
                else f"metric deviation {100.0 * metric_before:.1f}% -> {100.0 * metric_after:.1f}%"
            ),
            ha="center",
            va="center",
            fontsize=10,
        )
        figure.text(
            0.985,
            0.02,
            (
                f"参考：剛体整列RMSE={aligned_rmse:.3f}"
                if japanese
                else f"reference: rigid-aligned RMSE={aligned_rmse:.3f}"
            ),
            ha="right",
            va="bottom",
            fontsize=7.5,
            color="#555555",
        )
        figure.subplots_adjust(left=0.025, right=0.985, bottom=0.035, top=0.78)
        base = CONTROL_FIGURE_STEM.with_name(CONTROL_FIGURE_STEM.name + suffix)
        figure.savefig(base.with_suffix(".png"), dpi=220, bbox_inches="tight")
        figure.savefig(
            base.with_suffix(".pdf"),
            bbox_inches="tight",
            metadata={"Creator": "ot_isometric_visualization.py", "CreationDate": None},
        )
        plt.close(figure)


def run_controlled_experiment():
    chart, faces, truth, height = controlled_problem()
    targets = coordinate_face_lengths(truth, faces)
    config = SurfaceFlowConfig(
        seed=1701,
        hidden_width=40,
        coupling_blocks=12,
        metric_smoothing_strength=0.0,
        stage_one_steps=1_200,
        stage_two_steps=700,
        stage_one_learning_rate=4.0e-3,
        stage_two_learning_rate=1.5e-3,
        metric_relative_slack=0.08,
        metric_constraint_weight=160.0,
        record_every=20,
    )
    result = fit_surface_flow(chart, faces, targets, config=config)
    aligned, aligned_rmse = rigid_alignment(result.coordinates, truth)
    before = math.expm1(math.sqrt(result.initial_isometry_loss))
    after = math.expm1(math.sqrt(result.final_isometry_loss))
    reference_scale = float(
        np.sqrt(np.mean(np.sum((truth - np.mean(truth, axis=0)) ** 2, axis=1)))
    )
    with CONTROL_RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "vertex_count",
                "face_count",
                "initial_metric_deviation",
                "corrected_metric_deviation",
                "minimum_generalized_stretch",
                "maximum_generalized_stretch",
                "rigid_aligned_rmse",
                "relative_rigid_aligned_rmse",
            ),
        )
        writer.writeheader()
        writer.writerow(
            {
                "vertex_count": len(chart),
                "face_count": len(faces),
                "initial_metric_deviation": before,
                "corrected_metric_deviation": after,
                "minimum_generalized_stretch": result.generalized_stretch_min,
                "maximum_generalized_stretch": result.generalized_stretch_max,
                "rigid_aligned_rmse": aligned_rmse,
                "relative_rigid_aligned_rmse": aligned_rmse / reference_scale,
            }
        )
    for japanese in (False, True):
        save_control_figure(
            chart,
            faces,
            truth,
            aligned,
            height,
            before,
            after,
            aligned_rmse,
            japanese,
        )
    print(
        "controlled OT correction: "
        f"metric deviation {100.0 * before:.2f}% -> {100.0 * after:.2f}%, "
        f"aligned RMSE={aligned_rmse:.4f}",
        flush=True,
    )
    return result, aligned_rmse


def write_history(result) -> None:
    with HISTORY_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=(
                "stage",
                "step",
                "objective",
                "isometry_loss",
                "bending_loss",
                "metric_feasible",
            ),
        )
        writer.writeheader()
        for record in result.history:
            writer.writerow(asdict(record))


def write_results(rows: list[dict[str, float | str | int]]) -> None:
    with RESULTS_PATH.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_table(rows: list[dict[str, float | str | int]], japanese: bool) -> None:
    destination = TABLE_JA_PATH if japanese else TABLE_PATH
    labels = {
        "original_latent": "元の潜在平面" if japanese else "original latent plane",
        "ot_corrected": "OT補正曲面" if japanese else "OT-corrected surface",
    }
    lines = [
        r"\begin{tabular}{lrrr}",
        r"\toprule",
        (
            r"表現 & 局所計量偏差 & 辺W2 RMS & 構造 \\"
            if japanese
            else r"Representation & local metric dev. & edge $W_2$ RMS & structure \\"
        ),
        r"\midrule",
    ]
    for row in rows:
        label = labels[str(row["method"])]
        structure = (
            ("平面" if japanese else "plane")
            if row["method"] == "original_latent"
            else ("単射曲面" if japanese else "injective surface")
        )
        lines.append(
            f"{label} & "
            f"{100.0 * float(row['characteristic_metric_deviation']):.1f}\\% & "
            f"{100.0 * float(row['edge_relative_rms']):.1f}\\% & "
            f"{structure} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    destination.write_text("\n".join(lines) + "\n")


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group()
    selection.add_argument(
        "--control-only", action="store_true", help="run only the known control"
    )
    selection.add_argument(
        "--mnist-only", action="store_true", help="run only the MNIST correction"
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if not arguments.mnist_only:
        run_controlled_experiment()
    if arguments.control_only:
        return
    data = load_or_build_mnist_actual_posterior_surface()
    target_faces = face_edge_lengths(data.faces, data.edges, data.edge_distances)
    config = SurfaceFlowConfig(
        seed=20260802,
        hidden_width=48,
        coupling_blocks=12,
        metric_smoothing_strength=3.0,
        stage_one_steps=1_400,
        stage_two_steps=900,
        stage_one_learning_rate=4.0e-3,
        stage_two_learning_rate=1.5e-3,
        metric_relative_slack=0.10,
        metric_constraint_weight=160.0,
        record_every=20,
    )
    print(
        "fitting topology-preserving OT surface: "
        f"vertices={data.vertex_count}, faces={len(data.faces)}",
        flush=True,
    )
    result = fit_surface_flow(
        data.chart_coordinates,
        data.faces,
        target_faces,
        config=config,
    )

    edge_mask, two_hop_mask = query_masks(data.edges, data.query_pairs)
    if not np.array_equal(data.query_distances[edge_mask], data.edge_distances):
        raise AssertionError("query ordering no longer matches the cached edge set")
    baseline_scale = optimal_global_scale(
        data.chart_coordinates, data.edges, data.edge_distances
    )
    baseline = baseline_scale * data.chart_coordinates

    initial_deviation = math.expm1(math.sqrt(result.initial_isometry_loss))
    final_deviation = math.expm1(math.sqrt(result.final_isometry_loss))
    rows: list[dict[str, float | str | int]] = [
        {
            "method": "original_latent",
            "vertex_count": data.vertex_count,
            "face_count": len(data.faces),
            "characteristic_metric_deviation": initial_deviation,
            "edge_relative_rms": relative_rms(
                baseline, data.edges, data.edge_distances
            ),
            "two_hop_relative_rms": relative_rms(
                baseline,
                data.query_pairs[two_hop_mask],
                data.query_distances[two_hop_mask],
            ),
            "isometry_loss": result.initial_isometry_loss,
            "bending_loss": 0.0,
            "minimum_generalized_stretch": math.nan,
            "maximum_generalized_stretch": math.nan,
            "metric_smoothing_strength": result.metric_smoothing_strength,
            "raw_metric_log_roughness": result.raw_metric_log_roughness,
            "smoothed_metric_log_roughness": result.smoothed_metric_log_roughness,
        },
        {
            "method": "ot_corrected",
            "vertex_count": data.vertex_count,
            "face_count": len(data.faces),
            "characteristic_metric_deviation": final_deviation,
            "edge_relative_rms": relative_rms(
                result.coordinates, data.edges, data.edge_distances
            ),
            "two_hop_relative_rms": relative_rms(
                result.coordinates,
                data.query_pairs[two_hop_mask],
                data.query_distances[two_hop_mask],
            ),
            "isometry_loss": result.final_isometry_loss,
            "bending_loss": result.final_bending_loss,
            "minimum_generalized_stretch": result.generalized_stretch_min,
            "maximum_generalized_stretch": result.generalized_stretch_max,
            "metric_smoothing_strength": result.metric_smoothing_strength,
            "raw_metric_log_roughness": result.raw_metric_log_roughness,
            "smoothed_metric_log_roughness": result.smoothed_metric_log_roughness,
        },
    ]
    write_results(rows)
    write_history(result)
    write_table(rows, japanese=False)
    write_table(rows, japanese=True)

    figure_data = OTSurfaceFigureData(
        posterior_points=data.latent_codes,
        posterior_labels=data.source_labels,
        chart_points=data.latent_codes,
        faces=data.faces,
        edges=data.edges,
        w2_edge_lengths=data.edge_distances,
        corrected_points=result.coordinates,
        decoded_images=data.decoded_images,
        focus_label=data.digit,
        source_images=data.source_images,
        surface_evaluator=lambda raw_points: evaluate_surface(
            result.model,
            (raw_points - data.whitening_center) @ data.whitening_matrix,
        ),
    )
    save_bilingual_ot_surface_figures(
        figure_data,
        FIGURE_STEM,
        ellipse_count=9,
        distance_band_count=7,
        image_count=7,
    )

    print(
        "OT surface fitted: "
        f"metric deviation {100.0 * initial_deviation:.2f}% -> "
        f"{100.0 * final_deviation:.2f}%; "
        f"edge RMS {100.0 * float(rows[0]['edge_relative_rms']):.2f}% -> "
        f"{100.0 * float(rows[1]['edge_relative_rms']):.2f}%; "
        f"stretch=[{result.generalized_stretch_min:.3f}, "
        f"{result.generalized_stretch_max:.3f}]",
        flush=True,
    )
    print(f"wrote {FIGURE_STEM.name} bilingual figures and diagnostics", flush=True)


if __name__ == "__main__":
    main()
