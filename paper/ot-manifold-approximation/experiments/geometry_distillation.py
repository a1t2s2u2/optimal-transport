#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "scipy>=1.14",
#   "torch>=2.4",
# ]
# ///
"""Capacity sweep for geometry-preserving Gaussian decoder distillation.

The teacher is a diagonal-Gaussian decoder whose parameter surface is a flat
two-torus.  Its intrinsic distances are known exactly.  A second teacher uses
a smooth, high-frequency reparametrisation of the same torus, so that output
accuracy and first-derivative accuracy separate cleanly.

For each student width and objective, the script measures the quantities in
Theorem 3.1 of the paper on a dense, fixed evaluation grid:

    epsilon_1 = max ||J_student - J_teacher||_op,
    delta      = (2 M_1 epsilon_1 + epsilon_1^2) / s^2.

It then compares the resulting metric, all-pairs-distance, and triplet-order
bounds with the observed errors.  ``delta_cert`` in the CSV is a plug-in
certificate on the declared evaluation grid; a continuum certificate would
add a covering/Lipschitz remainder and is deliberately not claimed here.

Run from the repository root with:

    uv run --python 3.12 \
      paper/ot-manifold-approximation/experiments/geometry_distillation.py

Use ``--quick`` only for a smoke test; paper artefacts use the defaults.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from torch import Tensor, nn
from torch.func import jacrev, vmap

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
SEED = 20260731
MEAN_SCALE = 0.9
STD_SCALE = math.sqrt(1.0 - MEAN_SCALE**2)
STD_OFFSET = 0.75
MIN_STD = 0.05
# The stress test deliberately separates a value requirement that the largest
# output-only model meets from the stricter geometry requirement it violates.
OUTPUT_RMSE_TOLERANCE = 0.02
DISTANCE_DISTORTION_TOLERANCE = 0.05
JACOBIAN_LOSS_WEIGHT = 5.0


@dataclass(frozen=True)
class TeacherSpec:
    key: str
    label: str
    label_ja: str
    frequency: int
    derivative_amplitude: float

    @property
    def warp_amplitude(self) -> float:
        if self.frequency == 0:
            return 0.0
        return self.derivative_amplitude / self.frequency

    @property
    def lower_singular_value(self) -> float:
        # J_phi = I + [[0, c_2], [c_1, 0]], |c_i| <= c.
        return 1.0 - self.derivative_amplitude

    @property
    def upper_jacobian_norm(self) -> float:
        return 1.0 + self.derivative_amplitude


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    label_ja: str
    use_jacobian_loss: bool


TEACHERS = (
    TeacherSpec("plain", "plain torus", "単純トーラス", 0, 0.0),
    TeacherSpec("warped", "frequency-4 warped torus", "周波数4の変形トーラス", 4, 0.20),
)

METHODS = (
    MethodSpec("value", "output only", "出力のみ", False),
    MethodSpec("c1", "output + Jacobian", "出力 + Jacobian", True),
)


class StudentDecoder(nn.Module):
    """Smooth periodic student Gaussian decoder."""

    def __init__(self, width: int) -> None:
        super().__init__()
        self.width = width
        self.network = nn.Sequential(
            nn.Linear(4, width),
            nn.Tanh(),
            nn.Linear(width, width),
            nn.Tanh(),
            nn.Linear(width, 8),
        )
        self.double()
        with torch.no_grad():
            final = self.network[-1]
            assert isinstance(final, nn.Linear)
            final.bias[4:] = STD_OFFSET

    @staticmethod
    def features(z: Tensor) -> Tensor:
        return torch.stack(
            [torch.cos(z[..., 0]), torch.sin(z[..., 0]),
             torch.cos(z[..., 1]), torch.sin(z[..., 1])],
            dim=-1,
        )

    def forward(self, z: Tensor) -> Tensor:
        return self.network(self.features(z))

    @property
    def parameter_count(self) -> int:
        return sum(parameter.numel() for parameter in self.parameters())


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--quick", action="store_true", help="small smoke test")
    parser.add_argument("--steps", type=int, default=None)
    parser.add_argument("--repeats", type=int, default=None)
    parser.add_argument("--widths", type=str, default=None,
                        help="comma-separated hidden widths")
    parser.add_argument("--teachers", type=str, default=None,
                        help="comma-separated teacher keys")
    parser.add_argument("--methods", type=str, default=None,
                        help="comma-separated method keys")
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def wrap_angle(value: Tensor | np.ndarray) -> Tensor | np.ndarray:
    return (value + math.pi) % (2.0 * math.pi) - math.pi


def teacher_angles(z: Tensor, spec: TeacherSpec) -> Tensor:
    if spec.frequency == 0:
        return z
    amplitude = spec.warp_amplitude
    first = z[..., 0] + amplitude * torch.sin(spec.frequency * z[..., 1])
    second = z[..., 1] + amplitude * torch.sin(spec.frequency * z[..., 0])
    return torch.stack([first, second], dim=-1)


def teacher_parameters(z: Tensor, spec: TeacherSpec) -> Tensor:
    phi = teacher_angles(z, spec)
    embedding = torch.stack(
        [torch.cos(phi[..., 0]), torch.sin(phi[..., 0]),
         torch.cos(phi[..., 1]), torch.sin(phi[..., 1])],
        dim=-1,
    )
    mean = MEAN_SCALE * embedding
    standard_deviation = STD_OFFSET + STD_SCALE * embedding
    return torch.cat([mean, standard_deviation], dim=-1)


def teacher_jacobian(z: Tensor, spec: TeacherSpec) -> Tensor:
    """Exact Jacobian of the teacher with shape (..., 8, 2)."""
    phi = teacher_angles(z, spec)
    if spec.frequency == 0:
        cross_first = torch.zeros_like(z[..., 0])
        cross_second = torch.zeros_like(z[..., 0])
    else:
        cross_first = spec.derivative_amplitude * torch.cos(
            spec.frequency * z[..., 1]
        )
        cross_second = spec.derivative_amplitude * torch.cos(
            spec.frequency * z[..., 0]
        )

    zero = torch.zeros_like(z[..., 0])
    d_embedding_d_phi_first = torch.stack(
        [-torch.sin(phi[..., 0]), torch.cos(phi[..., 0]), zero, zero], dim=-1
    )
    d_embedding_d_phi_second = torch.stack(
        [zero, zero, -torch.sin(phi[..., 1]), torch.cos(phi[..., 1])], dim=-1
    )
    d_embedding_d_z_first = (
        d_embedding_d_phi_first
        + cross_second[..., None] * d_embedding_d_phi_second
    )
    d_embedding_d_z_second = (
        cross_first[..., None] * d_embedding_d_phi_first
        + d_embedding_d_phi_second
    )
    embedding_jacobian = torch.stack(
        [d_embedding_d_z_first, d_embedding_d_z_second], dim=-1
    )
    scales = torch.tensor(
        [MEAN_SCALE] * 4 + [STD_SCALE] * 4,
        dtype=z.dtype,
        device=z.device,
    )
    doubled = torch.cat([embedding_jacobian, embedding_jacobian], dim=-2)
    return scales[..., None] * doubled


def student_jacobian(model: StudentDecoder, z: Tensor) -> Tensor:
    def single(point: Tensor) -> Tensor:
        return model(point.unsqueeze(0)).squeeze(0)

    return vmap(jacrev(single))(z)


def sample_latents(count: int, generator: torch.Generator) -> Tensor:
    return (
        2.0 * math.pi * torch.rand((count, 2), generator=generator, dtype=torch.float64)
        - math.pi
    )


def train_student(
    teacher: TeacherSpec,
    method: MethodSpec,
    width: int,
    seed: int,
    steps: int,
    batch_size: int,
) -> tuple[StudentDecoder, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = StudentDecoder(width)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=3.0e-3, weight_decay=1.0e-7
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=steps, eta_min=2.0e-4
    )
    generator = torch.Generator().manual_seed(
        seed + 1009 * teacher.frequency + (17 if method.use_jacobian_loss else 0)
    )
    history: list[dict[str, float]] = []

    for step in range(1, steps + 1):
        z = sample_latents(batch_size, generator)
        target = teacher_parameters(z, teacher)
        predicted = model(z)
        value_loss = (predicted - target).square().mean()
        positivity_penalty = torch.relu(MIN_STD - predicted[:, 4:]).square().mean()
        jacobian_loss = torch.zeros((), dtype=torch.float64)
        if method.use_jacobian_loss:
            predicted_jacobian = student_jacobian(model, z)
            target_jacobian = teacher_jacobian(z, teacher)
            jacobian_loss = (predicted_jacobian - target_jacobian).square().mean()
        loss = (
            value_loss
            + JACOBIAN_LOSS_WEIGHT * jacobian_loss
            + 100.0 * positivity_penalty
        )
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        scheduler.step()

        if step == 1 or step % 100 == 0 or step == steps:
            history.append(
                {
                    "step": float(step),
                    "value_loss": float(value_loss.detach()),
                    "jacobian_loss": float(jacobian_loss.detach()),
                    "objective": float(loss.detach()),
                }
            )
    return model, history


def evaluation_grid(size: int) -> tuple[np.ndarray, Tensor]:
    values = np.linspace(-math.pi, math.pi, size, endpoint=False, dtype=np.float64)
    first, second = np.meshgrid(values, values, indexing="ij")
    coordinates = np.column_stack([first.ravel(), second.ravel()])
    return values, torch.as_tensor(coordinates, dtype=torch.float64)


def metric(jacobian: np.ndarray) -> np.ndarray:
    return np.einsum("noi,noj->nij", jacobian, jacobian)


def operator_norm(matrices: np.ndarray) -> np.ndarray:
    return np.linalg.svd(matrices, compute_uv=False)[..., 0]


def relative_metric_distortion(
    teacher_metric: np.ndarray, student_metric: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues = np.empty((len(teacher_metric), 2), dtype=np.float64)
    for index, (target, predicted) in enumerate(zip(teacher_metric, student_metric)):
        values = np.linalg.eigvals(np.linalg.solve(target, predicted)).real
        eigenvalues[index] = np.sort(values)
    squared_length_distortion = np.max(np.abs(eigenvalues - 1.0), axis=1)
    length_distortion = np.max(np.abs(np.sqrt(np.maximum(eigenvalues, 0.0)) - 1.0), axis=1)
    return squared_length_distortion, length_distortion


def torus_distances(phi: np.ndarray, sources: np.ndarray, targets: np.ndarray) -> np.ndarray:
    difference = wrap_angle(phi[sources] - phi[targets])
    assert isinstance(difference, np.ndarray)
    return np.linalg.norm(difference, axis=1)


def build_local_graph(parameters: np.ndarray, grid_size: int) -> coo_matrix:
    rows: list[int] = []
    columns: list[int] = []
    weights: list[float] = []
    offsets = [
        (first, second)
        for first in range(-2, 3)
        for second in range(-2, 3)
        if (first, second) != (0, 0)
    ]
    for first in range(grid_size):
        for second in range(grid_size):
            source = first * grid_size + second
            for delta_first, delta_second in offsets:
                target_first = (first + delta_first) % grid_size
                target_second = (second + delta_second) % grid_size
                target = target_first * grid_size + target_second
                if target <= source:
                    continue
                weight = float(np.linalg.norm(parameters[source] - parameters[target]))
                rows.extend([source, target])
                columns.extend([target, source])
                weights.extend([weight, weight])
    count = grid_size * grid_size
    return coo_matrix((weights, (rows, columns)), shape=(count, count))


def make_pair_and_triplet_indices(
    grid_size: int,
    rng: np.random.Generator,
    anchor_count: int,
    targets_per_anchor: int,
    teacher_phi: np.ndarray,
) -> dict[str, np.ndarray]:
    count = grid_size * grid_size
    anchors = rng.choice(count, size=anchor_count, replace=False)
    pair_sources = np.repeat(anchors, targets_per_anchor)
    pair_targets = rng.integers(0, count, size=len(pair_sources))
    collision = pair_sources == pair_targets
    while np.any(collision):
        pair_targets[collision] = rng.integers(0, count, size=int(collision.sum()))
        collision = pair_sources == pair_targets

    # Half of the triplets deliberately have a small teacher margin.  Purely
    # random triplets are almost always easy and hide geometry failures behind
    # accuracies above 98%.  The other half remains unconditioned, so the
    # theorem's margin-dependent certified lower bound is still informative.
    triplet_count = anchor_count * targets_per_anchor
    hard_count = triplet_count // 2
    pool_count = 30 * triplet_count
    pool_sources = rng.choice(anchors, size=pool_count, replace=True)
    pool_first = rng.integers(0, count, size=pool_count)
    pool_second = rng.integers(0, count, size=pool_count)
    valid = (
        (pool_first != pool_sources)
        & (pool_second != pool_sources)
        & (pool_first != pool_second)
    )
    first_distance = torus_distances(teacher_phi, pool_sources, pool_first)
    second_distance = torus_distances(teacher_phi, pool_sources, pool_second)
    gamma = np.abs(np.square(second_distance) - np.square(first_distance)) / (
        np.square(first_distance) + np.square(second_distance)
    )
    hard = np.flatnonzero(valid & (gamma > 1.0e-6) & (gamma <= 0.10))
    if len(hard) < hard_count:
        raise RuntimeError("not enough hard triplets; increase pool_count")
    hard = hard[:hard_count]
    random_pool = np.flatnonzero(valid & (gamma > 1.0e-6))
    random_choice = rng.choice(
        random_pool, size=triplet_count - hard_count, replace=False
    )
    selected = np.concatenate([hard, random_choice])
    rng.shuffle(selected)
    triplet_sources = pool_sources[selected]
    triplet_first = pool_first[selected]
    triplet_second = pool_second[selected]
    return {
        "anchors": anchors,
        "pair_sources": pair_sources,
        "pair_targets": pair_targets,
        "triplet_sources": triplet_sources,
        "triplet_first": triplet_first,
        "triplet_second": triplet_second,
    }


def theorem_distance_bound(delta: float) -> float:
    if not np.isfinite(delta) or delta >= 1.0:
        return float("inf")
    return max(1.0 - math.sqrt(1.0 - delta), math.sqrt(1.0 + delta) - 1.0)


def evaluate_student(
    model: StudentDecoder,
    teacher: TeacherSpec,
    grid_values: np.ndarray,
    grid: Tensor,
    indices: dict[str, np.ndarray],
) -> tuple[dict[str, float | str | int], dict[str, np.ndarray]]:
    with torch.no_grad():
        target_parameters_tensor = teacher_parameters(grid, teacher)
        student_parameters_tensor = model(grid)
    target_jacobian_tensor = teacher_jacobian(grid, teacher)
    student_jacobian_tensor = student_jacobian(model, grid).detach()

    target_parameters = target_parameters_tensor.cpu().numpy()
    student_parameters = student_parameters_tensor.cpu().numpy()
    target_jacobian = target_jacobian_tensor.cpu().numpy()
    predicted_jacobian = student_jacobian_tensor.cpu().numpy()
    jacobian_error = predicted_jacobian - target_jacobian
    epsilon_one = float(operator_norm(jacobian_error).max())
    q_value = (
        2.0 * teacher.upper_jacobian_norm * epsilon_one + epsilon_one**2
    )
    delta = q_value / teacher.lower_singular_value**2
    distance_bound = theorem_distance_bound(delta)

    target_metric = metric(target_jacobian)
    predicted_metric = metric(predicted_jacobian)
    metric_distortion, length_distortion = relative_metric_distortion(
        target_metric, predicted_metric
    )

    phi = teacher_angles(grid, teacher).detach().cpu().numpy()
    graph = build_local_graph(student_parameters, len(grid_values)).tocsr()
    graph_distances = dijkstra(graph, directed=False, indices=indices["anchors"])
    anchor_lookup = {anchor: row for row, anchor in enumerate(indices["anchors"])}

    pair_rows = np.asarray(
        [anchor_lookup[int(source)] for source in indices["pair_sources"]], dtype=int
    )
    student_pair_distance = graph_distances[pair_rows, indices["pair_targets"]]
    teacher_pair_distance = torus_distances(
        phi, indices["pair_sources"], indices["pair_targets"]
    )
    pair_relative_error = np.abs(student_pair_distance / teacher_pair_distance - 1.0)

    triplet_rows = np.asarray(
        [anchor_lookup[int(source)] for source in indices["triplet_sources"]], dtype=int
    )
    student_first = graph_distances[triplet_rows, indices["triplet_first"]]
    student_second = graph_distances[triplet_rows, indices["triplet_second"]]
    teacher_first = torus_distances(
        phi, indices["triplet_sources"], indices["triplet_first"]
    )
    teacher_second = torus_distances(
        phi, indices["triplet_sources"], indices["triplet_second"]
    )
    teacher_order = teacher_first < teacher_second
    student_order = student_first < student_second
    correct = teacher_order == student_order
    denominator = np.square(teacher_first) + np.square(teacher_second)
    gamma = np.abs(np.square(teacher_second) - np.square(teacher_first)) / denominator
    certified = gamma > delta

    output_rmse = float(np.sqrt(np.mean((student_parameters - target_parameters) ** 2)))
    actual_distance_max = float(pair_relative_error.max())
    actual_distance_p95 = float(np.quantile(pair_relative_error, 0.95))
    certified_all_distance_pass = (
        output_rmse <= OUTPUT_RMSE_TOLERANCE
        and distance_bound <= DISTANCE_DISTORTION_TOLERANCE
    )
    empirical_all_distance_pass = (
        output_rmse <= OUTPUT_RMSE_TOLERANCE
        and actual_distance_max <= DISTANCE_DISTORTION_TOLERANCE
    )
    metrics: dict[str, float | str | int] = {
        "parameter_count": model.parameter_count,
        "output_rmse": output_rmse,
        "minimum_student_std": float(student_parameters[:, 4:].min()),
        "epsilon1_grid_max": epsilon_one,
        "q_theory": q_value,
        "delta_cert": delta,
        "distance_distortion_bound": distance_bound,
        "metric_distortion_max": float(metric_distortion.max()),
        "metric_length_distortion_max": float(length_distortion.max()),
        "distance_distortion_max": actual_distance_max,
        "distance_distortion_p95": actual_distance_p95,
        "triplet_accuracy": float(correct.mean()),
        "hard_triplet_accuracy": float(correct[gamma <= 0.10].mean()),
        "hard_triplet_fraction": float((gamma <= 0.10).mean()),
        "certified_triplet_accuracy_lower_bound": float(certified.mean()),
        "triplet_accuracy_on_certified_subset": (
            float(correct[certified].mean()) if np.any(certified) else float("nan")
        ),
        "certified_all_distance_pass": int(certified_all_distance_pass),
        "empirical_all_distance_pass": int(empirical_all_distance_pass),
        "certificate_scope": "fixed_evaluation_grid",
    }
    diagnostics = {
        "metric_length_distortion": length_distortion.reshape(
            len(grid_values), len(grid_values)
        ),
        "teacher_pair_distance": teacher_pair_distance,
        "student_pair_distance": student_pair_distance,
        "gamma": gamma,
        "triplet_correct": correct,
    }
    return metrics, diagnostics


def mean_and_sample_std(values: Iterable[float]) -> tuple[float, float]:
    array = np.asarray(list(values), dtype=np.float64)
    return float(array.mean()), float(array.std(ddof=1)) if len(array) > 1 else 0.0


def summarize(rows: list[dict[str, float | str | int]]) -> list[dict[str, float | str | int]]:
    numeric = (
        "parameter_count", "output_rmse", "epsilon1_grid_max", "delta_cert",
        "distance_distortion_bound", "metric_distortion_max",
        "metric_length_distortion_max", "distance_distortion_max",
        "distance_distortion_p95", "triplet_accuracy",
        "hard_triplet_accuracy", "hard_triplet_fraction",
        "certified_triplet_accuracy_lower_bound",
        "triplet_accuracy_on_certified_subset",
        "certified_all_distance_pass", "empirical_all_distance_pass",
    )
    grouped: dict[tuple[str, str, int], list[dict[str, float | str | int]]] = {}
    for row in rows:
        key = (str(row["teacher"]), str(row["method"]), int(row["width"]))
        grouped.setdefault(key, []).append(row)
    output: list[dict[str, float | str | int]] = []
    for (teacher, method, width), group in sorted(grouped.items()):
        summary: dict[str, float | str | int] = {
            "teacher": teacher,
            "method": method,
            "width": width,
            "repeats": len(group),
        }
        for name in numeric:
            finite_values = [
                float(row[name]) for row in group if np.isfinite(float(row[name]))
            ]
            if not finite_values:
                summary[f"{name}_mean"] = float("nan")
                summary[f"{name}_std"] = float("nan")
                continue
            mean, standard_deviation = mean_and_sample_std(finite_values)
            summary[f"{name}_mean"] = mean
            summary[f"{name}_std"] = standard_deviation
        output.append(summary)
    return output


def threshold_rows(
    summaries: list[dict[str, float | str | int]],
) -> list[dict[str, float | str | int]]:
    output: list[dict[str, float | str | int]] = []
    for teacher in TEACHERS:
        for method in METHODS:
            candidates = [
                row for row in summaries
                if row["teacher"] == teacher.key and row["method"] == method.key
            ]
            if not candidates:
                continue
            candidates.sort(key=lambda row: int(row["parameter_count_mean"]))
            passing = [
                row for row in candidates
                if float(row["certified_all_distance_pass_mean"]) == 1.0
            ]
            selected = passing[0] if passing else candidates[-1]
            output.append(
                {
                    "teacher": teacher.key,
                    "method": method.key,
                    "status": "pass" if passing else "no_certified_pass",
                    "minimum_width": int(selected["width"]),
                    "minimum_parameter_count": int(selected["parameter_count_mean"]),
                    "output_rmse_mean": selected["output_rmse_mean"],
                    "epsilon1_grid_max_mean": selected["epsilon1_grid_max_mean"],
                    "delta_cert_mean": selected["delta_cert_mean"],
                    "distance_distortion_bound_mean": selected[
                        "distance_distortion_bound_mean"
                    ],
                    "distance_distortion_max_mean": selected[
                        "distance_distortion_max_mean"
                    ],
                    "triplet_accuracy_mean": selected["triplet_accuracy_mean"],
                    "certified_triplet_accuracy_lower_bound_mean": selected[
                        "certified_triplet_accuracy_lower_bound_mean"
                    ],
                }
            )
    return output


def write_csv(path: Path, rows: list[dict[str, object]]) -> None:
    if not rows:
        raise ValueError(f"no rows for {path}")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def japanese_font_family() -> str:
    installed = {font.name for font in font_manager.fontManager.ttflist}
    for candidate in (
        "Hiragino Sans", "Yu Gothic", "Noto Sans CJK JP", "IPAexGothic", "IPAGothic"
    ):
        if candidate in installed:
            return candidate
    return "sans-serif"


def find_summary(
    summaries: list[dict[str, float | str | int]], teacher: str, method: str
) -> list[dict[str, float | str | int]]:
    rows = [
        row for row in summaries
        if row["teacher"] == teacher and row["method"] == method
    ]
    return sorted(rows, key=lambda row: int(row["parameter_count_mean"]))


def save_tradeoff_figure(
    summaries: list[dict[str, float | str | int]], language: str
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    figure, axes = plt.subplots(2, 2, figsize=(10.8, 7.2), sharex="col")
    colours = {"value": "#c76534", "c1": "#2779ad"}
    markers = {"value": "o", "c1": "s"}
    for row_index, teacher in enumerate(TEACHERS):
        accuracy_axis = axes[row_index, 0]
        distortion_axis = axes[row_index, 1]
        for method in METHODS:
            rows = find_summary(summaries, teacher.key, method.key)
            parameters = np.asarray([row["parameter_count_mean"] for row in rows])
            actual_accuracy = 100.0 * np.asarray(
                [row["triplet_accuracy_mean"] for row in rows]
            )
            certified_accuracy = 100.0 * np.asarray(
                [row["certified_triplet_accuracy_lower_bound_mean"] for row in rows]
            )
            actual_distortion = 100.0 * np.asarray(
                [row["distance_distortion_max_mean"] for row in rows]
            )
            bound = 100.0 * np.asarray(
                [row["distance_distortion_bound_mean"] for row in rows]
            )
            label = method.label_ja if japanese else method.label
            accuracy_axis.plot(
                parameters, actual_accuracy, color=colours[method.key],
                marker=markers[method.key], label=label,
            )
            accuracy_axis.plot(
                parameters, certified_accuracy, color=colours[method.key],
                marker=markers[method.key], linestyle="--", alpha=0.75,
            )
            distortion_axis.plot(
                parameters, actual_distortion, color=colours[method.key],
                marker=markers[method.key], label=label,
            )
            finite = np.isfinite(bound)
            distortion_axis.plot(
                parameters[finite], bound[finite], color=colours[method.key],
                marker=markers[method.key], linestyle="--", alpha=0.75,
            )
        teacher_label = teacher.label_ja if japanese else teacher.label
        accuracy_axis.set_title(
            f"{teacher_label}: " + ("triplet近傍精度" if japanese else "triplet-neighbour accuracy")
        )
        distortion_axis.set_title(
            f"{teacher_label}: " + ("最大距離歪み" if japanese else "maximum distance distortion")
        )
        accuracy_axis.set_xscale("log")
        distortion_axis.set_xscale("log")
        accuracy_axis.set_ylim(45.0, 101.0)
        distortion_axis.set_yscale("log")
        distortion_axis.axhline(
            100.0 * DISTANCE_DISTORTION_TOLERANCE,
            color="black", linestyle=":", linewidth=1.2,
            label="5% " + ("許容値" if japanese else "tolerance"),
        )
        accuracy_axis.grid(alpha=0.2)
        distortion_axis.grid(alpha=0.2)
        accuracy_axis.set_ylabel("精度 [%]" if japanese else "accuracy [%]")
        distortion_axis.set_ylabel("歪み [%]" if japanese else "distortion [%]")
    axes[-1, 0].set_xlabel("学生パラメータ数" if japanese else "student parameters")
    axes[-1, 1].set_xlabel("学生パラメータ数" if japanese else "student parameters")
    axes[0, 0].legend(fontsize=8)
    axes[0, 1].legend(fontsize=8)
    figure.text(
        0.5, 0.005,
        ("実線：実測、破線：定理による保証下界／上界"
         if japanese else "solid: observed; dashed: theorem lower/upper bound"),
        ha="center", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.03, 1, 1))
    figure.savefig(HERE / f"distillation_tradeoff{suffix}.png", dpi=220)
    plt.close(figure)


def choose_before_after(
    rows: list[dict[str, float | str | int]],
) -> tuple[int, int, bool]:
    by_width: dict[int, list[dict[str, float | str | int]]] = {}
    for row in rows:
        by_width.setdefault(int(row["width"]), []).append(row)
    widths = sorted(by_width)
    passing_widths = [
        width for width in widths
        if all(int(row["certified_all_distance_pass"]) == 1 for row in by_width[width])
    ]
    if passing_widths:
        after = passing_widths[0]
        before_candidates = [width for width in widths if width < after]
        before = before_candidates[-1] if before_candidates else after
        return before, after, True
    return widths[-2], widths[-1], False


def save_threshold_figure(
    raw_rows: list[dict[str, float | str | int]],
    diagnostics: dict[tuple[str, str, int, int], dict[str, np.ndarray]],
    language: str,
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    figure, axes = plt.subplots(2, 4, figsize=(13.0, 6.4))
    scatter_max = 4.5
    image = None
    for row_index, teacher in enumerate(TEACHERS):
        c1_all_seeds = [
            row for row in raw_rows
            if row["teacher"] == teacher.key and row["method"] == "c1"
        ]
        c1_seed_zero = [row for row in c1_all_seeds if int(row["repeat"]) == 0]
        before_width, after_width, has_pass = choose_before_after(c1_all_seeds)
        for column_pair, (width, stage) in enumerate(
            ((before_width, "before"), (after_width, "after"))
        ):
            row = next(row for row in c1_seed_zero if int(row["width"]) == width)
            data = diagnostics[(teacher.key, "c1", 0, width)]
            scatter_axis = axes[row_index, 2 * column_pair]
            heatmap_axis = axes[row_index, 2 * column_pair + 1]
            teacher_distance = data["teacher_pair_distance"]
            student_distance = data["student_pair_distance"]
            scatter_axis.scatter(
                teacher_distance, student_distance, s=5, alpha=0.22, color="#2779ad"
            )
            line = np.linspace(0.0, scatter_max, 100)
            scatter_axis.plot(line, line, color="black", linewidth=1)
            scatter_axis.plot(line, 0.95 * line, color="#c33d38", linestyle=":")
            scatter_axis.plot(line, 1.05 * line, color="#c33d38", linestyle=":")
            scatter_axis.set_xlim(0.0, scatter_max)
            scatter_axis.set_ylim(0.0, scatter_max)
            scatter_axis.set_aspect("equal", adjustable="box")
            scatter_axis.grid(alpha=0.15)
            image = heatmap_axis.imshow(
                100.0 * data["metric_length_distortion"].T,
                origin="lower", extent=(-math.pi, math.pi, -math.pi, math.pi),
                cmap="magma", vmin=0.0, vmax=12.0, aspect="auto",
            )
            p_count = int(row["parameter_count"])
            robust_pass = stage == "after" and has_pass
            status = (
                ("全seed合格" if japanese else "all seeds pass")
                if robust_pass else ("seed間で未保証" if japanese else "not robust across seeds")
            )
            stage_label = (
                ("直前" if japanese else "before")
                if stage == "before" else ("直後" if japanese else "after")
            )
            if stage == "after" and not has_pass:
                stage_label = "最大（未合格）" if japanese else "largest (no pass)"
            scatter_axis.set_title(
                f"{stage_label}: w={width}, P={p_count}\n{status}, "
                + ("距離" if japanese else "distance")
            )
            heatmap_axis.set_title(
                ("局所長さ歪み" if japanese else "local length distortion")
                + f" (max={100.0 * float(row['metric_length_distortion_max']):.1f}%)"
            )
            scatter_axis.set_xlabel("教師距離" if japanese else "teacher distance")
            heatmap_axis.set_xlabel(r"$z_1$")
            if column_pair == 0:
                scatter_axis.set_ylabel(
                    (teacher.label_ja + "\n学生距離")
                    if japanese else (teacher.label + "\nstudent distance")
                )
                heatmap_axis.set_ylabel(r"$z_2$")
    assert image is not None
    colourbar = figure.colorbar(image, ax=axes[:, 1::2], fraction=0.025, pad=0.02)
    colourbar.set_label("局所長さ歪み [%]" if japanese else "local length distortion [%]")
    figure.suptitle(
        "5%距離保証を跨ぐ直前・直後（出力 + Jacobian蒸留）"
        if japanese
        else "Immediately before and after the 5% distance certificate"
    )
    figure.subplots_adjust(left=0.07, right=0.91, bottom=0.09, top=0.88, wspace=0.34, hspace=0.55)
    figure.savefig(HERE / f"distillation_threshold{suffix}.png", dpi=220)
    plt.close(figure)


def format_mean_std(mean: float, standard_deviation: float, digits: int = 3) -> str:
    return f"{mean:.{digits}f} $\\pm$ {standard_deviation:.{digits}f}"


def write_latex_table(summaries: list[dict[str, float | str | int]], language: str) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    lines = [
        r"\begin{tabular}{llrrrrrr}",
        r"\toprule",
        ((r"教師 & 蒸留 & $P$ & 出力RMSE & $\varepsilon_1$ & $\delta$ & "
          r"triplet精度 & 5\%保証 \\") if japanese else
         (r"teacher & distillation & $P$ & output RMSE & $\varepsilon_1$ & $\delta$ & "
          r"triplet acc. & 5\% cert. \\")),
        r"\midrule",
    ]
    for teacher in TEACHERS:
        for method in METHODS:
            rows = find_summary(summaries, teacher.key, method.key)
            for row_index, row in enumerate(rows):
                teacher_name = teacher.label_ja if japanese else teacher.label
                method_name = method.label_ja if japanese else method.label
                if row_index > 0:
                    teacher_name = ""
                    method_name = ""
                pass_fraction = float(row["certified_all_distance_pass_mean"])
                pass_text = ("yes" if pass_fraction == 1.0 else "no")
                if japanese:
                    pass_text = "可" if pass_fraction == 1.0 else "不可"
                lines.append(
                    f"{teacher_name} & {method_name} & {int(row['parameter_count_mean'])} & "
                    f"{float(row['output_rmse_mean']):.4f} & "
                    f"{float(row['epsilon1_grid_max_mean']):.4f} & "
                    f"{float(row['delta_cert_mean']):.3f} & "
                    f"{100.0 * float(row['triplet_accuracy_mean']):.1f} & {pass_text} \\\\"
                )
        lines.append(r"\midrule")
    lines[-1] = r"\bottomrule"
    lines.append(r"\end{tabular}")
    (HERE / f"distillation_table{suffix}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def main() -> None:
    arguments = parse_arguments()
    quick = bool(arguments.quick)
    widths = [4, 8, 16, 32, 64, 128, 256]
    if arguments.widths:
        widths = [int(value) for value in arguments.widths.split(",")]
    elif quick:
        widths = [8, 32]
    selected_teachers = TEACHERS
    if arguments.teachers:
        requested = set(arguments.teachers.split(","))
        selected_teachers = tuple(spec for spec in TEACHERS if spec.key in requested)
        if len(selected_teachers) != len(requested):
            raise ValueError(f"unknown teacher in {sorted(requested)}")
    selected_methods = METHODS
    if arguments.methods:
        requested = set(arguments.methods.split(","))
        selected_methods = tuple(spec for spec in METHODS if spec.key in requested)
        if len(selected_methods) != len(requested):
            raise ValueError(f"unknown method in {sorted(requested)}")
    repeats = arguments.repeats if arguments.repeats is not None else (1 if quick else 3)
    steps = arguments.steps if arguments.steps is not None else (250 if quick else 2500)
    batch_size = 96 if quick else 192
    grid_size = 24 if quick else 48
    anchor_count = 24 if quick else 96
    targets_per_anchor = 8 if quick else 32
    seed_everything(SEED)
    grid_values, grid = evaluation_grid(grid_size)
    rng = np.random.default_rng(SEED + 81)
    indices_by_teacher: dict[str, dict[str, np.ndarray]] = {}
    for teacher in selected_teachers:
        phi = teacher_angles(grid, teacher).detach().cpu().numpy()
        indices_by_teacher[teacher.key] = make_pair_and_triplet_indices(
            grid_size, rng, anchor_count, targets_per_anchor, phi
        )

    rows: list[dict[str, float | str | int]] = []
    histories: list[dict[str, float | str | int]] = []
    diagnostics: dict[tuple[str, str, int, int], dict[str, np.ndarray]] = {}
    total = len(selected_teachers) * len(selected_methods) * repeats * len(widths)
    completed = 0
    for teacher in selected_teachers:
        for method in selected_methods:
            for repeat in range(repeats):
                for width in widths:
                    run_seed = (
                        SEED + 100_000 * TEACHERS.index(teacher)
                        + 10_000 * METHODS.index(method) + 100 * repeat + width
                    )
                    model, run_history = train_student(
                        teacher, method, width, run_seed, steps, batch_size
                    )
                    metrics, run_diagnostics = evaluate_student(
                        model, teacher, grid_values, grid, indices_by_teacher[teacher.key]
                    )
                    completed += 1
                    row: dict[str, float | str | int] = {
                        "teacher": teacher.key,
                        "teacher_frequency": teacher.frequency,
                        "teacher_derivative_amplitude": teacher.derivative_amplitude,
                        "teacher_s": teacher.lower_singular_value,
                        "teacher_M1": teacher.upper_jacobian_norm,
                        "method": method.key,
                        "repeat": repeat,
                        "seed": run_seed,
                        "width": width,
                        "steps": steps,
                        "evaluation_grid_size": grid_size,
                        "output_rmse_tolerance": OUTPUT_RMSE_TOLERANCE,
                        "distance_distortion_tolerance": DISTANCE_DISTORTION_TOLERANCE,
                        **metrics,
                    }
                    rows.append(row)
                    for history_row in run_history:
                        histories.append(
                            {
                                "teacher": teacher.key,
                                "method": method.key,
                                "repeat": repeat,
                                "width": width,
                                **history_row,
                            }
                        )
                    if repeat == 0:
                        diagnostics[(teacher.key, method.key, repeat, width)] = run_diagnostics
                    print(
                        f"[{completed:02d}/{total}] {teacher.key:6s} {method.key:5s} "
                        f"w={width:3d} P={model.parameter_count:6d} "
                        f"rmse={float(metrics['output_rmse']):.4f} "
                        f"eps1={float(metrics['epsilon1_grid_max']):.4f} "
                        f"delta={float(metrics['delta_cert']):.3f} "
                        f"triplet={100.0 * float(metrics['triplet_accuracy']):5.1f}% "
                        f"pass={int(metrics['certified_all_distance_pass'])}",
                        flush=True,
                    )

    summaries = summarize(rows)
    thresholds = threshold_rows(summaries)
    write_csv(HERE / "distillation_results.csv", rows)
    write_csv(HERE / "distillation_summary.csv", summaries)
    write_csv(HERE / "distillation_thresholds.csv", thresholds)
    write_csv(HERE / "distillation_history.csv", histories)
    complete_sweep = selected_teachers == TEACHERS and selected_methods == METHODS
    if complete_sweep:
        save_tradeoff_figure(summaries, "en")
        save_tradeoff_figure(summaries, "ja")
        save_threshold_figure(rows, diagnostics, "en")
        save_threshold_figure(rows, diagnostics, "ja")
        write_latex_table(summaries, "en")
        write_latex_table(summaries, "ja")

    print("\nCertified thresholds (all repeats must pass):")
    for row in thresholds:
        print(
            f"  {row['teacher']:6s} {row['method']:5s}: {row['status']}, "
            f"width={row['minimum_width']}, P={row['minimum_parameter_count']}, "
            f"triplet={100.0 * float(row['triplet_accuracy_mean']):.1f}%"
        )


if __name__ == "__main__":
    main()
