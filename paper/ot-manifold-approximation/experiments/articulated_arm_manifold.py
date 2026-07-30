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
"""Infer the torus geometry behind images of a two-joint robot arm.

Each joint angle is periodic, so the true configuration space is
S^1 x S^1 = T^2.  Training uses only 32 x 32 RGB arm images and a pixel-space
neighbourhood graph; the two angles are retained exclusively for evaluation.
We compare R^2, S^2, and T^2 autoencoders using a deterministic-coupling
(Monge) upper bound on the Gromov--Wasserstein objective.

Run with:
    uv run --python 3.12 articulated_arm_manifold.py
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components, shortest_path
from scipy.spatial.distance import cdist
from torch import Tensor, nn

plt.switch_backend("Agg")


HERE = Path(__file__).resolve().parent
SEED = 20260730
IMAGE_SIZE = 32
ENCODED_SIZE = IMAGE_SIZE // 4
N_TRAIN = 900
N_VALID = 300
GRAPH_NEIGHBOURS = 4
STEPS = 1500
BATCH_SIZE = 96
METRIC_WEIGHT = 0.35
HISTORY_EVERY = 50
RESTARTS = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def sample_joint_angles(n: int, rng: np.random.Generator) -> np.ndarray:
    """Uniform shoulder and elbow angles on the flat torus."""
    return rng.uniform(-np.pi, np.pi, size=(n, 2)).astype(np.float32)


def render_arms(
    angles: np.ndarray, rng: np.random.Generator, noise: float = 0.006
) -> np.ndarray:
    """Render a coloured two-link arm with a base, elbow, and two-finger gripper."""
    n = len(angles)
    grid = np.linspace(-1.0, 1.0, IMAGE_SIZE, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)
    x_grid = np.broadcast_to(gx, (n, IMAGE_SIZE, IMAGE_SIZE))
    y_grid = np.broadcast_to(gy, (n, IMAGE_SIZE, IMAGE_SIZE))

    base_point = np.zeros((n, 2), dtype=np.float32)
    base_point[:, 1] = -0.08
    shoulder, elbow = angles.T
    first_tip = base_point + 0.43 * np.stack(
        [np.cos(shoulder), np.sin(shoulder)], axis=1
    )
    end_angle = shoulder + elbow
    second_tip = first_tip + 0.34 * np.stack(
        [np.cos(end_angle), np.sin(end_angle)], axis=1
    )

    def segment_blob(start: np.ndarray, end: np.ndarray, sigma: float) -> np.ndarray:
        vx = end[:, 0] - start[:, 0]
        vy = end[:, 1] - start[:, 1]
        wx = x_grid - start[:, 0, None, None]
        wy = y_grid - start[:, 1, None, None]
        length_sq = vx * vx + vy * vy
        projection = np.clip(
            (wx * vx[:, None, None] + wy * vy[:, None, None])
            / length_sq[:, None, None],
            0.0,
            1.0,
        )
        dx = x_grid - (start[:, 0, None, None] + projection * vx[:, None, None])
        dy = y_grid - (start[:, 1, None, None] + projection * vy[:, None, None])
        return np.exp(-(dx * dx + dy * dy) / (2.0 * sigma * sigma))

    def joint_blob(point: np.ndarray, sigma: float) -> np.ndarray:
        distance_sq = (x_grid - point[:, 0, None, None]) ** 2 + (
            y_grid - point[:, 1, None, None]
        ) ** 2
        return np.exp(-distance_sq / (2.0 * sigma * sigma))

    link_one = segment_blob(base_point, first_tip, 0.050)
    link_two = segment_blob(first_tip, second_tip, 0.046)
    forward = 0.09 * np.stack([np.cos(end_angle), np.sin(end_angle)], axis=1)
    across = 0.065 * np.stack([-np.sin(end_angle), np.cos(end_angle)], axis=1)
    palm_left = second_tip - across
    palm_right = second_tip + across
    gripper = (
        segment_blob(palm_left, palm_right, 0.032)
        + segment_blob(palm_left, palm_left + forward, 0.030)
        + segment_blob(palm_right, palm_right + forward, 0.030)
    )
    base = joint_blob(base_point, 0.080)
    elbow_joint = joint_blob(first_tip, 0.065)
    wrist = joint_blob(second_tip, 0.047)

    images = np.zeros((n, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    images[:, 0] = 1.0 - np.exp(
        -(1.55 * link_one + 1.05 * elbow_joint + 0.80 * base + 0.18 * gripper)
    )
    images[:, 1] = 1.0 - np.exp(
        -(
            0.22 * link_one
            + 0.30 * link_two
            + 1.10 * elbow_joint
            + 0.95 * wrist
            + 1.35 * gripper
            + 0.80 * base
        )
    )
    images[:, 2] = 1.0 - np.exp(
        -(0.12 * link_one + 1.55 * link_two + 0.22 * elbow_joint + 0.80 * base)
    )
    if noise:
        images += rng.normal(0.0, noise, size=images.shape).astype(np.float32)
    return np.clip(images, 0.0, 1.0)


def observation_graph_metric(
    images: np.ndarray,
) -> tuple[np.ndarray, int, float, np.ndarray]:
    """Approximate intrinsic distances from pixel observations by a k-NN graph."""
    flat = images.reshape(len(images), -1).astype(np.float64)
    distances = cdist(flat, flat, metric="euclidean") / math.sqrt(flat.shape[1])
    k = GRAPH_NEIGHBOURS
    while True:
        neighbours = np.argpartition(distances, kth=k, axis=1)[:, 1 : k + 1]
        rows = np.repeat(np.arange(len(images)), k)
        cols = neighbours.reshape(-1)
        vals = distances[rows, cols]
        graph = csr_matrix((vals, (rows, cols)), shape=distances.shape)
        # Union symmetrisation.  Averaging with implicit sparse zeros would halve
        # one-way k-NN edges and introduce artificial shortest-path shortcuts.
        graph = graph.maximum(graph.T)
        components, _ = connected_components(graph, directed=False)
        if components == 1:
            break
        k += 2
        if k >= min(40, len(images) - 1):
            raise RuntimeError("observation graph stayed disconnected")
    geodesic = shortest_path(graph, directed=False)
    finite = geodesic[np.isfinite(geodesic) & (geodesic > 0)]
    scale = float(np.quantile(finite, 0.95))
    return (
        np.clip(geodesic / scale, 0.0, 1.0).astype(np.float32),
        k,
        scale,
        geodesic.astype(np.float32),
    )


def out_of_sample_graph_metric(
    reference_images: np.ndarray,
    query_images: np.ndarray,
    reference_geodesic: np.ndarray,
    scale: float,
    k: int,
) -> np.ndarray:
    """Extend a fixed training graph metric to independent query points.

    Each query is connected to its k closest reference images, but query points
    never connect to one another.  Conditional on the training/reference split,
    this makes d_hat(x, x') a fixed two-sample kernel, as required by the
    validation U-statistic theorem in the paper.
    """
    reference = reference_images.reshape(len(reference_images), -1).astype(np.float64)
    query = query_images.reshape(len(query_images), -1).astype(np.float64)
    cross = cdist(query, reference, metric="euclidean") / math.sqrt(reference.shape[1])
    neighbours = np.argpartition(cross, kth=k - 1, axis=1)[:, :k]
    edge_costs = np.take_along_axis(cross, neighbours, axis=1)
    raw = np.empty((len(query), len(query)), dtype=np.float64)
    for i in range(len(query)):
        # Distance from query i to every reference node through its best anchor.
        to_reference = np.min(
            edge_costs[i, :, None] + reference_geodesic[neighbours[i], :], axis=0
        )
        raw[i] = np.min(to_reference[neighbours] + edge_costs, axis=1)
    raw = 0.5 * (raw + raw.T)
    np.fill_diagonal(raw, 0.0)
    return np.clip(raw / scale, 0.0, 1.0).astype(np.float32)


class ArmAutoencoder(nn.Module):
    def __init__(self, geometry: str) -> None:
        super().__init__()
        self.geometry = geometry
        raw_dim = {"euclidean": 2, "sphere": 3, "torus": 4}[geometry]
        decoder_dim = raw_dim
        # Without a prior, a bounded Euclidean bottleneck can fall into the
        # constant-code stationary point of pairwise stress.  Batch normalisation
        # is the standard minimal anti-collapse constraint for this baseline.
        self.euclidean_norm = nn.BatchNorm1d(raw_dim, affine=False)
        self.encoder_conv = nn.Sequential(
            nn.Conv2d(3, 16, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(16, 32, 3, stride=2, padding=1),
            nn.SiLU(),
        )
        self.encoder_head = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * ENCODED_SIZE * ENCODED_SIZE, 96),
            nn.SiLU(),
            nn.Linear(96, raw_dim),
        )
        self.decoder_head = nn.Sequential(
            nn.Linear(decoder_dim, 96),
            nn.SiLU(),
            nn.Linear(96, 32 * ENCODED_SIZE * ENCODED_SIZE),
            nn.SiLU(),
        )
        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(32, 16, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(16, 3, 4, stride=2, padding=1),
            nn.Sigmoid(),
        )

    def encode(self, x: Tensor) -> Tensor:
        raw = self.encoder_head(self.encoder_conv(x))
        if self.geometry == "euclidean":
            return torch.tanh(self.euclidean_norm(raw))
        if self.geometry == "torus":
            pairs = raw.reshape(-1, 2, 2)
            return nn.functional.normalize(pairs, dim=2, eps=1e-8).reshape(-1, 4)
        return nn.functional.normalize(raw, dim=1, eps=1e-8)

    def decoder_features(self, code: Tensor) -> Tensor:
        return code

    def decode(self, code: Tensor) -> Tensor:
        hidden = self.decoder_head(self.decoder_features(code)).reshape(
            -1, 32, ENCODED_SIZE, ENCODED_SIZE
        )
        return self.decoder_conv(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encode(x)
        return self.decode(code), code

    def pairwise_distance(self, code: Tensor) -> Tensor:
        if self.geometry == "euclidean":
            return torch.cdist(code, code) / (2.0 * math.sqrt(2.0))
        if self.geometry == "sphere":
            dots = torch.clamp(code @ code.T, -1.0 + 1e-6, 1.0 - 1e-6)
            return torch.acos(dots) / math.pi
        first = code[:, :2]
        second = code[:, 2:]
        first_dots = torch.clamp(first @ first.T, -1.0 + 1e-6, 1.0 - 1e-6)
        second_dots = torch.clamp(second @ second.T, -1.0 + 1e-6, 1.0 - 1e-6)
        first_angles = torch.acos(first_dots)
        second_angles = torch.acos(second_dots)
        return torch.sqrt(first_angles.square() + second_angles.square()) / (
            math.pi * math.sqrt(2.0)
        )


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    geometry: str
    metric_weight: float


SPECS = [
    ModelSpec("r2_gw", r"$\mathbb{R}^2$+MGW", "euclidean", METRIC_WEIGHT),
    ModelSpec("s2_gw", r"$S^2$+MGW", "sphere", METRIC_WEIGHT),
    ModelSpec("t2_recon", r"$\mathbb{T}^2$ recon-only", "torus", 0.0),
    ModelSpec("t2_gw", r"$\mathbb{T}^2$+MGW", "torus", METRIC_WEIGHT),
]


def upper_triangle_values(matrix: Tensor) -> Tensor:
    n = matrix.shape[0]
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=matrix.device), diagonal=1
    )
    return matrix[mask]


def weighted_reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    return torch.mean((prediction - target) ** 2 * (1.0 + 3.0 * target))


def train_model(
    spec: ModelSpec,
    train_images: Tensor,
    train_metric: Tensor,
    restart: int,
) -> tuple[ArmAutoencoder, list[dict[str, float]]]:
    restart_seed = SEED + restart + sum(ord(c) for c in spec.key)
    torch.manual_seed(restart_seed)
    model = ArmAutoencoder(spec.geometry)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=STEPS, eta_min=2e-4
    )
    generator = torch.Generator().manual_seed(SEED + restart + 11 * len(spec.key))
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, STEPS + 1):
        indices = torch.randint(
            0, len(train_images), (BATCH_SIZE,), generator=generator
        )
        batch = train_images[indices]
        target_metric = train_metric[indices][:, indices]
        reconstruction, code = model(batch)
        recon_loss = weighted_reconstruction_loss(reconstruction, batch)
        latent_metric = model.pairwise_distance(code)
        metric_error = upper_triangle_values(latent_metric - target_metric)
        metric_loss = torch.mean(metric_error * metric_error)
        # A short reconstruction warm-up prevents the metric term from selecting
        # an arbitrary orientation before the decoder sees the image content.
        ramp = min(1.0, max(0.0, (step - 150) / 350.0))
        loss = recon_loss + spec.metric_weight * ramp * metric_loss
        optimiser.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=5.0)
        optimiser.step()
        scheduler.step()
        if step == 1 or step % HISTORY_EVERY == 0 or step == STEPS:
            history.append(
                {
                    "step": float(step),
                    "objective": float(loss.detach()),
                    "reconstruction": float(recon_loss.detach()),
                    "mgw_squared": float(metric_loss.detach()),
                }
            )
    return model, history


def true_torus_distances(angles: np.ndarray) -> np.ndarray:
    delta = angles[:, None, :] - angles[None, :, :]
    wrapped = np.arctan2(np.sin(delta), np.cos(delta))
    return (
        np.sqrt(np.sum(wrapped * wrapped, axis=2)) / (math.pi * math.sqrt(2.0))
    ).astype(np.float32)


def pair_correlation(a: np.ndarray, b: np.ndarray) -> float:
    indices = np.triu_indices(len(a), k=1)
    if np.std(a[indices]) < 1e-12 or np.std(b[indices]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[indices], b[indices])[0, 1])


def neighbour_recall(reference: np.ndarray, estimate: np.ndarray, k: int = 10) -> float:
    ref_order = np.argsort(reference, axis=1)[:, 1 : k + 1]
    est_order = np.argsort(estimate, axis=1)[:, 1 : k + 1]
    recalls = [len(set(a).intersection(b)) / k for a, b in zip(ref_order, est_order)]
    return float(np.mean(recalls))


def evaluate_model(
    model: ArmAutoencoder,
    images: Tensor,
    observation_metric: np.ndarray,
    true_metric: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        reconstruction, code = model(images)
        recon = float(weighted_reconstruction_loss(reconstruction, images))
        latent_metric_t = model.pairwise_distance(code)
    latent_metric = latent_metric_t.cpu().numpy()
    pair_idx = np.triu_indices(len(images), k=1)
    mgw_sq = float(
        np.mean((latent_metric[pair_idx] - observation_metric[pair_idx]) ** 2)
    )
    true_sq = float(np.mean((latent_metric[pair_idx] - true_metric[pair_idx]) ** 2))
    result = {
        "reconstruction_mse": recon,
        "mgw_stress": math.sqrt(mgw_sq),
        # Every candidate is selected by the same held-out criterion, including
        # the reconstruction-only ablation.
        "selection_score": recon + METRIC_WEIGHT * mgw_sq,
        "true_torus_stress": math.sqrt(true_sq),
        "true_distance_correlation": pair_correlation(latent_metric, true_metric),
        "true_10nn_recall": neighbour_recall(true_metric, latent_metric),
        "parameters": float(sum(p.numel() for p in model.parameters())),
    }
    return (
        result,
        reconstruction.cpu().numpy(),
        latent_metric,
        code.cpu().numpy(),
    )


def write_csvs(
    results: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    scatter_rows: list[dict[str, str | float]],
) -> None:
    with (HERE / "arm_manifold_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    with (HERE / "arm_manifold_history.csv").open("w", newline="") as handle:
        fields = ["model", "step", "objective", "reconstruction", "mgw_squared"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, rows in histories.items():
            for row in rows:
                writer.writerow({"model": key, **row})
    with (HERE / "arm_distance_scatter.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(scatter_rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(scatter_rows)


def save_example_figure(images: np.ndarray) -> None:
    fig, axes = plt.subplots(2, 6, figsize=(8.4, 3.0))
    chosen = np.linspace(0, len(images) - 1, 12, dtype=int)
    for axis, idx in zip(axes.flat, chosen):
        axis.imshow(np.moveaxis(images[idx], 0, -1), interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
    fig.suptitle(
        "Observed data: two-joint robot arm images (angles hidden)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.25)
    fig.savefig(HERE / "arm_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_torus_explainer_figure() -> None:
    """Put the two physical joint cycles beside the two torus cycles."""
    figure = plt.figure(figsize=(10.4, 3.4))
    grid = figure.add_gridspec(
        2,
        12,
        width_ratios=[1.3, 1.3, 1.3, *([1.0] * 9)],
        wspace=0.05,
        hspace=0.10,
    )
    torus_axis = figure.add_subplot(grid[:, :3], projection="3d")
    u = np.linspace(0.0, 2.0 * np.pi, 72)
    v = np.linspace(0.0, 2.0 * np.pi, 36)
    u_surface, v_surface = np.meshgrid(u, v)
    major, minor = 1.45, 0.53
    x = (major + minor * np.cos(v_surface)) * np.cos(u_surface)
    y = (major + minor * np.cos(v_surface)) * np.sin(u_surface)
    z = minor * np.sin(v_surface)
    torus_axis.plot_surface(
        x,
        y,
        z,
        color="#d7dde2",
        alpha=0.55,
        linewidth=0,
        shade=True,
    )
    shoulder_v = 0.55
    torus_axis.plot(
        (major + minor * np.cos(shoulder_v)) * np.cos(u),
        (major + minor * np.cos(shoulder_v)) * np.sin(u),
        np.full_like(u, minor * np.sin(shoulder_v)),
        color="#d94b4b",
        linewidth=3.0,
    )
    elbow_u = 0.35
    torus_axis.plot(
        (major + minor * np.cos(v)) * np.full_like(v, np.cos(elbow_u)),
        (major + minor * np.cos(v)) * np.full_like(v, np.sin(elbow_u)),
        minor * np.sin(v),
        color="#3c78c2",
        linewidth=3.0,
    )
    torus_axis.set_title(r"configuration space $\mathbb{T}^2$", fontsize=10, pad=0)
    torus_axis.set_axis_off()
    torus_axis.view_init(elev=24, azim=-52)
    torus_axis.set_box_aspect((1, 1, 0.55))

    cycle = np.linspace(-np.pi, np.pi, 9, dtype=np.float32)
    shoulder_angles = np.column_stack([cycle, np.full_like(cycle, 0.75)])
    elbow_angles = np.column_stack([np.full_like(cycle, 0.35), cycle])
    explainer_rng = np.random.default_rng(SEED + 99)
    rows = [
        (
            "shoulder cycle",
            "#d94b4b",
            render_arms(shoulder_angles, explainer_rng, noise=0.0),
        ),
        (
            "elbow cycle",
            "#3c78c2",
            render_arms(elbow_angles, explainer_rng, noise=0.0),
        ),
    ]
    for row_index, (label, colour, images) in enumerate(rows):
        for column_index, image in enumerate(images):
            axis = figure.add_subplot(grid[row_index, 3 + column_index])
            axis.imshow(np.moveaxis(image, 0, -1), interpolation="nearest")
            axis.set_xticks([])
            axis.set_yticks([])
            for spine in axis.spines.values():
                spine.set_color(colour)
                spine.set_linewidth(1.2)
            if column_index == 0:
                axis.set_ylabel(
                    label,
                    color=colour,
                    rotation=0,
                    ha="right",
                    va="center",
                    fontsize=8,
                )
            if row_index == 0 and column_index == 4:
                axis.set_title(r"one full turn: $-\pi\rightarrow\pi$", fontsize=9)
            if column_index in {0, 8}:
                axis.text(
                    0.5,
                    -0.08,
                    r"same pose",
                    transform=axis.transAxes,
                    ha="center",
                    va="top",
                    fontsize=6.7,
                )
    figure.suptitle(
        "Two independent periodic motions produce two closed loops",
        fontsize=11,
        y=0.99,
    )
    figure.savefig(HERE / "arm_torus_explainer.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_figure(
    originals: np.ndarray,
    reconstructions: dict[str, np.ndarray],
) -> None:
    chosen = np.array([3, 91, 208, 277])
    columns = [("Input", originals)] + [
        (next(s.label for s in SPECS if s.key == key), reconstructions[key])
        for key in ["r2_gw", "s2_gw", "t2_recon", "t2_gw"]
    ]
    fig, axes = plt.subplots(len(chosen), len(columns), figsize=(8.4, 6.8))
    for row_index, idx in enumerate(chosen):
        for col_index, (label, images) in enumerate(columns):
            axes[row_index, col_index].imshow(
                np.moveaxis(images[idx], 0, -1), interpolation="nearest"
            )
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])
            if row_index == 0:
                axes[row_index, col_index].set_title(label, fontsize=10, pad=7)
        axes[row_index, 0].set_ylabel(
            f"pose {row_index + 1}", rotation=0, ha="right", va="center", fontsize=9
        )
    fig.tight_layout(pad=0.45, w_pad=0.35, h_pad=0.45)
    fig.savefig(HERE / "arm_reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics_figure(
    results: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    scatter: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    colours = {
        "r2_gw": "#777777",
        "s2_gw": "#d78b2d",
        "t2_recon": "#5e83ba",
        "t2_gw": "#b33a3a",
    }
    fig, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for spec in SPECS:
        rows = histories[spec.key]
        axes[0].plot(
            [r["step"] for r in rows],
            [r["reconstruction"] for r in rows],
            label=spec.label,
            color=colours[spec.key],
            linewidth=1.4,
        )
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("weighted reconstruction MSE")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    truth, learned = scatter["t2_gw"]
    axes[1].scatter(
        truth,
        learned,
        s=8,
        alpha=0.33,
        color=colours["t2_gw"],
        rasterized=True,
    )
    axes[1].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"true $\mathbb{T}^2$ distance")
    axes[1].set_ylabel("learned latent distance")
    axes[1].set_title(r"selected $\mathbb{T}^2$ model", fontsize=10)
    axes[1].grid(alpha=0.2)

    labels = [str(r["short_label"]) for r in results]
    x = np.arange(len(results))
    scores = [float(r["selection_score"]) for r in results]
    bar_colours = [colours[str(r["model"])] for r in results]
    bars = axes[2].bar(x, scores, width=0.68, color=bar_colours)
    best_index = int(np.argmin(scores))
    bars[best_index].set_edgecolor("black")
    bars[best_index].set_linewidth(1.5)
    for bar, score in zip(bars, scores):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            score + 0.00035,
            f"{score:.3f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("held-out selection score")
    axes[2].set_title(r"best: $\mathbb{T}^2$+MGW", fontsize=10)
    axes[2].set_ylim(0.0, max(scores) * 1.18)
    axes[2].grid(axis="y", alpha=0.25)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="upper center", ncol=4, frameon=False, fontsize=8
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.65)
    fig.savefig(HERE / "arm_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(results: list[dict[str, str | float]]) -> None:
    best_score = min(float(row["selection_score"]) for row in results)
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"latent model & recon. MSE & MGW stress & score & true stress & 10-NN recall \\",
        r"\midrule",
    ]
    for row in results:
        score = float(row["selection_score"])
        score_text = f"{score:.3f}"
        if math.isclose(score, best_score):
            score_text = rf"\textbf{{{score_text}}}"
        lines.append(
            f"{row['latex_label']} & {float(row['reconstruction_mse']):.4f} & "
            f"{float(row['mgw_stress']):.3f} & {score_text} & "
            f"{float(row['true_torus_stress']):.3f} & "
            f"{float(row['true_10nn_recall']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / "arm_manifold_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    train_angles = sample_joint_angles(N_TRAIN, rng)
    valid_angles = sample_joint_angles(N_VALID, rng)
    train_images_np = render_arms(train_angles, rng)
    valid_images_np = render_arms(valid_angles, rng)
    print("constructing observation metrics", flush=True)
    train_observation, train_k, train_scale, train_geodesic = observation_graph_metric(
        train_images_np
    )
    valid_k = train_k
    valid_observation = out_of_sample_graph_metric(
        train_images_np,
        valid_images_np,
        train_geodesic,
        train_scale,
        valid_k,
    )
    valid_scale = train_scale
    valid_true = true_torus_distances(valid_angles)
    graph_true_correlation = pair_correlation(valid_observation, valid_true)
    graph_pairs = np.triu_indices(N_VALID, k=1)
    graph_true_rmse = float(
        np.sqrt(
            np.mean((valid_observation[graph_pairs] - valid_true[graph_pairs]) ** 2)
        )
    )
    print(
        f"graph: train k={train_k}, valid k={valid_k}, "
        f"pixel-to-true correlation={graph_true_correlation:.4f}, "
        f"RMSE={graph_true_rmse:.4f}",
        flush=True,
    )

    train_images = torch.from_numpy(train_images_np)
    valid_images = torch.from_numpy(valid_images_np)
    train_metric = torch.from_numpy(train_observation)
    results: list[dict[str, str | float]] = []
    histories: dict[str, list[dict[str, float]]] = {}
    reconstructions: dict[str, np.ndarray] = {}
    scatter_for_plot: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    scatter_rows: list[dict[str, str | float]] = []

    for spec in SPECS:
        print(f"training {spec.key}", flush=True)
        candidates = []
        for restart in range(RESTARTS):
            model, history = train_model(
                spec, train_images, train_metric, restart=restart
            )
            metrics, reconstructed, latent_metric, _ = evaluate_model(
                model, valid_images, valid_observation, valid_true
            )
            candidates.append((metrics, reconstructed, latent_metric, history, restart))
            print(
                f"  restart={restart} score={metrics['selection_score']:.5f}",
                flush=True,
            )
        metrics, reconstructed, latent_metric, history, selected_restart = min(
            candidates, key=lambda candidate: candidate[0]["selection_score"]
        )
        result: dict[str, str | float] = {
            "model": spec.key,
            "short_label": {
                "r2_gw": "R2+MGW",
                "s2_gw": "S2+MGW",
                "t2_recon": "T2-only",
                "t2_gw": "T2+MGW",
            }[spec.key],
            "latex_label": spec.label,
            "geometry": spec.geometry,
            "metric_weight": spec.metric_weight,
            "selected_restart": float(selected_restart),
            **metrics,
            "graph_true_correlation": graph_true_correlation,
            "graph_true_rmse": graph_true_rmse,
            "train_graph_k": float(train_k),
            "valid_graph_k": float(valid_k),
            "train_graph_scale": train_scale,
            "valid_graph_scale": valid_scale,
        }
        results.append(result)
        histories[spec.key] = history
        reconstructions[spec.key] = reconstructed
        rng_scatter = np.random.default_rng(SEED + len(spec.key))
        all_pairs = np.column_stack(np.triu_indices(N_VALID, k=1))
        selected = all_pairs[rng_scatter.choice(len(all_pairs), 700, replace=False)]
        truth = valid_true[selected[:, 0], selected[:, 1]]
        learned = latent_metric[selected[:, 0], selected[:, 1]]
        scatter_for_plot[spec.key] = (truth, learned)
        scatter_rows.extend(
            {
                "model": spec.key,
                "true_torus_distance": float(t),
                "learned_distance": float(l),
            }
            for t, l in zip(truth, learned)
        )
        print(
            f"  recon={metrics['reconstruction_mse']:.5f} "
            f"MGW={metrics['mgw_stress']:.4f} "
            f"true={metrics['true_torus_stress']:.4f} "
            f"recall={metrics['true_10nn_recall']:.3f}",
            flush=True,
        )

    write_csvs(results, histories, scatter_rows)
    write_latex_table(results)
    save_example_figure(valid_images_np)
    save_torus_explainer_figure()
    save_reconstruction_figure(valid_images_np, reconstructions)
    save_diagnostics_figure(results, histories, scatter_for_plot)
    print(f"wrote results to {HERE}")


if __name__ == "__main__":
    main()
