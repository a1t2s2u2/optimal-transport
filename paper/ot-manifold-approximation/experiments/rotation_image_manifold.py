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
"""Infer the latent geometry of rendered rotation images.

The observations are 32 x 32 RGB mosaics from three synchronised cameras looking
at a coloured 3-D orientation marker.
Only images and their pixel-space neighbourhood graph are used for training.  The
ground-truth rotations are retained exclusively for evaluation.  We compare
autoencoders whose latent spaces are R^3, S^3, and SO(3), and train them with a
deterministic-coupling (Monge) upper bound on the Gromov--Wasserstein objective.

Run with:
    uv run --python 3.12 rotation_image_manifold.py
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
PANEL_SIZE = IMAGE_SIZE // 2
ENCODED_SIZE = IMAGE_SIZE // 4
N_TRAIN = 900
N_VALID = 300
GRAPH_NEIGHBOURS = 14
STEPS = 1500
BATCH_SIZE = 96
METRIC_WEIGHT = 0.35
HISTORY_EVERY = 50


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def sample_uniform_quaternions(n: int, rng: np.random.Generator) -> np.ndarray:
    """Shoemake's uniform sampling on S^3, in (w, x, y, z) order."""
    u1, u2, u3 = rng.random((3, n))
    x = np.sqrt(1.0 - u1) * np.sin(2.0 * np.pi * u2)
    y = np.sqrt(1.0 - u1) * np.cos(2.0 * np.pi * u2)
    z = np.sqrt(u1) * np.sin(2.0 * np.pi * u3)
    w = np.sqrt(u1) * np.cos(2.0 * np.pi * u3)
    return np.stack([w, x, y, z], axis=1).astype(np.float32)


def quaternion_to_matrix_numpy(q: np.ndarray) -> np.ndarray:
    q = q / np.linalg.norm(q, axis=1, keepdims=True)
    w, x, y, z = q.T
    return np.stack(
        [
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ],
        axis=1,
    ).reshape(-1, 3, 3)


def marker_points() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A right-handed RGB triad, represented by Gaussian splats."""
    points: list[np.ndarray] = []
    colours: list[np.ndarray] = []
    weights: list[float] = []
    axes = [
        (np.array([1.0, 0.0, 0.0]), np.array([1.0, 0.12, 0.08])),
        (np.array([0.0, 1.0, 0.0]), np.array([0.08, 1.0, 0.12])),
        (np.array([0.0, 0.0, 1.0]), np.array([0.10, 0.18, 1.0])),
    ]
    for axis, colour in axes:
        for t in np.linspace(0.10, 0.92, 15):
            points.append(t * axis)
            colours.append(colour)
            weights.append(0.72 if t < 0.85 else 1.15)
    # The grey hub makes the object read as a single physical marker.
    points.append(np.zeros(3))
    colours.append(np.array([0.78, 0.78, 0.78]))
    weights.append(1.8)
    return (
        np.asarray(points, dtype=np.float32),
        np.asarray(colours, dtype=np.float32),
        np.asarray(weights, dtype=np.float32),
    )


def render_rotations(
    quaternions: np.ndarray, rng: np.random.Generator, noise: float = 0.006
) -> np.ndarray:
    """Render three orthographic camera views into one RGB image.

    A single projection loses depth and is not injective.  The three-view sensor is
    a realistic way to make the inverse problem identifiable while leaving a
    strongly nonlinear 3,072-dimensional observation map.
    """
    rotations = quaternion_to_matrix_numpy(quaternions)
    points, colours, weights = marker_points()
    grid = np.linspace(-1.0, 1.0, PANEL_SIZE, dtype=np.float32)
    gx, gy = np.meshgrid(grid, grid)
    images = np.empty((len(quaternions), 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
    for start in range(0, len(quaternions), 96):
        stop = min(start + 96, len(quaternions))
        rotated = np.einsum("nij,pj->npi", rotations[start:stop], points)
        image = np.zeros((stop - start, 3, IMAGE_SIZE, IMAGE_SIZE), dtype=np.float32)
        camera_panels = [
            ((0, 1), (0, 0)),  # front: x--y
            ((0, 2), (0, PANEL_SIZE)),  # top: x--z
            ((2, 1), (PANEL_SIZE, 0)),  # side: z--y
        ]
        for (horizontal, vertical), (row, column) in camera_panels:
            px = rotated[:, :, horizontal]
            py = rotated[:, :, vertical]
            dist2 = (gx[None, None] - px[:, :, None, None]) ** 2 + (
                gy[None, None] - py[:, :, None, None]
            ) ** 2
            blobs = np.exp(-dist2 / (2.0 * 0.055**2))
            blobs *= weights[None, :, None, None]
            rgb = 1.0 - np.exp(-1.15 * np.einsum("nphw,pc->nchw", blobs, colours))
            image[:, :, row : row + PANEL_SIZE, column : column + PANEL_SIZE] = rgb
        if noise:
            image += rng.normal(0.0, noise, size=image.shape).astype(np.float32)
        images[start:stop] = np.clip(image, 0.0, 1.0)
    return images


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


def quaternion_to_matrix_torch(q: Tensor) -> Tensor:
    q = nn.functional.normalize(q, dim=1, eps=1e-8)
    w, x, y, z = q.unbind(dim=1)
    values = [
        1 - 2 * (y * y + z * z),
        2 * (x * y - z * w),
        2 * (x * z + y * w),
        2 * (x * y + z * w),
        1 - 2 * (x * x + z * z),
        2 * (y * z - x * w),
        2 * (x * z - y * w),
        2 * (y * z + x * w),
        1 - 2 * (x * x + y * y),
    ]
    return torch.stack(values, dim=1)


class RotationAutoencoder(nn.Module):
    def __init__(self, geometry: str) -> None:
        super().__init__()
        self.geometry = geometry
        raw_dim = 3 if geometry == "euclidean" else 4
        decoder_dim = 9 if geometry == "so3" else raw_dim
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
        return nn.functional.normalize(raw, dim=1, eps=1e-8)

    def decoder_features(self, code: Tensor) -> Tensor:
        if self.geometry == "so3":
            return quaternion_to_matrix_torch(code)
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
            return torch.cdist(code, code) / (2.0 * math.sqrt(3.0))
        dots = torch.clamp(code @ code.T, -1.0 + 1e-6, 1.0 - 1e-6)
        if self.geometry == "sphere":
            return torch.acos(dots) / math.pi
        return 2.0 * torch.acos(torch.clamp(torch.abs(dots), max=1.0 - 1e-6)) / math.pi


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    geometry: str
    metric_weight: float


SPECS = [
    ModelSpec("r3_gw", r"$\mathbb{R}^3$+MGW", "euclidean", METRIC_WEIGHT),
    ModelSpec("s3_gw", r"$S^3$+MGW", "sphere", METRIC_WEIGHT),
    ModelSpec("so3_recon", r"$SO(3)$ recon-only", "so3", 0.0),
    ModelSpec("so3_gw", r"$SO(3)$+MGW", "so3", METRIC_WEIGHT),
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
) -> tuple[RotationAutoencoder, list[dict[str, float]]]:
    torch.manual_seed(SEED + sum(ord(c) for c in spec.key))
    model = RotationAutoencoder(spec.geometry)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=STEPS, eta_min=2e-4
    )
    generator = torch.Generator().manual_seed(SEED + 11 * len(spec.key))
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


def true_so3_distances(quaternions: np.ndarray) -> np.ndarray:
    dots = np.clip(np.abs(quaternions @ quaternions.T), 0.0, 1.0)
    return (2.0 * np.arccos(dots) / np.pi).astype(np.float32)


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
    model: RotationAutoencoder,
    images: Tensor,
    observation_metric: np.ndarray,
    true_metric: np.ndarray,
    metric_weight: float,
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
        "true_so3_stress": math.sqrt(true_sq),
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
    with (HERE / "rotation_manifold_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(results[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    with (HERE / "rotation_manifold_history.csv").open("w", newline="") as handle:
        fields = ["model", "step", "objective", "reconstruction", "mgw_squared"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, rows in histories.items():
            for row in rows:
                writer.writerow({"model": key, **row})
    with (HERE / "rotation_distance_scatter.csv").open("w", newline="") as handle:
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
        r"Observed data: three-camera RGB mosaics (rotation labels hidden)", fontsize=11
    )
    fig.tight_layout(rect=(0, 0, 1, 0.92), pad=0.25)
    fig.savefig(HERE / "rotation_examples.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_reconstruction_figure(
    originals: np.ndarray,
    reconstructions: dict[str, np.ndarray],
) -> None:
    chosen = np.array([3, 47, 91, 136, 208, 277])
    rows = [("Input", originals)] + [
        (next(s.label for s in SPECS if s.key == key), reconstructions[key])
        for key in ["r3_gw", "s3_gw", "so3_recon", "so3_gw"]
    ]
    fig, axes = plt.subplots(len(rows), len(chosen), figsize=(8.4, 6.7))
    for row_index, (label, images) in enumerate(rows):
        for col_index, idx in enumerate(chosen):
            axes[row_index, col_index].imshow(
                np.moveaxis(images[idx], 0, -1), interpolation="nearest"
            )
            axes[row_index, col_index].set_xticks([])
            axes[row_index, col_index].set_yticks([])
            if col_index == 0:
                axes[row_index, col_index].set_ylabel(
                    label, rotation=0, ha="right", va="center"
                )
    fig.tight_layout(pad=0.25)
    fig.savefig(HERE / "rotation_reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(fig)


def save_diagnostics_figure(
    results: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    scatter: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    colours = {
        "r3_gw": "#777777",
        "s3_gw": "#d78b2d",
        "so3_recon": "#5e83ba",
        "so3_gw": "#b33a3a",
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

    for spec in SPECS:
        truth, learned = scatter[spec.key]
        axes[1].scatter(
            truth,
            learned,
            s=6,
            alpha=0.28,
            color=colours[spec.key],
            label=spec.label,
            rasterized=True,
        )
    axes[1].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"true $SO(3)$ distance")
    axes[1].set_ylabel("learned latent distance")
    axes[1].grid(alpha=0.2)

    labels = [str(r["short_label"]) for r in results]
    x = np.arange(len(results))
    width = 0.36
    axes[2].bar(
        x - width / 2,
        [float(r["mgw_stress"]) for r in results],
        width,
        label="observed MGW stress",
        color="#8bb8d8",
    )
    axes[2].bar(
        x + width / 2,
        [float(r["true_so3_stress"]) for r in results],
        width,
        label=r"true $SO(3)$ stress",
        color="#d9917b",
    )
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("lower is better")
    axes[2].grid(axis="y", alpha=0.25)
    axes[2].legend(fontsize=7, frameon=False)

    handles, labels_legend = axes[0].get_legend_handles_labels()
    fig.legend(
        handles, labels_legend, loc="upper center", ncol=4, frameon=False, fontsize=8
    )
    fig.tight_layout(rect=(0, 0, 1, 0.88), pad=0.65)
    fig.savefig(HERE / "rotation_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(fig)


def write_latex_table(results: list[dict[str, str | float]]) -> None:
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        r"latent model & recon. MSE & MGW stress & score & true stress & 10-NN recall \\",
        r"\midrule",
    ]
    for row in results:
        lines.append(
            f"{row['latex_label']} & {float(row['reconstruction_mse']):.4f} & "
            f"{float(row['mgw_stress']):.3f} & {float(row['selection_score']):.3f} & "
            f"{float(row['true_so3_stress']):.3f} & "
            f"{float(row['true_10nn_recall']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / "rotation_manifold_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    train_quaternions = sample_uniform_quaternions(N_TRAIN, rng)
    valid_quaternions = sample_uniform_quaternions(N_VALID, rng)
    train_images_np = render_rotations(train_quaternions, rng)
    valid_images_np = render_rotations(valid_quaternions, rng)
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
    valid_true = true_so3_distances(valid_quaternions)
    graph_true_correlation = pair_correlation(valid_observation, valid_true)
    print(
        f"graph: train k={train_k}, valid k={valid_k}, "
        f"pixel-to-true correlation={graph_true_correlation:.4f}",
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
        model, history = train_model(spec, train_images, train_metric)
        metrics, reconstructed, latent_metric, _ = evaluate_model(
            model, valid_images, valid_observation, valid_true, spec.metric_weight
        )
        result: dict[str, str | float] = {
            "model": spec.key,
            "short_label": {
                "r3_gw": "R3+MGW",
                "s3_gw": "S3+MGW",
                "so3_recon": "SO3-only",
                "so3_gw": "SO3+MGW",
            }[spec.key],
            "latex_label": spec.label,
            "geometry": spec.geometry,
            "metric_weight": spec.metric_weight,
            **metrics,
            "graph_true_correlation": graph_true_correlation,
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
                "true_so3_distance": float(t),
                "learned_distance": float(l),
            }
            for t, l in zip(truth, learned)
        )
        print(
            f"  recon={metrics['reconstruction_mse']:.5f} "
            f"MGW={metrics['mgw_stress']:.4f} "
            f"true={metrics['true_so3_stress']:.4f} "
            f"recall={metrics['true_10nn_recall']:.3f}",
            flush=True,
        )

    write_csvs(results, histories, scatter_rows)
    write_latex_table(results)
    save_example_figure(valid_images_np)
    save_reconstruction_figure(valid_images_np, reconstructions)
    save_diagnostics_figure(results, histories, scatter_for_plot)
    print(f"wrote results to {HERE}")


if __name__ == "__main__":
    main()
