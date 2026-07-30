#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "torch>=2.4",
# ]
# ///
"""Recover a spherical source manifold from high-dimensional sensor images.

A hidden source z lies on S^2.  A calibrated 32 x 32 sensor array turns z into
a three-channel response image, but the network never receives z.  We compare
R^2, T^2, and S^2 autoencoders with a deterministic-coupling
Monge--Gromov--Wasserstein objective.  The first sensor channel is linear and
therefore gives an explicit finite-sensor approximation to the normalized
chordal metric on S^2; the other channels make the inverse observation map
nonlinear and visually interpretable.

Run with:
    uv run --python 3.12 sensor_sphere_manifold.py
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
from torch import Tensor, nn

plt.switch_backend("Agg")


HERE = Path(__file__).resolve().parent
SEED = 20260730
IMAGE_SIZE = 32
ENCODED_SIZE = IMAGE_SIZE // 4
N_TRAIN = 900
N_VALID = 300
NOISE_STD = 0.006
LINEAR_AMPLITUDE = 0.42
STEPS = 1500
BATCH_SIZE = 96
METRIC_WEIGHT = 0.55
HISTORY_EVERY = 50
RESTARTS = 2


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def sample_sphere(n: int, rng: np.random.Generator) -> np.ndarray:
    points = rng.normal(size=(n, 3)).astype(np.float32)
    return points / np.linalg.norm(points, axis=1, keepdims=True)


def sensor_grid(side: int) -> tuple[np.ndarray, np.ndarray]:
    """Latitude-longitude sensors and normalized spherical quadrature weights."""
    theta = (np.arange(side, dtype=np.float64) + 0.5) * math.pi / side
    phi = (np.arange(side, dtype=np.float64) + 0.5) * 2.0 * math.pi / side
    theta_grid, phi_grid = np.meshgrid(theta, phi, indexing="ij")
    directions = np.stack(
        [
            np.sin(theta_grid) * np.cos(phi_grid),
            np.sin(theta_grid) * np.sin(phi_grid),
            np.cos(theta_grid),
        ],
        axis=-1,
    ).reshape(-1, 3)
    weights = np.broadcast_to(np.sin(theta)[:, None], (side, side)).copy().reshape(-1)
    weights /= weights.sum()
    return directions.astype(np.float32), weights.astype(np.float64)


def render_sensor_images(
    sources: np.ndarray,
    directions: np.ndarray,
    rng: np.random.Generator,
    noise_std: float = NOISE_STD,
) -> np.ndarray:
    """Render calibrated, hotspot, and wave-front response channels."""
    side = round(math.sqrt(len(directions)))
    if side * side != len(directions):
        raise ValueError("sensor count must be a square")
    dots = np.clip(sources @ directions.T, -1.0, 1.0)
    calibrated = 0.5 + LINEAR_AMPLITUDE * dots
    hotspot = np.exp(5.0 * (dots - 1.0))
    wavefront = np.exp(-0.5 * ((dots - 0.20) / 0.16) ** 2)
    images = np.stack([calibrated, hotspot, wavefront], axis=1).reshape(
        len(sources), 3, side, side
    )
    if noise_std:
        images += rng.normal(0.0, noise_std, size=images.shape)
    return np.clip(images, 0.0, 1.0).astype(np.float32)


def weighted_squared_distances(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    weighted = values * np.sqrt(weights)[None, :]
    squared_norms = np.sum(weighted * weighted, axis=1)
    squared = (
        squared_norms[:, None] + squared_norms[None, :] - 2.0 * weighted @ weighted.T
    )
    return np.maximum(squared, 0.0)


def calibrated_sensor_metric(
    images: np.ndarray,
    weights: np.ndarray,
    noise_std: float,
) -> np.ndarray:
    """Estimate normalized chordal S^2 distance without using source labels."""
    calibrated = images[:, 0].reshape(len(images), -1).astype(np.float64)
    response_sq = weighted_squared_distances(calibrated, weights)
    # Two independently noised images add 2 sigma^2 in expectation.  Sensor
    # calibration supplies sigma but never a source coordinate.
    debiased = np.maximum(response_sq - 2.0 * noise_std * noise_std, 0.0)
    metric_sq = 3.0 * debiased / (4.0 * LINEAR_AMPLITUDE * LINEAR_AMPLITUDE)
    metric = np.sqrt(np.clip(metric_sq, 0.0, 1.0))
    np.fill_diagonal(metric, 0.0)
    return metric.astype(np.float32)


def true_sphere_distances(sources: np.ndarray) -> np.ndarray:
    """Normalized chordal distance on S^2, with diameter one."""
    dots = np.clip(sources @ sources.T, -1.0, 1.0)
    return (0.5 * np.sqrt(np.maximum(2.0 - 2.0 * dots, 0.0))).astype(np.float32)


class SensorAutoencoder(nn.Module):
    def __init__(self, geometry: str) -> None:
        super().__init__()
        self.geometry = geometry
        raw_dim = {"euclidean": 2, "sphere": 3, "torus": 4}[geometry]
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
            nn.Linear(raw_dim, 96),
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

    def decode(self, code: Tensor) -> Tensor:
        hidden = self.decoder_head(code).reshape(-1, 32, ENCODED_SIZE, ENCODED_SIZE)
        return self.decoder_conv(hidden)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        code = self.encode(x)
        return self.decode(code), code

    def pairwise_distance(self, code: Tensor) -> Tensor:
        if self.geometry == "euclidean":
            return torch.cdist(code, code) / (2.0 * math.sqrt(2.0))
        if self.geometry == "sphere":
            return torch.cdist(code, code) / 2.0
        return torch.cdist(code, code) / (2.0 * math.sqrt(2.0))


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    geometry: str
    metric_weight: float


SPECS = [
    ModelSpec("r2_gw", r"$\mathbb{R}^2$+MGW", "euclidean", METRIC_WEIGHT),
    ModelSpec("t2_gw", r"$\mathbb{T}^2$+MGW", "torus", METRIC_WEIGHT),
    ModelSpec("s2_recon", r"$S^2$ recon-only", "sphere", 0.0),
    ModelSpec("s2_gw", r"$S^2$+MGW", "sphere", METRIC_WEIGHT),
]


def upper_triangle_values(matrix: Tensor) -> Tensor:
    n = matrix.shape[0]
    mask = torch.triu(
        torch.ones(n, n, dtype=torch.bool, device=matrix.device), diagonal=1
    )
    return matrix[mask]


def reconstruction_loss(prediction: Tensor, target: Tensor) -> Tensor:
    channel_weights = torch.tensor(
        [1.0, 1.4, 1.2], dtype=target.dtype, device=target.device
    ).reshape(1, 3, 1, 1)
    return torch.mean((prediction - target).square() * channel_weights)


def train_model(
    spec: ModelSpec,
    train_images: Tensor,
    train_metric: Tensor,
    restart: int,
) -> tuple[SensorAutoencoder, list[dict[str, float]]]:
    restart_seed = SEED + restart + sum(ord(c) for c in spec.key)
    torch.manual_seed(restart_seed)
    model = SensorAutoencoder(spec.geometry)
    optimiser = torch.optim.AdamW(model.parameters(), lr=1.5e-3, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimiser, T_max=STEPS, eta_min=2e-4
    )
    generator = torch.Generator().manual_seed(SEED + restart + 13 * len(spec.key))
    history: list[dict[str, float]] = []
    model.train()
    for step in range(1, STEPS + 1):
        indices = torch.randint(
            0, len(train_images), (BATCH_SIZE,), generator=generator
        )
        batch = train_images[indices]
        target_metric = train_metric[indices][:, indices]
        reconstruction, code = model(batch)
        recon_loss = reconstruction_loss(reconstruction, batch)
        latent_metric = model.pairwise_distance(code)
        metric_error = upper_triangle_values(latent_metric - target_metric)
        metric_loss = torch.mean(metric_error.square())
        ramp = min(1.0, max(0.0, (step - 100) / 300.0))
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


def pair_correlation(a: np.ndarray, b: np.ndarray) -> float:
    indices = np.triu_indices(len(a), k=1)
    if np.std(a[indices]) < 1e-12 or np.std(b[indices]) < 1e-12:
        return 0.0
    return float(np.corrcoef(a[indices], b[indices])[0, 1])


def pair_rmse(a: np.ndarray, b: np.ndarray) -> float:
    indices = np.triu_indices(len(a), k=1)
    return float(np.sqrt(np.mean((a[indices] - b[indices]) ** 2)))


def neighbour_recall(reference: np.ndarray, estimate: np.ndarray, k: int = 10) -> float:
    ref_order = np.argsort(reference, axis=1)[:, 1 : k + 1]
    est_order = np.argsort(estimate, axis=1)[:, 1 : k + 1]
    recalls = [len(set(a).intersection(b)) / k for a, b in zip(ref_order, est_order)]
    return float(np.mean(recalls))


def evaluate_model(
    model: SensorAutoencoder,
    images: Tensor,
    observation_metric: np.ndarray,
    true_metric: np.ndarray,
) -> tuple[dict[str, float], np.ndarray, np.ndarray, np.ndarray]:
    model.eval()
    with torch.no_grad():
        reconstruction, code = model(images)
        recon = float(reconstruction_loss(reconstruction, images))
        latent_metric_t = model.pairwise_distance(code)
    latent_metric = latent_metric_t.cpu().numpy()
    mgw = pair_rmse(latent_metric, observation_metric)
    true_stress = pair_rmse(latent_metric, true_metric)
    result = {
        "reconstruction_mse": recon,
        "mgw_stress": mgw,
        "selection_score": recon + METRIC_WEIGHT * mgw * mgw,
        "true_sphere_stress": true_stress,
        "true_distance_correlation": pair_correlation(latent_metric, true_metric),
        "true_10nn_recall": neighbour_recall(true_metric, latent_metric),
        "parameters": float(sum(p.numel() for p in model.parameters())),
    }
    return result, reconstruction.cpu().numpy(), latent_metric, code.cpu().numpy()


def sensor_accuracy_study(
    sources: np.ndarray,
) -> tuple[list[dict[str, float]], list[dict[str, float]]]:
    true_metric = true_sphere_distances(sources)
    resolution_rows: list[dict[str, float]] = []
    for side in [4, 6, 8, 12, 16, 24, 32, 48, 64]:
        directions, weights = sensor_grid(side)
        second_moment = directions.T @ (weights[:, None] * directions)
        moment_error = float(np.linalg.norm(second_moment - np.eye(3) / 3.0, ord=2))
        local_rng = np.random.default_rng(SEED + side)
        images = render_sensor_images(sources, directions, local_rng, noise_std=0.0)
        estimate = calibrated_sensor_metric(images, weights, noise_std=0.0)
        resolution_rows.append(
            {
                "sensor_side": float(side),
                "sensor_count": float(side * side),
                "rmse": pair_rmse(estimate, true_metric),
                "correlation": pair_correlation(estimate, true_metric),
                "moment_operator_error": moment_error,
                "squared_metric_uniform_bound": 3.0 * moment_error,
            }
        )

    noise_rows: list[dict[str, float]] = []
    directions, weights = sensor_grid(IMAGE_SIZE)
    for noise_index, noise in enumerate([0.0, 0.0025, 0.005, 0.01, 0.02, 0.04]):
        errors = []
        for repeat in range(6):
            local_rng = np.random.default_rng(SEED + 1000 * noise_index + repeat)
            images = render_sensor_images(
                sources, directions, local_rng, noise_std=noise
            )
            estimate = calibrated_sensor_metric(images, weights, noise_std=noise)
            errors.append(pair_rmse(estimate, true_metric))
        noise_rows.append(
            {
                "noise_std": noise,
                "rmse_mean": float(np.mean(errors)),
                "rmse_std": float(np.std(errors)),
            }
        )
    return resolution_rows, noise_rows


def write_csvs(
    results: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    scatter_rows: list[dict[str, str | float]],
    resolution_rows: list[dict[str, float]],
    noise_rows: list[dict[str, float]],
) -> None:
    outputs = [
        ("sensor_manifold_results.csv", results),
        ("sensor_distance_scatter.csv", scatter_rows),
        ("sensor_resolution_study.csv", resolution_rows),
        ("sensor_noise_study.csv", noise_rows),
    ]
    for filename, rows in outputs:
        with (HERE / filename).open("w", newline="") as handle:
            writer = csv.DictWriter(
                handle, fieldnames=list(rows[0]), lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
    with (HERE / "sensor_manifold_history.csv").open("w", newline="") as handle:
        fields = ["model", "step", "objective", "reconstruction", "mgw_squared"]
        writer = csv.DictWriter(handle, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for key, rows in histories.items():
            for row in rows:
                writer.writerow({"model": key, **row})


def save_sensor_explainer_figure(
    directions: np.ndarray,
) -> None:
    sources = np.asarray(
        [
            [0.92, 0.22, 0.32],
            [-0.25, 0.93, 0.27],
            [-0.75, -0.30, 0.59],
            [0.34, -0.55, -0.76],
            [-0.80, 0.40, -0.44],
            [0.58, 0.70, -0.41],
        ],
        dtype=np.float32,
    )
    sources /= np.linalg.norm(sources, axis=1, keepdims=True)
    images = render_sensor_images(
        sources, directions, np.random.default_rng(SEED + 99), noise_std=0.0
    )
    colours = ["#d94841", "#e79b28", "#5f9f42", "#3288bd", "#7455a4", "#c04c92"]

    figure = plt.figure(figsize=(10.4, 4.15))
    grid = figure.add_gridspec(2, 7, width_ratios=[1.45, 1.45, 1.45, 1, 1, 1, 0.08])
    sphere_axis = figure.add_subplot(grid[:, :3], projection="3d")
    u = np.linspace(0.0, 2.0 * np.pi, 72)
    v = np.linspace(0.0, np.pi, 36)
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones_like(u), np.cos(v))
    sphere_axis.plot_surface(
        x, y, z, color="#d9e3ea", alpha=0.28, linewidth=0, shade=False
    )
    for index, (source, colour) in enumerate(zip(sources, colours)):
        sphere_axis.scatter(
            *source, s=52, color=colour, edgecolor="black", linewidth=0.4
        )
        sphere_axis.text(
            *(1.10 * source), f"{index + 1}", color=colour, fontsize=9, weight="bold"
        )
    sphere_axis.set_axis_off()
    sphere_axis.set_box_aspect((1, 1, 1))
    sphere_axis.view_init(elev=18, azim=-58)
    sphere_axis.set_title(r"hidden source $z\in S^2$", fontsize=11, pad=0)

    for index, (image, colour) in enumerate(zip(images, colours)):
        row, column = divmod(index, 3)
        axis = figure.add_subplot(grid[row, 3 + column])
        axis.imshow(np.moveaxis(image, 0, -1), interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_title(f"sensor image {index + 1}", fontsize=8, color=colour)
        for spine in axis.spines.values():
            spine.set_color(colour)
            spine.set_linewidth(2.0)
    figure.text(0.46, 0.50, r"$z\longmapsto X(z)\in\mathbb{R}^{3072}$", fontsize=13)
    figure.text(
        0.70,
        0.025,
        "the model sees only these 32 x 32 x 3 response images",
        ha="center",
        fontsize=9,
    )
    figure.suptitle(
        "One low-dimensional source position generates one high-dimensional observation",
        fontsize=11,
        y=0.98,
    )
    figure.savefig(HERE / "sensor_source_to_image.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_example_figure(images: np.ndarray) -> None:
    figure, axes = plt.subplots(2, 6, figsize=(8.4, 3.0))
    chosen = np.linspace(0, len(images) - 1, 12, dtype=int)
    for axis, index in zip(axes.flat, chosen):
        axis.imshow(np.moveaxis(images[index], 0, -1), interpolation="nearest")
        axis.set_xticks([])
        axis.set_yticks([])
    figure.suptitle(
        "Observed data: 3,072-dimensional sensor images (source labels hidden)",
        fontsize=11,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.92), pad=0.25)
    figure.savefig(HERE / "sensor_examples.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_reconstruction_figure(
    originals: np.ndarray,
    reconstructions: dict[str, np.ndarray],
) -> None:
    chosen = np.array([3, 91, 208, 277])
    columns = [("Input", originals)] + [
        (next(s.label for s in SPECS if s.key == key), reconstructions[key])
        for key in ["r2_gw", "t2_gw", "s2_recon", "s2_gw"]
    ]
    figure, axes = plt.subplots(len(chosen), len(columns), figsize=(8.4, 6.8))
    for row_index, index in enumerate(chosen):
        for column_index, (label, images) in enumerate(columns):
            axes[row_index, column_index].imshow(
                np.moveaxis(images[index], 0, -1), interpolation="nearest"
            )
            axes[row_index, column_index].set_xticks([])
            axes[row_index, column_index].set_yticks([])
            if row_index == 0:
                axes[row_index, column_index].set_title(label, fontsize=10, pad=7)
        axes[row_index, 0].set_ylabel(
            f"source {row_index + 1}",
            rotation=0,
            ha="right",
            va="center",
            fontsize=9,
        )
    figure.tight_layout(pad=0.45, w_pad=0.35, h_pad=0.45)
    figure.savefig(HERE / "sensor_reconstructions.png", dpi=180, bbox_inches="tight")
    plt.close(figure)


def save_accuracy_figure(
    resolution_rows: list[dict[str, float]],
    noise_rows: list[dict[str, float]],
) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(8.4, 3.2))
    counts = np.asarray([row["sensor_count"] for row in resolution_rows])
    errors = np.asarray([row["rmse"] for row in resolution_rows])
    axes[0].loglog(counts, errors, "o-", color="#326b9b", linewidth=1.8)
    guide = errors[0] * counts[0] / counts
    axes[0].loglog(counts, guide, "--", color="#888888", label=r"$D^{-1}$ guide")
    axes[0].axvline(IMAGE_SIZE**2, color="#b43b3b", alpha=0.55, linewidth=1)
    axes[0].set_xlabel(r"number of sensors $D$")
    axes[0].set_ylabel(r"metric RMSE $\varepsilon_{\rm sensor}$")
    axes[0].set_title("finite-sensor discretization")
    axes[0].grid(alpha=0.25, which="both")
    axes[0].legend(frameon=False, fontsize=8)

    noise = np.asarray([row["noise_std"] for row in noise_rows])
    means = np.asarray([row["rmse_mean"] for row in noise_rows])
    stds = np.asarray([row["rmse_std"] for row in noise_rows])
    axes[1].plot(noise, means, "o-", color="#b43b3b", linewidth=1.8)
    axes[1].fill_between(noise, means - stds, means + stds, color="#b43b3b", alpha=0.18)
    axes[1].axvline(NOISE_STD, color="#333333", linestyle="--", linewidth=1)
    axes[1].set_xlabel(r"sensor noise $\sigma$")
    axes[1].set_ylabel("metric RMSE")
    axes[1].set_title("noise sensitivity at 1,024 sensors")
    axes[1].grid(alpha=0.25)
    figure.tight_layout(pad=0.8)
    figure.savefig(HERE / "sensor_accuracy.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


def save_diagnostics_figure(
    results: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
    scatter: dict[str, tuple[np.ndarray, np.ndarray]],
) -> None:
    colours = {
        "r2_gw": "#777777",
        "t2_gw": "#d78b2d",
        "s2_recon": "#5e83ba",
        "s2_gw": "#b33a3a",
    }
    figure, axes = plt.subplots(1, 3, figsize=(10.2, 3.1))
    for spec in SPECS:
        rows = histories[spec.key]
        axes[0].plot(
            [row["step"] for row in rows],
            [row["reconstruction"] for row in rows],
            label=spec.label,
            color=colours[spec.key],
            linewidth=1.4,
        )
    axes[0].set_xlabel("training step")
    axes[0].set_ylabel("reconstruction MSE")
    axes[0].set_yscale("log")
    axes[0].grid(alpha=0.25)

    truth, learned = scatter["s2_gw"]
    axes[1].scatter(
        truth,
        learned,
        s=8,
        alpha=0.33,
        color=colours["s2_gw"],
        rasterized=True,
    )
    axes[1].plot([0, 1], [0, 1], color="black", linewidth=1, linestyle="--")
    axes[1].set_xlim(0, 1)
    axes[1].set_ylim(0, 1)
    axes[1].set_xlabel(r"true $S^2$ chordal distance")
    axes[1].set_ylabel("learned latent distance")
    axes[1].set_title("distance recovery", fontsize=10)
    axes[1].grid(alpha=0.2)

    labels = [str(row["short_label"]) for row in results]
    x = np.arange(len(results))
    scores = [float(row["selection_score"]) for row in results]
    bar_colours = [colours[str(row["model"])] for row in results]
    bars = axes[2].bar(x, scores, width=0.68, color=bar_colours)
    best_index = int(np.argmin(scores))
    bars[best_index].set_edgecolor("black")
    bars[best_index].set_linewidth(1.5)
    for bar, score in zip(bars, scores):
        axes[2].text(
            bar.get_x() + bar.get_width() / 2,
            score * 1.16,
            f"{score:.4f}",
            ha="center",
            va="bottom",
            fontsize=7,
        )
    axes[2].set_xticks(x, labels, rotation=18, ha="right")
    axes[2].set_ylabel("held-out selection score")
    axes[2].set_title(r"model selection: $S^2$ wins", fontsize=10)
    axes[2].set_yscale("log")
    axes[2].set_ylim(min(scores) * 0.65, max(scores) * 1.55)
    axes[2].grid(axis="y", alpha=0.25)

    handles, legend_labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles, legend_labels, loc="upper center", ncol=4, frameon=False, fontsize=8
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88), pad=0.65)
    figure.savefig(HERE / "sensor_diagnostics.png", dpi=190, bbox_inches="tight")
    plt.close(figure)


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
        score_text = f"{score:.5f}"
        if math.isclose(score, best_score):
            score_text = rf"\textbf{{{score_text}}}"
        lines.append(
            f"{row['latex_label']} & {float(row['reconstruction_mse']):.4f} & "
            f"{float(row['mgw_stress']):.3f} & {score_text} & "
            f"{float(row['true_sphere_stress']):.3f} & "
            f"{float(row['true_10nn_recall']):.3f} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / "sensor_manifold_table.tex").write_text("\n".join(lines) + "\n")


def main() -> None:
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    directions, weights = sensor_grid(IMAGE_SIZE)
    train_sources = sample_sphere(N_TRAIN, rng)
    valid_sources = sample_sphere(N_VALID, rng)
    train_images_np = render_sensor_images(train_sources, directions, rng)
    valid_images_np = render_sensor_images(valid_sources, directions, rng)
    train_observation = calibrated_sensor_metric(train_images_np, weights, NOISE_STD)
    valid_observation = calibrated_sensor_metric(valid_images_np, weights, NOISE_STD)
    valid_true = true_sphere_distances(valid_sources)
    observation_correlation = pair_correlation(valid_observation, valid_true)
    observation_rmse = pair_rmse(valid_observation, valid_true)
    print(
        f"sensor metric: correlation={observation_correlation:.5f}, "
        f"RMSE={observation_rmse:.5f}",
        flush=True,
    )

    study_sources = sample_sphere(240, np.random.default_rng(SEED + 50))
    resolution_rows, noise_rows = sensor_accuracy_study(study_sources)

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
                "t2_gw": "T2+MGW",
                "s2_recon": "S2-only",
                "s2_gw": "S2+MGW",
            }[spec.key],
            "latex_label": spec.label,
            "geometry": spec.geometry,
            "metric_weight": spec.metric_weight,
            "selected_restart": float(selected_restart),
            **metrics,
            "observation_true_correlation": observation_correlation,
            "observation_true_rmse": observation_rmse,
            "sensor_count": float(IMAGE_SIZE * IMAGE_SIZE),
            "noise_std": NOISE_STD,
        }
        results.append(result)
        histories[spec.key] = history
        reconstructions[spec.key] = reconstructed
        scatter_rng = np.random.default_rng(SEED + len(spec.key))
        all_pairs = np.column_stack(np.triu_indices(N_VALID, k=1))
        selected = all_pairs[scatter_rng.choice(len(all_pairs), 700, replace=False)]
        truth = valid_true[selected[:, 0], selected[:, 1]]
        learned = latent_metric[selected[:, 0], selected[:, 1]]
        scatter_for_plot[spec.key] = (truth, learned)
        scatter_rows.extend(
            {
                "model": spec.key,
                "true_sphere_distance": float(truth_value),
                "learned_distance": float(learned_value),
            }
            for truth_value, learned_value in zip(truth, learned)
        )
        print(
            f"  recon={metrics['reconstruction_mse']:.5f} "
            f"MGW={metrics['mgw_stress']:.4f} "
            f"true={metrics['true_sphere_stress']:.4f} "
            f"recall={metrics['true_10nn_recall']:.3f}",
            flush=True,
        )

    write_csvs(
        results,
        histories,
        scatter_rows,
        resolution_rows,
        noise_rows,
    )
    write_latex_table(results)
    save_sensor_explainer_figure(directions)
    save_example_figure(valid_images_np)
    save_reconstruction_figure(valid_images_np, reconstructions)
    save_accuracy_figure(resolution_rows, noise_rows)
    save_diagnostics_figure(results, histories, scatter_for_plot)
    print(f"wrote results to {HERE}")


if __name__ == "__main__":
    main()
