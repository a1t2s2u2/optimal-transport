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
"""Controlled recovery of Wasserstein pullback geometry.

The latent coordinates are Mercator coordinates on a sphere.  At every
coordinate, a diagonal Gaussian distribution is observed.  A neural decoder
is fitted only from coordinates and Gaussian samples; the sphere, its metric,
and its curvature are withheld until evaluation.

Run with:
    uv run --python 3.12 wasserstein_cartography.py
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
from matplotlib import font_manager
from matplotlib.colors import Normalize
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from torch import Tensor, nn
from torch.func import jacrev, vmap

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
SEED = 20260730
N_LONGITUDE = 72
N_MERCATOR = 41
MERCATOR_LIMIT = 2.5
EVALUATION_LIMIT = 2.0
SAMPLES_PER_STATE = 64
TRAIN_FRACTION = 0.70
WIDTH = 96
DEPTH = 3
STEPS = 2400
BATCH_SIZE = 256
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 2.0e-6
HISTORY_EVERY = 40
REPEATS = 5

# The decoder F=(m,sigma) is an isometric embedding of the unit sphere:
# ||F(s)-F(t)||^2 = (MEAN_SCALE^2 + STD_SCALE^2)||s-t||^2.
MEAN_SCALE = 0.8
STD_SCALE = 0.6
STD_OFFSET = 0.75
MIN_STD = 0.04


@dataclass(frozen=True)
class ModelSpec:
    key: str
    label: str
    activation: str


MODEL_SPECS = (
    ModelSpec("tanh", "Tanh (smooth)", "tanh"),
    ModelSpec("softplus", "Softplus (smooth)", "softplus"),
    ModelSpec("relu", "ReLU (piecewise affine)", "relu"),
)

MODEL_LABELS_JA = {
    "tanh": "Tanh（滑らか）",
    "softplus": "Softplus（滑らか）",
    "relu": "ReLU（区分アフィン）",
}


def japanese_font_family() -> str:
    """Return an installed Japanese font without hard-coding a platform."""
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
    print(
        "warning: no Japanese font found; Japanese plot glyphs may be missing",
        flush=True,
    )
    return "sans-serif"


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(min(6, max(1, torch.get_num_threads())))


def mercator_grid() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    longitude = np.linspace(-math.pi, math.pi, N_LONGITUDE, endpoint=False)
    mercator = np.linspace(-MERCATOR_LIMIT, MERCATOR_LIMIT, N_MERCATOR)
    longitude_grid, mercator_grid_values = np.meshgrid(
        longitude, mercator, indexing="xy"
    )
    coordinates = np.column_stack(
        [longitude_grid.ravel(), mercator_grid_values.ravel()]
    )
    sphere = inverse_mercator(coordinates)
    return coordinates.astype(np.float64), sphere, mercator


def inverse_mercator(coordinates: np.ndarray) -> np.ndarray:
    longitude = coordinates[:, 0]
    mercator = coordinates[:, 1]
    radial = 1.0 / np.cosh(mercator)
    return np.column_stack(
        [radial * np.cos(longitude), radial * np.sin(longitude), np.tanh(mercator)]
    )


def true_parameters(sphere: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    mean = MEAN_SCALE * sphere
    standard_deviation = STD_OFFSET + STD_SCALE * sphere
    parameters = np.concatenate([mean, standard_deviation], axis=1)
    return mean, standard_deviation, parameters


def observed_statistics(
    mean: np.ndarray,
    standard_deviation: np.ndarray,
    rng: np.random.Generator,
    sample_count: int = SAMPLES_PER_STATE,
) -> tuple[np.ndarray, np.ndarray]:
    noise = rng.normal(size=(len(mean), sample_count, 3))
    samples = mean[:, None, :] + standard_deviation[:, None, :] * noise
    empirical_mean = samples.mean(axis=1)
    empirical_variance = ((samples - empirical_mean[:, None, :]) ** 2).mean(axis=1)
    return empirical_mean, empirical_variance


class GaussianDecoder(nn.Module):
    def __init__(self, activation: str) -> None:
        super().__init__()
        if activation == "tanh":
            activation_factory = nn.Tanh
        elif activation == "softplus":
            activation_factory = lambda: nn.Softplus(beta=2.0)
        elif activation == "relu":
            activation_factory = nn.ReLU
        else:
            raise ValueError(f"unknown activation: {activation}")

        layers: list[nn.Module] = []
        in_features = 3
        for _ in range(DEPTH):
            layers.extend([nn.Linear(in_features, WIDTH), activation_factory()])
            in_features = WIDTH
        layers.append(nn.Linear(WIDTH, 6))
        self.network = nn.Sequential(*layers)
        self.double()

    @staticmethod
    def features(coordinates: Tensor) -> Tensor:
        longitude = coordinates[..., 0]
        mercator = coordinates[..., 1]
        return torch.stack(
            [torch.cos(longitude), torch.sin(longitude), mercator / MERCATOR_LIMIT],
            dim=-1,
        )

    def raw_parameters(self, coordinates: Tensor) -> Tensor:
        raw = self.network(self.features(coordinates))
        offset = torch.tensor(
            [0.0, 0.0, 0.0, STD_OFFSET, STD_OFFSET, STD_OFFSET],
            dtype=raw.dtype,
            device=raw.device,
        )
        return raw + offset

    def forward(self, coordinates: Tensor) -> Tensor:
        parameters = self.raw_parameters(coordinates)
        mean = parameters[..., :3]
        standard_deviation = torch.clamp(parameters[..., 3:], min=MIN_STD)
        return torch.cat([mean, standard_deviation], dim=-1)


def gaussian_nll(
    parameters: Tensor, empirical_mean: Tensor, empirical_variance: Tensor
) -> Tensor:
    mean = parameters[:, :3]
    standard_deviation = parameters[:, 3:]
    variance = standard_deviation.square()
    return (
        torch.log(standard_deviation)
        + 0.5 * (empirical_variance + (empirical_mean - mean).square()) / variance
    ).mean()


def train_decoder(
    spec: ModelSpec,
    coordinates: Tensor,
    empirical_mean: Tensor,
    empirical_variance: Tensor,
    train_indices: np.ndarray,
    seed: int,
) -> tuple[GaussianDecoder, list[dict[str, float]]]:
    torch.manual_seed(seed)
    model = GaussianDecoder(spec.activation)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )
    generator = torch.Generator().manual_seed(seed + sum(map(ord, spec.key)))
    train_tensor = torch.as_tensor(train_indices, dtype=torch.long)
    history: list[dict[str, float]] = []

    for step in range(1, STEPS + 1):
        chosen = train_tensor[
            torch.randint(
                len(train_tensor),
                (min(BATCH_SIZE, len(train_tensor)),),
                generator=generator,
            )
        ]
        raw = model.raw_parameters(coordinates[chosen])
        predicted = torch.cat([raw[:, :3], torch.clamp(raw[:, 3:], min=MIN_STD)], dim=1)
        loss = gaussian_nll(
            predicted, empirical_mean[chosen], empirical_variance[chosen]
        )
        positivity_penalty = torch.relu(MIN_STD - raw[:, 3:]).square().mean()
        objective = loss + 100.0 * positivity_penalty
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        optimizer.step()

        if step == 1 or step % HISTORY_EVERY == 0 or step == STEPS:
            with torch.no_grad():
                full_parameters = model(coordinates[train_tensor])
                full_loss = gaussian_nll(
                    full_parameters,
                    empirical_mean[train_tensor],
                    empirical_variance[train_tensor],
                )
            history.append({"step": float(step), "nll": float(full_loss)})
    return model, history


def derivatives(
    model: GaussianDecoder, coordinates: Tensor
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    def single(point: Tensor) -> Tensor:
        return model(point.unsqueeze(0)).squeeze(0)

    jacobian_function = jacrev(single)
    hessian_function = jacrev(jacobian_function)
    jacobian = vmap(jacobian_function)(coordinates)
    hessian = vmap(hessian_function)(coordinates)
    return (
        model(coordinates).detach().cpu().numpy(),
        jacobian.detach().cpu().numpy(),
        hessian.detach().cpu().numpy(),
    )


def metric_and_curvature(
    jacobian: np.ndarray, hessian: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    metric = np.einsum("noi,noj->nij", jacobian, jacobian)
    curvature = np.empty(len(metric), dtype=np.float64)
    identity = np.eye(jacobian.shape[1])
    for index, (j_value, h_value, g_value) in enumerate(zip(jacobian, hessian, metric)):
        inverse = np.linalg.inv(g_value)
        projection = j_value @ inverse @ j_value.T
        normal = identity - projection
        b11 = normal @ h_value[:, 0, 0]
        b12 = normal @ h_value[:, 0, 1]
        b22 = normal @ h_value[:, 1, 1]
        numerator = float(b11 @ b22 - b12 @ b12)
        curvature[index] = numerator / np.linalg.det(g_value)
    return metric, curvature


def true_metric(coordinates: np.ndarray) -> np.ndarray:
    conformal_squared = 1.0 / np.cosh(coordinates[:, 1]) ** 2
    metric = np.zeros((len(coordinates), 2, 2), dtype=np.float64)
    metric[:, 0, 0] = conformal_squared
    metric[:, 1, 1] = conformal_squared
    return metric


def build_local_graph(parameters: np.ndarray) -> coo_matrix:
    rows: list[int] = []
    columns: list[int] = []
    values: list[float] = []
    offsets = [
        (delta_latitude, delta_longitude)
        for delta_latitude in range(-2, 3)
        for delta_longitude in range(-2, 3)
        if (delta_latitude, delta_longitude) != (0, 0)
    ]
    for latitude_index in range(N_MERCATOR):
        for longitude_index in range(N_LONGITUDE):
            source = latitude_index * N_LONGITUDE + longitude_index
            for delta_latitude, delta_longitude in offsets:
                target_latitude = latitude_index + delta_latitude
                if not 0 <= target_latitude < N_MERCATOR:
                    continue
                target_longitude = (longitude_index + delta_longitude) % N_LONGITUDE
                target = target_latitude * N_LONGITUDE + target_longitude
                if target <= source:
                    continue
                weight = float(np.linalg.norm(parameters[source] - parameters[target]))
                rows.extend([source, target])
                columns.extend([target, source])
                values.extend([weight, weight])
    size = N_LONGITUDE * N_MERCATOR
    return coo_matrix((values, (rows, columns)), shape=(size, size))


def wrapped_flat_distance(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    delta_longitude = np.abs(first[:, 0] - second[:, 0])
    delta_longitude = np.minimum(delta_longitude, 2.0 * math.pi - delta_longitude)
    delta_mercator = first[:, 1] - second[:, 1]
    return np.sqrt(delta_longitude**2 + delta_mercator**2)


def evaluation_pairs(
    coordinates: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray]:
    candidates = np.flatnonzero(np.abs(coordinates[:, 1]) <= EVALUATION_LIMIT)
    sources = rng.choice(candidates, size=80, replace=False)
    repeated_sources = np.repeat(sources, 10)
    targets = rng.choice(candidates, size=len(repeated_sources), replace=True)
    keep = targets != repeated_sources
    return repeated_sources[keep], targets[keep]


def pairwise_metrics(
    coordinates: np.ndarray,
    sphere: np.ndarray,
    parameters: np.ndarray,
    graph: coo_matrix,
    sources: np.ndarray,
    targets: np.ndarray,
) -> dict[str, np.ndarray]:
    unique_sources, inverse = np.unique(sources, return_inverse=True)
    graph_distances = dijkstra(graph.tocsr(), directed=False, indices=unique_sources)
    local_chain = graph_distances[inverse, targets]
    dot = np.sum(sphere[sources] * sphere[targets], axis=1)
    great_circle = np.arccos(np.clip(dot, -1.0, 1.0))
    chord = np.linalg.norm(parameters[sources] - parameters[targets], axis=1)
    flat = wrapped_flat_distance(coordinates[sources], coordinates[targets])
    return {
        "truth": great_circle,
        "local_chain": local_chain,
        "global_w2": chord,
        "flat": flat,
    }


def surface_error(
    parameters: np.ndarray, sphere: np.ndarray
) -> tuple[float, np.ndarray]:
    centered = parameters - parameters.mean(axis=0, keepdims=True)
    left, singular, _ = np.linalg.svd(centered, full_matrices=False)
    embedded = left[:, :3] * singular[:3]
    u_value, _, v_value = np.linalg.svd(embedded.T @ sphere)
    aligned = embedded @ (u_value @ v_value)
    error = float(np.sqrt(np.mean(np.sum((aligned - sphere) ** 2, axis=1))))
    return error, aligned


def rmse(estimate: np.ndarray, truth: np.ndarray) -> float:
    return float(np.sqrt(np.mean((estimate - truth) ** 2)))


def evaluate(
    model: GaussianDecoder,
    coordinates: np.ndarray,
    sphere: np.ndarray,
    true_parameters_value: np.ndarray,
    empirical_mean: Tensor,
    empirical_variance: Tensor,
    test_indices: np.ndarray,
    pair_sources: np.ndarray,
    pair_targets: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    geometry_indices = np.flatnonzero(np.abs(coordinates[:, 1]) <= EVALUATION_LIMIT)
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float64)
    geometry_tensor = coordinate_tensor[geometry_indices]
    _, jacobian, hessian = derivatives(model, geometry_tensor)
    predicted_metric, predicted_curvature = metric_and_curvature(jacobian, hessian)
    target_metric = true_metric(coordinates[geometry_indices])
    relative_metric_error = np.linalg.norm(
        predicted_metric - target_metric, axis=(1, 2)
    ) / np.linalg.norm(target_metric, axis=(1, 2))

    with torch.no_grad():
        all_parameters = model(coordinate_tensor).cpu().numpy()
        minimum_standard_deviation = float(all_parameters[:, 3:].min())
        if minimum_standard_deviation <= MIN_STD + 1.0e-8:
            raise RuntimeError(
                "the positivity clamp is active; the evaluated parameter map "
                "is not differentiable at all reported states"
            )
        test_nll = gaussian_nll(
            model(coordinate_tensor[test_indices]),
            empirical_mean[test_indices],
            empirical_variance[test_indices],
        )
    pair_values = pairwise_metrics(
        coordinates,
        sphere,
        all_parameters,
        build_local_graph(all_parameters),
        pair_sources,
        pair_targets,
    )
    pca_error, embedded = surface_error(all_parameters, sphere)
    test_parameter_rmse = rmse(
        all_parameters[test_indices], true_parameters_value[test_indices]
    )
    metrics = {
        "test_nll": float(test_nll),
        "minimum_standard_deviation": minimum_standard_deviation,
        "parameter_rmse": test_parameter_rmse,
        "metric_relative_error": float(np.mean(relative_metric_error)),
        "metric_relative_median": float(np.median(relative_metric_error)),
        "curvature_mae": float(np.mean(np.abs(predicted_curvature - 1.0))),
        "curvature_median_ae": float(np.median(np.abs(predicted_curvature - 1.0))),
        "geodesic_rmse": rmse(pair_values["local_chain"], pair_values["truth"]),
        "global_w2_rmse": rmse(pair_values["global_w2"], pair_values["truth"]),
        "flat_rmse": rmse(pair_values["flat"], pair_values["truth"]),
        "surface_rmse": pca_error,
    }
    diagnostics = {
        "geometry_indices": geometry_indices,
        "parameters": all_parameters,
        "metric": predicted_metric,
        "curvature": predicted_curvature,
        "embedded": embedded,
        **pair_values,
    }
    return metrics, diagnostics


def oracle_evaluation(
    coordinates: np.ndarray,
    sphere: np.ndarray,
    parameters: np.ndarray,
    pair_sources: np.ndarray,
    pair_targets: np.ndarray,
) -> tuple[dict[str, float], dict[str, np.ndarray]]:
    pairs = pairwise_metrics(
        coordinates,
        sphere,
        parameters,
        build_local_graph(parameters),
        pair_sources,
        pair_targets,
    )
    surface_rmse_value, embedded = surface_error(parameters, sphere)
    metrics = {
        "test_nll": float("nan"),
        "minimum_standard_deviation": float(parameters[:, 3:].min()),
        "parameter_rmse": 0.0,
        "metric_relative_error": 0.0,
        "metric_relative_median": 0.0,
        "curvature_mae": 0.0,
        "curvature_median_ae": 0.0,
        "geodesic_rmse": rmse(pairs["local_chain"], pairs["truth"]),
        "global_w2_rmse": rmse(pairs["global_w2"], pairs["truth"]),
        "flat_rmse": rmse(pairs["flat"], pairs["truth"]),
        "surface_rmse": surface_rmse_value,
    }
    diagnostics = {"parameters": parameters, "embedded": embedded, **pairs}
    return metrics, diagnostics


def gaussian_projection_image(
    mean: np.ndarray, standard_deviation: np.ndarray
) -> np.ndarray:
    grid = np.linspace(-3.0, 3.0, 64)
    first, second = np.meshgrid(grid, grid, indexing="xy")
    pairs = ((0, 1), (1, 2), (2, 0))
    channels = []
    for left_index, right_index in pairs:
        exponent = -0.5 * (
            ((first - mean[left_index]) / standard_deviation[left_index]) ** 2
            + ((second - mean[right_index]) / standard_deviation[right_index]) ** 2
        )
        density = np.exp(exponent)
        density /= max(float(density.max()), 1.0e-12)
        channels.append(density)
    return np.stack(channels, axis=-1)


def save_pipeline_figure(
    coordinates: np.ndarray,
    sphere: np.ndarray,
    true_parameters_value: np.ndarray,
    diagnostics: dict[str, np.ndarray],
    language: str = "en",
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    selected = [
        N_LONGITUDE * 8 + 8,
        N_LONGITUDE * 15 + 26,
        N_LONGITUDE * 25 + 44,
        N_LONGITUDE * 33 + 62,
    ]
    colours = ["#ca3b33", "#e49a26", "#4f9c45", "#2779ad"]
    figure = plt.figure(figsize=(12.2, 4.0))
    grid = figure.add_gridspec(2, 7, width_ratios=[2.0, 0.18, 1, 1, 0.18, 2.0, 0.08])
    map_axis = figure.add_subplot(grid[:, 0])
    sphere_axis = figure.add_subplot(grid[:, 5], projection="3d")

    map_axis.scatter(
        coordinates[:, 0], coordinates[:, 1], s=1.2, color="#a8b2ba", alpha=0.45
    )
    for number, (index, colour) in enumerate(zip(selected, colours), start=1):
        map_axis.scatter(*coordinates[index], s=45, color=colour, edgecolor="black")
        map_axis.text(
            coordinates[index, 0] + 0.10,
            coordinates[index, 1] + 0.08,
            str(number),
            color=colour,
            weight="bold",
        )
        image_axis = figure.add_subplot(grid[(number - 1) // 2, 2 + (number - 1) % 2])
        image = gaussian_projection_image(
            true_parameters_value[index, :3], true_parameters_value[index, 3:]
        )
        image_axis.imshow(image, origin="lower", interpolation="bilinear")
        image_axis.set_xticks([])
        image_axis.set_yticks([])
        distribution_label = "分布" if japanese else "distribution"
        image_axis.set_title(f"{distribution_label} {number}", color=colour, fontsize=9)
        for spine in image_axis.spines.values():
            spine.set_color(colour)
            spine.set_linewidth(1.8)

    map_axis.set_xlabel(r"経度 $\lambda$" if japanese else r"longitude $\lambda$")
    map_axis.set_ylabel(
        r"メルカトル座標 $y$" if japanese else r"Mercator coordinate $y$"
    )
    map_axis.set_title("歪んだ潜在座標" if japanese else "distorted latent chart")
    map_axis.set_xlim(-math.pi, math.pi)
    map_axis.set_ylim(-MERCATOR_LIMIT, MERCATOR_LIMIT)

    embedded = diagnostics["embedded"]
    sphere_axis.scatter(
        embedded[:, 0],
        embedded[:, 1],
        embedded[:, 2],
        c=sphere[:, 2],
        cmap="coolwarm",
        s=4,
        alpha=0.42,
    )
    for number, (index, colour) in enumerate(zip(selected, colours), start=1):
        sphere_axis.scatter(*embedded[index], s=50, color=colour, edgecolor="black")
        sphere_axis.text(
            *(1.08 * embedded[index]), str(number), color=colour, weight="bold"
        )
    sphere_axis.set_axis_off()
    sphere_axis.set_box_aspect((1, 1, 1))
    sphere_axis.view_init(elev=20, azim=-58)
    sphere_axis.set_title(
        "学習されたデコーダー曲面（PCA）"
        if japanese
        else "learned decoder surface (PCA)",
        pad=0,
    )
    figure.text(0.205, 0.49, r"$\longrightarrow$", fontsize=20)
    figure.text(0.665, 0.49, r"$\longrightarrow$", fontsize=20)
    title = (
        "Wasserstein地図学：平面座標の背後にある球面を分布から復元"
        if japanese
        else "Wasserstein cartography: distributions reveal the sphere hidden by a flat chart"
    )
    figure.suptitle(title, fontsize=12, y=0.99)
    figure.tight_layout()
    figure.savefig(
        HERE / f"cartography_pipeline{suffix}.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def save_geometry_figure(
    coordinates: np.ndarray,
    diagnostics_by_model: dict[str, dict[str, np.ndarray]],
    language: str = "en",
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    geometry_indices = diagnostics_by_model["tanh"]["geometry_indices"].astype(int)
    geometry_coordinates = coordinates[geometry_indices]
    longitude_values = np.unique(geometry_coordinates[:, 0])
    mercator_values = np.unique(geometry_coordinates[:, 1])
    extent = [-math.pi, math.pi, mercator_values.min(), mercator_values.max()]
    figure, axes = plt.subplots(2, 3, figsize=(11.6, 6.4), sharex=True, sharey=True)
    display_models = ("tanh", "softplus", "relu")
    for column, key in enumerate(display_models):
        metric = diagnostics_by_model[key]["metric"]
        scale = np.sqrt(np.sqrt(np.maximum(np.linalg.det(metric), 0.0)))
        curvature = diagnostics_by_model[key]["curvature"]
        scale_image = scale.reshape(len(mercator_values), len(longitude_values))
        curvature_image = curvature.reshape(len(mercator_values), len(longitude_values))
        first = axes[0, column].imshow(
            scale_image,
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="viridis",
            vmin=0.2,
            vmax=1.05,
        )
        second = axes[1, column].imshow(
            np.clip(curvature_image, -0.5, 2.5),
            origin="lower",
            aspect="auto",
            extent=extent,
            cmap="coolwarm",
            norm=Normalize(vmin=-0.5, vmax=2.5),
        )
        model_label = MODEL_LABELS_JA[key] if japanese else MODEL_SPECS[column].label
        axes[0, column].set_title(model_label)
        axes[1, column].set_xlabel(
            r"経度 $\lambda$" if japanese else r"longitude $\lambda$"
        )
    axes[0, 0].set_ylabel("$y$\n局所長さ尺度" if japanese else "$y$\nmetric scale")
    axes[1, 0].set_ylabel("$y$\nガウス曲率" if japanese else "$y$\nGaussian curvature")
    figure.colorbar(first, ax=axes[0, :], shrink=0.72, label=r"$(\det G)^{1/4}$")
    figure.colorbar(second, ax=axes[1, :], shrink=0.72, label=r"$K$")
    title = (
        "同程度の尤度でも微分幾何は一致しない"
        if japanese
        else "Comparable likelihood does not imply comparable differential geometry"
    )
    figure.suptitle(title, fontsize=12)
    figure.subplots_adjust(
        left=0.08, right=0.91, bottom=0.10, top=0.90, wspace=0.08, hspace=0.18
    )
    figure.savefig(
        HERE / f"cartography_geometry{suffix}.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def save_distance_figure(
    diagnostics_by_model: dict[str, dict[str, np.ndarray]],
    language: str = "en",
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    proposed = diagnostics_by_model["tanh"]
    truth = proposed["truth"]
    figure, axes = plt.subplots(1, 2, figsize=(9.8, 3.8))
    axes[0].scatter(
        truth,
        proposed["flat"],
        s=9,
        alpha=0.25,
        label="平面座標" if japanese else "flat chart",
    )
    axes[0].scatter(
        truth,
        proposed["global_w2"],
        s=9,
        alpha=0.25,
        label=r"大域 $W_2$ 弦" if japanese else r"global $W_2$ chord",
    )
    axes[0].scatter(
        truth,
        proposed["local_chain"],
        s=9,
        alpha=0.25,
        label=r"局所 $W_2$ 鎖" if japanese else r"local $W_2$ chain",
    )
    limit = float(max(truth.max(), proposed["flat"].max()))
    axes[0].plot([0, limit], [0, limit], "k--", linewidth=1)
    axes[0].set_xlim(0, math.pi)
    axes[0].set_ylim(0, min(limit, 5.0))
    axes[0].set_xlabel("大円距離" if japanese else "great-circle distance")
    axes[0].set_ylabel("推定距離" if japanese else "estimated distance")
    axes[0].legend(frameon=False, fontsize=8)
    axes[0].set_title(
        "局所輸送は内在距離を復元する"
        if japanese
        else "Local transport recovers intrinsic travel"
    )

    oracle = diagnostics_by_model["oracle"]
    gap = oracle["truth"] - oracle["global_w2"]
    order = np.argsort(oracle["truth"])
    distance_grid = np.linspace(0.0, math.pi, 200)
    axes[1].scatter(
        oracle["truth"],
        gap,
        s=9,
        alpha=0.25,
        color="#2779ad",
        label="観測された差" if japanese else "observed gap",
    )
    axes[1].plot(
        distance_grid,
        distance_grid - 2.0 * np.sin(distance_grid / 2.0),
        color="black",
        linewidth=1.6,
        label=r"$d-2\sin(d/2)$",
    )
    local = oracle["truth"][order] <= 0.9
    ordered_truth = oracle["truth"][order]
    axes[1].plot(
        ordered_truth[local],
        ordered_truth[local] ** 3 / 24.0,
        "--",
        color="#ca3b33",
        linewidth=1.4,
        label=r"局所則 $d^3/24$" if japanese else r"local law $d^3/24$",
    )
    axes[1].set_xlabel(r"内在距離 $d$" if japanese else r"intrinsic distance $d$")
    axes[1].set_ylabel(r"$d-W_2$")
    axes[1].set_title(
        "弦の近道誤差は局所的に3次"
        if japanese
        else "The chord shortcut is cubic locally"
    )
    axes[1].legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        HERE / f"cartography_distances{suffix}.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def save_training_figure(
    histories: dict[str, list[dict[str, float]]], language: str = "en"
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    figure, axis = plt.subplots(figsize=(5.5, 3.5))
    colours = {"tanh": "#2779ad", "softplus": "#4f9c45", "relu": "#ca3b33"}
    for spec in MODEL_SPECS:
        rows = histories[spec.key]
        axis.plot(
            [row["step"] for row in rows],
            [row["nll"] for row in rows],
            label=MODEL_LABELS_JA[spec.key] if japanese else spec.label,
            color=colours[spec.key],
        )
    axis.set_xlabel("最適化ステップ" if japanese else "optimization step")
    axis.set_ylabel("学習時ガウスNLL" if japanese else "training Gaussian NLL")
    axis.set_title(
        "すべてのデコーダーが観測分布に適合"
        if japanese
        else "All decoders fit the observed distributions"
    )
    axis.legend(frameon=False, fontsize=8)
    figure.tight_layout()
    figure.savefig(
        HERE / f"cartography_training{suffix}.png", dpi=220, bbox_inches="tight"
    )
    plt.close(figure)


def formatted_summary_metric(row: dict[str, str | float], metric: str) -> str:
    return f"${row[metric]:.4f}\\pm{row[f'{metric}_std']:.4f}$"


def write_outputs(
    results: list[dict[str, str | float]],
    summary: list[dict[str, str | float]],
    histories: dict[str, list[dict[str, float]]],
) -> None:
    with (HERE / "cartography_results.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(results[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(results)
    with (HERE / "cartography_summary.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(summary[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(summary)
    history_rows = [
        {"model": key, **row} for key, values in histories.items() for row in values
    ]
    with (HERE / "cartography_history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(history_rows[0].keys()), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history_rows)

    by_key = {row["model"]: row for row in summary}
    with (HERE / "cartography_table.tex").open("w") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(
            "Decoder & Param. RMSE & Metric rel. & Curvature MAE & Geodesic RMSE & Surface RMSE \\\\\n"
        )
        handle.write("\\midrule\n")
        for key in ("oracle", "tanh", "softplus", "relu"):
            row = by_key[key]
            label = {
                "oracle": "Oracle $F_*$",
                "tanh": "Tanh",
                "softplus": "Softplus",
                "relu": "ReLU",
            }[key]
            handle.write(
                f"{label} & {formatted_summary_metric(row, 'parameter_rmse')} & "
                f"{formatted_summary_metric(row, 'metric_relative_error')} & "
                f"{formatted_summary_metric(row, 'curvature_mae')} & "
                f"{formatted_summary_metric(row, 'geodesic_rmse')} & "
                f"{formatted_summary_metric(row, 'surface_rmse')} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")

    with (HERE / "cartography_table_ja.tex").open("w") as handle:
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\toprule\n")
        handle.write(
            "デコーダー & パラメータRMSE & 計量相対誤差 & 曲率MAE & "
            "測地RMSE & 曲面RMSE \\\\\n"
        )
        handle.write("\\midrule\n")
        for key in ("oracle", "tanh", "softplus", "relu"):
            row = by_key[key]
            label = {
                "oracle": "真の写像 $F_*$",
                "tanh": "Tanh",
                "softplus": "Softplus",
                "relu": "ReLU",
            }[key]

            handle.write(
                f"{label} & {formatted_summary_metric(row, 'parameter_rmse')} & "
                f"{formatted_summary_metric(row, 'metric_relative_error')} & "
                f"{formatted_summary_metric(row, 'curvature_mae')} & "
                f"{formatted_summary_metric(row, 'geodesic_rmse')} & "
                f"{formatted_summary_metric(row, 'surface_rmse')} \\\\\n"
            )
        handle.write("\\bottomrule\n")
        handle.write("\\end{tabular}\n")


def summarize_results(
    results: list[dict[str, str | float]],
) -> list[dict[str, str | float]]:
    metric_names = [
        key
        for key, value in results[0].items()
        if key not in {"trial", "model", "label"} and isinstance(value, float)
    ]
    summary: list[dict[str, str | float]] = []
    for key, label in [
        ("oracle", "Oracle decoder"),
        *[(spec.key, spec.label) for spec in MODEL_SPECS],
    ]:
        selected = [row for row in results if row["model"] == key]
        aggregate: dict[str, str | float] = {"model": key, "label": label}
        for metric in metric_names:
            values = np.asarray([float(row[metric]) for row in selected])
            finite = values[np.isfinite(values)]
            aggregate[metric] = float(finite.mean()) if len(finite) else float("nan")
            aggregate[f"{metric}_std"] = (
                float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
            )
        summary.append(aggregate)
    return summary


def main() -> None:
    seed_everything(SEED)
    coordinates, sphere, _ = mercator_grid()
    true_mean, true_standard_deviation, parameters = true_parameters(sphere)
    coordinate_tensor = torch.as_tensor(coordinates, dtype=torch.float64)
    results: list[dict[str, str | float]] = []
    diagnostics_by_model: dict[str, dict[str, np.ndarray]] = {}
    histories: dict[str, list[dict[str, float]]] = {}

    for trial in range(REPEATS):
        trial_seed = SEED + 1009 * trial
        seed_everything(trial_seed)
        rng = np.random.default_rng(trial_seed)
        empirical_mean_np, empirical_variance_np = observed_statistics(
            true_mean, true_standard_deviation, rng
        )
        empirical_mean = torch.as_tensor(empirical_mean_np, dtype=torch.float64)
        empirical_variance = torch.as_tensor(empirical_variance_np, dtype=torch.float64)
        permutation = rng.permutation(len(coordinates))
        split = int(TRAIN_FRACTION * len(permutation))
        train_indices = np.sort(permutation[:split])
        test_indices = np.sort(permutation[split:])
        pair_sources, pair_targets = evaluation_pairs(coordinates, rng)

        oracle_metrics, oracle_diagnostics = oracle_evaluation(
            coordinates, sphere, parameters, pair_sources, pair_targets
        )
        results.append(
            {
                "trial": float(trial),
                "model": "oracle",
                "label": "Oracle decoder",
                **oracle_metrics,
            }
        )
        if trial == 0:
            diagnostics_by_model["oracle"] = oracle_diagnostics

        for spec in MODEL_SPECS:
            print(f"trial {trial + 1}/{REPEATS}: training {spec.label}", flush=True)
            model, history = train_decoder(
                spec,
                coordinate_tensor,
                empirical_mean,
                empirical_variance,
                train_indices,
                seed=trial_seed,
            )
            metrics, diagnostics = evaluate(
                model,
                coordinates,
                sphere,
                parameters,
                empirical_mean,
                empirical_variance,
                test_indices,
                pair_sources,
                pair_targets,
            )
            results.append(
                {
                    "trial": float(trial),
                    "model": spec.key,
                    "label": spec.label,
                    **metrics,
                }
            )
            if trial == 0:
                diagnostics_by_model[spec.key] = diagnostics
                histories[spec.key] = history
            print(
                f"  parameter={metrics['parameter_rmse']:.4f} "
                f"metric={metrics['metric_relative_error']:.4f} "
                f"curvature={metrics['curvature_mae']:.4f} "
                f"geodesic={metrics['geodesic_rmse']:.4f}",
                flush=True,
            )

    summary = summarize_results(results)
    write_outputs(results, summary, histories)
    save_pipeline_figure(coordinates, sphere, parameters, diagnostics_by_model["tanh"])
    save_geometry_figure(coordinates, diagnostics_by_model)
    save_distance_figure(diagnostics_by_model)
    save_training_figure(histories)
    with plt.rc_context({"font.family": japanese_font_family()}):
        save_pipeline_figure(
            coordinates,
            sphere,
            parameters,
            diagnostics_by_model["tanh"],
            language="ja",
        )
        save_geometry_figure(coordinates, diagnostics_by_model, language="ja")
        save_distance_figure(diagnostics_by_model, language="ja")
        save_training_figure(histories, language="ja")
    print("wrote Wasserstein cartography results", flush=True)


if __name__ == "__main__":
    main()
