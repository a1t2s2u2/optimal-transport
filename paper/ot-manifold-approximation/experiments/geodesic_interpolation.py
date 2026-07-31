#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib>=3.9",
#   "numpy>=2.0",
#   "torch>=2.4",
#   "torchvision>=0.19",
# ]
# ///
"""Decoded straight lines and numerical intrinsic-path candidates of a 2-D VAE.

The frozen decoder has a smooth trunk ``h`` and a fixed-covariance Gaussian
mean head ``F(z) = W h(z) + b``.  Hence its exact pullback metric is

    G(z) = J_h(z).T @ W.T @ W @ J_h(z).

This script deliberately keeps the geometry on the *unclipped* Gaussian mean.
Clipping to [0, 1] is used only when rendering images.  It produces

* a latent plot and three decoded sequences: the ordinary affine straight
  line, the same straight route reparameterized by teacher arc length, and a
  constant-speed numerical teacher-geodesic candidate;
* a CSV comparing path length, decoded-frame W2 step uniformity, and teacher
  versus low-rank-student geometry; and
* an NPZ containing all paths so that a paper figure is auditable.

The computed path is a multistart discrete-energy minimizer, not a certified
global geodesic.  Lengths are recomputed at two quadrature resolutions and the
discrepancy is reported.  A lower-length candidate is nevertheless enough to
show that the affine straight line is not length-minimizing.

Train the requested teacher first with ``mnist_low_rank_geometry.py``.  Then,
from the repository root, run for example

    uv run --python 3.12 \
      paper/ot-manifold-approximation/experiments/geodesic_interpolation.py \
      --dataset mnist --student-ranks 8 12
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import dataclass
from itertools import pairwise
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torch import Tensor, nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DATA_CACHE = CACHE / "mnist"
SEED = 20260731
IMAGE_SIDE = 28
IMAGE_DIM = IMAGE_SIDE**2
LATENT_DIM = 2
ENCODER_WIDTH = 128
ENCODER_FEATURES = 64
TRUNK_WIDTH = 64
FEATURE_DIM = 64
GEOMETRY_TRACE_WEIGHT = 4.0
RIDGE_RELATIVE = 1.0e-6


@dataclass(frozen=True)
class DatasetSpec:
    cli_name: str
    prefix: str
    display_name: str
    class_names: tuple[str, ...]
    class_names_ja: tuple[str, ...]

    @property
    def checkpoint(self) -> Path:
        return CACHE / f"{self.prefix}_gaussian_vae_v2.pt"

    def artifact(self, suffix: str) -> Path:
        return HERE / f"{self.prefix}_{suffix}"


SPECS = {
    "mnist": DatasetSpec(
        "mnist",
        "mnist",
        "MNIST",
        tuple(str(index) for index in range(10)),
        tuple(str(index) for index in range(10)),
    ),
    "fashion-mnist": DatasetSpec(
        "fashion-mnist",
        "fashion_mnist",
        "FashionMNIST",
        (
            "T-shirt",
            "trouser",
            "pullover",
            "dress",
            "coat",
            "sandal",
            "shirt",
            "sneaker",
            "bag",
            "boot",
        ),
        (
            "Tシャツ",
            "ズボン",
            "プルオーバー",
            "ドレス",
            "コート",
            "サンダル",
            "シャツ",
            "スニーカー",
            "バッグ",
            "ブーツ",
        ),
    ),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", choices=tuple(SPECS), default="mnist")
    parser.add_argument("--student-ranks", nargs="*", type=int, default=[8, 12])
    parser.add_argument("--pool-size", type=int, default=3000)
    parser.add_argument("--calibration-size", type=int, default=4096)
    parser.add_argument("--candidate-pairs", type=int, default=800)
    parser.add_argument("--segments", type=int, default=48)
    parser.add_argument("--optimization-steps", type=int, default=700)
    parser.add_argument("--restarts", type=int, default=3)
    parser.add_argument("--frames", type=int, default=11)
    parser.add_argument("--aggregate-pairs", type=int, default=100)
    parser.add_argument("--bootstrap-resamples", type=int, default=2000)
    parser.add_argument("--geometry-trace-weight", type=float, default=4.0)
    return parser.parse_args()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)


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


class GaussianVAE(nn.Module):
    """Architecture shared with the dataset training script."""

    def __init__(self) -> None:
        super().__init__()
        self.encoder_trunk = nn.Sequential(
            nn.Linear(IMAGE_DIM, ENCODER_WIDTH),
            nn.Tanh(),
            nn.Linear(ENCODER_WIDTH, ENCODER_FEATURES),
            nn.Tanh(),
        )
        self.encoder_mean = nn.Linear(ENCODER_FEATURES, LATENT_DIM)
        self.encoder_log_variance = nn.Linear(ENCODER_FEATURES, LATENT_DIM)
        self.decoder_trunk = nn.Sequential(
            nn.Linear(LATENT_DIM, TRUNK_WIDTH),
            nn.Tanh(),
            nn.Linear(TRUNK_WIDTH, FEATURE_DIM),
            nn.Tanh(),
        )
        self.decoder_head = nn.Linear(FEATURE_DIM, IMAGE_DIM)

    def encode(self, images: Tensor) -> tuple[Tensor, Tensor]:
        features = self.encoder_trunk(images.flatten(start_dim=1))
        return self.encoder_mean(features), self.encoder_log_variance(features)

    def decoder_features(self, latent: Tensor) -> Tensor:
        return self.decoder_trunk(latent)


class MetricDecoder:
    """A fixed trunk and one mean-head weight, with analytic directional JVP."""

    def __init__(self, model: GaussianVAE, weight: np.ndarray, name: str) -> None:
        self.name = name
        self.first: nn.Linear = model.decoder_trunk[0]  # type: ignore[assignment]
        self.second: nn.Linear = model.decoder_trunk[2]  # type: ignore[assignment]
        self.weight = torch.as_tensor(weight, dtype=torch.float64)
        self.bias = model.decoder_head.bias.detach().to(dtype=torch.float64)
        self.gram = self.weight.T @ self.weight

    def features(self, latent: Tensor) -> Tensor:
        first = torch.tanh(latent @ self.first.weight.T + self.first.bias)
        return torch.tanh(first @ self.second.weight.T + self.second.bias)

    def decode(self, latent: Tensor) -> Tensor:
        return self.features(latent) @ self.weight.T + self.bias

    def feature_tangent(self, latent: Tensor, tangent: Tensor) -> Tensor:
        first = torch.tanh(latent @ self.first.weight.T + self.first.bias)
        first_tangent = (tangent @ self.first.weight.T) * (1.0 - first.square())
        second = torch.tanh(first @ self.second.weight.T + self.second.bias)
        return (first_tangent @ self.second.weight.T) * (1.0 - second.square())

    def squared_speed(self, latent: Tensor, tangent: Tensor) -> Tensor:
        feature_tangent = self.feature_tangent(latent, tangent)
        return torch.einsum(
            "...q,qr,...r->...",
            feature_tangent,
            self.gram,
            feature_tangent,
        )

    def metric_eigenvalues(self, latent: Tensor) -> Tensor:
        basis = torch.eye(LATENT_DIM, dtype=torch.float64)
        tangent = basis[None].expand(len(latent), -1, -1)
        points = latent[:, None, :].expand(-1, LATENT_DIM, -1)
        feature_jacobian_columns = self.feature_tangent(points, tangent)
        metric = torch.einsum(
            "ndq,qr,ner->nde",
            feature_jacobian_columns,
            self.gram,
            feature_jacobian_columns,
        )
        return torch.linalg.eigvalsh(metric)


def load_model(spec: DatasetSpec) -> GaussianVAE:
    if not spec.checkpoint.exists():
        raise FileNotFoundError(
            f"missing {spec.checkpoint}; train it with "
            f"mnist_low_rank_geometry.py --dataset {spec.cli_name}"
        )
    payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
    model = GaussianVAE()
    model.load_state_dict(payload["state_dict"])
    model.eval().double()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


def load_datasets(
    spec: DatasetSpec,
) -> tuple[
    datasets.MNIST | datasets.FashionMNIST,
    datasets.MNIST | datasets.FashionMNIST,
]:
    dataset_type = datasets.MNIST if spec.cli_name == "mnist" else datasets.FashionMNIST
    transform = transforms.ToTensor()
    return (
        dataset_type(DATA_CACHE, train=True, download=True, transform=transform),
        dataset_type(DATA_CACHE, train=False, download=True, transform=transform),
    )


@torch.no_grad()
def collect(
    model: GaussianVAE,
    dataset: datasets.MNIST | datasets.FashionMNIST,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    loader = DataLoader(Subset(dataset, indices), batch_size=512, num_workers=0)
    images: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        batch_images = batch_images.to(dtype=torch.float64)
        mean, _ = model.encode(batch_images)
        images.append(batch_images.flatten(start_dim=1).numpy())
        codes.append(mean.numpy())
        labels.append(batch_labels.numpy())
    return np.concatenate(images), np.concatenate(codes), np.concatenate(labels)


@torch.no_grad()
def features_and_jacobians(
    model: GaussianVAE, latent: np.ndarray, batch_size: int = 512
) -> tuple[np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    jacobian_batches: list[np.ndarray] = []
    first: nn.Linear = model.decoder_trunk[0]  # type: ignore[assignment]
    second: nn.Linear = model.decoder_trunk[2]  # type: ignore[assignment]
    for start in range(0, len(latent), batch_size):
        points = torch.as_tensor(
            latent[start : start + batch_size], dtype=torch.float64
        )
        first_value = torch.tanh(points @ first.weight.T + first.bias)
        feature = torch.tanh(first_value @ second.weight.T + second.bias)
        first_jacobian = (1.0 - first_value.square())[:, :, None] * first.weight[None]
        second_jacobian = torch.einsum("fq,nqd->nfd", second.weight, first_jacobian)
        second_jacobian *= (1.0 - feature.square())[:, :, None]
        feature_batches.append(feature.numpy())
        jacobian_batches.append(second_jacobian.numpy())
    return np.concatenate(feature_batches), np.concatenate(jacobian_batches)


def symmetric_square_roots(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(float(eigenvalues.max()) * 1.0e-12, 1.0e-14)
    eigenvalues = np.maximum(eigenvalues, floor)
    root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse = (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T
    return root, inverse


def geometry_students(
    model: GaussianVAE,
    calibration_codes: np.ndarray,
    ranks: list[int],
    trace_weight: float,
) -> dict[str, MetricDecoder]:
    features, jacobians = features_and_jacobians(model, calibration_codes)
    value = features.T @ features / len(features)
    jacobian = np.einsum("nqd,nrd->qr", jacobians, jacobians) / len(jacobians)
    multiplier = trace_weight * np.trace(value) / max(np.trace(jacobian), 1.0e-12)
    ridge = RIDGE_RELATIVE * np.trace(value) / FEATURE_DIM
    covariance = value + multiplier * jacobian + ridge * np.eye(FEATURE_DIM)
    root, inverse = symmetric_square_roots(covariance)
    teacher_weight = model.decoder_head.weight.detach().numpy()
    left, singular, right = np.linalg.svd(teacher_weight @ root, full_matrices=False)
    decoders: dict[str, MetricDecoder] = {
        "teacher": MetricDecoder(model, teacher_weight, "teacher")
    }
    for rank in sorted(set(ranks)):
        if not 1 <= rank <= FEATURE_DIM:
            raise ValueError(f"student rank must be in [1, {FEATURE_DIM}]: {rank}")
        truncated = (left[:, :rank] * singular[:rank]) @ right[:rank]
        weight = truncated @ inverse
        decoders[f"rank_{rank}"] = MetricDecoder(model, weight, f"rank {rank}")
    print(
        f"geometry covariance multiplier={multiplier:.6g}, ridge={ridge:.3e}",
        flush=True,
    )
    return decoders


def coefficient_of_variation(values: np.ndarray) -> float:
    mean = float(np.mean(values))
    return float(np.std(values) / max(mean, 1.0e-12))


@torch.no_grad()
def decoded_chords(decoder: MetricDecoder, nodes: np.ndarray) -> np.ndarray:
    values = decoder.decode(torch.as_tensor(nodes, dtype=torch.float64)).numpy()
    return np.linalg.norm(np.diff(values, axis=0), axis=1)


def straight_nodes(start: np.ndarray, end: np.ndarray, count: int) -> np.ndarray:
    time = np.linspace(0.0, 1.0, count)[:, None]
    return (1.0 - time) * start + time * end


def choose_endpoints(
    teacher: MetricDecoder,
    images: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    candidate_count: int,
    rng: np.random.Generator,
) -> tuple[int, int, np.ndarray, np.ndarray]:
    """Choose a diagnostic pair without using optimized-geodesic length.

    Endpoints must be central, well reconstructed posterior means with
    different labels and moderate standardized separation.  Among a fixed
    random candidate pool, the score is the ordinary straight interpolation's
    decoded-W2 step CV plus its output-curve excess over the endpoint chord.
    Aggregate experiments should use every eligible pair, not only this figure
    selection.
    """

    points = torch.as_tensor(codes, dtype=torch.float64)
    with torch.no_grad():
        reconstructions = teacher.decode(points).numpy()
    reconstruction_mse = np.mean((reconstructions - images) ** 2, axis=1)
    low = np.quantile(codes, 0.05, axis=0)
    high = np.quantile(codes, 0.95, axis=0)
    quality = np.quantile(reconstruction_mse, 0.75)
    eligible = np.flatnonzero(
        np.all((codes >= low) & (codes <= high), axis=1)
        & (reconstruction_mse <= quality)
    )
    if len(eligible) < 20:
        raise RuntimeError("too few central, well-reconstructed endpoint candidates")

    first = rng.choice(eligible, size=candidate_count, replace=True)
    second = rng.choice(eligible, size=candidate_count, replace=True)
    valid = (first != second) & (labels[first] != labels[second])
    first, second = first[valid], second[valid]
    covariance = np.cov(codes[eligible].T) + 1.0e-8 * np.eye(LATENT_DIM)
    inverse = np.linalg.inv(covariance)
    differences = codes[first] - codes[second]
    separation = np.sqrt(np.einsum("ni,ij,nj->n", differences, inverse, differences))
    lower, upper = np.quantile(separation, [0.35, 0.80])
    moderate = (separation >= lower) & (separation <= upper)
    first, second = first[moderate], second[moderate]
    if not len(first):
        raise RuntimeError("no endpoint pairs satisfy the separation protocol")

    best_score = -math.inf
    best = (int(first[0]), int(second[0]))
    for left, right in zip(first, second):
        path = straight_nodes(codes[left], codes[right], 17)
        steps = decoded_chords(teacher, path)
        endpoints = teacher.decode(
            torch.as_tensor(path[[0, -1]], dtype=torch.float64)
        ).numpy()
        endpoint_chord = float(np.linalg.norm(endpoints[1] - endpoints[0]))
        output_curve_excess = float(np.sum(steps) / max(endpoint_chord, 1.0e-12) - 1.0)
        score = coefficient_of_variation(steps) + max(output_curve_excess, 0.0)
        if score > best_score:
            best_score = score
            best = (int(left), int(right))

    domain_low = np.quantile(codes, 0.005, axis=0)
    domain_high = np.quantile(codes, 0.995, axis=0)
    span = domain_high - domain_low
    domain_low -= 0.03 * span
    domain_high += 0.03 * span
    print(
        f"diagnostic endpoints pool indices={best}, labels="
        f"{labels[best[0]]}->{labels[best[1]]}, selection score={best_score:.4f}",
        flush=True,
    )
    return best[0], best[1], domain_low, domain_high


def discrete_energy(decoder: MetricDecoder, nodes: Tensor) -> Tensor:
    """Midpoint discretization of the teacher pullback energy.

    Optimizing adjacent decoded chords can spuriously jump between distant
    latent points whose decoder outputs happen to be close.  The midpoint
    directional derivative instead evaluates the declared local pullback
    metric on every latent edge.  It converges to Riemannian energy as the
    maximum edge length vanishes.
    """

    increments = nodes[..., 1:, :] - nodes[..., :-1, :]
    midpoints = 0.5 * (nodes[..., 1:, :] + nodes[..., :-1, :])
    return increments.shape[-2] * decoder.squared_speed(midpoints, increments).sum(
        dim=-1
    )


@torch.no_grad()
def densify(path: np.ndarray, subdivisions: int) -> np.ndarray:
    pieces: list[np.ndarray] = []
    fractions = np.arange(subdivisions, dtype=np.float64) / subdivisions
    for start, end in pairwise(path):
        pieces.append((1.0 - fractions[:, None]) * start + fractions[:, None] * end)
    pieces.append(path[-1:])
    return np.concatenate(pieces)


@torch.no_grad()
def path_length(
    decoder: MetricDecoder, path: np.ndarray, subdivisions: int = 8
) -> float:
    dense = densify(path, subdivisions)
    increments = torch.as_tensor(np.diff(dense, axis=0), dtype=torch.float64)
    midpoints = torch.as_tensor(0.5 * (dense[1:] + dense[:-1]), dtype=torch.float64)
    return float(
        torch.sqrt(
            torch.clamp_min(decoder.squared_speed(midpoints, increments), 0.0)
        ).sum()
    )


@torch.no_grad()
def resample_by_length(
    decoder: MetricDecoder,
    path: np.ndarray,
    count: int,
    subdivisions: int = 12,
) -> np.ndarray:
    dense = densify(path, subdivisions)
    increments = torch.as_tensor(np.diff(dense, axis=0), dtype=torch.float64)
    midpoints = torch.as_tensor(0.5 * (dense[1:] + dense[:-1]), dtype=torch.float64)
    lengths = torch.sqrt(
        torch.clamp_min(decoder.squared_speed(midpoints, increments), 0.0)
    ).numpy()
    cumulative = np.concatenate([[0.0], np.cumsum(lengths)])
    if cumulative[-1] <= 1.0e-12:
        raise RuntimeError(f"degenerate path under {decoder.name} metric")
    targets = np.linspace(0.0, cumulative[-1], count)
    result = np.column_stack(
        [
            np.interp(targets, cumulative, dense[:, coordinate])
            for coordinate in range(2)
        ]
    )
    result[0], result[-1] = path[0], path[-1]
    return result


def optimize_geodesic(
    decoder: MetricDecoder,
    start: np.ndarray,
    end: np.ndarray,
    domain_low: np.ndarray,
    domain_high: np.ndarray,
    segments: int,
    steps: int,
    restarts: int,
) -> np.ndarray:
    start_tensor = torch.as_tensor(start, dtype=torch.float64)
    end_tensor = torch.as_tensor(end, dtype=torch.float64)
    low = torch.as_tensor(domain_low, dtype=torch.float64)
    high = torch.as_tensor(domain_high, dtype=torch.float64)
    line = torch.stack(
        [
            (1.0 - time) * start_tensor + time * end_tensor
            for time in torch.linspace(0.0, 1.0, segments + 1, dtype=torch.float64)
        ]
    )
    direction = end_tensor - start_tensor
    normal = torch.stack([-direction[1], direction[0]])
    normal /= torch.clamp_min(torch.linalg.vector_norm(normal), 1.0e-12)
    best_path: np.ndarray | None = None
    best_length = math.inf

    for restart in range(restarts):
        initial = line.clone()
        if restart:
            sign = -1.0 if restart % 2 else 1.0
            amplitude = (
                sign
                * (0.10 + 0.04 * (restart // 2))
                * torch.linalg.vector_norm(direction)
            )
            bump = torch.sin(
                torch.linspace(0.0, math.pi, segments + 1, dtype=torch.float64)
            )
            initial += amplitude * bump[:, None] * normal
            initial = torch.minimum(torch.maximum(initial, low), high)
            initial[0], initial[-1] = start_tensor, end_tensor

        interior = initial[1:-1].clone().requires_grad_(True)
        optimizer = torch.optim.Adam([interior], lr=0.025)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(steps, 1), eta_min=0.001
        )
        for _ in range(steps):
            nodes = torch.cat([start_tensor[None], interior, end_tensor[None]])
            objective = discrete_energy(decoder, nodes)
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            torch.nn.utils.clip_grad_norm_([interior], max_norm=50.0)
            optimizer.step()
            scheduler.step()
            with torch.no_grad():
                interior.copy_(torch.minimum(torch.maximum(interior, low), high))

        candidate = torch.cat(
            [start_tensor[None], interior.detach(), end_tensor[None]]
        ).numpy()
        candidate = resample_by_length(decoder, candidate, segments + 1)
        candidate_length = path_length(decoder, candidate, subdivisions=12)
        if candidate_length < best_length:
            best_length = candidate_length
            best_path = candidate

    if best_path is None:
        raise RuntimeError("geodesic optimization produced no path")
    print(f"{decoder.name}: numerical geodesic length={best_length:.6f}", flush=True)
    return best_path


def sample_endpoint_pairs(
    teacher: MetricDecoder,
    images: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    count: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    """Sample a fixed aggregate set without inspecting geodesic outcomes."""

    points = torch.as_tensor(codes, dtype=torch.float64)
    with torch.no_grad():
        reconstructions = teacher.decode(points).numpy()
    reconstruction_mse = np.mean((reconstructions - images) ** 2, axis=1)
    low = np.quantile(codes, 0.05, axis=0)
    high = np.quantile(codes, 0.95, axis=0)
    quality = np.quantile(reconstruction_mse, 0.75)
    eligible = np.flatnonzero(
        np.all((codes >= low) & (codes <= high), axis=1)
        & (reconstruction_mse <= quality)
    )
    pool_size = max(10_000, 100 * count)
    first = rng.choice(eligible, size=pool_size, replace=True)
    second = rng.choice(eligible, size=pool_size, replace=True)
    valid = (first != second) & (labels[first] != labels[second])
    first, second = first[valid], second[valid]
    covariance = np.cov(codes[eligible].T) + 1.0e-8 * np.eye(LATENT_DIM)
    inverse = np.linalg.inv(covariance)
    differences = codes[first] - codes[second]
    separation = np.sqrt(np.einsum("ni,ij,nj->n", differences, inverse, differences))
    separation_low, separation_high = np.quantile(separation, [0.35, 0.80])
    valid = (separation >= separation_low) & (separation <= separation_high)
    pairs: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    used_endpoints: set[int] = set()
    for left, right in zip(first[valid], second[valid]):
        left, right = int(left), int(right)
        key = tuple(sorted((int(left), int(right))))
        if key in seen or left in used_endpoints or right in used_endpoints:
            continue
        seen.add(key)
        used_endpoints.update((left, right))
        pairs.append((left, right))
        if len(pairs) == count:
            break
    if len(pairs) != count:
        raise RuntimeError(f"selected only {len(pairs)} aggregate endpoint pairs")
    return (
        np.asarray([pair[0] for pair in pairs]),
        np.asarray([pair[1] for pair in pairs]),
    )


def optimize_geodesic_batch(
    decoder: MetricDecoder,
    starts: np.ndarray,
    ends: np.ndarray,
    domain_low: np.ndarray,
    domain_high: np.ndarray,
    segments: int,
    steps: int,
    restarts: int,
) -> np.ndarray:
    """Vectorized multistart energy minimization for aggregate statistics."""

    start = torch.as_tensor(starts, dtype=torch.float64)
    end = torch.as_tensor(ends, dtype=torch.float64)
    low = torch.as_tensor(domain_low, dtype=torch.float64)
    high = torch.as_tensor(domain_high, dtype=torch.float64)
    times = torch.linspace(0.0, 1.0, segments + 1, dtype=torch.float64)
    line = (1.0 - times[None, :, None]) * start[:, None, :] + times[
        None, :, None
    ] * end[:, None, :]
    direction = end - start
    normal = torch.stack([-direction[:, 1], direction[:, 0]], dim=-1)
    normal /= torch.linalg.vector_norm(normal, dim=-1, keepdim=True).clamp_min(1.0e-12)
    envelope = torch.sin(math.pi * times)[None, :, None]
    initial_paths: list[Tensor] = []
    for restart in range(restarts):
        if restart == 0:
            initial = line.clone()
        else:
            sign = -1.0 if restart % 2 else 1.0
            amplitude = (
                sign
                * (0.10 + 0.04 * (restart // 2))
                * torch.linalg.vector_norm(direction, dim=-1)
            )
            initial = line + amplitude[:, None, None] * envelope * normal[:, None, :]
            initial = torch.minimum(torch.maximum(initial, low), high)
            initial[:, 0], initial[:, -1] = start, end
        initial_paths.append(initial)
    initial = torch.stack(initial_paths, dim=1)
    interior = initial[:, :, 1:-1].clone().requires_grad_(True)
    optimizer = torch.optim.Adam([interior], lr=0.025)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(steps, 1), eta_min=0.001
    )
    for step in range(steps):
        nodes = torch.cat(
            [
                start[:, None, None, :].expand(-1, restarts, 1, -1),
                interior,
                end[:, None, None, :].expand(-1, restarts, 1, -1),
            ],
            dim=2,
        )
        objective = discrete_energy(decoder, nodes).mean()
        optimizer.zero_grad(set_to_none=True)
        objective.backward()
        torch.nn.utils.clip_grad_norm_([interior], max_norm=50.0)
        optimizer.step()
        scheduler.step()
        with torch.no_grad():
            interior.copy_(torch.minimum(torch.maximum(interior, low), high))
        if step in {0, steps // 2, steps - 1}:
            print(
                f"  aggregate step {step + 1:4d}/{steps}: energy={float(objective.detach()):.5f}",
                flush=True,
            )

    candidates = torch.cat(
        [
            start[:, None, None, :].expand(-1, restarts, 1, -1),
            interior.detach(),
            end[:, None, None, :].expand(-1, restarts, 1, -1),
        ],
        dim=2,
    ).numpy()
    candidates = np.concatenate([candidates, line[:, None].numpy()], axis=1)
    chosen: list[np.ndarray] = []
    for pair_paths in candidates:
        lengths = [path_length(decoder, path, 8) for path in pair_paths]
        chosen.append(pair_paths[int(np.argmin(lengths))])
    return np.asarray(chosen)


def aggregate_rows(
    teacher: MetricDecoder,
    passing_student: MetricDecoder,
    codes: np.ndarray,
    labels: np.ndarray,
    endpoint_left: np.ndarray,
    endpoint_right: np.ndarray,
    paths: np.ndarray,
    domain_low: np.ndarray,
    domain_high: np.ndarray,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for pair_index, (left, right, path) in enumerate(
        zip(endpoint_left, endpoint_right, paths)
    ):
        straight = straight_nodes(codes[left], codes[right], len(path))
        straight_length = path_length(teacher, straight, 16)
        geodesic_length = path_length(teacher, path, 16)
        student_length = path_length(passing_student, path, 16)
        coarse = path_length(teacher, path, 8)
        minimum_eigenvalue, maximum_condition_number = metric_diagnostics(teacher, path)
        rows.append(
            {
                "pair_index": pair_index,
                "left_pool_index": int(left),
                "right_pool_index": int(right),
                "left_label": int(labels[left]),
                "right_label": int(labels[right]),
                "straight_teacher_length": straight_length,
                "geodesic_candidate_teacher_length": geodesic_length,
                "straight_relative_excess": max(
                    straight_length / geodesic_length - 1.0, 0.0
                ),
                "passing_student": passing_student.name,
                "fixed_teacher_path_student_length_error": abs(
                    student_length / geodesic_length - 1.0
                ),
                "length_resolution_relative_gap": abs(geodesic_length / coarse - 1.0),
                "maximum_latent_edge": maximum_latent_edge(path),
                "normalized_boundary_clearance": normalized_boundary_clearance(
                    path, domain_low, domain_high
                ),
                "minimum_teacher_metric_eigenvalue": minimum_eigenvalue,
                "maximum_teacher_metric_condition_number": maximum_condition_number,
            }
        )
    return rows


def bootstrap_median_interval(
    values: np.ndarray,
    resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    indices = rng.integers(0, len(values), size=(resamples, len(values)))
    medians = np.median(values[indices], axis=1)
    lower, upper = np.quantile(medians, [0.025, 0.975])
    return float(lower), float(upper)


def write_aggregate_artifacts(
    spec: DatasetSpec,
    rows: list[dict[str, object]],
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> None:
    write_rows(spec.artifact("geodesic_aggregate.csv"), rows)
    excess = np.asarray([float(row["straight_relative_excess"]) for row in rows])
    student_error = np.asarray(
        [float(row["fixed_teacher_path_student_length_error"]) for row in rows]
    )
    resolution_gap = np.asarray(
        [float(row["length_resolution_relative_gap"]) for row in rows]
    )
    boundary = np.asarray([float(row["normalized_boundary_clearance"]) for row in rows])
    minimum_eigenvalues = np.asarray(
        [float(row["minimum_teacher_metric_eigenvalue"]) for row in rows]
    )
    maximum_condition_numbers = np.asarray(
        [float(row["maximum_teacher_metric_condition_number"]) for row in rows]
    )
    confidence_low, confidence_high = bootstrap_median_interval(
        excess, bootstrap_resamples, rng
    )
    summary = {
        "dataset": spec.display_name,
        "pair_count": len(rows),
        "straight_excess_median": float(np.median(excess)),
        "straight_excess_q25": float(np.quantile(excess, 0.25)),
        "straight_excess_q75": float(np.quantile(excess, 0.75)),
        "straight_excess_bootstrap_ci_low": confidence_low,
        "straight_excess_bootstrap_ci_high": confidence_high,
        "straight_excess_max": float(excess.max()),
        "student_path_error_median": float(np.median(student_error)),
        "student_path_error_max": float(student_error.max()),
        "resolution_gap_max": float(resolution_gap.max()),
        "minimum_normalized_boundary_clearance": float(boundary.min()),
        "minimum_teacher_metric_eigenvalue": float(minimum_eigenvalues.min()),
        "maximum_teacher_metric_condition_number": float(
            maximum_condition_numbers.max()
        ),
    }
    write_rows(spec.artifact("geodesic_summary.csv"), [summary])


def write_combined_geodesic_table() -> None:
    summaries: list[dict[str, str]] = []
    for key in ("mnist", "fashion-mnist"):
        spec = SPECS[key]
        path = spec.artifact("geodesic_summary.csv")
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as handle:
            summaries.extend(csv.DictReader(handle))
    english = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Dataset & Pairs & Straight excess median [IQR] & 95\% pair-bootstrap CI & Student path error med. / max \\",
        r"\midrule",
    ]
    japanese = [
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"データ & 組数 & 直線超過 中央値 [IQR] & 95\% pair-bootstrap CI & 学生経路長誤差 中央 / 最大 \\",
        r"\midrule",
    ]
    for row in summaries:
        values = (
            f"{row['dataset']} & {row['pair_count']} & "
            f"{100 * float(row['straight_excess_median']):.2f}\\% "
            f"[{100 * float(row['straight_excess_q25']):.2f}, "
            f"{100 * float(row['straight_excess_q75']):.2f}] & "
            f"[{100 * float(row['straight_excess_bootstrap_ci_low']):.2f}, "
            f"{100 * float(row['straight_excess_bootstrap_ci_high']):.2f}] & "
            f"{100 * float(row['student_path_error_median']):.2f}\\% / "
            f"{100 * float(row['student_path_error_max']):.2f}\\%"
        )
        english.append(values + r" \\")
        japanese.append(values + r" \\")
    ending = [r"\bottomrule", r"\end{tabular}"]
    (HERE / "realdata_geodesic_table.tex").write_text(
        "\n".join(english + ending) + "\n", encoding="utf-8"
    )
    (HERE / "realdata_geodesic_table_ja.tex").write_text(
        "\n".join(japanese + ending) + "\n", encoding="utf-8"
    )


def metric_diagnostics(decoder: MetricDecoder, path: np.ndarray) -> tuple[float, float]:
    dense = densify(path, 4)
    with torch.no_grad():
        eigenvalues = decoder.metric_eigenvalues(
            torch.as_tensor(dense, dtype=torch.float64)
        )
    smallest = eigenvalues[:, 0]
    condition_numbers = eigenvalues[:, -1] / smallest.clamp_min(1.0e-300)
    return float(smallest.min()), float(condition_numbers.max())


def minimum_metric_eigenvalue(decoder: MetricDecoder, path: np.ndarray) -> float:
    return metric_diagnostics(decoder, path)[0]


def max_step_relative_error(reference: np.ndarray, candidate: np.ndarray) -> float:
    return float(np.max(np.abs(candidate / np.maximum(reference, 1.0e-12) - 1.0)))


def maximum_latent_edge(path: np.ndarray) -> float:
    return float(np.linalg.norm(np.diff(path, axis=0), axis=1).max())


def normalized_boundary_clearance(
    path: np.ndarray,
    domain_low: np.ndarray,
    domain_high: np.ndarray,
) -> float:
    span = np.maximum(domain_high - domain_low, 1.0e-12)
    lower_clearance = (path - domain_low) / span
    upper_clearance = (domain_high - path) / span
    return float(np.minimum(lower_clearance, upper_clearance).min())


def evaluation_rows(
    decoders: dict[str, MetricDecoder],
    straight: np.ndarray,
    teacher_geodesic: np.ndarray,
    own_geodesics: dict[str, np.ndarray],
    frame_count: int,
    domain_low: np.ndarray,
    domain_high: np.ndarray,
) -> list[dict[str, object]]:
    teacher = decoders["teacher"]
    teacher_distance = path_length(teacher, teacher_geodesic, 16)
    teacher_line_length = path_length(teacher, straight, 16)
    teacher_frames = resample_by_length(teacher, teacher_geodesic, frame_count)
    teacher_steps = decoded_chords(teacher, teacher_frames)
    with torch.no_grad():
        teacher_values = teacher.decode(
            torch.as_tensor(teacher_frames, dtype=torch.float64)
        ).numpy()

    rows: list[dict[str, object]] = []
    for key, decoder in decoders.items():
        own = own_geodesics[key]
        own_distance = path_length(decoder, own, 16)
        line_length = path_length(decoder, straight, 16)
        teacher_path_length = path_length(decoder, teacher_geodesic, 16)
        own_teacher_length = path_length(teacher, own, 16)
        coarse = path_length(decoder, own, 8)
        fine = path_length(decoder, own, 16)
        common_steps = decoded_chords(decoder, teacher_frames)
        with torch.no_grad():
            values = decoder.decode(
                torch.as_tensor(teacher_frames, dtype=torch.float64)
            ).numpy()
        rows.append(
            {
                "decoder": key,
                "estimated_own_distance": own_distance,
                "distance_relative_error_vs_teacher": abs(
                    own_distance / teacher_distance - 1.0
                ),
                "straight_path_length": line_length,
                "straight_excess_over_own_geodesic": line_length / own_distance - 1.0,
                "teacher_geodesic_length_under_decoder": teacher_path_length,
                "fixed_teacher_geodesic_length_error": abs(
                    teacher_path_length / teacher_distance - 1.0
                ),
                "student_regret_on_teacher_path": teacher_path_length / own_distance
                - 1.0,
                "teacher_regret_on_student_path": own_teacher_length / teacher_distance
                - 1.0,
                "teacher_straight_excess": teacher_line_length / teacher_distance - 1.0,
                "teacher_geodesic_step_cv": coefficient_of_variation(common_steps),
                "teacher_geodesic_step_profile_max_error": max_step_relative_error(
                    teacher_steps, common_steps
                ),
                "teacher_path_output_rmse_per_pixel": float(
                    np.sqrt(np.mean((values - teacher_values) ** 2))
                ),
                "minimum_metric_eigenvalue_on_own_path": minimum_metric_eigenvalue(
                    decoder, own
                ),
                "length_resolution_relative_gap": abs(fine / coarse - 1.0),
                "maximum_latent_edge": maximum_latent_edge(own),
                "normalized_boundary_clearance": normalized_boundary_clearance(
                    own, domain_low, domain_high
                ),
            }
        )
    return rows


def write_rows(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def save_figure(
    spec: DatasetSpec,
    images: np.ndarray,
    codes: np.ndarray,
    labels: np.ndarray,
    endpoint_indices: tuple[int, int],
    teacher: MetricDecoder,
    passing_student: MetricDecoder,
    straight_affine_frames: np.ndarray,
    straight_arc_frames: np.ndarray,
    geodesic_frames: np.ndarray,
    language: str,
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    frame_count = len(straight_affine_frames)
    figure = plt.figure(figsize=(10.4, 5.8))
    outer = figure.add_gridspec(2, 1, height_ratios=[1.7, 4.6], hspace=0.24)
    top = outer[0].subgridspec(1, 2, width_ratios=[1.0, 1.35], wspace=0.25)
    latent_axis = figure.add_subplot(top[0, 0])
    step_axis = figure.add_subplot(top[0, 1])
    bottom = outer[1].subgridspec(5, frame_count, hspace=0.08, wspace=0.025)

    subset = np.linspace(0, len(codes) - 1, min(len(codes), 1600), dtype=int)
    latent_axis.scatter(
        codes[subset, 0],
        codes[subset, 1],
        c=labels[subset],
        cmap="tab10",
        s=5,
        alpha=0.20,
        linewidths=0,
    )
    latent_axis.plot(
        straight_affine_frames[:, 0],
        straight_affine_frames[:, 1],
        "--",
        color="#555555",
        lw=1.8,
        label="ユークリッド直線" if japanese else "Euclidean straight line",
    )
    latent_axis.plot(
        geodesic_frames[:, 0],
        geodesic_frames[:, 1],
        color="#1769aa",
        lw=2.4,
        label=(
            "教師の数値測地線候補"
            if japanese
            else "numerical teacher-geodesic candidate"
        ),
    )
    latent_axis.scatter(
        codes[list(endpoint_indices), 0],
        codes[list(endpoint_indices), 1],
        color=["#c9342f", "#27854a"],
        s=55,
        zorder=5,
    )
    latent_axis.set_xlabel("latent $z_1$")
    latent_axis.set_ylabel("latent $z_2$")
    latent_axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        fontsize=9,
    )
    latent_axis.grid(alpha=0.15)

    sequence_specs = (
        (
            straight_affine_frames,
            "直線：latent等間隔" if japanese else "straight: affine latent time",
            "#555555",
        ),
        (
            straight_arc_frames,
            "同じ直線：$W_2$等間隔"
            if japanese
            else "same straight route: teacher arc-length time",
            "#d87928",
        ),
        (
            geodesic_frames,
            (
                "数値測地線候補：$W_2$等間隔"
                if japanese
                else "numerical teacher candidate: constant speed"
            ),
            "#1769aa",
        ),
    )
    times = np.linspace(0.0, 1.0, frame_count)
    for nodes, label, color in sequence_specs:
        steps = decoded_chords(teacher, nodes)
        step_axis.plot(
            0.5 * (times[:-1] + times[1:]),
            steps,
            marker="o",
            ms=3.5,
            lw=1.6,
            color=color,
            label=f"{label} (CV={coefficient_of_variation(steps):.3f})",
        )
    step_axis.set_xlabel(
        "共通補間時刻 $t$" if japanese else "common interpolation time $t$"
    )
    step_axis.set_ylabel("教師 $W_2$ chord" if japanese else "teacher $W_2$ chord")
    step_axis.legend(
        frameon=True,
        facecolor="white",
        framealpha=0.9,
        edgecolor="none",
        fontsize=8.5,
    )
    step_axis.grid(alpha=0.18)

    with torch.no_grad():
        decoded = [
            teacher.decode(torch.as_tensor(nodes, dtype=torch.float64)).numpy()
            for nodes, _, _ in sequence_specs
        ]
        student_geodesic = passing_student.decode(
            torch.as_tensor(geodesic_frames, dtype=torch.float64)
        ).numpy()

    left, right = endpoint_indices
    for column in range(frame_count):
        axis = figure.add_subplot(bottom[0, column])
        axis.set_xticks([])
        axis.set_yticks([])
        axis.set_frame_on(False)
        if column == 0:
            axis.imshow(
                images[left].reshape(IMAGE_SIDE, IMAGE_SIDE),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_ylabel(
                "端点入力" if japanese else "endpoint inputs",
                fontsize=9,
                rotation=0,
                ha="right",
                va="center",
                labelpad=8,
            )
        elif column == frame_count - 1:
            axis.imshow(
                images[right].reshape(IMAGE_SIDE, IMAGE_SIDE),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )

    for row, ((_, row_label, color), decoded_images) in enumerate(
        zip(sequence_specs, decoded), start=1
    ):
        for column, image in enumerate(decoded_images):
            axis = figure.add_subplot(bottom[row, column])
            axis.imshow(
                np.clip(image, 0.0, 1.0).reshape(IMAGE_SIDE, IMAGE_SIDE),
                cmap="gray",
                vmin=0.0,
                vmax=1.0,
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column == 0:
                axis.set_ylabel(
                    row_label,
                    fontsize=9,
                    color=color,
                    rotation=0,
                    ha="right",
                    va="center",
                    labelpad=8,
                )

    student_label = (
        f"{passing_student.name}／同じ教師候補"
        if japanese
        else f"{passing_student.name} / same teacher candidate"
    )
    for column, image in enumerate(student_geodesic):
        axis = figure.add_subplot(bottom[4, column])
        axis.imshow(
            np.clip(image, 0.0, 1.0).reshape(IMAGE_SIDE, IMAGE_SIDE),
            cmap="gray",
            vmin=0.0,
            vmax=1.0,
        )
        axis.set_xticks([])
        axis.set_yticks([])
        if column == 0:
            axis.set_ylabel(
                student_label,
                fontsize=9,
                color="#27854a",
                rotation=0,
                ha="right",
                va="center",
                labelpad=8,
            )

    names = spec.class_names_ja if japanese else spec.class_names
    title = (
        f"{spec.display_name} posterior means: {names[int(labels[left])]} → "
        f"{names[int(labels[right])]} | 表示のみ[0,1]にclip"
        if japanese
        else f"{spec.display_name} posterior means: {names[int(labels[left])]} to "
        f"{names[int(labels[right])]} | clipping is for display only"
    )
    figure.suptitle(title, fontsize=11.5)
    figure.savefig(
        spec.artifact(f"geodesic_interpolation{suffix}.png"),
        dpi=220,
        bbox_inches="tight",
    )
    plt.close(figure)


def main() -> None:
    args = parse_args()
    spec = SPECS[args.dataset]
    seed_everything(SEED)
    rng = np.random.default_rng(SEED)
    model = load_model(spec)
    train_dataset, test_dataset = load_datasets(spec)

    images, codes, labels = collect(model, test_dataset, args.pool_size, SEED + 1)
    _, calibration_codes, _ = collect(
        model, train_dataset, args.calibration_size, SEED + 2
    )
    decoders = geometry_students(
        model,
        calibration_codes,
        args.student_ranks,
        args.geometry_trace_weight,
    )
    teacher = decoders["teacher"]
    left, right, domain_low, domain_high = choose_endpoints(
        teacher, images, codes, labels, args.candidate_pairs, rng
    )
    start, end = codes[left], codes[right]
    straight = straight_nodes(start, end, args.segments + 1)

    own_geodesics: dict[str, np.ndarray] = {}
    for key, decoder in decoders.items():
        own_geodesics[key] = optimize_geodesic(
            decoder,
            start,
            end,
            domain_low,
            domain_high,
            args.segments,
            args.optimization_steps,
            args.restarts,
        )
    teacher_geodesic = own_geodesics["teacher"]

    rows = evaluation_rows(
        decoders,
        straight,
        teacher_geodesic,
        own_geodesics,
        args.frames,
        domain_low,
        domain_high,
    )
    for row in rows:
        row.update(
            {
                "dataset": spec.cli_name,
                "left_pool_index": left,
                "right_pool_index": right,
                "left_label": int(labels[left]),
                "right_label": int(labels[right]),
                "segments": args.segments,
                "optimization_steps": args.optimization_steps,
                "restarts": args.restarts,
                "seed": SEED,
            }
        )
    write_rows(spec.artifact("geodesic_metrics.csv"), rows)

    passing_student = decoders[f"rank_{max(args.student_ranks)}"]
    aggregate_left, aggregate_right = sample_endpoint_pairs(
        teacher,
        images,
        codes,
        labels,
        args.aggregate_pairs,
        np.random.default_rng(SEED + 101),
    )
    aggregate_paths = optimize_geodesic_batch(
        teacher,
        codes[aggregate_left],
        codes[aggregate_right],
        domain_low,
        domain_high,
        args.segments,
        args.optimization_steps,
        args.restarts,
    )
    np.savez(
        spec.artifact("geodesic_aggregate_paths.npz"),
        paths=aggregate_paths,
        left_pool_indices=aggregate_left,
        right_pool_indices=aggregate_right,
        domain_low=domain_low,
        domain_high=domain_high,
    )
    aggregate = aggregate_rows(
        teacher,
        passing_student,
        codes,
        labels,
        aggregate_left,
        aggregate_right,
        aggregate_paths,
        domain_low,
        domain_high,
    )
    write_aggregate_artifacts(
        spec,
        aggregate,
        args.bootstrap_resamples,
        np.random.default_rng(SEED + 202),
    )
    write_combined_geodesic_table()

    straight_affine_frames = straight_nodes(start, end, args.frames)
    straight_arc_frames = resample_by_length(teacher, straight, args.frames)
    geodesic_frames = resample_by_length(teacher, teacher_geodesic, args.frames)
    save_figure(
        spec,
        images,
        codes,
        labels,
        (left, right),
        teacher,
        passing_student,
        straight_affine_frames,
        straight_arc_frames,
        geodesic_frames,
        "en",
    )
    save_figure(
        spec,
        images,
        codes,
        labels,
        (left, right),
        teacher,
        passing_student,
        straight_affine_frames,
        straight_arc_frames,
        geodesic_frames,
        "ja",
    )
    arrays: dict[str, np.ndarray] = {
        "straight": straight,
        "straight_affine_frames": straight_affine_frames,
        "straight_arc_frames": straight_arc_frames,
        "teacher_geodesic_frames": geodesic_frames,
        "endpoint_indices": np.asarray([left, right]),
    }
    arrays.update({f"{key}_geodesic": value for key, value in own_geodesics.items()})
    np.savez(spec.artifact("geodesic_paths.npz"), **arrays)
    print(f"wrote {spec.display_name} geodesic artifacts", flush=True)


if __name__ == "__main__":
    main()
