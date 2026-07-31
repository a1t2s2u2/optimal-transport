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
"""Geometry-aware low-rank distillation on MNIST or FashionMNIST.

The teacher has a two-dimensional latent code, a smooth Tanh decoder trunk
``h(z)``, and a linear Gaussian-mean head ``W h(z) + b``.  Its observation
variance is fixed, so the diagonal-Gaussian W2 pullback metric is

    G(z) = J_h(z)^T W^T W J_h(z).

For a positive-definite feature covariance C, the optimal rank-r head for
``||(U-W) C^(1/2)||_F`` is available in closed form by truncating the SVD of
``W C^(1/2)``.  This script compares raw, value-weighted, and value+Jacobian
covariances using a sampled test-set version of the theorem's local length
distortion.

Run either dataset from the repository root with

    uv run --python 3.12 \
      paper/ot-manifold-approximation/experiments/mnist_low_rank_geometry.py \
      --dataset mnist

Use ``--dataset fashion-mnist`` for FashionMNIST.  Downloads and independent
dataset-specific teacher checkpoints live under ``experiments/.cache`` and are
ignored by git.  Pass ``--force-train`` to replace only the selected dataset's
checkpoint.
"""

from __future__ import annotations

import argparse
import csv
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib import font_manager
from torch import Tensor, nn
from torch.func import jacrev, vmap
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
CACHE = HERE / ".cache"
DATA_CACHE = CACHE / "mnist"
SEED = 20260730
IMAGE_SIDE = 28
IMAGE_DIM = IMAGE_SIDE * IMAGE_SIDE
LATENT_DIM = 2
ENCODER_WIDTH = 128
ENCODER_FEATURES = 64
TRUNK_WIDTH = 64
FEATURE_DIM = 64
DEFAULT_EPOCHS = 12
DEFAULT_TRAIN_EXAMPLES = 30_000
BATCH_SIZE = 256
LEARNING_RATE = 2.0e-3
WEIGHT_DECAY = 1.0e-6
KL_WEIGHT = 3.0e-3
COVARIANCE_SAMPLES = 4096
METRIC_SAMPLES = 1536
RECONSTRUCTION_SAMPLES = 4000
TRIPLET_COUNT = 6000
TRIPLET_POOL_FACTOR = 12
TRIPLET_MIN_MARGIN = 0.01
TRIPLET_MAX_MARGIN = 0.15
GEOMETRY_TRACE_WEIGHT = 4.0
RIDGE_RELATIVE = 1.0e-6
PRIMARY_TOLERANCE = 0.05
RANKS = tuple(range(1, FEATURE_DIM + 1))


@dataclass(frozen=True)
class DatasetSpec:
    cli_name: str
    artifact_prefix: str
    display_name: str
    display_name_ja: str
    class_names: tuple[str, ...]
    class_names_ja: tuple[str, ...]

    @property
    def checkpoint(self) -> Path:
        return CACHE / f"{self.artifact_prefix}_gaussian_vae_v2.pt"

    def artifact(self, suffix: str) -> Path:
        return HERE / f"{self.artifact_prefix}_{suffix}"


DATASET_SPECS = {
    "mnist": DatasetSpec(
        cli_name="mnist",
        artifact_prefix="mnist",
        display_name="MNIST",
        display_name_ja="MNIST",
        class_names=tuple(str(index) for index in range(10)),
        class_names_ja=tuple(str(index) for index in range(10)),
    ),
    "fashion-mnist": DatasetSpec(
        cli_name="fashion-mnist",
        artifact_prefix="fashion_mnist",
        display_name="FashionMNIST",
        display_name_ja="FashionMNIST",
        class_names=(
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
        class_names_ja=(
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


@dataclass(frozen=True)
class TrainingConfig:
    seed: int
    epochs: int
    train_examples: int
    batch_size: int = BATCH_SIZE
    learning_rate: float = LEARNING_RATE
    weight_decay: float = WEIGHT_DECAY
    kl_weight: float = KL_WEIGHT
    architecture_version: int = 2


@dataclass(frozen=True)
class MethodSpec:
    key: str
    label: str
    label_ja: str
    color: str


METHODS = (
    MethodSpec("raw", "ordinary SVD", "通常SVD", "#777777"),
    MethodSpec("value", "value-weighted SVD", "出力重み付きSVD", "#d87928"),
    MethodSpec(
        "geometry",
        "value + Jacobian SVD",
        "出力＋Jacobian SVD",
        "#1769aa",
    ),
)


class SVDCompression(NamedTuple):
    left: np.ndarray
    singular_values: np.ndarray
    right: np.ndarray
    covariance_inverse_sqrt: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset",
        choices=tuple(DATASET_SPECS),
        default="mnist",
        help="dataset to train and evaluate (default: mnist)",
    )
    parser.add_argument("--force-train", action="store_true")
    parser.add_argument("--epochs", type=int, default=DEFAULT_EPOCHS)
    parser.add_argument("--train-examples", type=int, default=DEFAULT_TRAIN_EXAMPLES)
    parser.add_argument(
        "--geometry-trace-weight", type=float, default=GEOMETRY_TRACE_WEIGHT
    )
    return parser.parse_args()


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


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)


class GaussianVAE(nn.Module):
    """Two-dimensional VAE with a smooth trunk and affine mean head."""

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
        mean = self.encoder_mean(features)
        log_variance = self.encoder_log_variance(features).clamp(-8.0, 5.0)
        return mean, log_variance

    def decoder_features(self, latent: Tensor) -> Tensor:
        return self.decoder_trunk(latent)

    def decode(self, latent: Tensor) -> Tensor:
        return self.decoder_head(self.decoder_features(latent))

    def forward(self, images: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        mean, log_variance = self.encode(images)
        noise = torch.randn_like(mean)
        latent = mean + torch.exp(0.5 * log_variance) * noise
        return self.decode(latent), mean, log_variance


def load_datasets(
    spec: DatasetSpec,
) -> tuple[
    datasets.MNIST | datasets.FashionMNIST,
    datasets.MNIST | datasets.FashionMNIST,
]:
    transform = transforms.ToTensor()
    dataset_type = datasets.MNIST if spec.cli_name == "mnist" else datasets.FashionMNIST
    train = dataset_type(DATA_CACHE, train=True, download=True, transform=transform)
    test = dataset_type(DATA_CACHE, train=False, download=True, transform=transform)
    return train, test


def selected_training_data(
    dataset: datasets.MNIST | datasets.FashionMNIST,
    count: int,
    seed: int,
    dataset_name: str,
) -> Subset:
    if count > len(dataset):
        raise ValueError(
            f"requested {count} examples from {len(dataset)} {dataset_name} images"
        )
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    return Subset(dataset, indices)


def vae_loss(
    reconstruction: Tensor,
    images: Tensor,
    mean: Tensor,
    log_variance: Tensor,
    kl_weight: float,
) -> tuple[Tensor, Tensor, Tensor]:
    target = images.flatten(start_dim=1)
    reconstruction_loss = (reconstruction - target).square().mean()
    kl = (
        -0.5
        * (1.0 + log_variance - mean.square() - log_variance.exp()).sum(dim=1).mean()
    )
    return reconstruction_loss + kl_weight * kl, reconstruction_loss, kl


def train_teacher(
    model: GaussianVAE,
    dataset: Subset,
    config: TrainingConfig,
) -> list[dict[str, float]]:
    generator = torch.Generator().manual_seed(config.seed + 1)
    loader = DataLoader(
        dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=0,
        generator=generator,
    )
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    history: list[dict[str, float]] = []
    model.train()
    for epoch in range(1, config.epochs + 1):
        total_objective = 0.0
        total_reconstruction = 0.0
        total_kl = 0.0
        examples = 0
        for images, _ in loader:
            reconstruction, mean, log_variance = model(images)
            objective, reconstruction_loss, kl = vae_loss(
                reconstruction, images, mean, log_variance, config.kl_weight
            )
            optimizer.zero_grad(set_to_none=True)
            objective.backward()
            optimizer.step()
            batch = len(images)
            total_objective += float(objective.detach()) * batch
            total_reconstruction += float(reconstruction_loss.detach()) * batch
            total_kl += float(kl.detach()) * batch
            examples += batch
        row = {
            "epoch": float(epoch),
            "objective": total_objective / examples,
            "reconstruction_mse": total_reconstruction / examples,
            "kl": total_kl / examples,
        }
        history.append(row)
        print(
            f"teacher epoch {epoch:02d}/{config.epochs}: "
            f"mse={row['reconstruction_mse']:.5f}, kl={row['kl']:.4f}",
            flush=True,
        )
    model.eval()
    return history


def load_or_train_teacher(
    train_dataset: datasets.MNIST | datasets.FashionMNIST,
    config: TrainingConfig,
    force_train: bool,
    spec: DatasetSpec,
) -> tuple[GaussianVAE, list[dict[str, float]]]:
    CACHE.mkdir(parents=True, exist_ok=True)
    model = GaussianVAE()
    expected_config = asdict(config)
    if spec.checkpoint.exists() and not force_train:
        payload = torch.load(spec.checkpoint, map_location="cpu", weights_only=False)
        if payload.get("config") == expected_config:
            model.load_state_dict(payload["state_dict"])
            model.eval()
            print(f"loaded cached teacher: {spec.checkpoint}", flush=True)
            return model, payload["history"]
        print("cached teacher configuration differs; retraining", flush=True)

    subset = selected_training_data(
        train_dataset,
        config.train_examples,
        config.seed,
        spec.display_name,
    )
    history = train_teacher(model, subset, config)
    torch.save(
        {
            "config": expected_config,
            "state_dict": model.state_dict(),
            "history": history,
        },
        spec.checkpoint,
    )
    print(f"saved teacher checkpoint: {spec.checkpoint}", flush=True)
    return model, history


@torch.no_grad()
def collect_images_and_codes(
    model: GaussianVAE,
    dataset: datasets.MNIST | datasets.FashionMNIST | Subset,
    count: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    count = min(count, len(dataset))
    generator = torch.Generator().manual_seed(seed)
    indices = torch.randperm(len(dataset), generator=generator)[:count].tolist()
    loader = DataLoader(
        Subset(dataset, indices), batch_size=512, shuffle=False, num_workers=0
    )
    images: list[np.ndarray] = []
    codes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    model.eval()
    for batch_images, batch_labels in loader:
        mean, _ = model.encode(batch_images)
        images.append(batch_images.flatten(start_dim=1).numpy())
        codes.append(mean.numpy())
        labels.append(batch_labels.numpy())
    return (
        np.concatenate(images).astype(np.float64),
        np.concatenate(codes).astype(np.float64),
        np.concatenate(labels),
    )


def features_and_jacobians(
    model: GaussianVAE, latent: np.ndarray, batch_size: int = 256
) -> tuple[np.ndarray, np.ndarray]:
    """Evaluate h(z) and J_h(z) in manageable vmap batches."""

    def single(point: Tensor) -> Tensor:
        return model.decoder_features(point.unsqueeze(0)).squeeze(0)

    jacobian_function = jacrev(single)
    all_features: list[np.ndarray] = []
    all_jacobians: list[np.ndarray] = []
    for start in range(0, len(latent), batch_size):
        points = torch.as_tensor(
            latent[start : start + batch_size], dtype=torch.float32
        )
        with torch.no_grad():
            feature = model.decoder_features(points)
        jacobian = vmap(jacobian_function)(points)
        all_features.append(feature.detach().numpy().astype(np.float64))
        all_jacobians.append(jacobian.detach().numpy().astype(np.float64))
    return np.concatenate(all_features), np.concatenate(all_jacobians)


def symmetric_square_roots(matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    eigenvalues, eigenvectors = np.linalg.eigh(0.5 * (matrix + matrix.T))
    floor = max(float(eigenvalues.max()) * 1.0e-12, 1.0e-14)
    eigenvalues = np.maximum(eigenvalues, floor)
    square_root = (eigenvectors * np.sqrt(eigenvalues)) @ eigenvectors.T
    inverse_square_root = (
        eigenvectors * np.reciprocal(np.sqrt(eigenvalues))
    ) @ eigenvectors.T
    return square_root, inverse_square_root


def svd_compression(weight: np.ndarray, covariance: np.ndarray) -> SVDCompression:
    square_root, inverse_square_root = symmetric_square_roots(covariance)
    weighted = weight @ square_root
    left, singular_values, right = np.linalg.svd(weighted, full_matrices=False)
    return SVDCompression(left, singular_values, right, inverse_square_root)


def compressed_weight(compression: SVDCompression, rank: int) -> np.ndarray:
    truncated = (
        compression.left[:, :rank] * compression.singular_values[:rank]
    ) @ compression.right[:rank]
    return truncated @ compression.covariance_inverse_sqrt


def feature_covariances(
    features: np.ndarray,
    jacobians: np.ndarray,
    geometry_trace_weight: float,
) -> tuple[dict[str, np.ndarray], float, float]:
    count = len(features)
    value = features.T @ features / count
    jacobian = np.einsum("nqd,nrd->qr", jacobians, jacobians, optimize=True) / count
    value_trace = float(np.trace(value))
    jacobian_trace = float(np.trace(jacobian))
    geometry_multiplier = (
        geometry_trace_weight * value_trace / max(jacobian_trace, 1.0e-12)
    )
    ridge = RIDGE_RELATIVE * value_trace / FEATURE_DIM
    identity = np.eye(FEATURE_DIM, dtype=np.float64)
    covariances = {
        "raw": identity,
        "value": value + ridge * identity,
        "geometry": value + geometry_multiplier * jacobian + ridge * identity,
    }
    return covariances, geometry_multiplier, ridge


def pullback_metrics(jacobians: np.ndarray, gram: np.ndarray) -> np.ndarray:
    return np.einsum("nqi,qr,nrj->nij", jacobians, gram, jacobians, optimize=True)


def local_length_distortion(
    teacher_metric: np.ndarray, student_metric: np.ndarray
) -> tuple[float, float, float]:
    identity = np.eye(LATENT_DIM, dtype=np.float64)
    eigenvalue_floor = max(
        1.0e-12,
        1.0e-10 * float(np.median(np.linalg.eigvalsh(teacher_metric)[:, 0])),
    )
    length_errors: list[np.ndarray] = []
    minimum_teacher_eigenvalue = math.inf
    for teacher, student in zip(teacher_metric, student_metric):
        minimum_teacher_eigenvalue = min(
            minimum_teacher_eigenvalue, float(np.linalg.eigvalsh(teacher)[0])
        )
        cholesky = np.linalg.cholesky(teacher + eigenvalue_floor * identity)
        inverse = np.linalg.inv(cholesky)
        relative = inverse @ student @ inverse.T
        relative = 0.5 * (relative + relative.T)
        eigenvalues = np.maximum(np.linalg.eigvalsh(relative), 0.0)
        length_errors.append(np.abs(np.sqrt(eigenvalues) - 1.0))
    errors = np.concatenate(length_errors)
    return (
        float(errors.max()),
        float(np.quantile(errors, 0.95)),
        minimum_teacher_eigenvalue,
    )


def quadratic_mse(
    student_weight: np.ndarray,
    teacher_weight: np.ndarray,
    feature_second_moment: np.ndarray,
    output_dimension: int,
) -> float:
    difference = student_weight - teacher_weight
    squared = np.einsum(
        "oq,qr,or->", difference, feature_second_moment, difference, optimize=True
    )
    return float(squared / output_dimension)


def reconstruction_mse_from_moments(
    student_weight: np.ndarray,
    feature_second_moment: np.ndarray,
    feature_target_moment: np.ndarray,
    target_squared_mean: float,
    output_dimension: int,
) -> float:
    predicted_squared = np.einsum(
        "oq,qr,or->",
        student_weight,
        feature_second_moment,
        student_weight,
        optimize=True,
    )
    cross = 2.0 * float(np.sum(student_weight * feature_target_moment.T))
    return float((predicted_squared - cross + target_squared_mean) / output_dimension)


def make_hard_triplets(
    model: GaussianVAE,
    latent: np.ndarray,
    teacher_gram: np.ndarray,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    scale = np.maximum(np.std(latent, axis=0), 0.25)
    pool = TRIPLET_COUNT * TRIPLET_POOL_FACTOR
    anchor_indices = rng.integers(0, len(latent), size=pool)
    anchors = latent[anchor_indices]
    directions = rng.normal(size=(pool, 2, LATENT_DIM))
    directions /= np.linalg.norm(directions, axis=2, keepdims=True).clip(1.0e-12)
    steps = rng.uniform(0.03, 0.22, size=(pool, 2, 1)) * scale[None, None, :]
    candidates = anchors[:, None, :] + directions * steps
    stacked = np.concatenate([anchors, candidates[:, 0], candidates[:, 1]])
    features, _ = features_and_jacobians(model, stacked, batch_size=512)
    anchor_features, first_features, second_features = np.split(features, 3)
    first_delta = first_features - anchor_features
    second_delta = second_features - anchor_features
    first_distance = np.sqrt(
        np.maximum(
            np.einsum(
                "nq,qr,nr->n", first_delta, teacher_gram, first_delta, optimize=True
            ),
            0.0,
        )
    )
    second_distance = np.sqrt(
        np.maximum(
            np.einsum(
                "nq,qr,nr->n", second_delta, teacher_gram, second_delta, optimize=True
            ),
            0.0,
        )
    )
    margin = np.abs(first_distance - second_distance) / np.maximum(
        np.maximum(first_distance, second_distance), 1.0e-12
    )
    valid = np.flatnonzero(
        (margin >= TRIPLET_MIN_MARGIN) & (margin <= TRIPLET_MAX_MARGIN)
    )
    if len(valid) < TRIPLET_COUNT:
        raise RuntimeError(
            f"only {len(valid)} hard triplets; increase TRIPLET_POOL_FACTOR"
        )
    chosen = valid[:TRIPLET_COUNT]
    teacher_order = first_distance[chosen] < second_distance[chosen]
    return first_delta[chosen], second_delta[chosen], teacher_order


def triplet_agreement(
    gram: np.ndarray,
    first_delta: np.ndarray,
    second_delta: np.ndarray,
    teacher_order: np.ndarray,
) -> float:
    first = np.einsum("nq,qr,nr->n", first_delta, gram, first_delta, optimize=True)
    second = np.einsum("nq,qr,nr->n", second_delta, gram, second_delta, optimize=True)
    return float(np.mean((first < second) == teacher_order))


def model_parameter_counts(model: GaussianVAE, rank: int) -> dict[str, int]:
    dense_head = IMAGE_DIM * FEATURE_DIM + IMAGE_DIM
    factorized_head = rank * (IMAGE_DIM + FEATURE_DIM) + IMAGE_DIM
    head = min(dense_head, factorized_head)
    trunk = sum(parameter.numel() for parameter in model.decoder_trunk.parameters())
    encoder = sum(
        parameter.numel()
        for name, parameter in model.named_parameters()
        if name.startswith("encoder")
    )
    return {
        "head_parameters": head,
        "decoder_parameters": trunk + head,
        "vae_parameters": encoder + trunk + head,
        "teacher_head_parameters": dense_head,
    }


def write_history(history: list[dict[str, float]], spec: DatasetSpec) -> None:
    path = spec.artifact("teacher_history.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(history[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(history)


def write_results(rows: list[dict[str, object]], spec: DatasetSpec) -> None:
    path = spec.artifact("low_rank_results.csv")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=list(rows[0]), lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def method_boundaries(
    rows: list[dict[str, object]], method: str
) -> tuple[dict[str, object], dict[str, object]]:
    selected = sorted(
        [row for row in rows if row["method"] == method],
        key=lambda row: int(row["rank"]),
    )
    passing = [row for row in selected if bool(row["pass_5_percent"])]
    after = passing[0] if passing else selected[-1]
    preceding = [row for row in selected if int(row["rank"]) < int(after["rank"])]
    before = preceding[-1] if preceding else after
    return before, after


def write_table(rows: list[dict[str, object]], spec: DatasetSpec) -> None:
    english = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Method & Rank & Head / decoder params & $\widehat D_{\mathrm{loc}}$ & Output RMSE & Chord triplet & PSNR \\",
        r"\midrule",
    ]
    japanese = [
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"手法 & rank & head / decoder変数 & $\widehat D_{\mathrm{loc}}$ & 出力RMSE & chord triplet & PSNR \\",
        r"\midrule",
    ]
    for method_spec in METHODS:
        before, after = method_boundaries(rows, method_spec.key)
        selected = {
            int(row["rank"]): row for row in rows if row["method"] == method_spec.key
        }
        chosen = [selected[8]]
        for boundary in (before, after):
            if int(boundary["rank"]) not in {int(row["rank"]) for row in chosen}:
                chosen.append(boundary)
        for index, row in enumerate(chosen):
            status = "pass" if bool(row["pass_5_percent"]) else "fail"
            rank_label = f"{int(row['rank'])} ({status})"
            values = (
                f"{rank_label} & {int(row['head_parameters']):,} / "
                f"{int(row['decoder_parameters']):,} & "
                f"{100.0 * float(row['worst_local_length_distortion']):.2f}\\% & "
                f"{float(row['teacher_output_rmse']):.4f} & "
                f"{100.0 * float(row['teacher_triplet_agreement']):.1f}\\% & "
                f"{float(row['reconstruction_psnr']):.2f}"
            )
            prefix_en = method_spec.label if index == 0 else ""
            prefix_ja = method_spec.label_ja if index == 0 else ""
            english.append(f"{prefix_en} & {values}" + r" \\")
            japanese.append(f"{prefix_ja} & {values}" + r" \\")
        english.append(r"\addlinespace")
        japanese.append(r"\addlinespace")
    ending = [r"\bottomrule", r"\end{tabular}"]
    spec.artifact("low_rank_table.tex").write_text(
        "\n".join(english + ending) + "\n", encoding="utf-8"
    )
    spec.artifact("low_rank_table_ja.tex").write_text(
        "\n".join(japanese + ending) + "\n", encoding="utf-8"
    )


def write_combined_realdata_table() -> None:
    """Write a compact MNIST/FashionMNIST crossing table when both exist."""

    dataset_rows: list[tuple[DatasetSpec, list[dict[str, str]]]] = []
    for key in ("mnist", "fashion-mnist"):
        spec = DATASET_SPECS[key]
        path = spec.artifact("low_rank_results.csv")
        if not path.exists():
            return
        with path.open(newline="", encoding="utf-8") as handle:
            dataset_rows.append((spec, list(csv.DictReader(handle))))

    english = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Method & Rank fail $\to$ pass & $\widehat D_{\rm loc}$ fail $\to$ pass & Head reduction & Triplet \\",
        r"\midrule",
    ]
    japanese = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"データ & 手法 & rank 不合格 $\to$ 合格 & $\widehat D_{\mathrm{loc}}$ 不合格 $\to$ 合格 & head圧縮 & triplet \\",
        r"\midrule",
    ]
    for dataset_index, (spec, rows) in enumerate(dataset_rows):
        for method_index, method_spec in enumerate(METHODS):
            selected = sorted(
                [row for row in rows if row["method"] == method_spec.key],
                key=lambda row: int(row["rank"]),
            )
            passing = [row for row in selected if row["pass_5_percent"] == "True"]
            after = passing[0] if passing else selected[-1]
            preceding = [
                row for row in selected if int(row["rank"]) < int(after["rank"])
            ]
            before = preceding[-1] if preceding else after
            dataset_label = spec.display_name if method_index == 0 else ""
            values = (
                f"{dataset_label} & {method_spec.label} & "
                f"{before['rank']} $\\to$ {after['rank']} & "
                f"{100 * float(before['worst_local_length_distortion']):.2f}\\% "
                f"$\\to$ {100 * float(after['worst_local_length_distortion']):.2f}\\% & "
                f"{float(after['head_compression']):.2f}$\\times$ & "
                f"{100 * float(after['teacher_triplet_agreement']):.2f}\\%"
            )
            english.append(values + r" \\")
            japanese_values = values.replace(method_spec.label, method_spec.label_ja)
            japanese.append(japanese_values + r" \\")
        if dataset_index + 1 < len(dataset_rows):
            english.append(r"\addlinespace")
            japanese.append(r"\addlinespace")
    ending = [r"\bottomrule", r"\end{tabular}"]
    (HERE / "realdata_low_rank_table.tex").write_text(
        "\n".join(english + ending) + "\n", encoding="utf-8"
    )
    (HERE / "realdata_low_rank_table_ja.tex").write_text(
        "\n".join(japanese + ending) + "\n", encoding="utf-8"
    )


def save_training_plot(history: list[dict[str, float]], spec: DatasetSpec) -> None:
    epochs = [row["epoch"] for row in history]
    figure, axis = plt.subplots(figsize=(5.2, 3.4))
    axis.plot(
        epochs,
        [row["reconstruction_mse"] for row in history],
        marker="o",
        ms=3,
        color="#1769aa",
        label="reconstruction MSE",
    )
    axis.set_xlabel("epoch")
    axis.set_ylabel("MSE")
    axis.set_title(f"{spec.display_name} Gaussian VAE teacher training")
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(spec.artifact("teacher_training.png"), dpi=210)
    plt.close(figure)


def save_compression_plot(
    rows: list[dict[str, object]], language: str, spec: DatasetSpec
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    figure, axes = plt.subplots(2, 2, figsize=(10.6, 7.2))
    for method_spec in METHODS:
        selected = sorted(
            [row for row in rows if row["method"] == method_spec.key],
            key=lambda row: int(row["rank"]),
        )
        rank = np.asarray([int(row["rank"]) for row in selected])
        axes[0, 0].plot(
            rank,
            100.0
            * np.asarray(
                [float(row["worst_local_length_distortion"]) for row in selected]
            ),
            marker="o",
            ms=2.8,
            lw=1.8,
            color=method_spec.color,
            label=method_spec.label_ja if japanese else method_spec.label,
        )
        axes[0, 1].plot(
            rank,
            [float(row["teacher_output_rmse"]) for row in selected],
            marker="o",
            ms=2.8,
            lw=1.8,
            color=method_spec.color,
        )
        axes[1, 0].plot(
            rank,
            100.0
            * np.asarray([float(row["teacher_triplet_agreement"]) for row in selected]),
            marker="o",
            ms=2.8,
            lw=1.8,
            color=method_spec.color,
        )
        axes[1, 1].plot(
            rank,
            [float(row["reconstruction_psnr"]) for row in selected],
            marker="o",
            ms=2.8,
            lw=1.8,
            color=method_spec.color,
        )
    geometry_before, geometry_after = method_boundaries(rows, "geometry")
    axes[0, 0].scatter(
        [int(geometry_before["rank"]), int(geometry_after["rank"])],
        [
            100.0 * float(geometry_before["worst_local_length_distortion"]),
            100.0 * float(geometry_after["worst_local_length_distortion"]),
        ],
        s=58,
        color=["#c9342f", "#27854a"],
        edgecolor="white",
        linewidth=0.7,
        zorder=5,
    )
    axes[0, 0].axhline(5.0, color="#222222", ls="--", lw=1.1)
    axes[0, 0].set_ylabel(
        "標本最大局所長歪み（%）"
        if japanese
        else "sampled worst local length distortion (%)"
    )
    axes[0, 0].set_yscale("log")
    axes[0, 0].legend(frameon=False, fontsize=8)
    axes[0, 1].set_ylabel("教師出力RMSE" if japanese else "teacher-output RMSE")
    axes[1, 0].set_ylabel(
        "局所chord-triplet一致率（%）"
        if japanese
        else "local teacher-W2 chord agreement (%)"
    )
    axes[1, 1].set_ylabel(
        "再構成PSNR（dB）" if japanese else "reconstruction PSNR (dB)"
    )
    for axis in axes.ravel():
        axis.set_xlabel("head rank")
        axis.grid(alpha=0.18, which="both")
    figure.suptitle(
        f"{spec.display_name_ja}：教師のWasserstein潜在幾何を保つ低rank蒸留"
        if japanese
        else (
            f"{spec.display_name}: low-rank preservation of teacher "
            "Wasserstein geometry"
        ),
        fontsize=12,
    )
    figure.tight_layout()
    figure.savefig(spec.artifact(f"low_rank_compression{suffix}.png"), dpi=210)
    plt.close(figure)


def save_reconstruction_figure(
    images: np.ndarray,
    labels: np.ndarray,
    features: np.ndarray,
    teacher_weight: np.ndarray,
    bias: np.ndarray,
    before_weight: np.ndarray,
    after_weight: np.ndarray,
    before_rank: int,
    after_rank: int,
    language: str,
    spec: DatasetSpec,
) -> None:
    japanese = language == "ja"
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    chosen: list[int] = []
    for class_index in range(len(spec.class_names)):
        matches = np.flatnonzero(labels == class_index)
        if len(matches):
            chosen.append(int(matches[0]))
    chosen = chosen[:10]
    original = images[chosen]
    h = features[chosen]
    teacher = h @ teacher_weight.T + bias
    before = h @ before_weight.T + bias
    after = h @ after_weight.T + bias
    rows = [original, teacher, before, after]
    row_labels = (
        [
            "入力",
            "教師再構成",
            f"rank {before_rank}（不合格）",
            f"rank {after_rank}（合格）",
        ]
        if japanese
        else [
            "input",
            "teacher recon.",
            f"rank {before_rank} (fail)",
            f"rank {after_rank} (pass)",
        ]
    )
    figure_width = 13.2 if spec.cli_name == "fashion-mnist" else 11.3
    figure, axes = plt.subplots(4, len(chosen), figsize=(figure_width, 4.8))
    class_names = spec.class_names_ja if japanese else spec.class_names
    for row_index, values in enumerate(rows):
        for column_index, image in enumerate(values):
            axis = axes[row_index, column_index]
            axis.imshow(
                image.reshape(IMAGE_SIDE, IMAGE_SIDE), cmap="gray", vmin=0, vmax=1
            )
            axis.set_xticks([])
            axis.set_yticks([])
            if column_index == 0:
                axis.set_ylabel(row_labels[row_index], fontsize=9)
            if row_index == 0:
                class_index = int(labels[chosen[column_index]])
                axis.set_title(class_names[class_index], fontsize=8)
    figure.suptitle(
        (f"{spec.display_name_ja}：標本5%幾何閾値の直前と直後（Jacobian重み付きSVD）")
        if japanese
        else (
            f"{spec.display_name}: immediately below and above the sampled "
            "5% geometry threshold"
        ),
        fontsize=11,
    )
    figure.tight_layout()
    figure.savefig(spec.artifact(f"low_rank_images{suffix}.png"), dpi=220)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    dataset_spec = DATASET_SPECS[args.dataset]
    config = TrainingConfig(SEED, args.epochs, args.train_examples)
    seed_everything(SEED)
    train_dataset, test_dataset = load_datasets(dataset_spec)
    teacher, history = load_or_train_teacher(
        train_dataset, config, args.force_train, dataset_spec
    )
    write_history(history, dataset_spec)
    save_training_plot(history, dataset_spec)

    covariance_images, covariance_codes, _ = collect_images_and_codes(
        teacher, train_dataset, COVARIANCE_SAMPLES, SEED + 11
    )
    del covariance_images
    covariance_features, covariance_jacobians = features_and_jacobians(
        teacher, covariance_codes
    )
    covariances, geometry_multiplier, ridge = feature_covariances(
        covariance_features,
        covariance_jacobians,
        args.geometry_trace_weight,
    )
    print(
        f"geometry covariance multiplier={geometry_multiplier:.6f}, ridge={ridge:.3e}",
        flush=True,
    )

    evaluation_images, evaluation_codes, evaluation_labels = collect_images_and_codes(
        teacher, test_dataset, RECONSTRUCTION_SAMPLES, SEED + 29
    )
    evaluation_features, evaluation_jacobians = features_and_jacobians(
        teacher, evaluation_codes
    )
    metric_jacobians = evaluation_jacobians[:METRIC_SAMPLES]

    teacher_weight = teacher.decoder_head.weight.detach().numpy().astype(np.float64)
    teacher_bias = teacher.decoder_head.bias.detach().numpy().astype(np.float64)
    teacher_gram = teacher_weight.T @ teacher_weight
    teacher_metric = pullback_metrics(metric_jacobians, teacher_gram)
    feature_second_moment = (
        evaluation_features.T @ evaluation_features / len(evaluation_features)
    )
    centered_targets = evaluation_images - teacher_bias
    feature_target_moment = (
        evaluation_features.T @ centered_targets / len(evaluation_features)
    )
    target_squared_mean = float(np.mean(np.sum(centered_targets**2, axis=1)))

    rng = np.random.default_rng(SEED + 47)
    first_delta, second_delta, teacher_order = make_hard_triplets(
        teacher, evaluation_codes, teacher_gram, rng
    )

    compressions = {
        key: svd_compression(teacher_weight, covariance)
        for key, covariance in covariances.items()
    }
    rows: list[dict[str, object]] = []
    for spec in METHODS:
        compression = compressions[spec.key]
        for rank in RANKS:
            student_weight = compressed_weight(compression, rank)
            student_gram = student_weight.T @ student_weight
            student_metric = pullback_metrics(metric_jacobians, student_gram)
            worst_distortion, p95_distortion, minimum_teacher_eigenvalue = (
                local_length_distortion(teacher_metric, student_metric)
            )
            teacher_output_mse = quadratic_mse(
                student_weight,
                teacher_weight,
                feature_second_moment,
                IMAGE_DIM,
            )
            reconstruction_mse = reconstruction_mse_from_moments(
                student_weight,
                feature_second_moment,
                feature_target_moment,
                target_squared_mean,
                IMAGE_DIM,
            )
            reconstruction_mse = max(reconstruction_mse, 1.0e-12)
            counts = model_parameter_counts(teacher, rank)
            row: dict[str, object] = {
                "method": spec.key,
                "rank": rank,
                **counts,
                "head_compression": counts["teacher_head_parameters"]
                / counts["head_parameters"],
                "teacher_output_rmse": math.sqrt(max(teacher_output_mse, 0.0)),
                "worst_local_length_distortion": worst_distortion,
                "p95_local_length_distortion": p95_distortion,
                "pass_5_percent": worst_distortion <= PRIMARY_TOLERANCE,
                "teacher_triplet_agreement": triplet_agreement(
                    student_gram, first_delta, second_delta, teacher_order
                ),
                "reconstruction_mse": reconstruction_mse,
                "reconstruction_psnr": 10.0 * math.log10(1.0 / reconstruction_mse),
                "minimum_teacher_metric_eigenvalue": minimum_teacher_eigenvalue,
                "weighted_svd_next_singular_value": float(
                    compression.singular_values[rank] if rank < FEATURE_DIM else 0.0
                ),
                "geometry_covariance_multiplier": geometry_multiplier,
                "geometry_trace_weight": args.geometry_trace_weight,
                "seed": config.seed,
                "teacher_epochs": config.epochs,
                "teacher_train_examples": config.train_examples,
                "covariance_sample_count": COVARIANCE_SAMPLES,
                "metric_sample_count": METRIC_SAMPLES,
                "reconstruction_sample_count": RECONSTRUCTION_SAMPLES,
            }
            rows.append(row)
        before, after = method_boundaries(rows, spec.key)
        print(
            f"{spec.key:8s}: 5% boundary rank "
            f"{int(before['rank'])} ({100 * float(before['worst_local_length_distortion']):.2f}%) -> "
            f"{int(after['rank'])} ({100 * float(after['worst_local_length_distortion']):.2f}%)",
            flush=True,
        )

    write_results(rows, dataset_spec)
    write_table(rows, dataset_spec)
    write_combined_realdata_table()
    save_compression_plot(rows, "en", dataset_spec)
    save_compression_plot(rows, "ja", dataset_spec)

    geometry_before, geometry_after = method_boundaries(rows, "geometry")
    before_rank = int(geometry_before["rank"])
    after_rank = int(geometry_after["rank"])
    before_weight = compressed_weight(compressions["geometry"], before_rank)
    after_weight = compressed_weight(compressions["geometry"], after_rank)
    save_reconstruction_figure(
        evaluation_images,
        evaluation_labels,
        evaluation_features,
        teacher_weight,
        teacher_bias,
        before_weight,
        after_weight,
        before_rank,
        after_rank,
        "en",
        dataset_spec,
    )
    save_reconstruction_figure(
        evaluation_images,
        evaluation_labels,
        evaluation_features,
        teacher_weight,
        teacher_bias,
        before_weight,
        after_weight,
        before_rank,
        after_rank,
        "ja",
        dataset_spec,
    )
    print(
        f"wrote {dataset_spec.display_name} low-rank geometry artifacts",
        flush=True,
    )


if __name__ == "__main__":
    main()
