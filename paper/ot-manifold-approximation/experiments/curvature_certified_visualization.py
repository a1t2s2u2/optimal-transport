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
"""Visualize decoder-induced Wasserstein geometry as a curved surface.

The script contains two deliberately different tests of the same estimator.

1. A controlled diagonal-Gaussian decoder has parameters

       F(u, v) = (mean=(u, v), std=(c + h(u, v), c)).

   The Gaussian W2 formula therefore equals the Euclidean chord distance of
   the graph (u, v, h(u, v)) *exactly*.  The smooth height h has two bumps and
   a saddle, hence both positive and negative, spatially varying curvature.

2. A cached two-dimensional MNIST VAE from ``mnist_low_rank_geometry.py`` is
   sampled on a shape-regular 317-point lattice ellipse fitted to digit 3's
   posterior.  Its decoded 14-by-14 images are normalized as pixel masses.
   Local W2 distances use unregularized discrete OT (POT ``emd2``).

For each test, only local W2 queries are passed to stress minimization.  A
scaled-latent two-dimensional baseline and three-dimensional
representations are fitted with deterministic Adam followed by L-BFGS.  The
controlled main branch is the graph-valued convention

       X_i = (A z_i, q_i),

where A is globally invertible and q has a small graph-Laplacian penalty.
For MNIST, the main branch instead adds adjacent-face normal bending to free
R3 edge stress.  On a fixed weight grid, the selected feasible candidate has
the smallest edge stress among those whose fifth normal-dot percentile is at
least 0.25 and whose smallest facewise affine singular value is at least 0.35.
A collapse penalty makes this shape audit meaningful.  Unregularized free R3
is retained as a visibly folded low-stress reference.  Unregularized discrete W2 on the
non-edge two-hop pairs is never optimized and is reported only as hold-out
error.

The triangular edges needed by the PL-curvature theorem are always included
among the queries.  Boundary angle defects are computed for visualization but
excluded from every reported curvature error.

Run from the repository root with

    uv run --python 3.12 \
      paper/ot-manifold-approximation/experiments/curvature_certified_visualization.py
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import NamedTuple

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import ot
import torch
from matplotlib import font_manager
from matplotlib.colors import BoundaryNorm, Normalize
from mnist_low_rank_geometry import (
    DATASET_SPECS,
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_EXAMPLES,
    GaussianVAE,
    TrainingConfig,
    load_datasets,
    load_or_train_teacher,
)
from mnist_low_rank_geometry import (
    SEED as MNIST_VAE_SEED,
)
from mnist_low_rank_geometry import (
    seed_everything as seed_vae_training,
)
from mpl_toolkits.mplot3d.art3d import Line3DCollection, Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import Delaunay
from torch import Tensor
from torch.nn import functional as torch_functional
from torch.utils.data import DataLoader
from torchvision import datasets

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent

MASTER_SEED = 20260802
GRID_SIDE = 21
CONTROL_QUERY_RADIUS = 4
MNIST_QUERY_HOPS = 2
MNIST_DIGIT = 3
MNIST_MASS_SIDE = 14
MNIST_POSTERIOR_LOW = 0.03
MNIST_POSTERIOR_HIGH = 0.97
OPTIMIZATION_SEEDS = (0, 1, 2)
ADAM_STEPS = 1_200
LBFGS_STEPS = 220
ADAM_LEARNING_RATE = 3.0e-2
BENDING_WEIGHT = 2.0e-5
SMOOTHNESS_LAMBDAS = (0.0, 0.003, 0.01, 0.03, 0.10)
SMOOTHNESS_NORMAL_QUANTILE = 0.05
SMOOTHNESS_MINIMUM_NORMAL_DOT = 0.25
FACE_SINGULAR_VALUE_FLOOR = 0.45
FACE_SHAPE_PENALTY_WEIGHT = 0.25
SMOOTHNESS_MINIMUM_GLOBAL_S_MIN = 0.35
EPSILON = 1.0e-12

CONTROL_COLOR = "#1769aa"
MNIST_COLOR = "#d87928"
HEIGHT_COLOR = "#27854a"
FREE_COLOR = "#777777"


@dataclass(frozen=True)
class SurfaceProblem:
    """Local metric observations on a known triangular complex."""

    key: str
    base_coordinates: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    query_pairs: np.ndarray
    query_distances: np.ndarray
    target_edge_distances: np.ndarray
    target_curvature_mass: np.ndarray
    target_curvature_density: np.ndarray
    boundary: np.ndarray
    true_coordinates: np.ndarray | None = None
    planar_base_coordinates: np.ndarray | None = None
    holdout_pairs: np.ndarray | None = None
    holdout_distances: np.ndarray | None = None


@dataclass(frozen=True)
class Embedding:
    """One fitted coordinate representation and its optimization trace."""

    method: str
    coordinates: np.ndarray
    seed: int
    objective: float
    query_stress: float
    history_steps: np.ndarray
    history_values: np.ndarray
    smoothing_lambda: float = math.nan


@dataclass(frozen=True)
class MnistDisplay:
    """MNIST-only arrays used to explain what each point represents."""

    decoder_images: np.ndarray
    mass_images: np.ndarray
    nearest_digits: np.ndarray
    nearest_real_images: np.ndarray
    nearest_real_digits: np.ndarray
    latent_quantile_low: np.ndarray
    latent_quantile_high: np.ndarray


@dataclass(frozen=True)
class SmoothnessCandidate:
    """One MNIST smooth-R3 solution along the observable Pareto path."""

    smoothing_lambda: float
    embedding: Embedding
    normal_dot_quantile_05: float
    negative_normal_fraction: float
    global_face_affine_s_min: float


class FitOutput(NamedTuple):
    coordinates: np.ndarray
    objective: float
    stress: float
    history_steps: np.ndarray
    history_values: np.ndarray


def seed_everything(seed: int) -> None:
    """Make all numerical choices repeatable on the CPU."""

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


def grid_mesh(side: int = GRID_SIDE) -> tuple[np.ndarray, np.ndarray]:
    """Return a square lattice and a checkerboard triangularization."""

    axis = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    coordinates = np.column_stack([xx.ravel(), yy.ravel()])
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


def disk_lattice(side: int = GRID_SIDE) -> np.ndarray:
    """Return the 317 lattice points in a radius-10 integer disk."""

    radius = (side - 1) // 2
    points = [
        (column / radius, row / radius)
        for row in range(-radius, radius + 1)
        for column in range(-radius, radius + 1)
        if row * row + column * column <= radius * radius
    ]
    result = np.asarray(points, dtype=np.float64)
    expected_count = 317 if side == 21 else len(result)
    if len(result) != expected_count:
        raise AssertionError(f"unexpected disk lattice size: {len(result)}")
    return result


def orient_faces(coordinates: np.ndarray, faces: np.ndarray) -> np.ndarray:
    """Orient two-dimensional Delaunay faces counter-clockwise."""

    oriented = np.asarray(faces, dtype=np.int64).copy()
    first = coordinates[oriented[:, 1]] - coordinates[oriented[:, 0]]
    second = coordinates[oriented[:, 2]] - coordinates[oriented[:, 0]]
    signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    reverse = signed_area < 0.0
    temporary = oriented[reverse, 1].copy()
    oriented[reverse, 1] = oriented[reverse, 2]
    oriented[reverse, 2] = temporary
    return oriented


def unique_edges(faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def boundary_vertices(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    edges, counts = np.unique(raw, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    boundary[np.unique(edges[counts == 1])] = True
    return boundary


def chebyshev_query_pairs(side: int, radius: int) -> np.ndarray:
    pairs: list[tuple[int, int]] = []
    for first_row in range(side):
        for first_column in range(side):
            first = first_row * side + first_column
            for second_row in range(
                max(0, first_row - radius), min(side, first_row + radius + 1)
            ):
                for second_column in range(
                    max(0, first_column - radius),
                    min(side, first_column + radius + 1),
                ):
                    second = second_row * side + second_column
                    if second > first:
                        pairs.append((first, second))
    return np.asarray(pairs, dtype=np.int64)


def graph_hop_query_pairs(
    vertex_count: int, edges: np.ndarray, maximum_hops: int
) -> np.ndarray:
    """Return all unordered pairs at graph distance at most maximum_hops."""

    neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for left, right in edges:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))
    pairs: list[tuple[int, int]] = []
    for source in range(vertex_count):
        visited = {source}
        frontier = {source}
        for _ in range(maximum_hops):
            frontier = {
                neighbour
                for vertex in frontier
                for neighbour in neighbours[vertex]
                if neighbour not in visited
            }
            visited.update(frontier)
        pairs.extend((source, target) for target in sorted(visited) if target > source)
    return np.asarray(pairs, dtype=np.int64)


def assert_edges_are_queried(edges: np.ndarray, query_pairs: np.ndarray) -> None:
    edge_set = {tuple(pair) for pair in edges.tolist()}
    query_set = {tuple(pair) for pair in query_pairs.tolist()}
    missing = edge_set - query_set
    if missing:
        raise AssertionError(f"{len(missing)} theoretical edges are not queried")


def graph_height(coordinates: np.ndarray) -> np.ndarray:
    """A smooth two-bump plus saddle height with mixed Gaussian curvature."""

    horizontal = coordinates[:, 0]
    vertical = coordinates[:, 1]
    first_bump = 0.48 * np.exp(
        -6.4 * ((horizontal + 0.42) ** 2 + 1.15 * (vertical - 0.24) ** 2)
    )
    second_bump = 0.38 * np.exp(
        -7.2 * (1.2 * (horizontal - 0.38) ** 2 + (vertical + 0.30) ** 2)
    )
    saddle = 0.30 * horizontal * vertical
    return first_bump + second_bump + saddle


def pairwise_coordinate_distances(
    coordinates: np.ndarray, pairs: np.ndarray
) -> np.ndarray:
    differences = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    return np.linalg.norm(differences, axis=1)


def edge_positions(edges: np.ndarray, query_pairs: np.ndarray) -> np.ndarray:
    lookup = {tuple(pair): index for index, pair in enumerate(query_pairs.tolist())}
    return np.asarray([lookup[tuple(edge)] for edge in edges.tolist()], dtype=np.int64)


def triangle_geometry_from_edge_lengths(
    vertex_count: int,
    faces: np.ndarray,
    edges: np.ndarray,
    edge_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return PL angle-defect mass, barycentric area, and density."""

    length_lookup = {
        tuple(edge): float(length) for edge, length in zip(edges.tolist(), edge_lengths)
    }
    angle_sum = np.zeros(vertex_count, dtype=np.float64)
    vertex_area = np.zeros(vertex_count, dtype=np.float64)
    for first, second, third in faces:
        vertices = (int(first), int(second), int(third))

        def length(left: int, right: int) -> float:
            return length_lookup[tuple(sorted((left, right)))]

        opposite = np.asarray(
            [
                length(vertices[1], vertices[2]),
                length(vertices[0], vertices[2]),
                length(vertices[0], vertices[1]),
            ],
            dtype=np.float64,
        )
        a, b, c = opposite
        cosines = np.asarray(
            [
                (b * b + c * c - a * a) / max(2.0 * b * c, EPSILON),
                (a * a + c * c - b * b) / max(2.0 * a * c, EPSILON),
                (a * a + b * b - c * c) / max(2.0 * a * b, EPSILON),
            ]
        )
        angles = np.arccos(np.clip(cosines, -1.0, 1.0))
        semiperimeter = 0.5 * (a + b + c)
        area_squared = max(
            semiperimeter
            * (semiperimeter - a)
            * (semiperimeter - b)
            * (semiperimeter - c),
            0.0,
        )
        area = math.sqrt(area_squared)
        for local_index, vertex in enumerate(vertices):
            angle_sum[vertex] += angles[local_index]
            vertex_area[vertex] += area / 3.0
    boundary = boundary_vertices(vertex_count, faces)
    target_angle = np.where(boundary, math.pi, 2.0 * math.pi)
    curvature_mass = target_angle - angle_sum
    density = curvature_mass / np.maximum(vertex_area, EPSILON)
    return curvature_mass, vertex_area, density


def triangle_quality(
    faces: np.ndarray, edges: np.ndarray, edge_lengths: np.ndarray
) -> tuple[float, float]:
    """Return the minimum angle and normalized strict-triangle slack."""

    lookup = {
        tuple(edge): float(length) for edge, length in zip(edges.tolist(), edge_lengths)
    }
    minimum_angle = math.inf
    minimum_slack = math.inf
    for face in faces:
        first, second, third = (int(vertex) for vertex in face)
        lengths = np.sort(
            np.asarray(
                [
                    lookup[tuple(sorted((first, second)))],
                    lookup[tuple(sorted((second, third)))],
                    lookup[tuple(sorted((third, first)))],
                ],
                dtype=np.float64,
            )
        )
        shortest, middle, longest = lengths
        normalized_slack = (shortest + middle - longest) / max(shortest, EPSILON)
        minimum_slack = min(minimum_slack, float(normalized_slack))
        a, b, c = lengths
        cosines = np.asarray(
            [
                (b * b + c * c - a * a) / max(2.0 * b * c, EPSILON),
                (a * a + c * c - b * b) / max(2.0 * a * c, EPSILON),
                (a * a + b * b - c * c) / max(2.0 * a * b, EPSILON),
            ]
        )
        minimum_angle = min(
            minimum_angle,
            float(np.min(np.arccos(np.clip(cosines, -1.0, 1.0)))),
        )
    return math.degrees(minimum_angle), minimum_slack


def target_triangle_quality(problem: SurfaceProblem) -> tuple[float, float]:
    return triangle_quality(problem.faces, problem.edges, problem.target_edge_distances)


def face_edge_length_array(
    faces: np.ndarray, edges: np.ndarray, edge_lengths: np.ndarray
) -> np.ndarray:
    lookup = {
        tuple(edge): float(length) for edge, length in zip(edges.tolist(), edge_lengths)
    }
    result = np.empty((len(faces), 3), dtype=np.float64)
    for face_index, (first, second, third) in enumerate(faces):
        result[face_index] = (
            lookup[tuple(sorted((int(first), int(second))))],
            lookup[tuple(sorted((int(second), int(third))))],
            lookup[tuple(sorted((int(third), int(first))))],
        )
    return result


def face_angles_from_edge_array(lengths: np.ndarray) -> np.ndarray:
    """Return angles at face vertices (v0,v1,v2) from (l01,l12,l20)."""

    first, opposite_first, second = lengths.T
    cosines = np.column_stack(
        [
            (first**2 + second**2 - opposite_first**2)
            / np.maximum(2.0 * first * second, EPSILON),
            (first**2 + opposite_first**2 - second**2)
            / np.maximum(2.0 * first * opposite_first, EPSILON),
            (second**2 + opposite_first**2 - first**2)
            / np.maximum(2.0 * second * opposite_first, EPSILON),
        ]
    )
    return np.arccos(np.clip(cosines, -1.0, 1.0))


def exact_distortion_and_curvature_certificate(
    problem: SurfaceProblem, display_edge_lengths: np.ndarray
) -> tuple[float, float, np.ndarray]:
    """Compute exact facewise metric distortion and angle-error certificates."""

    target = face_edge_length_array(
        problem.faces, problem.edges, problem.target_edge_distances
    )
    display = face_edge_length_array(problem.faces, problem.edges, display_edge_lengths)
    minimum_singular_value = math.inf
    maximum_singular_value = 0.0
    for target_lengths, display_lengths in zip(target, display):
        target_01, target_12, target_20 = target_lengths
        display_01, display_12, display_20 = display_lengths
        target_gram = np.asarray(
            [
                [
                    target_01**2,
                    0.5 * (target_01**2 + target_20**2 - target_12**2),
                ],
                [
                    0.5 * (target_01**2 + target_20**2 - target_12**2),
                    target_20**2,
                ],
            ]
        )
        display_gram = np.asarray(
            [
                [
                    display_01**2,
                    0.5 * (display_01**2 + display_20**2 - display_12**2),
                ],
                [
                    0.5 * (display_01**2 + display_20**2 - display_12**2),
                    display_20**2,
                ],
            ]
        )
        eigenvalues, eigenvectors = np.linalg.eigh(target_gram)
        inverse_square_root = (
            eigenvectors * np.reciprocal(np.sqrt(np.maximum(eigenvalues, EPSILON)))
        ) @ eigenvectors.T
        relative = inverse_square_root @ display_gram @ inverse_square_root
        relative = 0.5 * (relative + relative.T)
        singular_values = np.sqrt(np.maximum(np.linalg.eigvalsh(relative), 0.0))
        minimum_singular_value = min(minimum_singular_value, float(singular_values[0]))
        maximum_singular_value = max(maximum_singular_value, float(singular_values[-1]))

    target_angles = face_angles_from_edge_array(target)
    display_angles = face_angles_from_edge_array(display)
    certificate = np.zeros(len(problem.base_coordinates), dtype=np.float64)
    for face, face_error in zip(problem.faces, np.abs(display_angles - target_angles)):
        certificate[face] += face_error
    return minimum_singular_value, maximum_singular_value, certificate


def minimum_signed_area_ratio(
    problem: SurfaceProblem, planar_coordinates: np.ndarray
) -> float:
    target_lengths = face_edge_length_array(
        problem.faces, problem.edges, problem.target_edge_distances
    )
    semiperimeter = 0.5 * np.sum(target_lengths, axis=1)
    target_twice_area = 2.0 * np.sqrt(
        np.maximum(
            semiperimeter
            * (semiperimeter - target_lengths[:, 0])
            * (semiperimeter - target_lengths[:, 1])
            * (semiperimeter - target_lengths[:, 2]),
            EPSILON,
        )
    )
    first_edges = (
        planar_coordinates[problem.faces[:, 1]]
        - planar_coordinates[problem.faces[:, 0]]
    )
    second_edges = (
        planar_coordinates[problem.faces[:, 2]]
        - planar_coordinates[problem.faces[:, 0]]
    )
    signed_display_twice_area = (
        first_edges[:, 0] * second_edges[:, 1] - first_edges[:, 1] * second_edges[:, 0]
    )
    return float(np.min(signed_display_twice_area / target_twice_area))


def controlled_problem() -> tuple[SurfaceProblem, dict[str, float]]:
    base, faces = grid_mesh()
    edges = unique_edges(faces)
    queries = chebyshev_query_pairs(GRID_SIDE, CONTROL_QUERY_RADIUS)
    assert_edges_are_queried(edges, queries)
    height = graph_height(base)
    standard_deviation_constant = 0.80
    if np.min(standard_deviation_constant + height) <= 0.0:
        raise AssertionError(
            "the controlled Gaussian standard deviation is not positive"
        )

    true_coordinates = np.column_stack([base, height])
    gaussian_parameters = np.column_stack(
        [
            base[:, 0],
            base[:, 1],
            standard_deviation_constant + height,
            np.full(len(base), standard_deviation_constant),
        ]
    )
    w2_distances = pairwise_coordinate_distances(gaussian_parameters, queries)
    chords = pairwise_coordinate_distances(true_coordinates, queries)
    exactness = float(np.max(np.abs(w2_distances - chords)))

    positions = edge_positions(edges, queries)
    target_edges = w2_distances[positions]
    curvature_mass, _, curvature_density = triangle_geometry_from_edge_lengths(
        len(base), faces, edges, target_edges
    )
    boundary = boundary_vertices(len(base), faces)
    interior_density = curvature_density[~boundary]
    diagnostics = {
        "w2_chord_max_absolute_error": exactness,
        "minimum_gaussian_standard_deviation": float(
            np.min(standard_deviation_constant + height)
        ),
        "positive_curvature_vertex_fraction": float(np.mean(interior_density > 0.0)),
        "negative_curvature_vertex_fraction": float(np.mean(interior_density < 0.0)),
    }
    if not (np.any(interior_density > 0.0) and np.any(interior_density < 0.0)):
        raise AssertionError("controlled PL surface must contain both curvature signs")
    return (
        SurfaceProblem(
            key="controlled",
            base_coordinates=base,
            faces=faces,
            edges=edges,
            query_pairs=queries,
            query_distances=w2_distances,
            target_edge_distances=target_edges,
            target_curvature_mass=curvature_mass,
            target_curvature_density=curvature_density,
            boundary=boundary,
            true_coordinates=true_coordinates,
        ),
        diagnostics,
    )


def load_mnist_teacher() -> tuple[GaussianVAE, datasets.MNIST]:
    """Reuse, validate, or reproducibly train the prior paper's MNIST VAE."""

    specification = DATASET_SPECS["mnist"]
    configuration = TrainingConfig(
        seed=MNIST_VAE_SEED,
        epochs=DEFAULT_EPOCHS,
        train_examples=DEFAULT_TRAIN_EXAMPLES,
    )
    seed_vae_training(MNIST_VAE_SEED)
    train, test = load_datasets(specification)
    model, _ = load_or_train_teacher(
        train,
        configuration,
        force_train=False,
        spec=specification,
    )
    seed_everything(MASTER_SEED)
    return model, test


@torch.no_grad()
def encode_mnist_test(
    model: GaussianVAE, dataset: datasets.MNIST
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    codes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    image_batches: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        mean, _ = model.encode(batch_images)
        codes.append(mean.numpy().astype(np.float64))
        labels.append(batch_labels.numpy())
        image_batches.append(batch_images.numpy().astype(np.float64))
    return (
        np.concatenate(codes),
        np.concatenate(labels),
        np.concatenate(image_batches),
    )


def digit_three_ellipse(
    codes: np.ndarray, labels: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map the 317-point unit disk to a robust posterior quantile ellipse."""

    selected = codes[labels == MNIST_DIGIT]
    center = np.mean(selected, axis=0)
    covariance = np.cov(selected, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    order = np.argsort(eigenvalues)[::-1]
    basis = eigenvectors[:, order]
    projected = (selected - center) @ basis
    low = np.quantile(projected, MNIST_POSTERIOR_LOW, axis=0)
    high = np.quantile(projected, MNIST_POSTERIOR_HIGH, axis=0)
    midpoint = 0.5 * (low + high)
    half_width = 0.5 * (high - low)
    disk = disk_lattice()
    ellipse_principal = midpoint + disk * half_width
    ellipse = center + ellipse_principal @ basis.T
    return ellipse, center + low @ basis.T, center + high @ basis.T


@torch.no_grad()
def decode_images(model: GaussianVAE, latent: np.ndarray) -> np.ndarray:
    points = torch.as_tensor(latent, dtype=torch.float32)
    decoded = model.decode(points).reshape(-1, 28, 28)
    return decoded.numpy().astype(np.float64)


def images_to_pixel_masses(images: np.ndarray) -> np.ndarray:
    """Downsample decoded means to normalized nonnegative 14x14 pixel masses."""

    clipped = np.clip(images, 0.0, 1.0)
    pooled = clipped.reshape(-1, MNIST_MASS_SIDE, 2, MNIST_MASS_SIDE, 2).mean(
        axis=(2, 4)
    )
    masses = pooled.reshape(len(pooled), -1) + 1.0e-10
    masses /= np.sum(masses, axis=1, keepdims=True)
    return masses


def pixel_ground_cost() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, MNIST_MASS_SIDE, dtype=np.float64)
    xx, yy = np.meshgrid(axis, axis, indexing="xy")
    positions = np.column_stack([xx.ravel(), yy.ravel()])
    differences = positions[:, None, :] - positions[None, :, :]
    return np.sum(differences * differences, axis=2)


def exact_pixel_w2(masses: np.ndarray, query_pairs: np.ndarray) -> np.ndarray:
    """Evaluate unregularized discrete pixel-space W2, without Sinkhorn."""

    cost = pixel_ground_cost()
    distances = np.empty(len(query_pairs), dtype=np.float64)
    for index, (left, right) in enumerate(query_pairs):
        squared = ot.emd2(
            masses[int(left)],
            masses[int(right)],
            cost,
            numItermax=100_000,
            check_marginals=True,
        )
        distances[index] = math.sqrt(max(float(squared), 0.0))
        if (index + 1) % 1000 == 0:
            print(
                f"MNIST unregularized pixel OT: {index + 1}/{len(query_pairs)} pairs",
                flush=True,
            )
    return distances


def nearest_posterior_digits(
    query_codes: np.ndarray,
    reference_codes: np.ndarray,
    labels: np.ndarray,
    k: int = 15,
) -> np.ndarray:
    squared = np.sum(
        (query_codes[:, None, :] - reference_codes[None, :, :]) ** 2, axis=2
    )
    neighbours = np.argpartition(squared, kth=k - 1, axis=1)[:, :k]
    votes = labels[neighbours]
    counts = np.stack([np.sum(votes == digit, axis=1) for digit in range(10)], axis=1)
    return np.argmax(counts, axis=1)


def nearest_posterior_indices(
    query_codes: np.ndarray, reference_codes: np.ndarray
) -> np.ndarray:
    squared = np.sum(
        (query_codes[:, None, :] - reference_codes[None, :, :]) ** 2, axis=2
    )
    return np.argmin(squared, axis=1)


def mnist_problem() -> tuple[SurfaceProblem, MnistDisplay, dict[str, float]]:
    model, test = load_mnist_teacher()
    posterior_codes, posterior_labels, posterior_images = encode_mnist_test(model, test)
    latent, quantile_low, quantile_high = digit_three_ellipse(
        posterior_codes, posterior_labels
    )
    canonical_disk = disk_lattice()
    triangulation = Delaunay(latent, qhull_options="Qbb Qc Qz Q12")
    faces = orient_faces(canonical_disk, triangulation.simplices)
    canonical_first = canonical_disk[faces[:, 1]] - canonical_disk[faces[:, 0]]
    canonical_second = canonical_disk[faces[:, 2]] - canonical_disk[faces[:, 0]]
    canonical_twice_area = (
        canonical_first[:, 0] * canonical_second[:, 1]
        - canonical_first[:, 1] * canonical_second[:, 0]
    )
    faces = faces[np.abs(canonical_twice_area) > 1.0e-10]
    edges = unique_edges(faces)
    two_hop_pairs = graph_hop_query_pairs(len(latent), edges, MNIST_QUERY_HOPS)
    assert_edges_are_queried(edges, two_hop_pairs)

    decoded = decode_images(model, latent)
    masses = images_to_pixel_masses(decoded)
    two_hop_distances = exact_pixel_w2(masses, two_hop_pairs)
    positions = edge_positions(edges, two_hop_pairs)
    target_edges = two_hop_distances[positions]
    edge_set = {tuple(edge) for edge in edges.tolist()}
    holdout_mask = np.asarray(
        [tuple(pair) not in edge_set for pair in two_hop_pairs.tolist()], dtype=bool
    )
    holdout_pairs = two_hop_pairs[holdout_mask]
    holdout_distances = two_hop_distances[holdout_mask]
    curvature_mass, _, curvature_density = triangle_geometry_from_edge_lengths(
        len(latent), faces, edges, target_edges
    )
    boundary = boundary_vertices(len(latent), faces)
    nearest_digits = nearest_posterior_digits(latent, posterior_codes, posterior_labels)
    nearest_indices = nearest_posterior_indices(latent, posterior_codes)
    display = MnistDisplay(
        decoder_images=decoded,
        mass_images=masses.reshape(-1, MNIST_MASS_SIDE, MNIST_MASS_SIDE),
        nearest_digits=nearest_digits,
        nearest_real_images=posterior_images[nearest_indices, 0],
        nearest_real_digits=posterior_labels[nearest_indices],
        latent_quantile_low=quantile_low,
        latent_quantile_high=quantile_high,
    )
    diagnostics = {
        "mnist_vertex_count": float(len(latent)),
        "mnist_digit_three_reference_count": float(
            np.sum(posterior_labels == MNIST_DIGIT)
        ),
        "mnist_target_positive_curvature_fraction": float(
            np.mean(curvature_mass[~boundary] > 0.0)
        ),
        "mnist_target_negative_curvature_fraction": float(
            np.mean(curvature_mass[~boundary] < 0.0)
        ),
    }
    return (
        SurfaceProblem(
            key="mnist",
            base_coordinates=latent,
            faces=faces,
            edges=edges,
            query_pairs=edges,
            query_distances=target_edges,
            target_edge_distances=target_edges,
            target_curvature_mass=curvature_mass,
            target_curvature_density=curvature_density,
            boundary=boundary,
            planar_base_coordinates=canonical_disk,
            holdout_pairs=holdout_pairs,
            holdout_distances=holdout_distances,
        ),
        display,
        diagnostics,
    )


def adjacency_average_matrix(vertex_count: int, edges: np.ndarray) -> Tensor:
    adjacency = np.zeros((vertex_count, vertex_count), dtype=np.float64)
    adjacency[edges[:, 0], edges[:, 1]] = 1.0
    adjacency[edges[:, 1], edges[:, 0]] = 1.0
    degree = np.sum(adjacency, axis=1, keepdims=True)
    adjacency /= np.maximum(degree, 1.0)
    return torch.as_tensor(adjacency, dtype=torch.float64)


def initialization_scale(
    base: np.ndarray, pairs: np.ndarray, targets: np.ndarray
) -> float:
    base_distances = pairwise_coordinate_distances(base, pairs)
    valid = base_distances > EPSILON
    ratios = targets[valid] / base_distances[valid]
    return float(np.median(ratios))


def evaluate_stress_torch(
    coordinates: Tensor, pairs: Tensor, targets: Tensor
) -> Tensor:
    differences = coordinates[pairs[:, 0]] - coordinates[pairs[:, 1]]
    distances = torch.linalg.vector_norm(differences, dim=1).clamp_min(EPSILON)
    relative = (distances - targets) / targets.clamp_min(EPSILON)
    return torch.mean(relative.square())


def optimize_parameters(
    parameters: list[Tensor],
    objective,
    adam_steps: int = ADAM_STEPS,
    lbfgs_steps: int = LBFGS_STEPS,
    adam_learning_rate: float = ADAM_LEARNING_RATE,
) -> tuple[float, np.ndarray, np.ndarray]:
    """Run Adam then full-batch L-BFGS and retain a compact loss trace."""

    history_steps: list[int] = []
    history_values: list[float] = []
    adam = torch.optim.Adam(parameters, lr=adam_learning_rate)
    best_value = math.inf
    best_parameters = [parameter.detach().clone() for parameter in parameters]
    for step in range(adam_steps):
        adam.zero_grad(set_to_none=True)
        value = objective()
        scalar_value = float(value.detach())
        if math.isfinite(scalar_value) and scalar_value < best_value:
            best_value = scalar_value
            best_parameters = [parameter.detach().clone() for parameter in parameters]
        value.backward()
        adam.step()
        if step % 10 == 0 or step + 1 == adam_steps:
            history_steps.append(step)
            history_values.append(scalar_value)

    with torch.no_grad():
        for parameter, best_parameter in zip(parameters, best_parameters):
            parameter.copy_(best_parameter)

    evaluations = 0
    lbfgs = torch.optim.LBFGS(
        parameters,
        lr=0.8,
        max_iter=lbfgs_steps,
        max_eval=lbfgs_steps * 2,
        tolerance_grad=1.0e-10,
        tolerance_change=1.0e-13,
        history_size=80,
        line_search_fn="strong_wolfe",
    )

    def closure() -> Tensor:
        nonlocal best_parameters, best_value, evaluations
        lbfgs.zero_grad(set_to_none=True)
        value = objective()
        scalar_value = float(value.detach())
        if math.isfinite(scalar_value) and scalar_value < best_value:
            best_value = scalar_value
            best_parameters = [parameter.detach().clone() for parameter in parameters]
        value.backward()
        if evaluations % 5 == 0:
            history_steps.append(adam_steps + evaluations)
            history_values.append(scalar_value)
        evaluations += 1
        return value

    lbfgs.step(closure)
    current_value = float(objective().detach())
    if math.isfinite(current_value) and current_value < best_value:
        best_value = current_value
        best_parameters = [parameter.detach().clone() for parameter in parameters]
    with torch.no_grad():
        for parameter, best_parameter in zip(parameters, best_parameters):
            parameter.copy_(best_parameter)
    final = best_value
    history_steps.append(adam_steps + evaluations)
    history_values.append(final)
    return final, np.asarray(history_steps), np.asarray(history_values)


def fit_free_embedding(problem: SurfaceProblem, dimension: int, seed: int) -> FitOutput:
    rng = np.random.default_rng(MASTER_SEED + 100 * dimension + seed)
    centered = problem.base_coordinates - np.mean(problem.base_coordinates, axis=0)
    scale = initialization_scale(centered, problem.query_pairs, problem.query_distances)
    if dimension == 2:
        initial = scale * centered + 0.002 * scale * rng.normal(size=centered.shape)
    elif dimension == 3:
        initial = np.column_stack(
            [
                scale * centered,
                0.08 * scale * rng.normal(size=len(centered)),
            ]
        )
    else:
        raise ValueError("only dimensions two and three are supported")

    coordinates = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    pairs = torch.as_tensor(problem.query_pairs, dtype=torch.long)
    targets = torch.as_tensor(problem.query_distances, dtype=torch.float64)

    def objective() -> Tensor:
        centered_coordinates = coordinates - torch.mean(coordinates, dim=0)
        gauge = 1.0e-10 * torch.mean(centered_coordinates.square())
        return evaluate_stress_torch(centered_coordinates, pairs, targets) + gauge

    final_objective, history_steps, history_values = optimize_parameters(
        [coordinates], objective
    )
    fitted = coordinates.detach().numpy()
    fitted -= np.mean(fitted, axis=0)
    stress = float(
        np.mean(
            (
                (
                    pairwise_coordinate_distances(fitted, problem.query_pairs)
                    - problem.query_distances
                )
                / problem.query_distances
            )
            ** 2
        )
    )
    return FitOutput(
        fitted,
        final_objective,
        stress,
        history_steps,
        history_values,
    )


def fit_height_field(problem: SurfaceProblem, seed: int) -> FitOutput:
    """Fit X=(A z,q) with invertible triangular A and smooth graph height q."""

    rng = np.random.default_rng(MASTER_SEED + 300 + seed)
    base = problem.base_coordinates - np.mean(problem.base_coordinates, axis=0)
    scale = initialization_scale(base, problem.query_pairs, problem.query_distances)
    log_diagonal = torch.tensor(
        [math.log(scale), math.log(scale)], dtype=torch.float64, requires_grad=True
    )
    shear = torch.tensor(0.0, dtype=torch.float64, requires_grad=True)
    height = torch.tensor(
        0.10 * scale * rng.normal(size=len(base)),
        dtype=torch.float64,
        requires_grad=True,
    )
    base_tensor = torch.as_tensor(base, dtype=torch.float64)
    pairs = torch.as_tensor(problem.query_pairs, dtype=torch.long)
    targets = torch.as_tensor(problem.query_distances, dtype=torch.float64)
    adjacency_average = adjacency_average_matrix(len(base), problem.edges)
    target_scale = float(np.median(problem.target_edge_distances))

    def coordinates() -> Tensor:
        zero = torch.zeros((), dtype=torch.float64)
        matrix = torch.stack(
            [
                torch.stack([torch.exp(log_diagonal[0]), shear]),
                torch.stack([zero, torch.exp(log_diagonal[1])]),
            ]
        )
        planar = base_tensor @ matrix.T
        centered_height = height - torch.mean(height)
        return torch.cat([planar, centered_height[:, None]], dim=1)

    def objective() -> Tensor:
        fitted = coordinates()
        stress = evaluate_stress_torch(fitted, pairs, targets)
        centered_height = height - torch.mean(height)
        laplacian = centered_height - adjacency_average @ centered_height
        bending = torch.mean((laplacian / max(target_scale, EPSILON)).square())
        return stress + BENDING_WEIGHT * bending

    final_objective, history_steps, history_values = optimize_parameters(
        [log_diagonal, shear, height], objective
    )
    fitted = coordinates().detach().numpy()
    stress = float(
        np.mean(
            (
                (
                    pairwise_coordinate_distances(fitted, problem.query_pairs)
                    - problem.query_distances
                )
                / problem.query_distances
            )
            ** 2
        )
    )
    return FitOutput(
        fitted,
        final_objective,
        stress,
        history_steps,
        history_values,
    )


def adjacent_face_pairs(faces: np.ndarray) -> np.ndarray:
    """Return face pairs sharing an interior triangulation edge."""

    incident: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            incident.setdefault(edge, []).append(face_index)
    pairs = [tuple(indices) for indices in incident.values() if len(indices) == 2]
    return np.asarray(pairs, dtype=np.int64)


def face_normals_torch(coordinates: Tensor, faces: Tensor) -> Tensor:
    first = coordinates[faces[:, 1]] - coordinates[faces[:, 0]]
    second = coordinates[faces[:, 2]] - coordinates[faces[:, 0]]
    normals = torch.linalg.cross(first, second, dim=1)
    return normals / torch.linalg.vector_norm(normals, dim=1, keepdim=True).clamp_min(
        1.0e-10
    )


def normal_smoothness_statistics(
    coordinates: np.ndarray, faces: np.ndarray
) -> tuple[float, float, float]:
    first = coordinates[faces[:, 1]] - coordinates[faces[:, 0]]
    second = coordinates[faces[:, 2]] - coordinates[faces[:, 0]]
    normals = np.cross(first, second)
    normals /= np.maximum(np.linalg.norm(normals, axis=1, keepdims=True), EPSILON)
    adjacent = adjacent_face_pairs(faces)
    dots = np.sum(normals[adjacent[:, 0]] * normals[adjacent[:, 1]], axis=1)
    return (
        float(np.quantile(dots, SMOOTHNESS_NORMAL_QUANTILE)),
        float(np.mean(dots < 0.0)),
        float(np.mean(1.0 - dots)),
    )


def target_face_basis_inverses(problem: SurfaceProblem) -> Tensor:
    """Return inverse 2D edge bases for every target metric triangle."""

    lengths = face_edge_length_array(
        problem.faces, problem.edges, problem.target_edge_distances
    )
    first = lengths[:, 0]
    opposite = lengths[:, 1]
    second = lengths[:, 2]
    horizontal = (first**2 + second**2 - opposite**2) / (2.0 * first)
    vertical = np.sqrt(np.maximum(second**2 - horizontal**2, EPSILON))
    bases = np.zeros((len(lengths), 2, 2), dtype=np.float64)
    bases[:, 0, 0] = first
    bases[:, 0, 1] = horizontal
    bases[:, 1, 1] = vertical
    return torch.as_tensor(np.linalg.inv(bases), dtype=torch.float64)


def fit_scaled_latent_2d(problem: SurfaceProblem, seed: int) -> FitOutput:
    """Fit the positive global scale of a whitened latent plane."""

    planar_base = (
        problem.base_coordinates
        if problem.planar_base_coordinates is None
        else problem.planar_base_coordinates
    )
    centered = planar_base - np.mean(planar_base, axis=0)
    covariance = np.cov(centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    inverse_square_root = (
        eigenvectors * np.reciprocal(np.sqrt(np.maximum(eigenvalues, EPSILON)))
    ) @ eigenvectors.T
    whitened = centered @ inverse_square_root
    stress_scale = initialization_scale(
        whitened, problem.query_pairs, problem.query_distances
    )
    initial_raw = math.log(max(stress_scale, EPSILON)) + 0.03 * (seed - 1)
    raw_scale = torch.tensor(initial_raw, dtype=torch.float64, requires_grad=True)
    base_tensor = torch.as_tensor(whitened, dtype=torch.float64)
    pairs = torch.as_tensor(problem.query_pairs, dtype=torch.long)
    targets = torch.as_tensor(problem.query_distances, dtype=torch.float64)

    def coordinates() -> Tensor:
        scale = torch.exp(raw_scale)
        return scale * base_tensor

    def objective() -> Tensor:
        return evaluate_stress_torch(coordinates(), pairs, targets)

    final_objective, history_steps, history_values = optimize_parameters(
        [raw_scale], objective, adam_steps=900, adam_learning_rate=2.0e-2
    )
    fitted = coordinates().detach().numpy()
    stress = float(
        np.mean(
            (
                (
                    pairwise_coordinate_distances(fitted, problem.query_pairs)
                    - problem.query_distances
                )
                / problem.query_distances
            )
            ** 2
        )
    )
    return FitOutput(
        fitted,
        final_objective,
        stress,
        history_steps,
        history_values,
    )


def fit_smooth_embedding(
    problem: SurfaceProblem, smoothing_lambda: float, seed: int
) -> FitOutput:
    """Fit free R3 coordinates with observable adjacent-normal bending."""

    rng = np.random.default_rng(MASTER_SEED + 500 + 1000 * seed)
    centered = problem.base_coordinates - np.mean(problem.base_coordinates, axis=0)
    scale = initialization_scale(centered, problem.query_pairs, problem.query_distances)
    if smoothing_lambda == 0.0:
        initial_height = 0.08 * scale * rng.normal(size=len(centered))
    else:
        normalized = centered / np.maximum(np.std(centered, axis=0), 0.1)
        phase = 0.7 * seed
        initial_height = (
            0.08
            * scale
            * (
                0.45 * normalized[:, 0] * normalized[:, 1]
                + 0.25
                * np.sin(1.1 * normalized[:, 0] + phase)
                * np.cos(0.9 * normalized[:, 1] - phase)
            )
        )
    initial = np.column_stack([scale * centered, initial_height])
    coordinates = torch.tensor(initial, dtype=torch.float64, requires_grad=True)
    pairs = torch.as_tensor(problem.query_pairs, dtype=torch.long)
    targets = torch.as_tensor(problem.query_distances, dtype=torch.float64)
    faces = torch.as_tensor(problem.faces, dtype=torch.long)
    adjacent = torch.as_tensor(adjacent_face_pairs(problem.faces), dtype=torch.long)
    target_basis_inverse = target_face_basis_inverses(problem)

    def objective() -> Tensor:
        centered_coordinates = coordinates - torch.mean(coordinates, dim=0)
        stress = evaluate_stress_torch(centered_coordinates, pairs, targets)
        normals = face_normals_torch(centered_coordinates, faces)
        normal_dots = torch.sum(
            normals[adjacent[:, 0]] * normals[adjacent[:, 1]], dim=1
        )
        bending = torch.mean(1.0 - normal_dots)
        first_edges = (
            centered_coordinates[faces[:, 1]] - centered_coordinates[faces[:, 0]]
        )
        second_edges = (
            centered_coordinates[faces[:, 2]] - centered_coordinates[faces[:, 0]]
        )
        display_edge_bases = torch.stack([first_edges, second_edges], dim=2)
        face_affine_maps = display_edge_bases @ target_basis_inverse
        face_singular_values = torch.linalg.svdvals(face_affine_maps)
        collapse = torch_functional.relu(
            FACE_SINGULAR_VALUE_FLOOR - face_singular_values[:, -1]
        )
        shape_penalty = torch.mean(collapse.square()) + torch.max(collapse).square()
        gauge = 1.0e-10 * torch.mean(centered_coordinates.square())
        return (
            stress
            + smoothing_lambda * bending
            + (FACE_SHAPE_PENALTY_WEIGHT if smoothing_lambda > 0.0 else 0.0)
            * shape_penalty
            + gauge
        )

    final_objective, history_steps, history_values = optimize_parameters(
        [coordinates],
        objective,
        adam_steps=1_800,
        adam_learning_rate=5.0e-3,
    )
    fitted = coordinates.detach().numpy()
    fitted -= np.mean(fitted, axis=0)
    stress = float(
        np.mean(
            (
                (
                    pairwise_coordinate_distances(fitted, problem.query_pairs)
                    - problem.query_distances
                )
                / problem.query_distances
            )
            ** 2
        )
    )
    return FitOutput(
        fitted,
        final_objective,
        stress,
        history_steps,
        history_values,
    )


def mnist_smoothness_path(
    problem: SurfaceProblem,
) -> tuple[Embedding, Embedding, list[SmoothnessCandidate]]:
    """Select minimum stress subject to predeclared smoothness/shape rules."""

    path: list[SmoothnessCandidate] = []
    for smoothing_lambda in SMOOTHNESS_LAMBDAS:
        candidates: list[Embedding] = []
        for seed in OPTIMIZATION_SEEDS:
            output = fit_smooth_embedding(problem, smoothing_lambda, seed)
            candidates.append(
                Embedding(
                    method="free_3d" if smoothing_lambda == 0.0 else "smooth_3d",
                    coordinates=output.coordinates,
                    seed=seed,
                    objective=output.objective,
                    query_stress=output.stress,
                    history_steps=output.history_steps,
                    history_values=output.history_values,
                    smoothing_lambda=smoothing_lambda,
                )
            )
        best = min(
            candidates, key=lambda candidate: (candidate.objective, candidate.seed)
        )
        quantile, negative_fraction, _ = normal_smoothness_statistics(
            best.coordinates, problem.faces
        )
        display_edges = pairwise_coordinate_distances(best.coordinates, problem.edges)
        global_s_min, _, _ = exact_distortion_and_curvature_certificate(
            problem, display_edges
        )
        path.append(
            SmoothnessCandidate(
                smoothing_lambda=smoothing_lambda,
                embedding=best,
                normal_dot_quantile_05=quantile,
                negative_normal_fraction=negative_fraction,
                global_face_affine_s_min=global_s_min,
            )
        )
        print(
            f"mnist smoothness lambda={smoothing_lambda:.3g}: "
            f"RMS={100.0 * math.sqrt(best.query_stress):.4f}%, "
            f"normal-dot q05={quantile:.4f}, negative={100.0 * negative_fraction:.2f}%, "
            f"global s_min={global_s_min:.4f}",
            flush=True,
        )
    passing = [
        candidate
        for candidate in path
        if candidate.smoothing_lambda > 0.0
        and candidate.normal_dot_quantile_05 >= SMOOTHNESS_MINIMUM_NORMAL_DOT
        and candidate.global_face_affine_s_min >= SMOOTHNESS_MINIMUM_GLOBAL_S_MIN
    ]
    selected = (
        min(
            passing,
            key=lambda candidate: (
                candidate.embedding.query_stress,
                candidate.smoothing_lambda,
            ),
        )
        if passing
        else path[-1]
    )
    selected_embedding = Embedding(
        method="smooth_3d",
        coordinates=selected.embedding.coordinates,
        seed=selected.embedding.seed,
        objective=selected.embedding.objective,
        query_stress=selected.embedding.query_stress,
        history_steps=selected.embedding.history_steps,
        history_values=selected.embedding.history_values,
        smoothing_lambda=selected.smoothing_lambda,
    )
    free = path[0].embedding
    return selected_embedding, free, path


def best_embedding(problem: SurfaceProblem, method: str) -> Embedding:
    candidates: list[Embedding] = []
    for seed in OPTIMIZATION_SEEDS:
        if method == "scaled_latent_2d":
            output = fit_scaled_latent_2d(problem, seed)
        elif method == "free_2d_reference":
            output = fit_free_embedding(problem, 2, seed)
        elif method == "height_field_3d":
            output = fit_height_field(problem, seed)
        elif method == "free_3d":
            output = fit_free_embedding(problem, 3, seed)
        else:
            raise ValueError(f"unknown method: {method}")
        candidate = Embedding(
            method=method,
            coordinates=output.coordinates,
            seed=seed,
            objective=output.objective,
            query_stress=output.stress,
            history_steps=output.history_steps,
            history_values=output.history_values,
        )
        candidates.append(candidate)
        print(
            f"{problem.key:10s} {method:15s} seed={seed}: "
            f"relative RMS stress={100.0 * math.sqrt(output.stress):.4f}%",
            flush=True,
        )
    if method == "scaled_latent_2d":
        for candidate in candidates:
            display_edges = pairwise_coordinate_distances(
                candidate.coordinates, problem.edges
            )
            global_s_min, _, _ = exact_distortion_and_curvature_certificate(
                problem, display_edges
            )
            signed_area_ratio = minimum_signed_area_ratio(
                problem, candidate.coordinates
            )
            print(
                f"{problem.key:10s} scaled-latent audit seed={candidate.seed}: "
                f"s_min={global_s_min:.4f}, signed-area ratio={signed_area_ratio:.4f}",
                flush=True,
            )
    return min(candidates, key=lambda candidate: (candidate.objective, candidate.seed))


def rigid_alignment(
    coordinates: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, float]:
    if coordinates.shape[1] != reference.shape[1]:
        raise ValueError("alignment dimensions differ")
    centered = coordinates - np.mean(coordinates, axis=0)
    reference_centered = reference - np.mean(reference, axis=0)
    left, _, right = np.linalg.svd(centered.T @ reference_centered)
    rotation = left @ right
    aligned = centered @ rotation + np.mean(reference, axis=0)
    rmse = float(np.sqrt(np.mean(np.sum((aligned - reference) ** 2, axis=1))))
    return aligned, rmse


def graph_all_pairs(
    vertex_count: int, edges: np.ndarray, lengths: np.ndarray
) -> np.ndarray:
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([lengths, lengths])
    graph = coo_matrix((values, (rows, columns)), shape=(vertex_count, vertex_count))
    return np.asarray(dijkstra(graph.tocsr(), directed=False), dtype=np.float64)


def embedding_metrics(
    problem: SurfaceProblem, embedding: Embedding
) -> dict[str, object]:
    query_estimate = pairwise_coordinate_distances(
        embedding.coordinates, problem.query_pairs
    )
    query_relative = (
        query_estimate - problem.query_distances
    ) / problem.query_distances
    edge_estimate = pairwise_coordinate_distances(embedding.coordinates, problem.edges)
    edge_residual = edge_estimate - problem.target_edge_distances
    edge_relative = edge_residual / problem.target_edge_distances

    curvature_mass, _, curvature_density = triangle_geometry_from_edge_lengths(
        len(problem.base_coordinates), problem.faces, problem.edges, edge_estimate
    )
    interior = ~problem.boundary
    curvature_error = curvature_mass[interior] - problem.target_curvature_mass[interior]
    display_minimum_angle, display_minimum_slack = triangle_quality(
        problem.faces, problem.edges, edge_estimate
    )
    global_s_min, global_s_max, curvature_certificate = (
        exact_distortion_and_curvature_certificate(problem, edge_estimate)
    )
    if np.any(np.abs(curvature_error) > curvature_certificate[interior] + 1.0e-9):
        raise AssertionError("computed curvature certificate was violated")

    target_graph = graph_all_pairs(
        len(problem.base_coordinates), problem.edges, problem.target_edge_distances
    )
    estimate_graph = graph_all_pairs(
        len(problem.base_coordinates), problem.edges, edge_estimate
    )
    upper = np.triu_indices(len(problem.base_coordinates), k=1)
    graph_residual = estimate_graph[upper] - target_graph[upper]
    graph_relative = graph_residual / np.maximum(target_graph[upper], EPSILON)

    aligned_rmse = math.nan
    aligned_relative_rmse = math.nan
    if problem.true_coordinates is not None and embedding.coordinates.shape[1] == 3:
        _, aligned_rmse = rigid_alignment(
            embedding.coordinates, problem.true_coordinates
        )
        reference_scale = float(
            np.sqrt(
                np.mean(
                    np.sum(
                        (
                            problem.true_coordinates
                            - np.mean(problem.true_coordinates, axis=0)
                        )
                        ** 2,
                        axis=1,
                    )
                )
            )
        )
        aligned_relative_rmse = aligned_rmse / reference_scale

    holdout_relative_rms = math.nan
    holdout_relative_max = math.nan
    if problem.holdout_pairs is not None and problem.holdout_distances is not None:
        holdout_estimate = pairwise_coordinate_distances(
            embedding.coordinates, problem.holdout_pairs
        )
        holdout_relative = (
            holdout_estimate - problem.holdout_distances
        ) / problem.holdout_distances
        holdout_relative_rms = float(np.sqrt(np.mean(holdout_relative**2)))
        holdout_relative_max = float(np.max(np.abs(holdout_relative)))

    normal_quantile = math.nan
    negative_normal_fraction = math.nan
    mean_bending = math.nan
    if embedding.coordinates.shape[1] == 3:
        normal_quantile, negative_normal_fraction, mean_bending = (
            normal_smoothness_statistics(embedding.coordinates, problem.faces)
        )

    minimum_orientation_area_ratio = math.nan
    if embedding.coordinates.shape[1] == 2:
        minimum_orientation_area_ratio = minimum_signed_area_ratio(
            problem, embedding.coordinates
        )

    minimum_angle, minimum_slack = target_triangle_quality(problem)
    minimum_edge = float(np.min(problem.target_edge_distances))

    vertex_count = len(problem.base_coordinates)
    query_fraction = (
        2.0 * len(problem.query_pairs) / (vertex_count * (vertex_count - 1))
    )
    return {
        "experiment": problem.key,
        "method": embedding.method,
        "embedding_dimension": embedding.coordinates.shape[1],
        "selected_seed": embedding.seed,
        "vertex_count": vertex_count,
        "face_count": len(problem.faces),
        "edge_count": len(problem.edges),
        "query_count": len(problem.query_pairs),
        "query_fraction": query_fraction,
        "query_relative_rms": float(np.sqrt(np.mean(query_relative**2))),
        "query_relative_max": float(np.max(np.abs(query_relative))),
        "holdout_two_hop_relative_rms": holdout_relative_rms,
        "holdout_two_hop_relative_max": holdout_relative_max,
        "edge_max_absolute_residual": float(np.max(np.abs(edge_residual))),
        "edge_max_relative_residual": float(np.max(np.abs(edge_relative))),
        "minimum_target_edge_length": minimum_edge,
        "display_delta_over_ell_min": float(
            np.max(np.abs(edge_residual)) / minimum_edge
        ),
        "minimum_target_triangle_angle_degrees": minimum_angle,
        "minimum_target_normalized_triangle_slack": minimum_slack,
        "minimum_display_triangle_angle_degrees": display_minimum_angle,
        "minimum_display_normalized_triangle_slack": display_minimum_slack,
        "global_face_affine_s_min": global_s_min,
        "global_face_affine_s_max": global_s_max,
        "interior_pl_curvature_mass_rmse": float(np.sqrt(np.mean(curvature_error**2))),
        "interior_pl_curvature_mass_mean_absolute_error": float(
            np.mean(np.abs(curvature_error))
        ),
        "interior_pl_curvature_mass_max_absolute_error": float(
            np.max(np.abs(curvature_error))
        ),
        "interior_curvature_certificate_delta_mean": float(
            np.mean(curvature_certificate[interior])
        ),
        "interior_curvature_certificate_delta_rms": float(
            np.sqrt(np.mean(curvature_certificate[interior] ** 2))
        ),
        "interior_curvature_certificate_delta_max": float(
            np.max(curvature_certificate[interior])
        ),
        "edge_graph_all_pairs_distance_rmse": float(
            np.sqrt(np.mean(graph_residual**2))
        ),
        "edge_graph_all_pairs_relative_rmse": float(
            np.sqrt(np.mean(graph_relative**2))
        ),
        "controlled_aligned_rmse": aligned_rmse,
        "controlled_aligned_relative_rmse": aligned_relative_rmse,
        "optimizer_objective": embedding.objective,
        "smoothing_lambda": embedding.smoothing_lambda,
        "adjacent_normal_dot_q05": normal_quantile,
        "negative_adjacent_normal_fraction": negative_normal_fraction,
        "mean_normal_bending": mean_bending,
        "minimum_signed_display_to_target_area_ratio": minimum_orientation_area_ratio,
        "target_positive_curvature_fraction": float(
            np.mean(problem.target_curvature_mass[interior] > 0.0)
        ),
        "target_negative_curvature_fraction": float(
            np.mean(problem.target_curvature_mass[interior] < 0.0)
        ),
        "reconstructed_positive_curvature_fraction": float(
            np.mean(curvature_mass[interior] > 0.0)
        ),
        "reconstructed_negative_curvature_fraction": float(
            np.mean(curvature_mass[interior] < 0.0)
        ),
        "reconstructed_curvature_density_min": float(
            np.min(curvature_density[interior])
        ),
        "reconstructed_curvature_density_max": float(
            np.max(curvature_density[interior])
        ),
    }


def write_results(rows: list[dict[str, object]]) -> None:
    path = HERE / "curvature_certified_visualization_results.csv"
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def write_history(
    problems: dict[str, SurfaceProblem], embeddings: dict[tuple[str, str], Embedding]
) -> None:
    path = HERE / "curvature_certified_visualization_history.csv"
    fieldnames = ["experiment", "method", "selected_seed", "step", "objective"]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for experiment in problems:
            methods = (
                ("scaled_latent_2d", "height_field_3d", "free_3d")
                if experiment == "controlled"
                else (
                    "scaled_latent_2d",
                    "free_2d_reference",
                    "smooth_3d",
                    "free_3d",
                )
            )
            for method in methods:
                embedding = embeddings[(experiment, method)]
                for step, value in zip(
                    embedding.history_steps, embedding.history_values
                ):
                    writer.writerow(
                        {
                            "experiment": experiment,
                            "method": method,
                            "selected_seed": embedding.seed,
                            "step": int(step),
                            "objective": f"{value:.17g}",
                        }
                    )


def write_smoothness_path(path: list[SmoothnessCandidate]) -> None:
    output = HERE / "curvature_certified_visualization_smoothness.csv"
    fieldnames = [
        "smoothing_lambda",
        "selected_seed",
        "query_relative_rms",
        "normal_dot_q05",
        "negative_normal_fraction",
        "global_face_affine_s_min",
        "passes_q05_and_shape_rule",
    ]
    with output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for candidate in path:
            writer.writerow(
                {
                    "smoothing_lambda": candidate.smoothing_lambda,
                    "selected_seed": candidate.embedding.seed,
                    "query_relative_rms": math.sqrt(candidate.embedding.query_stress),
                    "normal_dot_q05": candidate.normal_dot_quantile_05,
                    "negative_normal_fraction": candidate.negative_normal_fraction,
                    "global_face_affine_s_min": candidate.global_face_affine_s_min,
                    "passes_q05_and_shape_rule": (
                        candidate.smoothing_lambda > 0.0
                        and candidate.normal_dot_quantile_05
                        >= SMOOTHNESS_MINIMUM_NORMAL_DOT
                        and candidate.global_face_affine_s_min
                        >= SMOOTHNESS_MINIMUM_GLOBAL_S_MIN
                    ),
                }
            )


def method_label(method: str, japanese: bool) -> str:
    labels = {
        "scaled_latent_2d": ("scaled latent R2", "尺度調整した潜在平面R2"),
        "free_2d_reference": ("free R2 reference", "自由R2参照"),
        "height_field_3d": ("height-field 3D", "高さ場3D"),
        "smooth_3d": ("smooth R3", "滑らかR3"),
        "free_3d": ("free R3 reference", "自由R3参照"),
    }
    return labels[method][1 if japanese else 0]


def write_table(rows: list[dict[str, object]], japanese: bool) -> None:
    suffix = "_ja" if japanese else ""
    headers = (
        [
            r"Data / representation",
            r"Query RMS",
            r"Edge max",
            r"Curvature RMSE",
            r"Graph-distance RMS",
            r"Aligned RMSE",
        ]
        if not japanese
        else [
            r"データ / 表示",
            r"query RMS",
            r"辺最大誤差",
            r"曲率質量RMSE",
            r"graph距離RMS",
            r"整列RMSE",
        ]
    )
    lines = [
        r"\begin{tabular}{lrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    experiment_labels = {
        "controlled": ("Controlled", "制御例"),
        "mnist": ("MNIST digit-3-centered", "MNIST数字3中心領域"),
    }
    for experiment_index, experiment in enumerate(("controlled", "mnist")):
        selected = [row for row in rows if row["experiment"] == experiment]
        methods = (
            ("scaled_latent_2d", "height_field_3d", "free_3d")
            if experiment == "controlled"
            else (
                "scaled_latent_2d",
                "free_2d_reference",
                "smooth_3d",
                "free_3d",
            )
        )
        for method_index, method in enumerate(methods):
            row = next(item for item in selected if item["method"] == method)
            data_label = (
                experiment_labels[experiment][1 if japanese else 0]
                if method_index == 0
                else ""
            )
            aligned = float(row["controlled_aligned_rmse"])
            aligned_text = "--" if math.isnan(aligned) else f"{aligned:.4f}"
            lines.append(
                f"{data_label} {method_label(method, japanese)} & "
                f"{100.0 * float(row['query_relative_rms']):.3f}\\% & "
                f"{100.0 * float(row['edge_max_relative_residual']):.3f}\\% & "
                f"{float(row['interior_pl_curvature_mass_rmse']):.4f} & "
                f"{100.0 * float(row['edge_graph_all_pairs_relative_rmse']):.3f}\\% & "
                f"{aligned_text} \\\\"
            )
        if experiment_index == 0:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    (HERE / f"curvature_certified_visualization_table{suffix}.tex").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )


def write_certificate_table(rows: list[dict[str, object]], japanese: bool) -> None:
    """Write the fully computable shape/distortion/curvature audit."""

    suffix = "_ja" if japanese else ""
    headers = (
        [
            "Data / representation",
            "Min angle observed/display",
            "Slack observed/display",
            r"$\delta/\ell_{\min}$",
            r"$[s_{\min},s_{\max}]$",
            r"Curv. RMSE / max",
            r"$\max_i\Delta_i$",
        ]
        if not japanese
        else [
            "データ / 表示",
            "最小角 観測/表示",
            "三角余裕 観測/表示",
            r"$\delta/\ell_{\min}$",
            r"$[s_{\min},s_{\max}]$",
            r"曲率RMSE / 最大",
            r"$\max_i\Delta_i$",
        ]
    )
    lines = [
        r"\begin{tabular}{lrrrrrr}",
        r"\toprule",
        " & ".join(headers) + r" \\",
        r"\midrule",
    ]
    experiment_labels = {
        "controlled": ("Controlled", "制御例"),
        "mnist": ("MNIST digit-3-centered", "MNIST数字3中心領域"),
    }
    for experiment_index, experiment in enumerate(("controlled", "mnist")):
        methods = (
            ("scaled_latent_2d", "height_field_3d", "free_3d")
            if experiment == "controlled"
            else (
                "scaled_latent_2d",
                "free_2d_reference",
                "smooth_3d",
                "free_3d",
            )
        )
        for method_index, method in enumerate(methods):
            row = next(
                item
                for item in rows
                if item["experiment"] == experiment and item["method"] == method
            )
            data_label = (
                experiment_labels[experiment][1 if japanese else 0]
                if method_index == 0
                else ""
            )
            lines.append(
                f"{data_label} {method_label(method, japanese)} & "
                f"{float(row['minimum_target_triangle_angle_degrees']):.1f}/"
                f"{float(row['minimum_display_triangle_angle_degrees']):.1f} & "
                f"{float(row['minimum_target_normalized_triangle_slack']):.3f}/"
                f"{float(row['minimum_display_normalized_triangle_slack']):.3f} & "
                f"{float(row['display_delta_over_ell_min']):.3f} & "
                f"[{float(row['global_face_affine_s_min']):.3f},"
                f"{float(row['global_face_affine_s_max']):.3f}] & "
                f"{float(row['interior_pl_curvature_mass_rmse']):.3f}/"
                f"{float(row['interior_pl_curvature_mass_max_absolute_error']):.3f} & "
                f"{float(row['interior_curvature_certificate_delta_max']):.3f} \\\\"
            )
        if experiment_index == 0:
            lines.append(r"\addlinespace")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    output = HERE / f"curvature_certified_visualization_certificate_table{suffix}.tex"
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def vertex_curvature_density(
    problem: SurfaceProblem, coordinates: np.ndarray
) -> np.ndarray:
    edge_lengths = pairwise_coordinate_distances(coordinates, problem.edges)
    _, _, density = triangle_geometry_from_edge_lengths(
        len(coordinates), problem.faces, problem.edges, edge_lengths
    )
    density = density.copy()
    density[problem.boundary] = np.nan
    return density


def vertex_curvature_mass(
    problem: SurfaceProblem, coordinates: np.ndarray
) -> np.ndarray:
    edge_lengths = pairwise_coordinate_distances(coordinates, problem.edges)
    mass, _, _ = triangle_geometry_from_edge_lengths(
        len(coordinates), problem.faces, problem.edges, edge_lengths
    )
    mass = mass.copy()
    mass[problem.boundary] = np.nan
    return mass


def robust_symmetric_limit(*values: np.ndarray) -> float:
    finite = np.concatenate([value[np.isfinite(value)] for value in values])
    return max(float(np.quantile(np.abs(finite), 0.96)), 1.0e-4)


def add_surface(
    axis,
    coordinates: np.ndarray,
    faces: np.ndarray,
    values: np.ndarray,
    norm: Normalize,
    cmap: mpl.colors.Colormap,
    alpha: float = 1.0,
) -> None:
    polygons = coordinates[faces]
    face_values = np.nanmean(values[faces], axis=1)
    collection = Poly3DCollection(
        polygons,
        facecolors=cmap(norm(face_values)),
        edgecolor=(0.12, 0.12, 0.12, 0.18),
        linewidth=0.16,
        alpha=alpha,
    )
    axis.add_collection3d(collection)
    mins = np.min(coordinates, axis=0)
    maxs = np.max(coordinates, axis=0)
    axis.set_xlim(mins[0], maxs[0])
    axis.set_ylim(mins[1], maxs[1])
    axis.set_zlim(mins[2], maxs[2] if maxs[2] > mins[2] else mins[2] + 1.0)
    axis.set_box_aspect(np.maximum(maxs - mins, 0.2))


def style_3d(axis) -> None:
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(False)
    axis.view_init(elev=27, azim=-62)


def save_figure(figure: mpl.figure.Figure, stem: str) -> None:
    figure.savefig(
        HERE / f"{stem}.png",
        dpi=220,
        bbox_inches="tight",
        metadata={"Software": "curvature_certified_visualization.py"},
    )
    figure.savefig(
        HERE / f"{stem}.pdf",
        bbox_inches="tight",
        metadata={
            "Creator": "curvature_certified_visualization.py",
            "CreationDate": None,
            "ModDate": None,
        },
    )
    plt.close(figure)


def save_control_figure(
    problem: SurfaceProblem,
    height_embedding: Embedding,
    metrics: dict[str, object],
    japanese: bool,
) -> None:
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    target_density = problem.target_curvature_density.copy()
    target_density[problem.boundary] = np.nan
    recovered_density = vertex_curvature_density(problem, height_embedding.coordinates)
    limit = robust_symmetric_limit(target_density, recovered_density)
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = mpl.colormaps["coolwarm"]
    figure = plt.figure(figsize=(12.8, 7.4))
    axes = [figure.add_subplot(2, 2, index + 1, projection="3d") for index in range(4)]

    flat = np.column_stack(
        [problem.base_coordinates, np.zeros(len(problem.base_coordinates))]
    )
    add_surface(axes[0], flat, problem.faces, target_density, norm, cmap)
    axes[0].set_title(
        "入力：平面潜在座標 $(u,v)$\n色＝W2辺長から得る目標PL曲率"
        if japanese
        else "INPUT: flat latent coordinates $(u,v)$\ncolor = target PL curvature from W2 edge lengths",
        fontsize=10,
    )
    add_surface(
        axes[1],
        problem.true_coordinates,
        problem.faces,
        target_density,
        norm,
        cmap,
    )
    axes[1].set_title(
        "正解（評価時のみ）：2-bump＋saddle曲面"
        if japanese
        else "TRUTH (evaluation only): two-bump + saddle graph",
        fontsize=10,
    )
    aligned, _ = rigid_alignment(height_embedding.coordinates, problem.true_coordinates)
    add_surface(axes[2], aligned, problem.faces, recovered_density, norm, cmap)
    axes[2].set_title(
        "出力：局所W2距離から復元した高さ場3D\n色＝復元PL曲率"
        if japanese
        else "OUTPUT: height-field 3D fitted to local W2\ncolor = reconstructed PL curvature",
        fontsize=10,
    )

    true_segments = problem.true_coordinates[problem.edges]
    recovered_segments = aligned[problem.edges]
    axes[3].add_collection3d(
        Line3DCollection(true_segments, colors="#555555", linewidths=0.35, alpha=0.45)
    )
    axes[3].add_collection3d(
        Line3DCollection(
            recovered_segments, colors="#d22d2d", linewidths=0.40, alpha=0.55
        )
    )
    combined = np.concatenate([problem.true_coordinates, aligned], axis=0)
    mins = np.min(combined, axis=0)
    maxs = np.max(combined, axis=0)
    axes[3].set_xlim(mins[0], maxs[0])
    axes[3].set_ylim(mins[1], maxs[1])
    axes[3].set_zlim(mins[2], maxs[2])
    axes[3].set_box_aspect(np.maximum(maxs - mins, 0.2))
    axes[3].set_title(
        "重ね合わせ：正解＝灰、出力＝赤\n剛体整列は評価にのみ使用"
        if japanese
        else "OVERLAY: truth = gray, output = red\nrigid alignment is used only for evaluation",
        fontsize=10,
    )
    for axis in axes:
        style_3d(axis)
    scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    scalar.set_array([])
    figure.colorbar(
        scalar,
        ax=axes[:3],
        fraction=0.025,
        pad=0.02,
        label="PL曲率密度" if japanese else "PL curvature density",
    )
    figure.suptitle(
        (
            "制御例：対角Gaussian W2は真の3D chordと厳密一致\n"
            f"局所query RMS={100.0 * float(metrics['query_relative_rms']):.3f}%, "
            f"整列RMSE={float(metrics['controlled_aligned_rmse']):.4f}"
        )
        if japanese
        else (
            "Controlled decoder: diagonal-Gaussian W2 equals the true 3D chord exactly\n"
            f"local-query RMS={100.0 * float(metrics['query_relative_rms']):.3f}%, "
            f"aligned RMSE={float(metrics['controlled_aligned_rmse']):.4f}"
        ),
        fontsize=13,
    )
    figure.subplots_adjust(left=0.02, right=0.91, bottom=0.03, top=0.88, wspace=0.03)
    save_figure(figure, f"curvature_certified_control{suffix}")


def categorical_face_colors(labels: np.ndarray, faces: np.ndarray) -> np.ndarray:
    face_labels = []
    for face in faces:
        counts = np.bincount(labels[face], minlength=10)
        face_labels.append(int(np.argmax(counts)))
    cmap = mpl.colormaps["tab10"]
    return cmap(np.asarray(face_labels) / 9.0)


def add_digit_surface(
    axis, coordinates: np.ndarray, faces: np.ndarray, digits: np.ndarray
) -> None:
    if coordinates.shape[1] == 2:
        coordinates = np.column_stack([coordinates, np.zeros(len(coordinates))])
    collection = Poly3DCollection(
        coordinates[faces],
        facecolors=categorical_face_colors(digits, faces),
        edgecolor=(0.1, 0.1, 0.1, 0.18),
        linewidth=0.15,
    )
    axis.add_collection3d(collection)
    mins = np.min(coordinates, axis=0)
    maxs = np.max(coordinates, axis=0)
    axis.set_xlim(mins[0], maxs[0])
    axis.set_ylim(mins[1], maxs[1])
    axis.set_zlim(mins[2], maxs[2] if maxs[2] > mins[2] else mins[2] + 1.0)
    axis.set_box_aspect(np.maximum(maxs - mins, 0.2))


def representative_indices(coordinates: np.ndarray, count: int = 7) -> np.ndarray:
    """Deterministic farthest-point landmarks for the decoded-image strip."""

    center = np.argmin(
        np.sum((coordinates - np.mean(coordinates, axis=0)) ** 2, axis=1)
    )
    chosen = [int(center)]
    minimum_squared = np.sum((coordinates - coordinates[center]) ** 2, axis=1)
    for _ in range(count - 1):
        candidate = int(np.argmax(minimum_squared))
        chosen.append(candidate)
        squared = np.sum((coordinates - coordinates[candidate]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.asarray(chosen, dtype=np.int64)


def save_mnist_figure(
    problem: SurfaceProblem,
    flat_embedding: Embedding,
    smooth_embedding: Embedding,
    display: MnistDisplay,
    flat_metrics: dict[str, object],
    smooth_metrics: dict[str, object],
    japanese: bool,
) -> None:
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    figure = plt.figure(figsize=(14.2, 8.3))
    grid = figure.add_gridspec(
        2, 3, height_ratios=[3.15, 2.0], hspace=0.16, wspace=0.05
    )
    flat_axis = figure.add_subplot(grid[0, 0])
    smooth_axis = figure.add_subplot(grid[0, 1], projection="3d")
    curvature_axis = figure.add_subplot(grid[0, 2], projection="3d")
    strip_axis = figure.add_subplot(grid[1, :])

    face_digits = np.asarray(
        [
            np.argmax(np.bincount(display.nearest_digits[face], minlength=10))
            for face in problem.faces
        ]
    )
    digit_cmap = mpl.colormaps["tab10"]
    digit_norm = BoundaryNorm(np.arange(-0.5, 10.5), digit_cmap.N)
    flat_axis.tripcolor(
        flat_embedding.coordinates[:, 0],
        flat_embedding.coordinates[:, 1],
        problem.faces,
        facecolors=face_digits,
        cmap=digit_cmap,
        norm=digit_norm,
        edgecolors="none",
    )
    flat_axis.triplot(
        flat_embedding.coordinates[:, 0],
        flat_embedding.coordinates[:, 1],
        problem.faces,
        color=(0.1, 0.1, 0.1, 0.20),
        linewidth=0.18,
    )
    flat_axis.set_aspect("equal", adjustable="box")
    flat_axis.set_xticks([])
    flat_axis.set_yticks([])
    flat_axis.set_title(
        "尺度調整した潜在平面R2：global scaleのみ最適化\n色＝15近傍posteriorの数字"
        if japanese
        else "SCALED LATENT R2: one global scale fitted\ncolor = 15-NN posterior digit",
        fontsize=10,
    )
    add_digit_surface(
        smooth_axis,
        smooth_embedding.coordinates,
        problem.faces,
        display.nearest_digits,
    )
    smooth_axis.set_title(
        "滑らか出力R3：辺stress＋隣接面normal bending\n色＝15近傍posteriorの数字"
        if japanese
        else "SMOOTH OUTPUT R3: edge stress + normal bending\ncolor = 15-NN posterior digit",
        fontsize=10,
    )
    recovered_mass = vertex_curvature_mass(problem, smooth_embedding.coordinates)
    target_mass = problem.target_curvature_mass.copy()
    target_mass[problem.boundary] = np.nan
    limit = robust_symmetric_limit(target_mass, recovered_mass)
    norm = Normalize(vmin=-limit, vmax=limit)
    cmap = mpl.colormaps["coolwarm"]
    add_surface(
        curvature_axis,
        smooth_embedding.coordinates,
        problem.faces,
        recovered_mass,
        norm,
        cmap,
    )
    curvature_axis.set_title(
        "同じ滑らかR3出力\n色＝角欠損PL曲率質量（radian、境界除外）"
        if japanese
        else "SAME SMOOTH R3 OUTPUT\ncolor = angle-defect PL curvature mass (rad; boundary omitted)",
        fontsize=10,
    )
    for axis in (smooth_axis, curvature_axis):
        style_3d(axis)

    landmarks = representative_indices(smooth_embedding.coordinates)
    for number, index in enumerate(landmarks, start=1):
        point = smooth_embedding.coordinates[index]
        smooth_axis.text(
            point[0], point[1], point[2], str(number), fontsize=8, color="black"
        )
    strip_axis.set_axis_off()
    for column, index in enumerate(landmarks):
        left = 0.09 + column * (0.89 / len(landmarks))
        real_axis = strip_axis.inset_axes([left, 0.53, 0.09, 0.40])
        decoded_axis = strip_axis.inset_axes([left, 0.04, 0.09, 0.40])
        real_axis.imshow(
            display.nearest_real_images[index], cmap="gray", vmin=0.0, vmax=1.0
        )
        decoded_axis.imshow(
            display.decoder_images[index], cmap="gray", vmin=0.0, vmax=1.0
        )
        for image_axis in (real_axis, decoded_axis):
            image_axis.set_xticks([])
            image_axis.set_yticks([])
        real_axis.set_title(
            f"#{column + 1}: y={int(display.nearest_real_digits[index])}",
            fontsize=8,
        )
    strip_axis.text(
        0.0,
        0.73,
        "最近傍の実入力" if japanese else "nearest real input",
        transform=strip_axis.transAxes,
        va="center",
        fontsize=9,
    )
    strip_axis.text(
        0.0,
        0.24,
        "復号measure\nsource（VAE平均）\n14×14でOT"
        if japanese
        else "decoded measure\nsource (VAE mean)\n14x14 for OT",
        transform=strip_axis.transAxes,
        va="center",
        fontsize=8,
    )

    digit_scalar = mpl.cm.ScalarMappable(norm=digit_norm, cmap=digit_cmap)
    digit_scalar.set_array([])
    digit_color_axis = figure.add_axes([0.15, 0.365, 0.23, 0.014])
    figure.colorbar(
        digit_scalar,
        cax=digit_color_axis,
        orientation="horizontal",
        ticks=range(10),
        label="近傍数字" if japanese else "nearest posterior digit",
    )
    digit_color_axis.xaxis.set_label_position("top")
    curvature_scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
    curvature_scalar.set_array([])
    figure.colorbar(
        curvature_scalar,
        ax=curvature_axis,
        fraction=0.045,
        pad=0.01,
        label="PL曲率質量 (rad)" if japanese else "PL curvature mass (rad)",
    )
    figure.suptitle(
        (
            "MNIST数字3中心posterior領域：14×14 pixel質量の非正則化離散W2で曲面化\n"
            f"観測＝{len(problem.edges)}三角辺；smooth選択規約：q05(normal内積)>=0.25かつ face s_min>=0.35；"
            f"R2 RMS={100.0 * float(flat_metrics['query_relative_rms']):.2f}% → "
            f"smooth R3 RMS={100.0 * float(smooth_metrics['query_relative_rms']):.2f}% "
            f"(λ={float(smooth_metrics['smoothing_lambda']):.3g})"
        )
        if japanese
        else (
            "MNIST digit-3-centered region: unregularized discrete W2 on 14x14 pixel masses\n"
            f"observed = {len(problem.edges)} edges; smooth selection: q05(normal dot) >= 0.25 and face s_min >= 0.35; "
            f"R2 RMS={100.0 * float(flat_metrics['query_relative_rms']):.2f}% → "
            f"smooth R3 RMS={100.0 * float(smooth_metrics['query_relative_rms']):.2f}% "
            f"(λ={float(smooth_metrics['smoothing_lambda']):.3g})"
        ),
        fontsize=12,
    )
    figure.subplots_adjust(left=0.03, right=0.96, bottom=0.02, top=0.85)
    save_figure(figure, f"curvature_certified_mnist{suffix}")


def save_diagnostics_figure(
    problems: dict[str, SurfaceProblem],
    embeddings: dict[tuple[str, str], Embedding],
    rows: list[dict[str, object]],
    smoothness_path: list[SmoothnessCandidate],
    japanese: bool,
) -> None:
    suffix = "_ja" if japanese else ""
    if japanese:
        plt.rcParams["font.family"] = japanese_font_family()
    figure, axes = plt.subplots(2, 2, figsize=(11.4, 7.5))
    method_colors = {
        "scaled_latent_2d": CONTROL_COLOR,
        "height_field_3d": HEIGHT_COLOR,
        "smooth_3d": MNIST_COLOR,
        "free_3d": FREE_COLOR,
    }
    loss_axis = axes[0, 0]
    for method in ("scaled_latent_2d", "height_field_3d", "free_3d"):
        embedding = embeddings[("controlled", method)]
        loss_axis.plot(
            embedding.history_steps,
            np.maximum(embedding.history_values, 1.0e-14),
            color=method_colors[method],
            lw=1.6,
            label=method_label(method, japanese),
        )
    loss_axis.set_yscale("log")
    loss_axis.set_xlabel("optimizer evaluation" if not japanese else "最適化評価回数")
    loss_axis.set_ylabel("objective")
    loss_axis.set_title("Controlled optimization" if not japanese else "制御例の最適化")
    loss_axis.grid(alpha=0.18, which="both")
    loss_axis.legend(frameon=False, fontsize=8)

    pareto_axis = axes[0, 1]
    pareto_rms = np.asarray(
        [100.0 * math.sqrt(item.embedding.query_stress) for item in smoothness_path]
    )
    pareto_q05 = np.asarray([item.normal_dot_quantile_05 for item in smoothness_path])
    pareto_axis.plot(pareto_rms, pareto_q05, color=MNIST_COLOR, lw=1.4)
    pareto_axis.scatter(
        pareto_rms,
        pareto_q05,
        c=np.arange(len(smoothness_path)),
        cmap="viridis",
        s=42,
        zorder=3,
    )
    for rms, quantile, item in zip(pareto_rms, pareto_q05, smoothness_path):
        pareto_axis.annotate(
            f"λ={item.smoothing_lambda:g}\ns={item.global_face_affine_s_min:.2f}",
            (rms, quantile),
            xytext=(4, 3),
            textcoords="offset points",
            fontsize=7,
        )
    pareto_axis.axhline(SMOOTHNESS_MINIMUM_NORMAL_DOT, color="#222222", ls="--", lw=1.0)
    pareto_axis.set_xlabel("edge relative RMS (%)")
    pareto_axis.set_ylabel("adjacent-normal dot q05")
    pareto_axis.set_title(
        "MNIST：観測可能な滑らかさ規約"
        if japanese
        else "MNIST: observable smoothness selection"
    )
    pareto_axis.grid(alpha=0.18)

    x_positions = np.arange(4)
    width = 0.34
    for experiment_index, experiment in enumerate(("controlled", "mnist")):
        selected = [row for row in rows if row["experiment"] == experiment]
        methods = (
            ("scaled_latent_2d", None, "height_field_3d", "free_3d")
            if experiment == "controlled"
            else (
                "scaled_latent_2d",
                "free_2d_reference",
                "smooth_3d",
                "free_3d",
            )
        )
        values = [
            (
                math.nan
                if method is None
                else 100.0
                * float(
                    next(row for row in selected if row["method"] == method)[
                        "query_relative_rms"
                    ]
                )
            )
            for method in methods
        ]
        axes[1, 0].bar(
            x_positions + (experiment_index - 0.5) * width,
            values,
            width=width,
            color=CONTROL_COLOR if experiment == "controlled" else MNIST_COLOR,
            alpha=0.85,
            label="制御例"
            if japanese and experiment == "controlled"
            else ("Controlled" if experiment == "controlled" else "MNIST"),
        )
    axes[1, 0].set_xticks(
        x_positions,
        [
            "scaled R2",
            "自由R2" if japanese else "free R2",
            "主3D" if japanese else "main 3D",
            "自由R3" if japanese else "free R3",
        ],
    )
    axes[1, 0].set_ylabel("query relative RMS (%)")
    axes[1, 0].set_yscale("log")
    axes[1, 0].set_title(
        "局所W2距離の再現誤差" if japanese else "Local W2 realization error"
    )
    free_2d_row = next(
        row
        for row in rows
        if row["experiment"] == "mnist" and row["method"] == "free_2d_reference"
    )
    free_2d_rms = 100.0 * float(free_2d_row["query_relative_rms"])
    free_2d_s_min = float(free_2d_row["global_face_affine_s_min"])
    free_2d_flipped = (
        float(free_2d_row["minimum_signed_display_to_target_area_ratio"]) < 0.0
    )
    free_2d_status = (
        ("反転あり" if free_2d_flipped else "向き保存")
        if japanese
        else ("flipped" if free_2d_flipped else "orientation preserved")
    )
    axes[1, 0].annotate(
        f"{'面' if japanese else 'face'} s_min={free_2d_s_min:.3f}; {free_2d_status}",
        (x_positions[1] + 0.5 * width, free_2d_rms),
        xytext=(0, 4),
        textcoords="offset points",
        ha="center",
        va="bottom",
        fontsize=7,
    )
    axes[1, 0].legend(frameon=False, fontsize=8)
    axes[1, 0].grid(axis="y", alpha=0.18, which="both")

    markers = {"controlled": "o", "mnist": "s"}
    for experiment in ("controlled", "mnist"):
        problem = problems[experiment]
        method = "height_field_3d" if experiment == "controlled" else "smooth_3d"
        embedding = embeddings[(experiment, method)]
        edge_lengths = pairwise_coordinate_distances(
            embedding.coordinates, problem.edges
        )
        reconstructed, _, _ = triangle_geometry_from_edge_lengths(
            len(problem.base_coordinates), problem.faces, problem.edges, edge_lengths
        )
        interior = ~problem.boundary
        axes[1, 1].scatter(
            problem.target_curvature_mass[interior],
            reconstructed[interior],
            s=11,
            alpha=0.48,
            marker=markers[experiment],
            color=CONTROL_COLOR if experiment == "controlled" else MNIST_COLOR,
            label="制御例"
            if japanese and experiment == "controlled"
            else ("Controlled" if experiment == "controlled" else "MNIST"),
        )
    limits = axes[1, 1].get_xlim()
    lower = min(limits[0], axes[1, 1].get_ylim()[0])
    upper = max(limits[1], axes[1, 1].get_ylim()[1])
    axes[1, 1].plot([lower, upper], [lower, upper], color="#222222", ls="--", lw=1.0)
    axes[1, 1].set_xlim(lower, upper)
    axes[1, 1].set_ylim(lower, upper)
    axes[1, 1].set_xlabel(
        "観測辺W2によるPL曲率質量" if japanese else "observed edge-W2 PL curvature mass"
    )
    axes[1, 1].set_ylabel(
        "表示PL曲率質量" if japanese else "displayed PL curvature mass"
    )
    axes[1, 1].set_title(
        "辺長が曲率を保証（内部頂点のみ）"
        if japanese
        else "Edge-length certificate for curvature (interior only)"
    )
    axes[1, 1].legend(frameon=False, fontsize=8)
    axes[1, 1].grid(alpha=0.18)
    figure.suptitle(
        "最適化と幾何診断" if japanese else "Optimization and geometric diagnostics",
        fontsize=13,
    )
    figure.tight_layout()
    save_figure(figure, f"curvature_certified_diagnostics{suffix}")


def print_summary(rows: list[dict[str, object]], diagnostics: dict[str, float]) -> None:
    for key, value in diagnostics.items():
        print(f"{key}={value:.10g}", flush=True)
    for row in rows:
        print(
            f"{row['experiment']:10s} {row['method']:15s}: "
            f"query RMS={100.0 * float(row['query_relative_rms']):.4f}%, "
            f"edge max={100.0 * float(row['edge_max_relative_residual']):.4f}%, "
            f"curvature RMSE={float(row['interior_pl_curvature_mass_rmse']):.5f}, "
            f"graph RMS={100.0 * float(row['edge_graph_all_pairs_relative_rmse']):.4f}%",
            flush=True,
        )


def main() -> None:
    seed_everything(MASTER_SEED)
    controlled, controlled_diagnostics = controlled_problem()
    mnist, mnist_display, mnist_diagnostics = mnist_problem()
    problems = {"controlled": controlled, "mnist": mnist}

    embeddings: dict[tuple[str, str], Embedding] = {}
    for method in ("scaled_latent_2d", "height_field_3d", "free_3d"):
        embeddings[("controlled", method)] = best_embedding(controlled, method)
    embeddings[("mnist", "scaled_latent_2d")] = best_embedding(
        mnist, "scaled_latent_2d"
    )
    embeddings[("mnist", "free_2d_reference")] = best_embedding(
        mnist, "free_2d_reference"
    )
    mnist_smooth, mnist_free, smoothness_path = mnist_smoothness_path(mnist)
    embeddings[("mnist", "smooth_3d")] = mnist_smooth
    embeddings[("mnist", "free_3d")] = mnist_free

    rows: list[dict[str, object]] = []
    for experiment, methods in (
        ("controlled", ("scaled_latent_2d", "height_field_3d", "free_3d")),
        (
            "mnist",
            (
                "scaled_latent_2d",
                "free_2d_reference",
                "smooth_3d",
                "free_3d",
            ),
        ),
    ):
        problem = problems[experiment]
        rows.extend(
            embedding_metrics(problem, embeddings[(experiment, method)])
            for method in methods
        )
    write_results(rows)
    write_history(problems, embeddings)
    write_smoothness_path(smoothness_path)
    write_table(rows, japanese=False)
    write_table(rows, japanese=True)
    write_certificate_table(rows, japanese=False)
    write_certificate_table(rows, japanese=True)

    controlled_height_metrics = next(
        row
        for row in rows
        if row["experiment"] == "controlled" and row["method"] == "height_field_3d"
    )
    mnist_flat_metrics = next(
        row
        for row in rows
        if row["experiment"] == "mnist" and row["method"] == "scaled_latent_2d"
    )
    mnist_smooth_metrics = next(
        row
        for row in rows
        if row["experiment"] == "mnist" and row["method"] == "smooth_3d"
    )
    for japanese in (False, True):
        save_control_figure(
            controlled,
            embeddings[("controlled", "height_field_3d")],
            controlled_height_metrics,
            japanese,
        )
        save_mnist_figure(
            mnist,
            embeddings[("mnist", "scaled_latent_2d")],
            embeddings[("mnist", "smooth_3d")],
            mnist_display,
            mnist_flat_metrics,
            mnist_smooth_metrics,
            japanese,
        )
        save_diagnostics_figure(problems, embeddings, rows, smoothness_path, japanese)

    quality_diagnostics: dict[str, float] = {}
    for key, problem in problems.items():
        minimum_angle, minimum_slack = target_triangle_quality(problem)
        quality_diagnostics[f"{key}_minimum_target_triangle_angle_degrees"] = (
            minimum_angle
        )
        quality_diagnostics[f"{key}_minimum_normalized_triangle_slack"] = minimum_slack
    print_summary(
        rows,
        controlled_diagnostics | mnist_diagnostics | quality_diagnostics,
    )
    print("wrote curvature-certified visualization artifacts", flush=True)


if __name__ == "__main__":
    main()
