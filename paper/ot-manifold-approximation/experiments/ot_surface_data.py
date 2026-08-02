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
"""Build and cache an actual-posterior MNIST surface problem.

The vertices returned here are posterior means of real test-set digit-3
images, not points from a filled ellipse.  The construction is deliberately
independent of any downstream embedding algorithm:

1. encode the MNIST test set with the cached two-dimensional VAE;
2. robustly remove posterior outliers and low-density points;
3. choose a spatially spread subset by deterministic farthest-point sampling;
4. triangulate whitened chart coordinates and remove pathological faces;
5. decode every retained vertex and evaluate exact pixel-space W2 on every
   graph pair at one or two hops.

The expensive finite OT problem is cached under ``experiments/.cache``.  The
explicit cache version and teacher-checkpoint digest make stale artifacts fail
closed.  A downstream experiment should import
``load_or_build_mnist_actual_posterior_surface`` and need not know how the
posterior mesh or OT observations were constructed.
"""

from __future__ import annotations

import argparse
import hashlib
import math
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import ot
import torch
from mnist_low_rank_geometry import (
    DATASET_SPECS,
    DEFAULT_EPOCHS,
    DEFAULT_TRAIN_EXAMPLES,
    GaussianVAE,
    TrainingConfig,
    load_datasets,
    load_or_train_teacher,
)
from mnist_low_rank_geometry import SEED as MNIST_VAE_SEED
from mnist_low_rank_geometry import seed_everything as seed_vae_training
from scipy.spatial import Delaunay
from torch.utils.data import DataLoader

HERE = Path(__file__).resolve().parent
CACHE_DIRECTORY = HERE / ".cache"

# Bump this manually whenever any semantic part of the finite problem changes.
CACHE_VERSION = 1
CACHE_PATH = (
    CACHE_DIRECTORY / f"mnist_digit3_actual_posterior_surface_v{CACHE_VERSION}.npz"
)

MASTER_SEED = 20260802
MNIST_DIGIT = 3
TARGET_VERTEX_COUNT = 240
MASS_IMAGE_SIDE = 14
GRAPH_QUERY_HOPS = 2

# Robust trimming is performed before spatial subsampling.  The first trim is
# a robust Mahalanobis-radius trim; the second removes unusually sparse points
# according to their 12-neighbour radius in the robustly standardized chart.
INITIAL_BOX_QUANTILE = 0.02
OUTLIER_RADIUS_QUANTILE = 0.95
DENSITY_NEIGHBOURS = 12
DENSITY_RADIUS_QUANTILE = 0.88
MINIMUM_CANDIDATE_MULTIPLIER = 1.35

# Delaunay filtering is intentionally mild after farthest-point sampling.  It
# removes long bridges across posterior gaps and numerically thin triangles.
LOCAL_MESH_NEIGHBOURS = 6
MAXIMUM_EDGE_TO_LOCAL_SCALE = 2.50
MINIMUM_TRIANGLE_ANGLE_DEGREES = 8.0
MINIMUM_RETAINED_VERTEX_FRACTION = 0.88

EPSILON = 1.0e-12


@dataclass(frozen=True)
class ActualPosteriorSurfaceData:
    """Finite local-Wasserstein geometry on actual MNIST posterior points."""

    cache_version: int
    checkpoint_sha256: str
    digit: int
    digit_reference_count: int
    trimmed_candidate_count: int
    source_indices: np.ndarray
    source_labels: np.ndarray
    source_images: np.ndarray
    latent_codes: np.ndarray
    chart_coordinates: np.ndarray
    whitening_center: np.ndarray
    whitening_matrix: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    boundary_vertices: np.ndarray
    decoded_images: np.ndarray
    mass_images: np.ndarray
    query_pairs: np.ndarray
    query_distances: np.ndarray
    edge_distances: np.ndarray
    minimum_chart_triangle_angle_degrees: float
    maximum_chart_edge_to_local_scale: float

    @property
    def vertex_count(self) -> int:
        return len(self.latent_codes)


def _seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.set_num_threads(min(8, max(1, torch.get_num_threads())))
    torch.use_deterministic_algorithms(True)


def _checkpoint_digest(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _load_teacher_and_test_set():
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
    model.eval()
    _seed_everything(MASTER_SEED)
    return model, test, specification.checkpoint


@torch.no_grad()
def _encode_test_set(
    model: GaussianVAE, dataset
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    loader = DataLoader(dataset, batch_size=512, shuffle=False, num_workers=0)
    codes: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    images: list[np.ndarray] = []
    for batch_images, batch_labels in loader:
        means, _ = model.encode(batch_images)
        codes.append(means.numpy().astype(np.float64))
        labels.append(batch_labels.numpy().astype(np.int64))
        images.append(batch_images[:, 0].numpy().astype(np.float64))
    all_codes = np.concatenate(codes)
    all_labels = np.concatenate(labels)
    all_images = np.concatenate(images)
    indices = np.arange(len(all_codes), dtype=np.int64)
    return all_codes, all_labels, all_images, indices


def _symmetric_inverse_square_root(covariance: np.ndarray) -> np.ndarray:
    symmetric = 0.5 * (covariance + covariance.T)
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    largest = max(float(np.max(eigenvalues)), EPSILON)
    floor = largest * 1.0e-8
    eigenvalues = np.maximum(eigenvalues, floor)
    return (eigenvectors * np.reciprocal(np.sqrt(eigenvalues))) @ eigenvectors.T


def _robust_trim_indices(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return dense inlier indices and robustly standardized coordinates."""

    if codes.ndim != 2 or codes.shape[1] != 2:
        raise ValueError("the cached MNIST teacher must have a 2D latent code")
    if len(codes) < TARGET_VERTEX_COUNT * 2:
        raise ValueError("too few digit references for robust posterior trimming")

    coordinate_low = np.quantile(codes, INITIAL_BOX_QUANTILE, axis=0)
    coordinate_high = np.quantile(codes, 1.0 - INITIAL_BOX_QUANTILE, axis=0)
    central = np.all((codes >= coordinate_low) & (codes <= coordinate_high), axis=1)
    if np.sum(central) < TARGET_VERTEX_COUNT * 1.5:
        raise AssertionError("initial robust posterior box retained too few points")

    robust_center = np.median(codes[central], axis=0)
    robust_covariance = np.cov(codes[central] - robust_center, rowvar=False)
    robust_inverse_sqrt = _symmetric_inverse_square_root(robust_covariance)
    standardized = (codes - robust_center) @ robust_inverse_sqrt
    squared_radius = np.sum(standardized * standardized, axis=1)
    radius_threshold = float(
        np.quantile(squared_radius[central], OUTLIER_RADIUS_QUANTILE)
    )
    radial_inliers = central & (squared_radius <= radius_threshold)
    radial_indices = np.flatnonzero(radial_inliers)

    radial_coordinates = standardized[radial_indices]
    differences = radial_coordinates[:, None, :] - radial_coordinates[None, :, :]
    squared_distances = np.sum(differences * differences, axis=2)
    neighbour = min(DENSITY_NEIGHBOURS, len(radial_coordinates) - 1)
    if neighbour < 2:
        raise AssertionError("not enough radial inliers for a density estimate")
    neighbour_radius_squared = np.partition(squared_distances, kth=neighbour, axis=1)[
        :, neighbour
    ]
    density_threshold = float(
        np.quantile(neighbour_radius_squared, DENSITY_RADIUS_QUANTILE)
    )
    dense = neighbour_radius_squared <= density_threshold
    dense_indices = radial_indices[dense]
    minimum_candidates = math.ceil(TARGET_VERTEX_COUNT * MINIMUM_CANDIDATE_MULTIPLIER)
    if len(dense_indices) < minimum_candidates:
        raise AssertionError(
            f"robust trimming retained {len(dense_indices)} candidates; "
            f"expected at least {minimum_candidates}"
        )
    return dense_indices, standardized[dense_indices]


def _whiten(codes: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    center = np.mean(codes, axis=0)
    covariance = np.cov(codes - center, rowvar=False)
    matrix = _symmetric_inverse_square_root(covariance)
    chart = (codes - center) @ matrix
    return chart, center, matrix


def _farthest_point_indices(coordinates: np.ndarray, count: int) -> np.ndarray:
    """Deterministic Euclidean farthest-point sampling, seeded at the medoid."""

    if count <= 0 or count > len(coordinates):
        raise ValueError("invalid farthest-point sample count")
    centered = coordinates - np.mean(coordinates, axis=0)
    selected = np.empty(count, dtype=np.int64)
    selected[0] = int(np.argmin(np.sum(centered * centered, axis=1)))
    differences = coordinates - coordinates[selected[0]]
    minimum_squared_distance = np.sum(differences * differences, axis=1)
    minimum_squared_distance[selected[0]] = -math.inf
    for position in range(1, count):
        next_index = int(np.argmax(minimum_squared_distance))
        if minimum_squared_distance[next_index] <= EPSILON:
            raise AssertionError(
                "posterior candidates do not contain enough unique points"
            )
        selected[position] = next_index
        differences = coordinates - coordinates[next_index]
        candidate_squared_distance = np.sum(differences * differences, axis=1)
        minimum_squared_distance = np.minimum(
            minimum_squared_distance, candidate_squared_distance
        )
        minimum_squared_distance[selected[: position + 1]] = -math.inf
    if len(np.unique(selected)) != count:
        raise AssertionError("farthest-point sampling returned duplicate vertices")
    return selected


def _orient_faces(coordinates: np.ndarray, faces: np.ndarray) -> np.ndarray:
    oriented = np.asarray(faces, dtype=np.int64).copy()
    first = coordinates[oriented[:, 1]] - coordinates[oriented[:, 0]]
    second = coordinates[oriented[:, 2]] - coordinates[oriented[:, 0]]
    signed_twice_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    reverse = signed_twice_area < 0.0
    temporary = oriented[reverse, 1].copy()
    oriented[reverse, 1] = oriented[reverse, 2]
    oriented[reverse, 2] = temporary
    return oriented


def _triangle_geometry(
    coordinates: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = coordinates[faces]
    edge_lengths = np.stack(
        [
            np.linalg.norm(points[:, 1] - points[:, 0], axis=1),
            np.linalg.norm(points[:, 2] - points[:, 1], axis=1),
            np.linalg.norm(points[:, 0] - points[:, 2], axis=1),
        ],
        axis=1,
    )
    a = edge_lengths[:, 1]
    b = edge_lengths[:, 2]
    c = edge_lengths[:, 0]
    cosines = np.stack(
        [
            (b * b + c * c - a * a) / np.maximum(2.0 * b * c, EPSILON),
            (a * a + c * c - b * b) / np.maximum(2.0 * a * c, EPSILON),
            (a * a + b * b - c * c) / np.maximum(2.0 * a * b, EPSILON),
        ],
        axis=1,
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    return edge_lengths, angles, np.min(angles, axis=1)


def _vertex_local_scales(coordinates: np.ndarray) -> np.ndarray:
    differences = coordinates[:, None, :] - coordinates[None, :, :]
    distances = np.linalg.norm(differences, axis=2)
    neighbour = min(LOCAL_MESH_NEIGHBOURS, len(coordinates) - 1)
    return np.partition(distances, kth=neighbour, axis=1)[:, neighbour]


def _largest_face_component(faces: np.ndarray) -> np.ndarray:
    incident: dict[tuple[int, int], list[int]] = {}
    for face_index, face in enumerate(faces):
        for first, second in (
            (face[0], face[1]),
            (face[1], face[2]),
            (face[2], face[0]),
        ):
            edge = tuple(sorted((int(first), int(second))))
            incident.setdefault(edge, []).append(face_index)
    adjacency: list[set[int]] = [set() for _ in range(len(faces))]
    for indices in incident.values():
        if len(indices) == 2:
            first, second = indices
            adjacency[first].add(second)
            adjacency[second].add(first)

    unvisited = set(range(len(faces)))
    components: list[list[int]] = []
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        unvisited.remove(seed)
        component: list[int] = []
        while stack:
            current = stack.pop()
            component.append(current)
            new = sorted(adjacency[current] & unvisited, reverse=True)
            for neighbour in new:
                unvisited.remove(neighbour)
                stack.append(neighbour)
        components.append(sorted(component))
    largest = max(components, key=lambda component: (len(component), -component[0]))
    return faces[np.asarray(largest, dtype=np.int64)]


def _filtered_delaunay_mesh(
    coordinates: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float, float]:
    triangulation = Delaunay(coordinates, qhull_options="Qbb Qc Qz Q12")
    faces = _orient_faces(coordinates, triangulation.simplices)
    edge_lengths, _, minimum_angles = _triangle_geometry(coordinates, faces)
    local_scales = _vertex_local_scales(coordinates)
    face_scales = np.median(local_scales[faces], axis=1)
    edge_to_scale = np.max(edge_lengths, axis=1) / np.maximum(face_scales, EPSILON)
    keep = (minimum_angles >= MINIMUM_TRIANGLE_ANGLE_DEGREES) & (
        edge_to_scale <= MAXIMUM_EDGE_TO_LOCAL_SCALE
    )
    faces = faces[keep]
    if len(faces) == 0:
        raise AssertionError("Delaunay quality filtering removed every triangle")
    faces = _largest_face_component(faces)

    retained_vertices = np.unique(faces)
    minimum_vertices = math.ceil(TARGET_VERTEX_COUNT * MINIMUM_RETAINED_VERTEX_FRACTION)
    if len(retained_vertices) < minimum_vertices:
        raise AssertionError(
            f"mesh filtering retained {len(retained_vertices)} vertices; "
            f"expected at least {minimum_vertices}"
        )
    remap = np.full(len(coordinates), -1, dtype=np.int64)
    remap[retained_vertices] = np.arange(len(retained_vertices), dtype=np.int64)
    faces = remap[faces]
    retained_coordinates = coordinates[retained_vertices]
    faces = _orient_faces(retained_coordinates, faces)
    final_edge_lengths, _, final_minimum_angles = _triangle_geometry(
        retained_coordinates, faces
    )
    final_scales = _vertex_local_scales(retained_coordinates)
    final_ratio = np.max(final_edge_lengths, axis=1) / np.maximum(
        np.median(final_scales[faces], axis=1), EPSILON
    )
    return (
        retained_vertices,
        retained_coordinates,
        faces,
        float(np.min(final_minimum_angles)),
        float(np.max(final_ratio)),
    )


def _unique_edges(faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def _boundary_vertices(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    edges, counts = np.unique(raw, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    boundary[np.unique(edges[counts == 1])] = True
    return boundary


def _validate_disk_topology(vertex_count: int, faces: np.ndarray) -> None:
    """Require a connected orientable triangular disk without pinched holes."""

    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    edges, counts = np.unique(raw, axis=0, return_counts=True)
    if np.any((counts < 1) | (counts > 2)):
        raise ValueError("the chart is not an edge-manifold")
    if vertex_count - len(edges) + len(faces) != 1:
        raise ValueError("the chart is not topologically a disk")

    boundary_edges = edges[counts == 1]
    boundary_degrees = np.bincount(boundary_edges.ravel(), minlength=vertex_count)
    boundary_mask = boundary_degrees > 0
    if not np.all(boundary_degrees[boundary_mask] == 2):
        raise ValueError("the chart boundary is pinched or branched")

    boundary_neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for first, second in boundary_edges:
        boundary_neighbours[int(first)].add(int(second))
        boundary_neighbours[int(second)].add(int(first))
    start = int(np.flatnonzero(boundary_mask)[0])
    visited = {start}
    frontier = {start}
    while frontier:
        frontier = {
            neighbour
            for vertex in frontier
            for neighbour in boundary_neighbours[vertex]
            if neighbour not in visited
        }
        visited.update(frontier)
    if len(visited) != int(np.sum(boundary_mask)):
        raise ValueError("the chart has more than one boundary component")


def _graph_hop_pairs(
    vertex_count: int, edges: np.ndarray, maximum_hops: int
) -> np.ndarray:
    neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for first, second in edges:
        neighbours[int(first)].add(int(second))
        neighbours[int(second)].add(int(first))
    pairs: list[tuple[int, int]] = []
    for source in range(vertex_count):
        visited = {source}
        frontier = {source}
        for _ in range(maximum_hops):
            next_frontier = {
                neighbour
                for vertex in frontier
                for neighbour in neighbours[vertex]
                if neighbour not in visited
            }
            visited.update(next_frontier)
            frontier = next_frontier
        pairs.extend((source, target) for target in sorted(visited) if target > source)
    return np.asarray(pairs, dtype=np.int64)


@torch.no_grad()
def _decode_images(model: GaussianVAE, latent_codes: np.ndarray) -> np.ndarray:
    points = torch.as_tensor(latent_codes, dtype=torch.float32)
    decoded = model.decode(points).reshape(-1, 28, 28)
    return decoded.numpy().astype(np.float64)


def _images_to_masses(images: np.ndarray) -> np.ndarray:
    clipped = np.clip(images, 0.0, 1.0)
    pooled = clipped.reshape(-1, MASS_IMAGE_SIDE, 2, MASS_IMAGE_SIDE, 2).mean(
        axis=(2, 4)
    )
    masses = pooled.reshape(len(pooled), -1) + 1.0e-10
    masses /= np.sum(masses, axis=1, keepdims=True)
    return masses.reshape(-1, MASS_IMAGE_SIDE, MASS_IMAGE_SIDE)


def _pixel_ground_cost() -> np.ndarray:
    axis = np.linspace(0.0, 1.0, MASS_IMAGE_SIDE, dtype=np.float64)
    horizontal, vertical = np.meshgrid(axis, axis, indexing="xy")
    positions = np.column_stack([horizontal.ravel(), vertical.ravel()])
    differences = positions[:, None, :] - positions[None, :, :]
    return np.sum(differences * differences, axis=2)


def _exact_pixel_w2(mass_images: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    masses = mass_images.reshape(len(mass_images), -1)
    cost = _pixel_ground_cost()
    distances = np.empty(len(pairs), dtype=np.float64)
    for index, (first, second) in enumerate(pairs):
        squared = ot.emd2(
            masses[int(first)],
            masses[int(second)],
            cost,
            numItermax=100_000,
            check_marginals=True,
        )
        distances[index] = math.sqrt(max(float(squared), 0.0))
        if (index + 1) % 500 == 0 or index + 1 == len(pairs):
            print(
                f"actual-posterior exact pixel OT: {index + 1}/{len(pairs)} pairs",
                flush=True,
            )
    return distances


def _edge_distances_from_queries(
    edges: np.ndarray, query_pairs: np.ndarray, query_distances: np.ndarray
) -> np.ndarray:
    lookup = {
        tuple(pair): float(distance)
        for pair, distance in zip(query_pairs.tolist(), query_distances)
    }
    missing = [tuple(edge) for edge in edges.tolist() if tuple(edge) not in lookup]
    if missing:
        raise AssertionError(f"{len(missing)} mesh edges are missing OT observations")
    return np.asarray([lookup[tuple(edge)] for edge in edges.tolist()])


def _validate_surface_data(data: ActualPosteriorSurfaceData) -> None:
    if data.cache_version != CACHE_VERSION:
        raise ValueError(
            f"cache version {data.cache_version} does not match {CACHE_VERSION}"
        )
    if data.digit != MNIST_DIGIT:
        raise ValueError("cached surface uses the wrong MNIST digit")
    if len(data.checkpoint_sha256) != 64:
        raise ValueError("cached teacher digest is malformed")

    vertex_count = data.vertex_count
    minimum_vertices = math.ceil(TARGET_VERTEX_COUNT * MINIMUM_RETAINED_VERTEX_FRACTION)
    if not minimum_vertices <= vertex_count <= TARGET_VERTEX_COUNT:
        raise ValueError(f"invalid retained vertex count: {vertex_count}")
    expected_shapes = {
        "source_indices": (vertex_count,),
        "source_labels": (vertex_count,),
        "source_images": (vertex_count, 28, 28),
        "latent_codes": (vertex_count, 2),
        "chart_coordinates": (vertex_count, 2),
        "whitening_center": (2,),
        "whitening_matrix": (2, 2),
        "decoded_images": (vertex_count, 28, 28),
        "mass_images": (vertex_count, MASS_IMAGE_SIDE, MASS_IMAGE_SIDE),
        "boundary_vertices": (vertex_count,),
    }
    for name, expected in expected_shapes.items():
        if getattr(data, name).shape != expected:
            raise ValueError(
                f"cached {name} has shape {getattr(data, name).shape}, "
                f"expected {expected}"
            )
    for name in (
        "source_images",
        "latent_codes",
        "chart_coordinates",
        "whitening_center",
        "whitening_matrix",
        "decoded_images",
        "mass_images",
        "query_distances",
        "edge_distances",
    ):
        if not np.all(np.isfinite(getattr(data, name))):
            raise ValueError(f"cached {name} contains a non-finite value")
    if len(np.unique(data.source_indices)) != vertex_count:
        raise ValueError("cached source indices are not unique")
    if not np.all(data.source_labels == MNIST_DIGIT):
        raise ValueError("cached mesh contains a non-target source label")
    if np.min(data.source_images) < 0.0 or np.max(data.source_images) > 1.0:
        raise ValueError("cached real images lie outside [0,1]")
    if np.min(data.mass_images) <= 0.0:
        raise ValueError("cached pixel masses must be strictly positive")
    if not np.allclose(np.sum(data.mass_images, axis=(1, 2)), 1.0, atol=2.0e-12):
        raise ValueError("cached pixel masses are not normalized")
    reconstructed_chart = (
        data.latent_codes - data.whitening_center
    ) @ data.whitening_matrix
    if not np.allclose(reconstructed_chart, data.chart_coordinates, atol=2.0e-10):
        raise ValueError("cached chart coordinates do not match the whitening map")

    if data.faces.ndim != 2 or data.faces.shape[1] != 3 or len(data.faces) == 0:
        raise ValueError("cached triangular faces are malformed")
    if np.min(data.faces) < 0 or np.max(data.faces) >= vertex_count:
        raise ValueError("cached face index is outside the vertex range")
    if len(np.unique(data.faces)) != vertex_count:
        raise ValueError("cached mesh contains an unused vertex")
    first = (
        data.chart_coordinates[data.faces[:, 1]]
        - data.chart_coordinates[data.faces[:, 0]]
    )
    second = (
        data.chart_coordinates[data.faces[:, 2]]
        - data.chart_coordinates[data.faces[:, 0]]
    )
    signed_area = first[:, 0] * second[:, 1] - first[:, 1] * second[:, 0]
    if np.any(signed_area <= EPSILON):
        raise ValueError("cached chart contains a degenerate or reversed face")
    _, _, minimum_angles = _triangle_geometry(data.chart_coordinates, data.faces)
    if np.min(minimum_angles) + 1.0e-9 < MINIMUM_TRIANGLE_ANGLE_DEGREES:
        raise ValueError("cached chart violates the minimum triangle angle")
    local_scales = _vertex_local_scales(data.chart_coordinates)
    chart_edge_lengths, _, _ = _triangle_geometry(data.chart_coordinates, data.faces)
    edge_to_local_scale = np.max(chart_edge_lengths, axis=1) / np.maximum(
        np.median(local_scales[data.faces], axis=1), EPSILON
    )
    if np.max(edge_to_local_scale) > MAXIMUM_EDGE_TO_LOCAL_SCALE + 1.0e-9:
        raise ValueError("cached chart contains a pathologically long face edge")
    if not math.isclose(
        data.minimum_chart_triangle_angle_degrees,
        float(np.min(minimum_angles)),
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise ValueError("cached minimum chart angle diagnostic is inconsistent")
    if not math.isclose(
        data.maximum_chart_edge_to_local_scale,
        float(np.max(edge_to_local_scale)),
        rel_tol=1.0e-10,
        abs_tol=1.0e-10,
    ):
        raise ValueError("cached chart edge-scale diagnostic is inconsistent")
    if not np.array_equal(data.edges, _unique_edges(data.faces)):
        raise ValueError("cached edge set does not match the triangular faces")
    if not np.array_equal(
        data.boundary_vertices, _boundary_vertices(vertex_count, data.faces)
    ):
        raise ValueError("cached boundary mask does not match the triangular faces")
    _validate_disk_topology(vertex_count, data.faces)

    expected_pairs = _graph_hop_pairs(vertex_count, data.edges, GRAPH_QUERY_HOPS)
    if not np.array_equal(data.query_pairs, expected_pairs):
        raise ValueError("cached OT pairs are not exactly the two-hop graph pairs")
    if data.query_distances.shape != (len(data.query_pairs),):
        raise ValueError("cached OT distances have the wrong shape")
    if np.any(data.query_distances <= 0.0):
        raise ValueError("cached OT distances must be strictly positive")
    expected_edge_distances = _edge_distances_from_queries(
        data.edges, data.query_pairs, data.query_distances
    )
    if not np.array_equal(data.edge_distances, expected_edge_distances):
        raise ValueError("cached edge distances disagree with the query observations")

    graph_neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for first_vertex, second_vertex in data.edges:
        graph_neighbours[int(first_vertex)].add(int(second_vertex))
        graph_neighbours[int(second_vertex)].add(int(first_vertex))
    visited = {0}
    frontier = {0}
    while frontier:
        frontier = {
            neighbour
            for vertex in frontier
            for neighbour in graph_neighbours[vertex]
            if neighbour not in visited
        }
        visited.update(frontier)
    if len(visited) != vertex_count:
        raise ValueError("cached triangular mesh is disconnected")


def _build_surface_data() -> ActualPosteriorSurfaceData:
    model, test, checkpoint_path = _load_teacher_and_test_set()
    checkpoint_sha256 = _checkpoint_digest(checkpoint_path)
    codes, labels, images, source_indices = _encode_test_set(model, test)
    digit_mask = labels == MNIST_DIGIT
    digit_codes = codes[digit_mask]
    digit_labels = labels[digit_mask]
    digit_images = images[digit_mask]
    digit_source_indices = source_indices[digit_mask]

    candidate_indices, _ = _robust_trim_indices(digit_codes)
    candidate_codes = digit_codes[candidate_indices]
    candidate_chart, whitening_center, whitening_matrix = _whiten(candidate_codes)
    sampled_candidates = _farthest_point_indices(candidate_chart, TARGET_VERTEX_COUNT)
    sampled_codes = candidate_codes[sampled_candidates]
    sampled_chart = candidate_chart[sampled_candidates]
    sampled_labels = digit_labels[candidate_indices][sampled_candidates]
    sampled_images = digit_images[candidate_indices][sampled_candidates]
    sampled_source_indices = digit_source_indices[candidate_indices][sampled_candidates]

    (
        retained_sample_indices,
        retained_chart,
        faces,
        minimum_angle,
        maximum_edge_ratio,
    ) = _filtered_delaunay_mesh(sampled_chart)
    latent_codes = sampled_codes[retained_sample_indices]
    source_labels = sampled_labels[retained_sample_indices]
    source_images = sampled_images[retained_sample_indices]
    retained_source_indices = sampled_source_indices[retained_sample_indices]
    edges = _unique_edges(faces)
    boundary = _boundary_vertices(len(latent_codes), faces)
    query_pairs = _graph_hop_pairs(len(latent_codes), edges, GRAPH_QUERY_HOPS)

    decoded_images = _decode_images(model, latent_codes)
    mass_images = _images_to_masses(decoded_images)
    query_distances = _exact_pixel_w2(mass_images, query_pairs)
    edge_distances = _edge_distances_from_queries(edges, query_pairs, query_distances)
    data = ActualPosteriorSurfaceData(
        cache_version=CACHE_VERSION,
        checkpoint_sha256=checkpoint_sha256,
        digit=MNIST_DIGIT,
        digit_reference_count=int(np.sum(digit_mask)),
        trimmed_candidate_count=len(candidate_indices),
        source_indices=retained_source_indices,
        source_labels=source_labels,
        source_images=source_images,
        latent_codes=latent_codes,
        chart_coordinates=retained_chart,
        whitening_center=whitening_center,
        whitening_matrix=whitening_matrix,
        faces=faces,
        edges=edges,
        boundary_vertices=boundary,
        decoded_images=decoded_images,
        mass_images=mass_images,
        query_pairs=query_pairs,
        query_distances=query_distances,
        edge_distances=edge_distances,
        minimum_chart_triangle_angle_degrees=minimum_angle,
        maximum_chart_edge_to_local_scale=maximum_edge_ratio,
    )
    _validate_surface_data(data)
    return data


def _write_cache(data: ActualPosteriorSurfaceData) -> None:
    CACHE_DIRECTORY.mkdir(parents=True, exist_ok=True)
    temporary_path = CACHE_PATH.with_name(CACHE_PATH.stem + ".tmp.npz")
    np.savez_compressed(
        temporary_path,
        cache_version=np.asarray(data.cache_version, dtype=np.int64),
        checkpoint_sha256=np.asarray(data.checkpoint_sha256),
        digit=np.asarray(data.digit, dtype=np.int64),
        digit_reference_count=np.asarray(data.digit_reference_count, dtype=np.int64),
        trimmed_candidate_count=np.asarray(
            data.trimmed_candidate_count, dtype=np.int64
        ),
        source_indices=data.source_indices,
        source_labels=data.source_labels,
        source_images=data.source_images,
        latent_codes=data.latent_codes,
        chart_coordinates=data.chart_coordinates,
        whitening_center=data.whitening_center,
        whitening_matrix=data.whitening_matrix,
        faces=data.faces,
        edges=data.edges,
        boundary_vertices=data.boundary_vertices,
        decoded_images=data.decoded_images,
        mass_images=data.mass_images,
        query_pairs=data.query_pairs,
        query_distances=data.query_distances,
        edge_distances=data.edge_distances,
        minimum_chart_triangle_angle_degrees=np.asarray(
            data.minimum_chart_triangle_angle_degrees, dtype=np.float64
        ),
        maximum_chart_edge_to_local_scale=np.asarray(
            data.maximum_chart_edge_to_local_scale, dtype=np.float64
        ),
    )
    os.replace(temporary_path, CACHE_PATH)
    print(f"saved actual-posterior surface cache: {CACHE_PATH}", flush=True)


def _read_cache() -> ActualPosteriorSurfaceData:
    with np.load(CACHE_PATH, allow_pickle=False) as payload:
        data = ActualPosteriorSurfaceData(
            cache_version=int(payload["cache_version"]),
            checkpoint_sha256=str(payload["checkpoint_sha256"]),
            digit=int(payload["digit"]),
            digit_reference_count=int(payload["digit_reference_count"]),
            trimmed_candidate_count=int(payload["trimmed_candidate_count"]),
            source_indices=payload["source_indices"].copy(),
            source_labels=payload["source_labels"].copy(),
            source_images=payload["source_images"].copy(),
            latent_codes=payload["latent_codes"].copy(),
            chart_coordinates=payload["chart_coordinates"].copy(),
            whitening_center=payload["whitening_center"].copy(),
            whitening_matrix=payload["whitening_matrix"].copy(),
            faces=payload["faces"].copy(),
            edges=payload["edges"].copy(),
            boundary_vertices=payload["boundary_vertices"].copy(),
            decoded_images=payload["decoded_images"].copy(),
            mass_images=payload["mass_images"].copy(),
            query_pairs=payload["query_pairs"].copy(),
            query_distances=payload["query_distances"].copy(),
            edge_distances=payload["edge_distances"].copy(),
            minimum_chart_triangle_angle_degrees=float(
                payload["minimum_chart_triangle_angle_degrees"]
            ),
            maximum_chart_edge_to_local_scale=float(
                payload["maximum_chart_edge_to_local_scale"]
            ),
        )
    _validate_surface_data(data)
    return data


def load_or_build_mnist_actual_posterior_surface(
    *, force_rebuild: bool = False
) -> ActualPosteriorSurfaceData:
    """Load the validated v1 cache, or build it once from the cached VAE."""

    if CACHE_PATH.exists() and not force_rebuild:
        data = _read_cache()
        checkpoint = DATASET_SPECS["mnist"].checkpoint
        if checkpoint.exists():
            current_digest = _checkpoint_digest(checkpoint)
            if current_digest != data.checkpoint_sha256:
                raise RuntimeError(
                    "the MNIST teacher checkpoint changed; increment CACHE_VERSION "
                    "or call with force_rebuild=True after reviewing the change"
                )
        print(f"loaded actual-posterior surface cache: {CACHE_PATH}", flush=True)
        return data

    data = _build_surface_data()
    _write_cache(data)
    return data


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--force-rebuild",
        action="store_true",
        help="recompute exact OT and replace only this versioned cache",
    )
    arguments = parser.parse_args()
    data = load_or_build_mnist_actual_posterior_surface(
        force_rebuild=arguments.force_rebuild
    )
    print(
        "actual-posterior surface: "
        f"vertices={data.vertex_count}, faces={len(data.faces)}, "
        f"edges={len(data.edges)}, two-hop queries={len(data.query_pairs)}, "
        f"min angle={data.minimum_chart_triangle_angle_degrees:.2f} deg, "
        f"max edge/local-scale={data.maximum_chart_edge_to_local_scale:.3f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
