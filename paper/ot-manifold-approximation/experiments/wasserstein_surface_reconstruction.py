#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.5.1",
#   "scipy==1.18.0",
# ]
# ///
"""Recover a sphere from connectivity and local Wasserstein travel times.

The controlled observation at a point ``x=(x1,x2,x3)`` on the unit sphere is
a full-rank two-dimensional Gaussian.  Its mean ``(x1,x2)`` is only a doubly
covered disk, while its 2-by-2 covariance follows an explicit noncommuting
Bures--Wasserstein geodesic parameterized by ``x3``.  Consequently, the
general Gaussian 2-Wasserstein formula gives exactly the three-dimensional
chord distance.  Samples are smoothly redistributed on the sphere, so the
uniform icosphere determined by topology is not the answer.
The proposed estimator is deliberately given *only* the triangulation and
noisy W2 chords for vertex pairs within three topology hops.  These O(n) local
queries remain sparse among the O(n^2) possible pairs.  Ground-truth
coordinates are used afterwards for alignment and evaluation.

The reconstruction pipeline is

    local W2 chords -> angle-deficit curvature -> radius from Gauss--Bonnet
                    -> chord-to-arc correction -> graph geodesics
                    -> round-sphere diameter calibration
                    -> spherical classical scaling.

The radius/correction/scaling stages deliberately impose the paper's
constant-curvature S^2 model class.  They are not a claim that an arbitrary
surface is determined by angle deficits alone.  Curvature error is reported
both against the noiseless discrete angle-deficit target and against the
smooth unit-sphere value K=1.

For comparison, the script also computes a topology-only uniform icosphere,
ordinary Euclidean MDS from the same graph distances, and the mean-only disk.
It runs several mesh resolutions, noise levels, and seeds; writes trial/summary
CSV files and EN/JA LaTeX tables; and renders bilingual paper figures.

Run with:
    uv run --python 3.12 wasserstein_surface_reconstruction.py

A quick smoke test is available with ``--quick``.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy import linalg
from scipy.optimize import minimize_scalar
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra
from scipy.spatial import cKDTree

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
MASTER_SEED = 20260731
GAUSSIAN_HEIGHT_RADIUS = 1.0
GAUSSIAN_GEODESIC_COUPLING = 0.5
LATITUDE_WARP_FACTOR = 1.35
LONGITUDE_TWIST_RADIANS = 0.24
LOCAL_QUERY_HOPS = 3
DEFAULT_SUBDIVISIONS = (0, 1, 2, 3)
DEFAULT_NOISE_LEVELS = (0.0, 0.005, 0.01, 0.02)
DEFAULT_SEEDS = (0, 1, 2)
SHOWCASE_NOISE = 0.01
EPSILON = 1.0e-12


@dataclass(frozen=True)
class Mesh:
    """A triangular mesh used to generate controlled observations."""

    vertices: np.ndarray
    uniform_vertices: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    query_pairs: np.ndarray


@dataclass(frozen=True)
class IntrinsicEstimate:
    """Quantities inferred without access to ground-truth coordinates."""

    curvature: np.ndarray
    vertex_areas: np.ndarray
    angle_deficits: np.ndarray
    graph_distances: np.ndarray
    spherical_coordinates: np.ndarray
    ordinary_mds_coordinates: np.ndarray
    learned_radius: float
    total_area: float
    euler_characteristic: int
    invalid_triangle_fraction: float
    clipped_chord_to_arc_fraction: float
    graph_diameter_calibration_factor: float
    clipped_spherical_pair_fraction: float


@dataclass(frozen=True)
class Trial:
    """One evaluated reconstruction and the arrays needed for a showcase."""

    metrics: dict[str, float]
    estimate: IntrinsicEstimate
    aligned_spherical: np.ndarray
    aligned_topology_uniform: np.ndarray
    aligned_ordinary_mds: np.ndarray
    aligned_mean_disk: np.ndarray
    noisy_query_lengths: np.ndarray


def japanese_font_family() -> str:
    """Return a usable Japanese font family when one is installed."""

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
        "warning: no Japanese font found; Japanese glyphs may be missing",
        flush=True,
    )
    return "sans-serif"


def normalize_rows(values: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(values, axis=1, keepdims=True)
    return values / np.maximum(norms, EPSILON)


def unique_edges(faces: np.ndarray) -> np.ndarray:
    """Return sorted, unique undirected edges of triangular faces."""

    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def local_query_pairs(
    vertex_count: int,
    edges: np.ndarray,
    maximum_hops: int = LOCAL_QUERY_HOPS,
) -> np.ndarray:
    """Return every unordered vertex pair within a fixed topology radius.

    The maximum degree of an icosphere is bounded.  Thus a fixed hop radius
    gives O(vertex_count) queries, rather than the quadratic complete graph.
    """

    if maximum_hops < 1:
        raise ValueError("maximum_hops must be positive")
    neighbours: list[set[int]] = [set() for _ in range(vertex_count)]
    for left, right in edges:
        neighbours[int(left)].add(int(right))
        neighbours[int(right)].add(int(left))

    pairs: list[tuple[int, int]] = []
    for source in range(vertex_count):
        visited = {source}
        frontier = {source}
        for _ in range(maximum_hops):
            next_frontier: set[int] = set()
            for vertex in frontier:
                next_frontier.update(neighbours[vertex])
            next_frontier.difference_update(visited)
            visited.update(next_frontier)
            frontier = next_frontier
        pairs.extend((source, target) for target in sorted(visited) if source < target)

    query_pairs = np.asarray(pairs, dtype=np.int64)
    query_set = {tuple(pair) for pair in query_pairs.tolist()}
    if any(tuple(edge) not in query_set for edge in edges.tolist()):
        raise RuntimeError("triangulation edges must be a subset of local queries")
    return query_pairs


def warp_sphere_samples(uniform_sphere: np.ndarray) -> np.ndarray:
    """Smoothly redistribute samples while keeping them on the unit sphere.

    The latitude map has derivative ``LATITUDE_WARP_FACTOR`` at the equator
    and remains strictly monotone up to the poles.  A z-dependent axial
    rotation adds a mild longitude twist.  Both are deterministic diffeomorphisms
    of S^2; only sampling locations change, not the underlying round geometry.
    """

    longitude = np.arctan2(uniform_sphere[:, 1], uniform_sphere[:, 0])
    latitude = np.arcsin(np.clip(uniform_sphere[:, 2], -1.0, 1.0))
    latitude_amplitude = 0.5 * (LATITUDE_WARP_FACTOR - 1.0)
    warped_latitude = latitude + latitude_amplitude * np.sin(2.0 * latitude)
    warped_height = np.sin(warped_latitude)
    # An even function of height preserves the mesh's antipodal sample pairs,
    # while still twisting latitude bands relative to one another.
    warped_longitude = longitude + LONGITUDE_TWIST_RADIANS * warped_height**2
    radial = np.cos(warped_latitude)
    warped = np.column_stack(
        [
            radial * np.cos(warped_longitude),
            radial * np.sin(warped_longitude),
            warped_height,
        ]
    )
    return normalize_rows(warped)


def validate_spherical_mesh(vertices: np.ndarray, faces: np.ndarray) -> None:
    """Reject degenerate or orientation-flipped warped triangles."""

    norm_error = float(np.max(np.abs(np.linalg.norm(vertices, axis=1) - 1.0)))
    if norm_error > 1.0e-12:
        raise RuntimeError(f"warped vertices left the unit sphere: {norm_error:.3e}")
    first = vertices[faces[:, 0]]
    second = vertices[faces[:, 1]]
    third = vertices[faces[:, 2]]
    normals = np.cross(second - first, third - first)
    signed_orientation = np.einsum("ij,ij->i", normals, first + second + third)
    if np.any(signed_orientation <= 1.0e-12):
        raise RuntimeError("latitude/longitude warp flipped or collapsed a face")


def base_icosahedron() -> tuple[np.ndarray, np.ndarray]:
    """Construct a consistently indexed unit icosahedron."""

    phi = (1.0 + math.sqrt(5.0)) / 2.0
    vertices = np.asarray(
        [
            (-1, phi, 0),
            (1, phi, 0),
            (-1, -phi, 0),
            (1, -phi, 0),
            (0, -1, phi),
            (0, 1, phi),
            (0, -1, -phi),
            (0, 1, -phi),
            (phi, 0, -1),
            (phi, 0, 1),
            (-phi, 0, -1),
            (-phi, 0, 1),
        ],
        dtype=np.float64,
    )
    vertices = normalize_rows(vertices)
    faces = np.asarray(
        [
            (0, 11, 5),
            (0, 5, 1),
            (0, 1, 7),
            (0, 7, 10),
            (0, 10, 11),
            (1, 5, 9),
            (5, 11, 4),
            (11, 10, 2),
            (10, 7, 6),
            (7, 1, 8),
            (3, 9, 4),
            (3, 4, 2),
            (3, 2, 6),
            (3, 6, 8),
            (3, 8, 9),
            (4, 9, 5),
            (2, 4, 11),
            (6, 2, 10),
            (8, 6, 7),
            (9, 8, 1),
        ],
        dtype=np.int64,
    )
    return vertices, faces


def subdivide_icosphere(
    vertices: np.ndarray, faces: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Split every face into four triangles and project midpoints to S^2."""

    vertex_list = [vertex.copy() for vertex in vertices]
    midpoint_cache: dict[tuple[int, int], int] = {}

    def midpoint(left: int, right: int) -> int:
        key = (min(left, right), max(left, right))
        cached = midpoint_cache.get(key)
        if cached is not None:
            return cached
        point = vertices[left] + vertices[right]
        point /= np.linalg.norm(point)
        index = len(vertex_list)
        vertex_list.append(point)
        midpoint_cache[key] = index
        return index

    refined_faces: list[tuple[int, int, int]] = []
    for first, second, third in faces:
        first_second = midpoint(int(first), int(second))
        second_third = midpoint(int(second), int(third))
        third_first = midpoint(int(third), int(first))
        refined_faces.extend(
            [
                (int(first), first_second, third_first),
                (int(second), second_third, first_second),
                (int(third), third_first, second_third),
                (first_second, second_third, third_first),
            ]
        )
    return np.asarray(vertex_list), np.asarray(refined_faces, dtype=np.int64)


def icosphere(subdivisions: int) -> Mesh:
    """Return uniform topology and smoothly redistributed sphere samples."""

    if subdivisions < 0:
        raise ValueError("subdivisions must be nonnegative")
    uniform_vertices, faces = base_icosahedron()
    for _ in range(subdivisions):
        uniform_vertices, faces = subdivide_icosphere(uniform_vertices, faces)
    edges = unique_edges(faces)
    expected_vertices = 10 * (4**subdivisions) + 2
    if len(uniform_vertices) != expected_vertices:
        raise RuntimeError(
            f"icosphere indexing failed: {len(uniform_vertices)} != {expected_vertices}"
        )
    vertices = warp_sphere_samples(uniform_vertices)
    validate_spherical_mesh(vertices, faces)
    query_pairs = local_query_pairs(len(vertices), edges)
    return Mesh(
        vertices=vertices,
        uniform_vertices=uniform_vertices,
        faces=faces,
        edges=edges,
        query_pairs=query_pairs,
    )


def noncommuting_gaussian_observations(
    surface: np.ndarray,
    height_radius: float = GAUSSIAN_HEIGHT_RADIUS,
    coupling: float = GAUSSIAN_GEODESIC_COUPLING,
) -> tuple[np.ndarray, np.ndarray]:
    """Return disk means and full-rank covariances carrying the third coordinate.

    For ``D=diag(1,2)``, ``H=coupling * [[0,1],[1,0]]``, and
    ``A_t=I+tH``, the covariance path ``scale * A_t D A_t`` is a
    constant-speed Bures geodesic.  The chosen scale makes covariance-sector
    W2 distance equal the difference of the input third coordinates.
    """

    if surface.ndim != 2 or surface.shape[1] < 3:
        raise ValueError("surface coordinates must have shape (n, d) with d >= 3")
    if not np.isfinite(surface).all():
        raise ValueError("surface coordinates must be finite")
    if height_radius <= 0.0:
        raise ValueError("height_radius must be positive")
    if not 0.0 < coupling < 1.0:
        raise ValueError("coupling must lie strictly between zero and one")
    heights = surface[:, 2]
    if np.max(np.abs(heights), initial=0.0) > height_radius + 64.0 * EPSILON:
        raise ValueError("third coordinates must lie inside the height range")

    times = (heights + height_radius) / (2.0 * height_radius)
    base_covariance = np.diag(np.asarray([1.0, 2.0], dtype=np.float64))
    generator = coupling * np.asarray([[0.0, 1.0], [1.0, 0.0]])
    transports = np.eye(2)[None, :, :] + times[:, None, None] * generator
    scale = 4.0 * height_radius**2 / (3.0 * coupling**2)
    covariances = scale * (
        transports @ base_covariance[None, :, :] @ transports.swapaxes(-1, -2)
    )
    means = surface[:, :2].copy()
    return means, covariances


def gaussian_w2_edges(
    mean: np.ndarray,
    covariance: np.ndarray,
    pairs: np.ndarray,
) -> np.ndarray:
    """Evaluate the general 2-by-2 Gaussian Bures/W2 formula.

    If ``C=L L^T`` and ``D=R R^T``, the Bures term is the orthogonal
    Procrustes value ``min_Q ||L-RQ||_F``.  Computing the minimizing residual
    directly is equivalent to the usual matrix-square-root formula and avoids
    cancellation when the covariances are close.  No hidden chord coordinate
    enters this evaluation.
    """

    if mean.ndim != 2 or mean.shape[1] != 2:
        raise ValueError("the observation means must have shape (n, 2)")
    if covariance.shape != (len(mean), 2, 2):
        raise ValueError("the observation covariances must have shape (n, 2, 2)")
    left = np.asarray(covariance[pairs[:, 0]], dtype=np.float64)
    right = np.asarray(covariance[pairs[:, 1]], dtype=np.float64)
    left = 0.5 * (left + left.swapaxes(-1, -2))
    right = 0.5 * (right + right.swapaxes(-1, -2))
    try:
        left_factor = np.linalg.cholesky(left)
        right_factor = np.linalg.cholesky(right)
    except np.linalg.LinAlgError as error:
        raise ValueError("observation covariance must be positive definite") from error
    cross_factor = left_factor.swapaxes(-1, -2) @ right_factor
    left_singular_vectors, _, right_singular_vectors_transpose = np.linalg.svd(
        cross_factor
    )
    alignment = right_singular_vectors_transpose.swapaxes(
        -1, -2
    ) @ left_singular_vectors.swapaxes(-1, -2)
    covariance_residual = left_factor - right_factor @ alignment
    covariance_cost = np.square(covariance_residual).sum(axis=(1, 2))
    mean_difference = mean[pairs[:, 0]] - mean[pairs[:, 1]]
    squared_w2 = np.square(mean_difference).sum(axis=1) + covariance_cost
    return np.sqrt(np.maximum(squared_w2, 0.0))


def oracle_scaled_raw_parameter_distortion(
    mean: np.ndarray,
    covariance: np.ndarray,
    pairs: np.ndarray,
    target_distances: np.ndarray,
) -> tuple[float, float]:
    """Audit the best globally scaled raw covariance-parameter baseline.

    The parameter distance is
    ``sqrt(||delta mean||^2 + alpha^2 ||delta covariance||_F^2)``.  The oracle
    chooses the nonnegative ``alpha^2`` that minimizes L2 error against the
    target unsquared distances.
    """

    mean_difference = mean[pairs[:, 0]] - mean[pairs[:, 1]]
    covariance_difference = covariance[pairs[:, 0]] - covariance[pairs[:, 1]]
    mean_squared = np.square(mean_difference).sum(axis=1)
    covariance_frobenius_squared = np.square(covariance_difference).sum(axis=(1, 2))
    missing_squared = np.maximum(np.square(target_distances) - mean_squared, 0.0)
    positive = covariance_frobenius_squared > EPSILON
    if not np.any(positive):
        return 0.0, math.inf
    pointwise_scales = (
        missing_squared[positive] / covariance_frobenius_squared[positive]
    )
    upper_bound = 1.01 * float(np.max(pointwise_scales, initial=0.0))
    if upper_bound <= EPSILON:
        scale_squared = 0.0
    else:
        result = minimize_scalar(
            lambda candidate: float(
                np.square(
                    np.sqrt(mean_squared + candidate * covariance_frobenius_squared)
                    - target_distances
                ).sum()
            ),
            bounds=(0.0, upper_bound),
            method="bounded",
            options={"xatol": 1.0e-15},
        )
        if not result.success:
            raise RuntimeError("oracle raw-parameter scale optimization failed")
        scale_squared = float(result.x)
    parameter_distances = np.sqrt(
        mean_squared + scale_squared * covariance_frobenius_squared
    )
    relative_distortion = float(
        np.linalg.norm(parameter_distances - target_distances)
        / np.linalg.norm(target_distances)
    )
    return math.sqrt(scale_squared), relative_distortion


def noncommuting_pair_fraction(covariance: np.ndarray, pairs: np.ndarray) -> float:
    """Return the fraction of queried covariance pairs with nonzero commutator."""

    left = covariance[pairs[:, 0]]
    right = covariance[pairs[:, 1]]
    commutators = left @ right - right @ left
    commutator_norm = np.linalg.norm(commutators, axis=(1, 2))
    relative_scale = np.linalg.norm(left, axis=(1, 2)) * np.linalg.norm(
        right, axis=(1, 2)
    )
    threshold = 1024.0 * np.finfo(np.float64).eps * np.maximum(1.0, relative_scale)
    return float(np.mean(commutator_norm > threshold))


def noisy_lengths(
    exact_lengths: np.ndarray,
    relative_noise: float,
    seed: int,
) -> np.ndarray:
    """Apply positive, mean-one multiplicative observation noise."""

    if relative_noise < 0.0:
        raise ValueError("relative_noise must be nonnegative")
    if relative_noise == 0.0:
        return exact_lengths.copy()
    rng = np.random.default_rng(seed)
    perturbation = np.exp(
        relative_noise * rng.standard_normal(len(exact_lengths))
        - 0.5 * relative_noise**2
    )
    return exact_lengths * perturbation


def subset_pair_lengths(
    query_pairs: np.ndarray,
    query_lengths: np.ndarray,
    subset_pairs: np.ndarray,
) -> np.ndarray:
    """Select measured lengths for a pair subset without re-observation."""

    lookup = {
        (int(left), int(right)): float(length)
        for (left, right), length in zip(query_pairs, query_lengths, strict=True)
    }
    try:
        return np.asarray(
            [lookup[(int(left), int(right))] for left, right in subset_pairs],
            dtype=np.float64,
        )
    except KeyError as error:
        raise ValueError("every triangulation edge must be locally queried") from error


def spherical_chord_to_arc(
    chord_lengths: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Apply the constant-curvature S^2 chord-to-arc correction."""

    ratios = chord_lengths / max(2.0 * radius, EPSILON)
    clipped_fraction = float(np.mean(ratios > 1.0))
    angular_half_lengths = np.arcsin(np.clip(ratios, 0.0, 1.0))
    return 2.0 * radius * angular_half_lengths, clipped_fraction


def edge_length_lookup(
    vertex_count: int,
    edges: np.ndarray,
    lengths: np.ndarray,
) -> coo_matrix:
    """Store symmetric edge lengths in sparse matrix form."""

    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([lengths, lengths])
    return coo_matrix((values, (rows, columns)), shape=(vertex_count, vertex_count))


def triangle_geometry_from_lengths(
    vertex_count: int,
    faces: np.ndarray,
    edges: np.ndarray,
    edge_lengths: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    """Compute angle deficits and barycentric areas from edge lengths only.

    A rare noise realization can violate a triangle inequality.  For the local
    curvature diagnostic only, the longest side of such a face is projected
    infinitesimally inside the valid cone.  Graph distances continue to use the
    unmodified measurements, and the affected fraction is reported.
    """

    lookup = {
        (int(left), int(right)): float(length)
        for (left, right), length in zip(edges, edge_lengths, strict=True)
    }

    def length(left: int, right: int) -> float:
        key = (min(left, right), max(left, right))
        return lookup[key]

    angle_sums = np.zeros(vertex_count, dtype=np.float64)
    vertex_areas = np.zeros(vertex_count, dtype=np.float64)
    face_areas = np.zeros(len(faces), dtype=np.float64)
    invalid_count = 0

    for face_index, (first, second, third) in enumerate(faces):
        # opposite[i] is the side opposite local vertex i.
        opposite = np.asarray(
            [
                length(int(second), int(third)),
                length(int(third), int(first)),
                length(int(first), int(second)),
            ],
            dtype=np.float64,
        )
        longest = int(np.argmax(opposite))
        other_sum = float(opposite.sum() - opposite[longest])
        if opposite[longest] >= other_sum:
            invalid_count += 1
            opposite[longest] = other_sum * (1.0 - 1.0e-10)

        angles = np.empty(3, dtype=np.float64)
        for local_vertex in range(3):
            side_opposite = opposite[local_vertex]
            side_left = opposite[(local_vertex + 1) % 3]
            side_right = opposite[(local_vertex + 2) % 3]
            cosine = (side_left**2 + side_right**2 - side_opposite**2) / max(
                2.0 * side_left * side_right, EPSILON
            )
            angles[local_vertex] = math.acos(float(np.clip(cosine, -1.0, 1.0)))

        semiperimeter = 0.5 * float(opposite.sum())
        area_squared = semiperimeter
        for side in opposite:
            area_squared *= max(semiperimeter - float(side), 0.0)
        area = math.sqrt(max(area_squared, EPSILON**2))
        face_areas[face_index] = area
        for vertex, angle in zip(
            (int(first), int(second), int(third)), angles, strict=True
        ):
            angle_sums[vertex] += angle
            vertex_areas[vertex] += area / 3.0

    angle_deficits = 2.0 * math.pi - angle_sums
    curvature = angle_deficits / np.maximum(vertex_areas, EPSILON)
    return (
        curvature,
        vertex_areas,
        angle_deficits,
        invalid_count / len(faces),
    )


def top_eigen_embedding(matrix: np.ndarray, dimensions: int) -> np.ndarray:
    """Embed a symmetric Gram matrix using its leading positive eigenpairs."""

    symmetric = 0.5 * (matrix + matrix.T)
    vertex_count = len(symmetric)
    first = max(0, vertex_count - dimensions)
    eigenvalues, eigenvectors = linalg.eigh(
        symmetric,
        subset_by_index=(first, vertex_count - 1),
        check_finite=False,
        driver="evr",
    )
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = np.maximum(eigenvalues[order], 0.0)
    eigenvectors = eigenvectors[:, order]
    return eigenvectors * np.sqrt(eigenvalues)[None, :]


def center_gram(matrix: np.ndarray) -> np.ndarray:
    """Apply H matrix H without constructing a dense centering matrix."""

    row_mean = matrix.mean(axis=1, keepdims=True)
    column_mean = matrix.mean(axis=0, keepdims=True)
    return matrix - row_mean - column_mean + float(matrix.mean())


def ordinary_classical_mds(distances: np.ndarray) -> np.ndarray:
    gram = -0.5 * center_gram(np.square(distances))
    return top_eigen_embedding(gram, dimensions=3)


def spherical_classical_mds(
    distances: np.ndarray,
    radius: float,
) -> tuple[np.ndarray, float]:
    """Recover points using <x_i,x_j>=r^2 cos(d_ij/r)."""

    angular_distance = distances / max(radius, EPSILON)
    off_diagonal = ~np.eye(len(distances), dtype=bool)
    clipped_fraction = float(np.mean(angular_distance[off_diagonal] > math.pi))
    angular_distance = np.clip(angular_distance, 0.0, math.pi)
    spherical_gram = radius**2 * np.cos(angular_distance)
    # Unlike Euclidean MDS, spherical scaling does not center this Gram matrix:
    # the origin is the sphere center, and samples are intentionally nonuniform.
    coordinates = top_eigen_embedding(spherical_gram, dimensions=3)
    coordinates = radius * normalize_rows(coordinates)
    return coordinates, clipped_fraction


def infer_from_local_lengths(
    vertex_count: int,
    faces: np.ndarray,
    query_pairs: np.ndarray,
    query_chord_lengths: np.ndarray,
) -> IntrinsicEstimate:
    """Infer geometry using only topology and sparse local measured chords.

    Keeping this function's interface free of true coordinates and Gaussian
    parameters is an explicit guard against evaluation leakage.
    """

    if len(query_chord_lengths) != len(query_pairs):
        raise ValueError("one positive chord is required for every local query")
    if not np.isfinite(query_chord_lengths).all() or np.any(query_chord_lengths <= 0.0):
        raise ValueError("all measured query chords must be finite and positive")

    triangulation_edges = unique_edges(faces)
    triangle_edge_chords = subset_pair_lengths(
        query_pairs, query_chord_lengths, triangulation_edges
    )

    curvature, vertex_areas, deficits, invalid_fraction = (
        triangle_geometry_from_lengths(
            vertex_count, faces, triangulation_edges, triangle_edge_chords
        )
    )
    euler_characteristic = vertex_count - len(triangulation_edges) + len(faces)
    total_curvature = 2.0 * math.pi * euler_characteristic
    if euler_characteristic != 2:
        raise ValueError(
            "the spherical reconstruction stage explicitly assumes S^2 topology "
            "(Euler characteristic two)"
        )
    total_area = float(vertex_areas.sum())
    learned_radius = math.sqrt(total_area / total_curvature)

    query_arc_lengths, chord_clipped_fraction = spherical_chord_to_arc(
        query_chord_lengths, learned_radius
    )
    graph = edge_length_lookup(vertex_count, query_pairs, query_arc_lengths).tocsr()
    graph_distances = np.asarray(
        dijkstra(graph, directed=False, return_predecessors=False)
    )
    if not np.isfinite(graph_distances).all():
        raise RuntimeError("the local-distance graph is disconnected")
    # A round sphere of learned radius R has diameter pi R.  The warped mesh
    # retains antipodal samples, so this topology/constant-curvature calibration
    # removes the fixed-direction stretch of finite-hop graph shortest paths.
    raw_graph_diameter = float(graph_distances.max())
    diameter_calibration = math.pi * learned_radius / max(raw_graph_diameter, EPSILON)
    graph_distances *= diameter_calibration

    spherical, clipped_fraction = spherical_classical_mds(
        graph_distances, learned_radius
    )
    ordinary = ordinary_classical_mds(graph_distances)
    return IntrinsicEstimate(
        curvature=curvature,
        vertex_areas=vertex_areas,
        angle_deficits=deficits,
        graph_distances=graph_distances,
        spherical_coordinates=spherical,
        ordinary_mds_coordinates=ordinary,
        learned_radius=learned_radius,
        total_area=total_area,
        euler_characteristic=euler_characteristic,
        invalid_triangle_fraction=invalid_fraction,
        clipped_chord_to_arc_fraction=chord_clipped_fraction,
        graph_diameter_calibration_factor=diameter_calibration,
        clipped_spherical_pair_fraction=clipped_fraction,
    )


def orthogonal_alignment(
    source: np.ndarray,
    target: np.ndarray,
) -> tuple[np.ndarray, float]:
    """Align by translation and an orthogonal map, never by scaling."""

    source_center = source.mean(axis=0)
    target_center = target.mean(axis=0)
    source_centered = source - source_center
    target_centered = target - target_center
    left, _, right_transpose = np.linalg.svd(
        source_centered.T @ target_centered, full_matrices=False
    )
    rotation = left @ right_transpose
    aligned = source_centered @ rotation + target_center
    rmse = math.sqrt(float(np.square(aligned - target).sum(axis=1).mean()))
    return aligned, rmse


def symmetric_hausdorff(source: np.ndarray, target: np.ndarray) -> float:
    source_to_target = cKDTree(target).query(source, k=1)[0]
    target_to_source = cKDTree(source).query(target, k=1)[0]
    return float(max(source_to_target.max(), target_to_source.max()))


def symmetric_chamfer_rms(source: np.ndarray, target: np.ndarray) -> float:
    source_to_target = cKDTree(target).query(source, k=1)[0]
    target_to_source = cKDTree(source).query(target, k=1)[0]
    return math.sqrt(
        0.5
        * float(np.square(source_to_target).mean() + np.square(target_to_source).mean())
    )


def true_great_circle_distances(sphere: np.ndarray) -> np.ndarray:
    cosine = np.clip(sphere @ sphere.T, -1.0, 1.0)
    return np.arccos(cosine)


def relative_frobenius_error(
    estimate: np.ndarray,
    truth: np.ndarray,
) -> float:
    return float(np.linalg.norm(estimate - truth) / np.linalg.norm(truth))


def weighted_smooth_curvature_rmse(estimate: IntrinsicEstimate) -> float:
    """Error to smooth K=1, including the mesh discretization error."""

    weights = estimate.vertex_areas / estimate.total_area
    return math.sqrt(float(np.sum(weights * np.square(estimate.curvature - 1.0))))


def weighted_oracle_curvature_rmse(
    estimate: IntrinsicEstimate,
    oracle_curvature: np.ndarray,
    oracle_areas: np.ndarray,
) -> float:
    """Noise error relative to the same mesh with exact W2 edge lengths."""

    weights = oracle_areas / oracle_areas.sum()
    return math.sqrt(
        float(np.sum(weights * np.square(estimate.curvature - oracle_curvature)))
    )


def run_trial(
    mesh: Mesh,
    subdivision: int,
    noise_level: float,
    seed_index: int,
) -> Trial:
    """Generate observations, infer from local W2, then evaluate."""

    mean, covariance = noncommuting_gaussian_observations(mesh.vertices)
    exact_w2 = gaussian_w2_edges(mean, covariance, mesh.query_pairs)
    exact_chords = np.linalg.norm(
        mesh.vertices[mesh.query_pairs[:, 0]] - mesh.vertices[mesh.query_pairs[:, 1]],
        axis=1,
    )
    w2_identity_error = float(np.max(np.abs(exact_w2 - exact_chords)))
    if w2_identity_error > 2.0e-12:
        raise RuntimeError(
            f"Gaussian W2/chord identity failed: {w2_identity_error:.3e}"
        )
    parameter_scale, parameter_distortion = oracle_scaled_raw_parameter_distortion(
        mean, covariance, mesh.query_pairs, exact_chords
    )

    # This noiseless, same-triangulation quantity is an evaluation target only.
    # It never enters infer_from_local_lengths or the reconstructed coordinates.
    exact_triangle_edge_w2 = subset_pair_lengths(mesh.query_pairs, exact_w2, mesh.edges)
    oracle_curvature, oracle_areas, _, _ = triangle_geometry_from_lengths(
        len(mesh.vertices), mesh.faces, mesh.edges, exact_triangle_edge_w2
    )

    noise_seed = (
        MASTER_SEED
        + 1_000_003 * subdivision
        + 101 * round(1_000_000 * noise_level)
        + seed_index
    )
    measured_query_w2 = noisy_lengths(exact_w2, noise_level, noise_seed)
    measured_triangle_edge_w2 = subset_pair_lengths(
        mesh.query_pairs, measured_query_w2, mesh.edges
    )

    # The estimator sees no mesh.vertices, uniform_vertices, Gaussian
    # parameters, or target distances.
    estimate = infer_from_local_lengths(
        len(mesh.vertices), mesh.faces, mesh.query_pairs, measured_query_w2
    )

    aligned_spherical, spherical_rmse = orthogonal_alignment(
        estimate.spherical_coordinates, mesh.vertices
    )
    aligned_topology, topology_rmse = orthogonal_alignment(
        mesh.uniform_vertices, mesh.vertices
    )
    aligned_ordinary, ordinary_rmse = orthogonal_alignment(
        estimate.ordinary_mds_coordinates, mesh.vertices
    )
    aligned_mean, mean_rmse = orthogonal_alignment(mean, mesh.vertices)
    true_distances = true_great_circle_distances(mesh.vertices)
    possible_pairs = len(mesh.vertices) * (len(mesh.vertices) - 1) / 2
    reconstructed_query_chords = np.linalg.norm(
        estimate.spherical_coordinates[mesh.query_pairs[:, 0]]
        - estimate.spherical_coordinates[mesh.query_pairs[:, 1]],
        axis=1,
    )
    reconstructed_edge_chords = subset_pair_lengths(
        mesh.query_pairs, reconstructed_query_chords, mesh.edges
    )

    metrics = {
        "subdivision": float(subdivision),
        "vertices": float(len(mesh.vertices)),
        "faces": float(len(mesh.faces)),
        "edges": float(len(mesh.edges)),
        "latitude_warp_factor": LATITUDE_WARP_FACTOR,
        "longitude_twist_radians": LONGITUDE_TWIST_RADIANS,
        "local_query_hops": float(LOCAL_QUERY_HOPS),
        "local_query_pairs": float(len(mesh.query_pairs)),
        "local_query_pair_fraction": float(len(mesh.query_pairs) / possible_pairs),
        "local_queries_per_vertex": float(
            2.0 * len(mesh.query_pairs) / len(mesh.vertices)
        ),
        "relative_edge_noise": float(noise_level),
        "seed": float(seed_index),
        "w2_chord_max_abs_error": w2_identity_error,
        "gaussian_covariance_minimum_eigenvalue": float(
            np.linalg.eigvalsh(covariance).min()
        ),
        "noncommuting_covariance_query_fraction": noncommuting_pair_fraction(
            covariance, mesh.query_pairs
        ),
        "oracle_covariance_parameter_scale": parameter_scale,
        "oracle_scaled_raw_parameter_distance_relative_rmse": parameter_distortion,
        "query_chord_relative_rmse": float(
            np.linalg.norm(measured_query_w2 - exact_w2) / np.linalg.norm(exact_w2)
        ),
        "triangle_edge_relative_rmse": float(
            np.linalg.norm(measured_triangle_edge_w2 - exact_triangle_edge_w2)
            / np.linalg.norm(exact_triangle_edge_w2)
        ),
        "euler_characteristic": float(estimate.euler_characteristic),
        "integrated_curvature": float(estimate.angle_deficits.sum()),
        "total_area": estimate.total_area,
        "learned_radius": estimate.learned_radius,
        "radius_absolute_error": abs(estimate.learned_radius - 1.0),
        "curvature_oracle_weighted_rmse": weighted_oracle_curvature_rmse(
            estimate, oracle_curvature, oracle_areas
        ),
        "curvature_smooth_k1_weighted_rmse": (weighted_smooth_curvature_rmse(estimate)),
        "nonpositive_angle_defect_fraction": float(
            np.mean(estimate.angle_deficits <= 0.0)
        ),
        "graph_geodesic_relative_error": relative_frobenius_error(
            estimate.graph_distances, true_distances
        ),
        "spherical_reconstruction_rmse": spherical_rmse,
        "spherical_reconstruction_hausdorff": symmetric_hausdorff(
            aligned_spherical, mesh.vertices
        ),
        "spherical_reconstruction_chamfer_rms": symmetric_chamfer_rms(
            aligned_spherical, mesh.vertices
        ),
        "reconstruction_query_chord_relative_stress": float(
            np.linalg.norm(reconstructed_query_chords - measured_query_w2)
            / np.linalg.norm(measured_query_w2)
        ),
        "reconstruction_edge_chord_relative_stress": float(
            np.linalg.norm(reconstructed_edge_chords - measured_triangle_edge_w2)
            / np.linalg.norm(measured_triangle_edge_w2)
        ),
        "ordinary_mds_rmse": ordinary_rmse,
        "ordinary_mds_hausdorff": symmetric_hausdorff(aligned_ordinary, mesh.vertices),
        "topology_uniform_rmse": topology_rmse,
        "topology_uniform_hausdorff": symmetric_hausdorff(
            aligned_topology, mesh.vertices
        ),
        "mean_disk_rmse": mean_rmse,
        "mean_disk_hausdorff": symmetric_hausdorff(aligned_mean, mesh.vertices),
        "invalid_triangle_fraction": estimate.invalid_triangle_fraction,
        "clipped_chord_to_arc_fraction": (estimate.clipped_chord_to_arc_fraction),
        "graph_diameter_calibration_factor": (
            estimate.graph_diameter_calibration_factor
        ),
        "clipped_spherical_pair_fraction": (estimate.clipped_spherical_pair_fraction),
    }
    return Trial(
        metrics=metrics,
        estimate=estimate,
        aligned_spherical=aligned_spherical,
        aligned_topology_uniform=aligned_topology,
        aligned_ordinary_mds=aligned_ordinary,
        aligned_mean_disk=aligned_mean,
        noisy_query_lengths=measured_query_w2,
    )


def aggregate_rows(rows: Sequence[dict[str, float]]) -> list[dict[str, float]]:
    """Aggregate numeric trial columns by resolution and noise level."""

    groups: dict[tuple[int, float], list[dict[str, float]]] = {}
    for row in rows:
        key = (int(row["subdivision"]), row["relative_edge_noise"])
        groups.setdefault(key, []).append(row)

    identifier_columns = {
        "subdivision",
        "vertices",
        "faces",
        "edges",
        "latitude_warp_factor",
        "longitude_twist_radians",
        "local_query_hops",
        "local_query_pairs",
        "local_query_pair_fraction",
        "local_queries_per_vertex",
        "relative_edge_noise",
        "seed",
    }
    metric_columns = [key for key in rows[0] if key not in identifier_columns]
    summary: list[dict[str, float]] = []
    for (subdivision, noise), group in sorted(groups.items()):
        output: dict[str, float] = {
            "subdivision": float(subdivision),
            "vertices": group[0]["vertices"],
            "faces": group[0]["faces"],
            "edges": group[0]["edges"],
            "latitude_warp_factor": group[0]["latitude_warp_factor"],
            "longitude_twist_radians": group[0]["longitude_twist_radians"],
            "local_query_hops": group[0]["local_query_hops"],
            "local_query_pairs": group[0]["local_query_pairs"],
            "local_query_pair_fraction": group[0]["local_query_pair_fraction"],
            "local_queries_per_vertex": group[0]["local_queries_per_vertex"],
            "relative_edge_noise": noise,
            "trials": float(len(group)),
        }
        for column in metric_columns:
            values = np.asarray([row[column] for row in group], dtype=np.float64)
            output[f"{column}_mean"] = float(values.mean())
            output[f"{column}_std"] = float(values.std(ddof=0))
        summary.append(output)
    return summary


def write_dict_csv(path: Path, rows: Sequence[dict[str, float]]) -> None:
    if not rows:
        raise ValueError("cannot write an empty CSV")
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def latex_noise(noise: float) -> str:
    return f"{100.0 * noise:.1f}\\%"


def write_latex_table(
    path: Path,
    summary: Sequence[dict[str, float]],
    finest_subdivision: int,
    japanese: bool,
) -> None:
    selected = [row for row in summary if int(row["subdivision"]) == finest_subdivision]
    if japanese:
        header = (
            "ノイズ & 測地誤差 & 曲率RMSE（PL） & 非正欠損率 "
            "& 球面3D RMSE & 位相のみ & 通常MDS & 平均のみ \\\\"
        )
        caption = (
            "疎な3-hop局所Wasserstein弦長と定曲率 $S^2$ の弦--弧・直径補正"
            "による球面復元（最細メッシュ、平均 $\\pm$ 標準偏差）。"
        )
        label = "tab:wasserstein-surface-reconstruction-ja"
    else:
        header = (
            "Noise & Geodesic error & Curvature RMSE (vs PL) "
            "& Nonpositive defects & Spherical 3D RMSE "
            "& Topology only & Euclidean MDS & Mean only \\\\"
        )
        caption = (
            "Sphere recovery from sparse three-hop local Wasserstein chords with "
            "constant-curvature $S^2$ chord--arc and diameter corrections "
            "on the finest mesh ($\\mathrm{mean}\\pm\\mathrm{std}$)."
        )
        label = "tab:wasserstein-surface-reconstruction"

    def value(row: dict[str, float], key: str) -> str:
        precision = (
            4
            if key
            in {
                "graph_geodesic_relative_error",
                "spherical_reconstruction_rmse",
                "topology_uniform_rmse",
                "ordinary_mds_rmse",
                "mean_disk_rmse",
            }
            else 3
        )
        return (
            f"{row[key + '_mean']:.{precision}f} "
            f"$\\pm$ {row[key + '_std']:.{precision}f}"
        )

    lines = [
        "\\begin{table*}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{0.98\\textwidth}{!}{%",
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in selected:
        lines.append(
            " & ".join(
                [
                    latex_noise(row["relative_edge_noise"]),
                    value(row, "graph_geodesic_relative_error"),
                    value(row, "curvature_oracle_weighted_rmse"),
                    value(row, "nonpositive_angle_defect_fraction"),
                    value(row, "spherical_reconstruction_rmse"),
                    value(row, "topology_uniform_rmse"),
                    value(row, "ordinary_mds_rmse"),
                    value(row, "mean_disk_rmse"),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", "}%", "\\end{table*}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def set_equal_3d_axes(axis: plt.Axes, limit: float = 1.08) -> None:
    axis.set_xlim(-limit, limit)
    axis.set_ylim(-limit, limit)
    axis.set_zlim(-limit, limit)
    axis.set_box_aspect((1.0, 1.0, 1.0))
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])


def add_mesh_surface(
    axis: plt.Axes,
    coordinates: np.ndarray,
    faces: np.ndarray,
    vertex_values: np.ndarray,
    cmap: str = "coolwarm",
    alpha: float = 0.9,
    edgecolor: str = "white",
    linewidth: float = 0.08,
) -> Poly3DCollection:
    face_values = vertex_values[faces].mean(axis=1)
    minimum = float(vertex_values.min())
    maximum = float(vertex_values.max())
    if math.isclose(minimum, maximum, abs_tol=EPSILON):
        maximum = minimum + 1.0
    normalization = plt.Normalize(vmin=minimum, vmax=maximum)
    color_map = plt.get_cmap(cmap)
    collection = Poly3DCollection(
        coordinates[faces],
        facecolors=color_map(normalization(face_values)),
        edgecolors=edgecolor,
        linewidths=linewidth,
        alpha=alpha,
    )
    collection.set_array(face_values)
    collection.set_cmap(cmap)
    collection.set_norm(normalization)
    axis.add_collection3d(collection)
    return collection


def flat_map_segments(mesh: Mesh) -> tuple[np.ndarray, np.ndarray]:
    longitude = np.arctan2(mesh.vertices[:, 1], mesh.vertices[:, 0])
    latitude = np.arcsin(np.clip(mesh.vertices[:, 2], -1.0, 1.0))
    flat = np.column_stack([longitude, latitude])
    segments = []
    for left, right in mesh.edges:
        if abs(longitude[left] - longitude[right]) < math.pi:
            segments.append(flat[[left, right]])
    return flat, np.asarray(segments)


def save_figure(figure: plt.Figure, output_stem: Path) -> None:
    figure.savefig(output_stem.with_suffix(".png"), dpi=220, bbox_inches="tight")
    figure.savefig(
        output_stem.with_suffix(".pdf"),
        bbox_inches="tight",
        metadata={"CreationDate": None, "ModDate": None},
    )
    plt.close(figure)


def plot_pipeline(
    mesh: Mesh,
    trial: Trial,
    output_dir: Path,
    japanese: bool,
) -> None:
    font = japanese_font_family() if japanese else "DejaVu Sans"
    suffix = "_ja" if japanese else ""
    possible_pairs = len(mesh.vertices) * (len(mesh.vertices) - 1) / 2
    query_percent = 100.0 * len(mesh.query_pairs) / possible_pairs
    labels = (
        {
            "flat": "ワープした平面地図（表示のみ）",
            "disk": "Gaussian平均：潰れた円板",
            "recovered": "3-hop局所 $W_2$ からの復元",
            "overlay": "正解球との比較（評価時のみ整列）",
            "truth": "正解",
            "estimate": "復元",
            "caption": (
                f"入力：接続 + 3-hop局所 $W_2$ 弦 "
                f"（{len(mesh.query_pairs):,}対、全対の{query_percent:.1f}%）; "
                "$S^2$ 弦–弧・直径補正"
            ),
        }
        if japanese
        else {
            "flat": "Warped flat map (display only)",
            "disk": "Gaussian means: collapsed disk",
            "recovered": "Recovered from three-hop local $W_2$",
            "overlay": "Truth vs recovery (aligned for evaluation only)",
            "truth": "truth",
            "estimate": "recovered",
            "caption": (
                f"Input: topology + {len(mesh.query_pairs):,} sparse three-hop $W_2$ "
                f"chords ({query_percent:.1f}% of all pairs); $S^2$ chord–arc/diameter correction"
            ),
        }
    )

    with plt.rc_context({"font.family": font, "font.size": 9.5}):
        figure = plt.figure(figsize=(13.2, 4.05), constrained_layout=True)
        grid = figure.add_gridspec(
            2, 4, height_ratios=(0.12, 0.88), hspace=0.01, wspace=0.08
        )
        color = mesh.vertices[:, 2]

        panel_titles = [
            "(a) " + labels["flat"],
            "(b) " + labels["disk"],
            "(c) " + labels["recovered"],
            "(d) " + labels["overlay"],
        ]
        for column, title in enumerate(panel_titles):
            title_axis = figure.add_subplot(grid[0, column])
            title_axis.axis("off")
            title_axis.text(0.5, 0.35, title, ha="center", va="center", fontsize=9.7)

        flat_axis = figure.add_subplot(grid[1, 0])
        flat, segments = flat_map_segments(mesh)
        flat_axis.add_collection(
            LineCollection(segments, colors="#94a3b8", linewidths=0.25, alpha=0.35)
        )
        flat_axis.scatter(
            flat[:, 0], flat[:, 1], c=color, cmap="coolwarm", s=6, zorder=2
        )
        flat_axis.set_xlim(-math.pi, math.pi)
        flat_axis.set_ylim(-0.5 * math.pi, 0.5 * math.pi)
        flat_axis.set_xticks([-math.pi, 0.0, math.pi], [r"$-\pi$", "0", r"$\pi$"])
        flat_axis.set_yticks(
            [-0.5 * math.pi, 0.0, 0.5 * math.pi],
            [r"$-\pi/2$", "0", r"$\pi/2$"],
        )
        flat_axis.set_aspect("equal")

        mean_axis = figure.add_subplot(grid[1, 1], projection="3d")
        mean_axis.scatter(
            trial.aligned_mean_disk[:, 0],
            trial.aligned_mean_disk[:, 1],
            trial.aligned_mean_disk[:, 2],
            c=color,
            cmap="coolwarm",
            s=8,
            alpha=0.8,
            depthshade=False,
        )
        set_equal_3d_axes(mean_axis)
        mean_axis.view_init(elev=23, azim=-55)

        recovered_axis = figure.add_subplot(grid[1, 2], projection="3d")
        add_mesh_surface(
            recovered_axis,
            trial.aligned_spherical,
            mesh.faces,
            color,
            alpha=0.96,
        )
        set_equal_3d_axes(recovered_axis)
        recovered_axis.view_init(elev=23, azim=-55)

        overlay_axis = figure.add_subplot(grid[1, 3], projection="3d")
        add_mesh_surface(
            overlay_axis,
            mesh.vertices,
            mesh.faces,
            np.ones(len(mesh.vertices)),
            cmap="Greys",
            alpha=0.18,
            edgecolor="#64748b",
            linewidth=0.16,
        )
        overlay_axis.scatter(
            trial.aligned_spherical[:, 0],
            trial.aligned_spherical[:, 1],
            trial.aligned_spherical[:, 2],
            c="#e11d48",
            s=3.5,
            alpha=0.75,
            label=labels["estimate"],
            depthshade=False,
        )
        overlay_axis.plot([], [], [], color="#64748b", label=labels["truth"])
        set_equal_3d_axes(overlay_axis)
        overlay_axis.view_init(elev=23, azim=-55)
        overlay_axis.legend(loc="lower center", frameon=False, fontsize=8)

        figure.suptitle(labels["caption"], fontsize=11.5)
        save_figure(
            figure,
            output_dir / f"wasserstein_surface_reconstruction_pipeline{suffix}",
        )


def plot_curvature(
    mesh: Mesh,
    trial: Trial,
    output_dir: Path,
    japanese: bool,
) -> None:
    font = japanese_font_family() if japanese else "DejaVu Sans"
    suffix = "_ja" if japanese else ""
    labels = (
        {
            "map": "角度欠損による未平滑化Gauss曲率",
            "hist": "未平滑化の頂点曲率（ノイズ感度）",
            "curvature": "推定曲率 $\\widehat K$",
            "density": "面積加重密度",
            "truth": "正解 $K=1$",
            "radius": "球面仮定 + Gauss–Bonnetで学習した半径",
        }
        if japanese
        else {
            "map": "Raw angle-deficit Gaussian curvature",
            "hist": "Raw vertex curvature (noise sensitivity)",
            "curvature": r"estimated curvature $\widehat K$",
            "density": "area-weighted density",
            "truth": "truth $K=1$",
            "radius": r"$S^2$ prior + Gauss–Bonnet radius",
        }
    )
    curvature = trial.estimate.curvature
    lower, upper = np.quantile(curvature, [0.03, 0.97])
    clipped = np.clip(curvature, lower, upper)

    with plt.rc_context({"font.family": font, "font.size": 10}):
        figure = plt.figure(figsize=(9.3, 3.65), constrained_layout=True)
        surface_axis = figure.add_subplot(1, 2, 1, projection="3d")
        collection = add_mesh_surface(
            surface_axis,
            trial.aligned_spherical,
            mesh.faces,
            clipped,
            cmap="viridis",
            alpha=0.97,
        )
        set_equal_3d_axes(surface_axis)
        surface_axis.view_init(elev=23, azim=-55)
        surface_axis.set_title("(d) " + labels["map"])
        colorbar = figure.colorbar(collection, ax=surface_axis, shrink=0.68, pad=0.02)
        colorbar.set_label(labels["curvature"])

        histogram_axis = figure.add_subplot(1, 2, 2)
        histogram_axis.hist(
            curvature,
            bins=34,
            weights=trial.estimate.vertex_areas / trial.estimate.total_area,
            color="#2563eb",
            alpha=0.82,
            edgecolor="white",
            linewidth=0.4,
            density=True,
        )
        histogram_axis.axvline(
            1.0, color="#dc2626", linestyle="--", linewidth=1.8, label=labels["truth"]
        )
        histogram_axis.set_xlabel(labels["curvature"])
        histogram_axis.set_ylabel(labels["density"])
        histogram_axis.set_title("(e) " + labels["hist"])
        histogram_axis.legend(frameon=False)
        histogram_axis.grid(alpha=0.2)
        histogram_axis.text(
            0.98,
            0.96,
            f"{labels['radius']}:  $\\widehat r={trial.estimate.learned_radius:.3f}$",
            transform=histogram_axis.transAxes,
            ha="right",
            va="top",
            bbox={"facecolor": "white", "edgecolor": "#cbd5e1", "alpha": 0.9},
        )
        save_figure(
            figure,
            output_dir / f"wasserstein_surface_reconstruction_curvature{suffix}",
        )


def summary_metric(
    summary: Sequence[dict[str, float]],
    subdivision: int,
    noise: float,
    metric: str,
) -> tuple[float, float]:
    for row in summary:
        if int(row["subdivision"]) == subdivision and math.isclose(
            row["relative_edge_noise"], noise, abs_tol=1.0e-12
        ):
            return row[f"{metric}_mean"], row[f"{metric}_std"]
    raise KeyError((subdivision, noise, metric))


def plot_errors(
    summary: Sequence[dict[str, float]],
    subdivisions: Sequence[int],
    noise_levels: Sequence[float],
    output_dir: Path,
    japanese: bool,
) -> None:
    font = japanese_font_family() if japanese else "DejaVu Sans"
    suffix = "_ja" if japanese else ""
    labels = (
        {
            "curvature": "曲率RMSE（無雑音の離散曲率に対して）",
            "oracle": "無雑音の離散曲率に対して",
            "smooth": "滑らかな球面 $K=1$ に対して",
            "shape": "3D復元RMSE",
            "intrinsic": "測地距離の相対誤差",
            "vertices": "頂点数（解像度）",
            "spherical": "提案法：球面MDS",
            "topology": "位相のみ：一様icosphere",
            "ordinary": "通常のMDS",
            "mean": "平均のみの円板",
            "noise": "局所弦長ノイズ",
            "tradeoff": "相対ノイズ固定：高解像度ほど未平滑化曲率誤差を増幅",
            "zero": "0%では離散曲率を数値精度で復元",
        }
        if japanese
        else {
            "curvature": "curvature RMSE (vs discrete target)",
            "oracle": "vs noiseless discrete target",
            "smooth": "vs smooth sphere $K=1$",
            "shape": "3D reconstruction RMSE",
            "intrinsic": "relative geodesic-distance error",
            "vertices": "vertices (resolution)",
            "spherical": "ours: spherical MDS",
            "topology": "topology-only uniform mesh",
            "ordinary": "ordinary MDS",
            "mean": "mean-only disk",
            "noise": "local-chord noise",
            "tradeoff": (
                "fixed relative noise: finer meshes amplify raw curvature error"
            ),
            "zero": "0% recovers the discrete target to numerical precision",
        }
    )
    vertex_counts = np.asarray([10 * 4**level + 2 for level in subdivisions])
    palette = plt.get_cmap("viridis")
    colors = [palette(value) for value in np.linspace(0.08, 0.88, len(noise_levels))]

    with plt.rc_context({"font.family": font, "font.size": 9.5}):
        figure, axes = plt.subplots(1, 3, figsize=(12.7, 3.45), constrained_layout=True)
        for noise, color in zip(noise_levels, colors, strict=True):
            curvature_oracle_mean = []
            curvature_oracle_std = []
            curvature_smooth_mean = []
            curvature_smooth_std = []
            geodesic_mean = []
            geodesic_std = []
            spherical_mean = []
            spherical_std = []
            ordinary_mean = []
            ordinary_std = []
            topology_uniform = []
            mean_disk = []
            for subdivision in subdivisions:
                mean_value, std_value = summary_metric(
                    summary, subdivision, noise, "curvature_oracle_weighted_rmse"
                )
                curvature_oracle_mean.append(mean_value)
                curvature_oracle_std.append(std_value)
                mean_value, std_value = summary_metric(
                    summary,
                    subdivision,
                    noise,
                    "curvature_smooth_k1_weighted_rmse",
                )
                curvature_smooth_mean.append(mean_value)
                curvature_smooth_std.append(std_value)
                mean_value, std_value = summary_metric(
                    summary, subdivision, noise, "graph_geodesic_relative_error"
                )
                geodesic_mean.append(mean_value)
                geodesic_std.append(std_value)
                mean_value, std_value = summary_metric(
                    summary, subdivision, noise, "spherical_reconstruction_rmse"
                )
                spherical_mean.append(mean_value)
                spherical_std.append(std_value)
                mean_value, std_value = summary_metric(
                    summary, subdivision, noise, "ordinary_mds_rmse"
                )
                ordinary_mean.append(mean_value)
                ordinary_std.append(std_value)
                topology_uniform.append(
                    summary_metric(
                        summary, subdivision, noise, "topology_uniform_rmse"
                    )[0]
                )
                mean_disk.append(
                    summary_metric(summary, subdivision, noise, "mean_disk_rmse")[0]
                )

            label = f"{labels['noise']} {100.0 * noise:.1f}%"
            axes[0].errorbar(
                vertex_counts,
                curvature_oracle_mean,
                yerr=curvature_oracle_std,
                marker="o",
                ms=4,
                linewidth=1.4,
                capsize=2,
                color=color,
                label=label,
            )
            if math.isclose(noise, noise_levels[0], abs_tol=EPSILON):
                axes[0].plot(
                    vertex_counts,
                    curvature_smooth_mean,
                    marker="s",
                    ms=4,
                    linestyle="--",
                    color="#dc2626",
                    label=labels["smooth"],
                )
            axes[1].errorbar(
                vertex_counts,
                spherical_mean,
                yerr=spherical_std,
                marker="o",
                ms=4,
                linewidth=1.5,
                capsize=2,
                color=color,
                label=label,
            )
            # Noise-independent baselines and ordinary MDS at zero noise are
            # shown once; their styles identify them without duplicating lines.
            if math.isclose(noise, noise_levels[0], abs_tol=EPSILON):
                axes[1].plot(
                    vertex_counts,
                    topology_uniform,
                    marker="D",
                    ms=3.8,
                    linestyle="-.",
                    color="#a21caf",
                    label=labels["topology"],
                )
                axes[1].plot(
                    vertex_counts,
                    ordinary_mean,
                    marker="s",
                    ms=4,
                    linestyle="--",
                    color="#f97316",
                    label=labels["ordinary"],
                )
                axes[1].plot(
                    vertex_counts,
                    mean_disk,
                    marker="^",
                    ms=4,
                    linestyle=":",
                    color="#64748b",
                    label=labels["mean"],
                )
            axes[2].errorbar(
                vertex_counts,
                geodesic_mean,
                yerr=geodesic_std,
                marker="o",
                ms=4,
                linewidth=1.4,
                capsize=2,
                color=color,
                label=label,
            )

        titles = [labels["curvature"], labels["shape"], labels["intrinsic"]]
        for index, (axis, title) in enumerate(zip(axes, titles, strict=True)):
            axis.set_xscale("log", base=2)
            if index == 0:
                axis.set_yscale("linear")
                axis.set_ylim(bottom=0.0)
            else:
                axis.set_yscale("log")
            axis.set_xticks(vertex_counts, [str(value) for value in vertex_counts])
            if japanese:
                axis.tick_params(axis="x", labelsize=8)
                for tick_label in axis.get_xticklabels():
                    tick_label.set_rotation(28)
                    tick_label.set_ha("right")
            axis.set_xlabel(labels["vertices"])
            axis.set_ylabel(title)
            axis.set_title(f"({chr(ord('a') + index)}) {title}")
            axis.grid(alpha=0.22, which="both")
        axes[0].legend(
            loc="upper left",
            bbox_to_anchor=(0.0, 0.87),
            frameon=True,
            framealpha=0.88,
            facecolor="white",
            edgecolor="none",
            fontsize=7.2,
        )
        axes[1].legend(
            loc="upper right",
            frameon=True,
            framealpha=0.88,
            facecolor="white",
            edgecolor="none",
            fontsize=6.9,
        )
        axes[0].text(
            0.02,
            0.98,
            labels["tradeoff"],
            transform=axes[0].transAxes,
            ha="left",
            va="top",
            fontsize=7.4,
            color="#7c2d12",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        axes[0].text(
            0.02,
            0.03,
            labels["zero"],
            transform=axes[0].transAxes,
            ha="left",
            va="bottom",
            fontsize=7.2,
            color="#312e81",
            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.72},
        )
        save_figure(
            figure,
            output_dir / f"wasserstein_surface_reconstruction_errors{suffix}",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subdivisions",
        nargs="+",
        type=int,
        default=list(DEFAULT_SUBDIVISIONS),
        help="icosphere subdivision levels",
    )
    parser.add_argument(
        "--noise-levels",
        nargs="+",
        type=float,
        default=list(DEFAULT_NOISE_LEVELS),
        help="relative standard deviations of multiplicative local-chord noise",
    )
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=list(DEFAULT_SEEDS),
        help="noise-replicate indices",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="directory for CSV, TeX, PNG, and PDF outputs",
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="run levels 0--2, noise 0/1%%, and one seed",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    subdivisions = tuple(sorted(set(arguments.subdivisions)))
    noise_levels = tuple(sorted(set(arguments.noise_levels)))
    seeds = tuple(sorted(set(arguments.seeds)))
    if arguments.quick:
        subdivisions = (0, 1, 2)
        noise_levels = (0.0, 0.01)
        seeds = (0,)
    if not subdivisions or not noise_levels or not seeds:
        raise ValueError("subdivisions, noise levels, and seeds cannot be empty")
    if min(subdivisions) < 0 or min(noise_levels) < 0.0:
        raise ValueError("subdivisions and noise levels must be nonnegative")

    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, float]] = []
    meshes = {level: icosphere(level) for level in subdivisions}
    showcase_trial: Trial | None = None
    showcase_mesh: Mesh | None = None
    desired_noise = min(noise_levels, key=lambda value: abs(value - SHOWCASE_NOISE))

    trials_per_mesh = sum(1 if noise == 0.0 else len(seeds) for noise in noise_levels)
    total_trials = len(subdivisions) * trials_per_mesh
    completed = 0
    for subdivision in subdivisions:
        mesh = meshes[subdivision]
        for noise in noise_levels:
            seed_indices = (seeds[0],) if noise == 0.0 else seeds
            for seed in seed_indices:
                trial = run_trial(mesh, subdivision, noise, seed)
                rows.append(trial.metrics)
                completed += 1
                print(
                    f"[{completed:02d}/{total_trials:02d}] "
                    f"level={subdivision} V={len(mesh.vertices)} "
                    f"noise={100.0 * noise:.2f}% seed={seed} "
                    f"r={trial.estimate.learned_radius:.4f} "
                    f"shape_rmse={trial.metrics['spherical_reconstruction_rmse']:.4f}",
                    flush=True,
                )
                if (
                    subdivision == max(subdivisions)
                    and math.isclose(noise, desired_noise, abs_tol=EPSILON)
                    and seed == seeds[0]
                ):
                    showcase_trial = trial
                    showcase_mesh = mesh

    if showcase_trial is None or showcase_mesh is None:
        raise RuntimeError("failed to select a showcase trial")
    summary = aggregate_rows(rows)
    write_dict_csv(output_dir / "wasserstein_surface_reconstruction_results.csv", rows)
    write_dict_csv(
        output_dir / "wasserstein_surface_reconstruction_summary.csv", summary
    )
    write_latex_table(
        output_dir / "wasserstein_surface_reconstruction_table.tex",
        summary,
        max(subdivisions),
        japanese=False,
    )
    write_latex_table(
        output_dir / "wasserstein_surface_reconstruction_table_ja.tex",
        summary,
        max(subdivisions),
        japanese=True,
    )
    for japanese in (False, True):
        plot_pipeline(showcase_mesh, showcase_trial, output_dir, japanese=japanese)
        plot_curvature(showcase_mesh, showcase_trial, output_dir, japanese=japanese)
        plot_errors(
            summary,
            subdivisions,
            noise_levels,
            output_dir,
            japanese=japanese,
        )

    print(
        "Wrote Wasserstein surface reconstruction CSV/TeX/PNG/PDF outputs to "
        f"{output_dir}",
        flush=True,
    )


if __name__ == "__main__":
    main()
