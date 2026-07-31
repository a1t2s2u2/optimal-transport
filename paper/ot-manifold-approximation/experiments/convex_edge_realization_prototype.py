#!/usr/bin/env python3
# /// script
# requires-python = ">=3.11,<3.14"
# dependencies = [
#   "matplotlib==3.11.1",
#   "numpy==2.5.1",
#   "scipy==1.18.0",
#   "torch==2.13.0",
# ]
# ///
"""Prototype: recover a general convex polyhedron from local W2 edge chords.

This experiment is the numerical counterpart of same-skeleton convex
rigidity.  A full-rank Gaussian observation with noncommuting covariances is
attached to every vertex so that W2 on every mesh edge equals its Euclidean
chord.  The estimator receives only

    (number of vertices, oriented triangular faces, edges, measured chords).

It first constructs a spherical-MDS initializer.  That initializer is used
only to choose the convex realization branch.  A continuation method then
deforms the initializer toward the observed edge lengths while penalizing
locally nonconvex dihedral configurations.  The final least-squares solve has
no radius, row-normalization, or spherical term: after fixing one triangular
face to remove E(3), its residuals are mesh-edge chord residuals only.

The prototype evaluates a round sphere and a linearly transformed ellipsoid,
both with the same convex triangular skeleton, at exact, 0.5%, and 1% noisy
W2 chords.  Each noisy condition uses three deterministic seeds.  It also
audits the local rigidity estimate

    ||X_hat - X||_F <= 2 ||ell_hat - ell||_2 / s_X,

where s_X is the smallest nonzero singular value of the true edge-length
rigidity matrix after quotienting Euclidean motions.  ConvexHull must retain
every vertex and exactly the input facets in every trial.

Run with:
    uv run --python 3.12 convex_edge_realization_prototype.py
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
import torch
import wasserstein_surface_reconstruction as spherical_experiment
from scipy.optimize import least_squares
from scipy.spatial import ConvexHull
from torch import Tensor

plt.switch_backend("Agg")

HERE = Path(__file__).resolve().parent
MASTER_SEED = 20260801
SUBDIVISION = 1
NOISE_LEVELS = (0.0, 0.005, 0.01)
NOISY_SEEDS = (0, 1, 2)
# This remains visibly nonspherical while keeping 1% independent edge noise
# inside the empirically audited convex same-skeleton neighborhood.
ELLIPSOID_AXES = (1.20, 1.00, 0.80)
CONTINUATION_STAGES = 41
LBFGS_ITERATIONS = 100
CONVEXITY_WEIGHT = 1_000.0
CONVEXITY_MARGIN = 1.0e-7
EPSILON = 1.0e-12


@dataclass(frozen=True)
class Realization:
    """Distance-only reconstruction before ground-truth evaluation."""

    coordinates: np.ndarray
    initializer: np.ndarray
    history: list[dict[str, float]]
    final_edge_stress: float
    final_edge_relative_rmse: float
    final_edge_max_relative_error: float
    convex_hull_all_vertices: bool
    convex_hull_facet_match: bool
    hull_vertex_count: int
    hull_facet_count: int
    local_convex_max_signed: float
    least_squares_nfev: int


@dataclass(frozen=True)
class EvaluatedCase:
    """A reconstruction plus evaluation-only ground truth and alignment."""

    shape: str
    noise_level: float
    seed_index: int
    truth: np.ndarray
    realization: Realization
    aligned_initializer: np.ndarray
    aligned_reconstruction: np.ndarray
    metrics: dict[str, float | str | bool]


def edge_length_map(
    edges: np.ndarray, lengths: np.ndarray
) -> dict[tuple[int, int], float]:
    return {
        (int(left), int(right)): float(length)
        for (left, right), length in zip(edges, lengths, strict=True)
    }


def canonical_anchor_positions(
    anchor_face: Sequence[int],
    edges: np.ndarray,
    lengths: np.ndarray,
) -> np.ndarray:
    """Place an anchor triangle canonically using only its three edge lengths."""

    first, second, third = (int(vertex) for vertex in anchor_face)
    lookup = edge_length_map(edges, lengths)

    def length(left: int, right: int) -> float:
        return lookup[(min(left, right), max(left, right))]

    first_second = length(first, second)
    first_third = length(first, third)
    second_third = length(second, third)
    third_x = (first_second**2 + first_third**2 - second_third**2) / max(
        2.0 * first_second, EPSILON
    )
    third_y_squared = first_third**2 - third_x**2
    if third_y_squared <= 0.0:
        raise ValueError("the noisy anchor lengths do not form a triangle")
    return np.asarray(
        [
            (0.0, 0.0, 0.0),
            (first_second, 0.0, 0.0),
            (third_x, math.sqrt(third_y_squared), 0.0),
        ],
        dtype=np.float64,
    )


def canonicalize_initializer(
    coordinates: np.ndarray,
    anchor_face: Sequence[int],
    anchor_positions: np.ndarray,
) -> np.ndarray:
    """Remove the initializer's E(3) gauge and put the mesh inside z <= 0."""

    first, second, third = (int(vertex) for vertex in anchor_face)
    origin = coordinates[first]
    first_axis = coordinates[second] - origin
    first_axis /= np.linalg.norm(first_axis)
    second_axis = coordinates[third] - origin
    second_axis -= first_axis * np.dot(second_axis, first_axis)
    second_axis /= np.linalg.norm(second_axis)
    third_axis = np.cross(first_axis, second_axis)
    basis = np.column_stack([first_axis, second_axis, third_axis])
    canonical = (coordinates - origin) @ basis

    nonanchors = np.asarray(
        [
            vertex
            for vertex in range(len(coordinates))
            if vertex not in (first, second, third)
        ],
        dtype=np.int64,
    )
    # The oriented anchor face is outward, so the convex body belongs below it.
    if np.median(canonical[nonanchors, 2]) > 0.0:
        canonical[:, 2] *= -1.0
    canonical[[first, second, third]] = anchor_positions
    return canonical


def adjacent_opposite_constraints(faces: np.ndarray) -> np.ndarray:
    """Return (a,b,c,d): d must lie inside the oriented face (a,b,c)."""

    edge_incidence: dict[tuple[int, int], list[tuple[int, int]]] = {}
    for face_index, (first, second, third) in enumerate(faces.tolist()):
        for left, right, opposite in (
            (first, second, third),
            (second, third, first),
            (third, first, second),
        ):
            edge_incidence.setdefault(tuple(sorted((left, right))), []).append(
                (face_index, opposite)
            )

    constraints: list[tuple[int, int, int, int]] = []
    for face_index, (first, second, third) in enumerate(faces.tolist()):
        for left, right in (
            (first, second),
            (second, third),
            (third, first),
        ):
            incident = edge_incidence[tuple(sorted((left, right)))]
            if len(incident) != 2:
                raise ValueError("the prototype requires a closed two-manifold mesh")
            opposite = next(
                vertex
                for adjacent_face, vertex in incident
                if adjacent_face != face_index
            )
            constraints.append((first, second, third, opposite))
    return np.asarray(constraints, dtype=np.int64)


def assemble_torch_coordinates(
    vertex_count: int,
    free_indices: Tensor,
    free_coordinates: Tensor,
    anchor_indices: Tensor,
    anchor_positions: Tensor,
) -> Tensor:
    coordinates = torch.zeros(
        (vertex_count, 3),
        dtype=free_coordinates.dtype,
        device=free_coordinates.device,
    )
    coordinates[free_indices] = free_coordinates
    coordinates[anchor_indices] = anchor_positions
    return coordinates


def normalized_local_convexity(
    coordinates: Tensor,
    constraints: Tensor,
    length_scale: float,
) -> Tensor:
    first, second, third, opposite = [constraints[:, index] for index in range(4)]
    normals = torch.cross(
        coordinates[second] - coordinates[first],
        coordinates[third] - coordinates[first],
        dim=1,
    )
    return torch.sum(
        normals * (coordinates[opposite] - coordinates[first]), dim=1
    ) / max(length_scale**3, EPSILON)


def edge_statistics(
    coordinates: np.ndarray,
    edges: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[float, float, float]:
    realized = np.linalg.norm(
        coordinates[edges[:, 0]] - coordinates[edges[:, 1]], axis=1
    )
    residual = realized - target_lengths
    length_scale = float(target_lengths.mean())
    stress = float(np.mean(np.square(residual / length_scale)))
    relative_rmse = float(np.linalg.norm(residual) / np.linalg.norm(target_lengths))
    max_relative = float(np.max(np.abs(residual) / target_lengths))
    return stress, relative_rmse, max_relative


def rigidity_smallest_nonzero_singular_value(
    coordinates: np.ndarray,
    edges: np.ndarray,
) -> tuple[float, int]:
    """Return s_X for the edge-length rigidity matrix at a true realization."""

    vertex_count = len(coordinates)
    rigidity = np.zeros((len(edges), 3 * vertex_count), dtype=np.float64)
    for row, (left, right) in enumerate(edges.tolist()):
        difference = coordinates[left] - coordinates[right]
        unit_direction = difference / max(np.linalg.norm(difference), EPSILON)
        rigidity[row, 3 * left : 3 * left + 3] = unit_direction
        rigidity[row, 3 * right : 3 * right + 3] = -unit_direction
    singular_values = np.linalg.svd(rigidity, compute_uv=False)
    tolerance = (
        max(rigidity.shape) * np.finfo(np.float64).eps * float(singular_values.max())
    )
    nonzero = singular_values[singular_values > tolerance]
    rank = len(nonzero)
    expected_rank = 3 * vertex_count - 6
    if rank != expected_rank:
        raise RuntimeError(
            f"true framework is not infinitesimally rigid: {rank} != {expected_rank}"
        )
    return float(nonzero.min()), rank


def hull_diagnostics(
    coordinates: np.ndarray,
    faces: np.ndarray,
) -> tuple[bool, bool, int, int]:
    hull = ConvexHull(coordinates)
    target_facets = {tuple(sorted(face)) for face in faces.tolist()}
    hull_facets = {tuple(sorted(face)) for face in hull.simplices.tolist()}
    all_vertices = set(hull.vertices.tolist()) == set(range(len(coordinates)))
    return (
        all_vertices,
        hull_facets == target_facets,
        len(hull.vertices),
        len(hull_facets),
    )


def edge_only_least_squares(
    initial_free: np.ndarray,
    vertex_count: int,
    free_indices: np.ndarray,
    anchor_face: tuple[int, int, int],
    anchor_positions: np.ndarray,
    edges: np.ndarray,
    target_lengths: np.ndarray,
) -> tuple[np.ndarray, int]:
    """Final refinement: only edge residuals, with E(3) removed by anchors."""

    free_position = {
        int(vertex): position for position, vertex in enumerate(free_indices.tolist())
    }
    anchor_edge_set = {
        tuple(sorted((anchor_face[0], anchor_face[1]))),
        tuple(sorted((anchor_face[1], anchor_face[2]))),
        tuple(sorted((anchor_face[2], anchor_face[0]))),
    }
    active_edge_indices = np.asarray(
        [
            index
            for index, edge in enumerate(edges.tolist())
            if tuple(edge) not in anchor_edge_set
        ],
        dtype=np.int64,
    )
    length_scale = float(target_lengths.mean())

    def coordinates(vector: np.ndarray) -> np.ndarray:
        output = np.zeros((vertex_count, 3), dtype=np.float64)
        output[free_indices] = vector.reshape(-1, 3)
        output[np.asarray(anchor_face)] = anchor_positions
        return output

    def residual(vector: np.ndarray) -> np.ndarray:
        output = coordinates(vector)
        active_edges = edges[active_edge_indices]
        realized = np.linalg.norm(
            output[active_edges[:, 0]] - output[active_edges[:, 1]], axis=1
        )
        return (realized - target_lengths[active_edge_indices]) / length_scale

    def jacobian(vector: np.ndarray) -> np.ndarray:
        output = coordinates(vector)
        active_edges = edges[active_edge_indices]
        difference = output[active_edges[:, 0]] - output[active_edges[:, 1]]
        distance = np.maximum(np.linalg.norm(difference, axis=1), EPSILON)
        direction = difference / distance[:, None] / length_scale
        matrix = np.zeros(
            (len(active_edge_indices), 3 * len(free_indices)), dtype=np.float64
        )
        for row, (left, right) in enumerate(active_edges.tolist()):
            if left in free_position:
                column = 3 * free_position[left]
                matrix[row, column : column + 3] = direction[row]
            if right in free_position:
                column = 3 * free_position[right]
                matrix[row, column : column + 3] = -direction[row]
        return matrix

    result = least_squares(
        residual,
        initial_free.reshape(-1),
        jac=jacobian,
        method="trf",
        ftol=1.0e-13,
        xtol=1.0e-13,
        gtol=1.0e-13,
        max_nfev=500,
    )
    if not result.success:
        raise RuntimeError(f"edge-only least squares failed: {result.message}")
    return coordinates(result.x), int(result.nfev)


def convex_edge_realization(
    vertex_count: int,
    faces: np.ndarray,
    edges: np.ndarray,
    measured_edge_chords: np.ndarray,
) -> Realization:
    """Recover a convex realization without receiving ground-truth positions."""

    if len(measured_edge_chords) != len(edges):
        raise ValueError("one measured W2 chord is required for every mesh edge")
    if (
        np.any(measured_edge_chords <= 0.0)
        or not np.isfinite(measured_edge_chords).all()
    ):
        raise ValueError("all measured edge chords must be finite and positive")

    # Spherical assumptions occur only here, before edge-realization refinement.
    initializer = spherical_experiment.infer_from_local_lengths(
        vertex_count, faces, edges, measured_edge_chords
    ).spherical_coordinates

    anchor_face = tuple(int(vertex) for vertex in faces[0])
    final_anchor_positions = canonical_anchor_positions(
        anchor_face, edges, measured_edge_chords
    )
    canonical_initializer = canonicalize_initializer(
        initializer, anchor_face, final_anchor_positions
    )
    initial_edge_lengths = np.linalg.norm(
        canonical_initializer[edges[:, 0]] - canonical_initializer[edges[:, 1]],
        axis=1,
    )

    free_indices_numpy = np.asarray(
        [vertex for vertex in range(vertex_count) if vertex not in anchor_face],
        dtype=np.int64,
    )
    free_indices = torch.as_tensor(free_indices_numpy, dtype=torch.long)
    anchor_indices = torch.as_tensor(anchor_face, dtype=torch.long)
    edge_tensor = torch.as_tensor(edges, dtype=torch.long)
    convexity_constraints_numpy = adjacent_opposite_constraints(faces)
    convexity_constraints = torch.as_tensor(
        convexity_constraints_numpy, dtype=torch.long
    )
    free_coordinates = torch.tensor(
        canonical_initializer[free_indices_numpy],
        dtype=torch.float64,
        requires_grad=True,
    )
    history: list[dict[str, float]] = []

    for stage, alpha in enumerate(np.linspace(0.0, 1.0, CONTINUATION_STAGES)):
        continuation_lengths = (
            1.0 - alpha
        ) * initial_edge_lengths + alpha * measured_edge_chords
        anchor_positions_numpy = canonical_anchor_positions(
            anchor_face, edges, continuation_lengths
        )
        anchor_positions = torch.as_tensor(anchor_positions_numpy, dtype=torch.float64)
        target = torch.as_tensor(continuation_lengths, dtype=torch.float64)
        length_scale = float(continuation_lengths.mean())
        optimizer = torch.optim.LBFGS(
            [free_coordinates],
            lr=0.8,
            max_iter=LBFGS_ITERATIONS,
            tolerance_grad=1.0e-11,
            tolerance_change=1.0e-14,
            line_search_fn="strong_wolfe",
        )

        def closure(
            current_optimizer: torch.optim.Optimizer = optimizer,
            current_anchor_positions: Tensor = anchor_positions,
            current_target: Tensor = target,
            current_length_scale: float = length_scale,
        ) -> Tensor:
            current_optimizer.zero_grad()
            coordinates = assemble_torch_coordinates(
                vertex_count,
                free_indices,
                free_coordinates,
                anchor_indices,
                current_anchor_positions,
            )
            realized = torch.linalg.vector_norm(
                coordinates[edge_tensor[:, 0]] - coordinates[edge_tensor[:, 1]],
                dim=1,
            )
            edge_stress = torch.mean(
                ((realized - current_target) / current_length_scale) ** 2
            )
            signed = normalized_local_convexity(
                coordinates, convexity_constraints, current_length_scale
            )
            convexity_penalty = torch.mean(torch.relu(signed + CONVEXITY_MARGIN) ** 2)
            objective = edge_stress + CONVEXITY_WEIGHT * convexity_penalty
            objective.backward()
            return objective

        optimizer.step(closure)
        with torch.no_grad():
            coordinates = assemble_torch_coordinates(
                vertex_count,
                free_indices,
                free_coordinates,
                anchor_indices,
                anchor_positions,
            )
            realized = torch.linalg.vector_norm(
                coordinates[edge_tensor[:, 0]] - coordinates[edge_tensor[:, 1]],
                dim=1,
            )
            edge_stress = torch.mean(((realized - target) / length_scale) ** 2)
            signed = normalized_local_convexity(
                coordinates, convexity_constraints, length_scale
            )
            convexity_penalty = torch.mean(torch.relu(signed + CONVEXITY_MARGIN) ** 2)
        history.append(
            {
                "stage": float(stage),
                "alpha": float(alpha),
                "edge_stress": float(edge_stress),
                "convexity_penalty": float(convexity_penalty),
                "local_convex_max_signed": float(signed.max()),
            }
        )

    # The last solver contains no spherical or convexity penalty.  Strict
    # convexity from continuation selects the branch; edge residuals determine
    # the final free coordinates in the E(3)-fixed gauge.
    final_coordinates, least_squares_nfev = edge_only_least_squares(
        free_coordinates.detach().cpu().numpy(),
        vertex_count,
        free_indices_numpy,
        anchor_face,
        final_anchor_positions,
        edges,
        measured_edge_chords,
    )
    stress, relative_rmse, max_relative = edge_statistics(
        final_coordinates, edges, measured_edge_chords
    )
    all_vertices, facets_match, hull_vertices, hull_facets = hull_diagnostics(
        final_coordinates, faces
    )
    constraint_tensor = torch.as_tensor(convexity_constraints_numpy, dtype=torch.long)
    with torch.no_grad():
        signed = normalized_local_convexity(
            torch.as_tensor(final_coordinates, dtype=torch.float64),
            constraint_tensor,
            float(measured_edge_chords.mean()),
        )
    local_convex_max_signed = float(signed.max())
    if not all_vertices or not facets_match or local_convex_max_signed >= 1.0e-8:
        raise RuntimeError(
            "edge realization left the required convex same-skeleton chamber: "
            f"all_vertices={all_vertices}, facets_match={facets_match}, "
            f"max_signed={local_convex_max_signed:.3e}"
        )

    return Realization(
        coordinates=final_coordinates,
        initializer=initializer,
        history=history,
        final_edge_stress=stress,
        final_edge_relative_rmse=relative_rmse,
        final_edge_max_relative_error=max_relative,
        convex_hull_all_vertices=all_vertices,
        convex_hull_facet_match=facets_match,
        hull_vertex_count=hull_vertices,
        hull_facet_count=hull_facets,
        local_convex_max_signed=local_convex_max_signed,
        least_squares_nfev=least_squares_nfev,
    )


def generated_shapes() -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    mesh = spherical_experiment.icosphere(SUBDIVISION)
    sphere = mesh.vertices
    ellipsoid = sphere @ np.diag(np.asarray(ELLIPSOID_AXES, dtype=np.float64))
    return mesh.faces, mesh.edges, {"sphere": sphere, "ellipsoid": ellipsoid}


def evaluate_case(
    shape: str,
    truth: np.ndarray,
    faces: np.ndarray,
    edges: np.ndarray,
    noise_level: float,
    seed_index: int,
) -> EvaluatedCase:
    mean, covariance = spherical_experiment.noncommuting_gaussian_observations(truth)
    exact_w2 = spherical_experiment.gaussian_w2_edges(mean, covariance, edges)
    exact_chords = np.linalg.norm(truth[edges[:, 0]] - truth[edges[:, 1]], axis=1)
    w2_identity_error = float(np.max(np.abs(exact_w2 - exact_chords)))
    if w2_identity_error > 2.0e-12:
        raise RuntimeError(f"W2/chord identity failed: {w2_identity_error:.3e}")
    parameter_scale, parameter_distortion = (
        spherical_experiment.oracle_scaled_raw_parameter_distortion(
            mean, covariance, edges, exact_chords
        )
    )
    seed = (
        MASTER_SEED
        + sum(map(ord, shape))
        + round(1.0e6 * noise_level)
        + 10_000_019 * seed_index
    )
    measured = spherical_experiment.noisy_lengths(exact_w2, noise_level, seed)

    # No truth is passed across this estimator boundary.
    realization = convex_edge_realization(len(truth), faces, edges, measured)
    aligned_initializer, initializer_rmse = spherical_experiment.orthogonal_alignment(
        realization.initializer, truth
    )
    aligned_reconstruction, reconstruction_rmse = (
        spherical_experiment.orthogonal_alignment(realization.coordinates, truth)
    )
    rigidity_singular_value, rigidity_rank = rigidity_smallest_nonzero_singular_value(
        truth, edges
    )
    edge_noise_l2 = float(np.linalg.norm(measured - exact_w2))
    aligned_frobenius = reconstruction_rmse * math.sqrt(len(truth))
    local_rigidity_bound = 2.0 * edge_noise_l2 / rigidity_singular_value
    if local_rigidity_bound > EPSILON:
        error_to_bound_ratio = aligned_frobenius / local_rigidity_bound
    else:
        error_to_bound_ratio = math.nan
    bound_satisfied = aligned_frobenius <= local_rigidity_bound + 1.0e-10
    metrics: dict[str, float | str | bool] = {
        "shape": shape,
        "relative_edge_noise": noise_level,
        "seed": float(seed_index),
        "vertices": float(len(truth)),
        "faces": float(len(faces)),
        "edges": float(len(edges)),
        "free_coordinates_after_e3_fix": float(3 * (len(truth) - 3)),
        "active_edge_residuals": float(len(edges) - 3),
        "w2_chord_max_abs_error": w2_identity_error,
        "gaussian_covariance_minimum_eigenvalue": float(
            np.linalg.eigvalsh(covariance).min()
        ),
        "noncommuting_covariance_edge_fraction": (
            spherical_experiment.noncommuting_pair_fraction(covariance, edges)
        ),
        "oracle_covariance_parameter_scale": parameter_scale,
        "oracle_scaled_raw_parameter_distance_relative_rmse": parameter_distortion,
        "initializer_e3_rmse": initializer_rmse,
        "final_e3_rmse": reconstruction_rmse,
        "rigidity_rank": float(rigidity_rank),
        "rigidity_expected_rank": float(3 * len(truth) - 6),
        "rigidity_smallest_nonzero_singular_value": rigidity_singular_value,
        "edge_noise_l2": edge_noise_l2,
        "aligned_frobenius_error": aligned_frobenius,
        "local_rigidity_bound": local_rigidity_bound,
        "error_to_bound_ratio": error_to_bound_ratio,
        "local_bound_satisfied_with_tolerance": bound_satisfied,
        "local_bound_slack": local_rigidity_bound - aligned_frobenius,
        "final_edge_stress": realization.final_edge_stress,
        "final_edge_relative_rmse": realization.final_edge_relative_rmse,
        "final_edge_max_relative_error": realization.final_edge_max_relative_error,
        "convex_hull_all_vertices": realization.convex_hull_all_vertices,
        "convex_hull_facet_match": realization.convex_hull_facet_match,
        "same_skeleton_verified": (
            realization.convex_hull_all_vertices and realization.convex_hull_facet_match
        ),
        "final_uses_radius_constraint": False,
        "final_uses_row_normalization": False,
        "hull_vertex_count": float(realization.hull_vertex_count),
        "hull_facet_count": float(realization.hull_facet_count),
        "local_convex_max_signed": realization.local_convex_max_signed,
        "least_squares_nfev": float(realization.least_squares_nfev),
        "continuation_stages": float(CONTINUATION_STAGES),
        "final_radial_standard_deviation": float(
            np.linalg.norm(
                realization.coordinates - realization.coordinates.mean(axis=0), axis=1
            ).std()
        ),
    }
    return EvaluatedCase(
        shape=shape,
        noise_level=noise_level,
        seed_index=seed_index,
        truth=truth,
        realization=realization,
        aligned_initializer=aligned_initializer,
        aligned_reconstruction=aligned_reconstruction,
        metrics=metrics,
    )


def write_dict_csv(
    path: Path,
    rows: Sequence[dict[str, float | str | bool]],
) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(rows[0].keys()),
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


SUMMARY_METRICS = (
    "w2_chord_max_abs_error",
    "gaussian_covariance_minimum_eigenvalue",
    "noncommuting_covariance_edge_fraction",
    "oracle_covariance_parameter_scale",
    "oracle_scaled_raw_parameter_distance_relative_rmse",
    "rigidity_smallest_nonzero_singular_value",
    "edge_noise_l2",
    "aligned_frobenius_error",
    "local_rigidity_bound",
    "error_to_bound_ratio",
    "final_e3_rmse",
    "final_edge_relative_rmse",
    "initializer_e3_rmse",
)


def aggregate_summary(
    cases: Sequence[EvaluatedCase],
) -> list[dict[str, float | str | bool]]:
    groups: dict[tuple[str, float], list[EvaluatedCase]] = {}
    for case in cases:
        groups.setdefault((case.shape, case.noise_level), []).append(case)

    shape_order = {"sphere": 0, "ellipsoid": 1}
    summary: list[dict[str, float | str | bool]] = []
    for (shape, noise), group in sorted(
        groups.items(), key=lambda item: (shape_order[item[0][0]], item[0][1])
    ):
        output: dict[str, float | str | bool] = {
            "shape": shape,
            "relative_edge_noise": noise,
            "trials": float(len(group)),
        }
        for metric in SUMMARY_METRICS:
            values = np.asarray(
                [float(case.metrics[metric]) for case in group], dtype=np.float64
            )
            finite = values[np.isfinite(values)]
            if len(finite) == 0:
                output[f"{metric}_mean"] = math.nan
                output[f"{metric}_std"] = math.nan
            else:
                output[f"{metric}_mean"] = float(finite.mean())
                output[f"{metric}_std"] = (
                    0.0 if np.all(finite == finite[0]) else float(finite.std(ddof=0))
                )
        hull_passes = sum(
            bool(case.metrics["same_skeleton_verified"]) for case in group
        )
        bound_passes = sum(
            bool(case.metrics["local_bound_satisfied_with_tolerance"]) for case in group
        )
        output["same_skeleton_passes"] = float(hull_passes)
        output["same_skeleton_pass_rate"] = hull_passes / len(group)
        output["local_bound_passes"] = float(bound_passes)
        output["local_bound_pass_rate"] = bound_passes / len(group)
        output["local_bound_violations"] = float(len(group) - bound_passes)
        summary.append(output)
    return summary


def write_latex_summary_table(
    path: Path,
    summary: Sequence[dict[str, float | str | bool]],
    japanese: bool,
) -> None:
    if japanese:
        header = (
            "条件 & $\\|\\Delta\\ell\\|_2$ & $\\|\\widehat X-X\\|_F$ "
            "& 局所上界 & 比 & Hull \\\\"
        )
        caption = (
            "ノイズ下の同一骨格剛性監査。Frobenius誤差は最適な $E(3)$ 整列後、"
            "上界は $2\\|\\Delta\\ell\\|_2/s_X$、比は誤差/上界である。"
            "数値は平均 $\\pm$ 標準偏差、Hullは全頂点・facet一致数を表す。"
        )
        shape_labels = {"sphere": "球面", "ellipsoid": "楕円体"}
        label = "tab:convex-edge-rigidity-ja"
    else:
        header = (
            "Case & $\\|\\Delta\\ell\\|_2$ & $\\|\\widehat X-X\\|_F$ "
            "& Local bound & Ratio & Hull \\\\"
        )
        caption = (
            "Noisy local same-skeleton rigidity audit (mean $\\pm$ standard deviation). "
            "Frobenius error follows optimal $E(3)$ alignment, the bound is "
            "$2\\|\\Delta\\ell\\|_2/s_X$, and Ratio is error/bound. Hull reports "
            "trials retaining every vertex and input facet."
        )
        shape_labels = {"sphere": "Sphere", "ellipsoid": "Ellipsoid"}
        label = "tab:convex-edge-rigidity"

    def mean_std(row: dict[str, float | str | bool], metric: str) -> str:
        mean = float(row[f"{metric}_mean"])
        standard_deviation = float(row[f"{metric}_std"])
        if not math.isfinite(mean):
            return "--"
        return f"{mean:.3g} $\\pm$ {standard_deviation:.2g}"

    lines = [
        "\\begin{table}[t]",
        "\\centering",
        "\\small",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\resizebox{\\linewidth}{!}{%",
        "\\begin{tabular}{@{}lccccc@{}}",
        "\\toprule",
        header,
        "\\midrule",
    ]
    for row in summary:
        if float(row["relative_edge_noise"]) == 0.0:
            continue
        trials = int(float(row["trials"]))
        hull_passes = int(float(row["same_skeleton_passes"]))
        lines.append(
            " & ".join(
                [
                    (
                        f"{shape_labels[str(row['shape'])]} "
                        f"{100.0 * float(row['relative_edge_noise']):.1f}\\%"
                    ),
                    mean_std(row, "edge_noise_l2"),
                    mean_std(row, "aligned_frobenius_error"),
                    mean_std(row, "local_rigidity_bound"),
                    mean_std(row, "error_to_bound_ratio"),
                    f"{hull_passes}/{trials}",
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}%", "}", "\\end{table}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_cases(
    cases: Sequence[EvaluatedCase],
    faces: np.ndarray,
    output_dir: Path,
    japanese: bool,
) -> None:
    font = spherical_experiment.japanese_font_family() if japanese else "DejaVu Sans"
    suffix = "_ja" if japanese else ""
    labels = (
        {
            "truth": "正解凸面",
            "initial": "球面MDS初期値のみ",
            "exact": "辺長実現：無雑音",
            "noisy": "辺長実現：0.5%ノイズ",
            "caption": "最終段はmesh辺の $W_2$ 弦長残差のみ（半径制約なし）",
            "rmse": "整列後RMSE",
        }
        if japanese
        else {
            "truth": "ground-truth convex surface",
            "initial": "spherical MDS initializer only",
            "exact": "edge realization: exact",
            "noisy": "edge realization: 0.5% noise",
            "caption": (
                "Final stage uses mesh-edge $W_2$ chord residuals only "
                "(no radius constraint)"
            ),
            "rmse": "aligned RMSE",
        }
    )
    by_key = {
        (case.shape, case.noise_level): case for case in cases if case.seed_index == 0
    }

    with plt.rc_context({"font.family": font, "font.size": 8.5}):
        figure = plt.figure(figsize=(12.0, 3.25), constrained_layout=True)
        grid = figure.add_gridspec(1, 4, wspace=0.02)
        exact = by_key[("ellipsoid", 0.0)]
        noisy = by_key[("ellipsoid", 0.005)]
        panels = (
            (exact.truth, labels["truth"], None),
            (
                exact.aligned_initializer,
                labels["initial"],
                float(exact.metrics["initializer_e3_rmse"]),
            ),
            (
                exact.aligned_reconstruction,
                labels["exact"],
                float(exact.metrics["final_e3_rmse"]),
            ),
            (
                noisy.aligned_reconstruction,
                labels["noisy"],
                float(noisy.metrics["final_e3_rmse"]),
            ),
        )
        color = exact.truth[:, 2]
        for column, (coordinates, title, rmse) in enumerate(panels):
            axis = figure.add_subplot(grid[0, column], projection="3d")
            spherical_experiment.add_mesh_surface(
                axis,
                coordinates,
                faces,
                color,
                alpha=0.96,
                linewidth=0.13,
            )
            spherical_experiment.set_equal_3d_axes(axis, limit=1.32)
            axis.view_init(elev=22, azim=-56)
            axis.set_title(title, pad=3, fontsize=8.7)
            if rmse is not None:
                axis.text2D(
                    0.5,
                    0.02,
                    f"{labels['rmse']} = {rmse:.4f}",
                    transform=axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7.8,
                    bbox={
                        "facecolor": "white",
                        "edgecolor": "#cbd5e1",
                        "alpha": 0.88,
                    },
                )
        figure.suptitle(labels["caption"], fontsize=11.2)
        spherical_experiment.save_figure(
            figure,
            output_dir / f"convex_edge_realization_prototype{suffix}",
        )


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=HERE,
        help="directory for prototype CSV and figures",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    output_dir = arguments.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    faces, edges, shapes = generated_shapes()
    cases: list[EvaluatedCase] = []
    history_rows: list[dict[str, float | str | bool]] = []

    for shape, truth in shapes.items():
        true_hull = ConvexHull(truth)
        true_facets = {tuple(sorted(face)) for face in true_hull.simplices.tolist()}
        expected_facets = {tuple(sorted(face)) for face in faces.tolist()}
        if set(true_hull.vertices.tolist()) != set(range(len(truth))):
            raise RuntimeError(f"{shape} ground truth lost a convex-hull vertex")
        if true_facets != expected_facets:
            raise RuntimeError(f"{shape} ground truth changed the mesh facets")

        for noise_level in NOISE_LEVELS:
            seed_indices = (0,) if noise_level == 0.0 else NOISY_SEEDS
            for seed_index in seed_indices:
                case = evaluate_case(
                    shape,
                    truth,
                    faces,
                    edges,
                    noise_level,
                    seed_index,
                )
                cases.append(case)
                for row in case.realization.history:
                    history_rows.append(
                        {
                            "shape": shape,
                            "relative_edge_noise": noise_level,
                            "seed": float(seed_index),
                            **row,
                        }
                    )
                print(
                    f"{shape:9s} noise={100.0 * noise_level:4.1f}% "
                    f"seed={seed_index} "
                    f"edge_rel={case.realization.final_edge_relative_rmse:.3e} "
                    f"E3_RMSE={float(case.metrics['final_e3_rmse']):.5f} "
                    f"bound_ratio={float(case.metrics['error_to_bound_ratio']):.3g} "
                    f"hull={case.realization.hull_vertex_count}V/"
                    f"{case.realization.hull_facet_count}F match="
                    f"{case.realization.convex_hull_facet_match}",
                    flush=True,
                )

    summary = aggregate_summary(cases)

    write_dict_csv(
        output_dir / "convex_edge_realization_prototype_results.csv",
        [case.metrics for case in cases],
    )
    write_dict_csv(
        output_dir / "convex_edge_realization_prototype_history.csv",
        history_rows,
    )
    write_dict_csv(
        output_dir / "convex_edge_realization_prototype_summary.csv",
        summary,
    )
    write_latex_summary_table(
        output_dir / "convex_edge_realization_prototype_table.tex",
        summary,
        japanese=False,
    )
    write_latex_summary_table(
        output_dir / "convex_edge_realization_prototype_table_ja.tex",
        summary,
        japanese=True,
    )
    for japanese in (False, True):
        plot_cases(cases, faces, output_dir, japanese=japanese)

    noisy_cases = [case for case in cases if case.noise_level > 0.0]
    violations = [
        case
        for case in noisy_cases
        if not bool(case.metrics["local_bound_satisfied_with_tolerance"])
    ]
    if violations:
        print(
            "WARNING: the finite-noise local rigidity estimate was violated in "
            f"{len(violations)}/{len(noisy_cases)} trials. The estimate uses "
            "the derivative at the truth and is only a first-order local "
            "prediction; finite perturbations add nonlinear remainder terms.",
            flush=True,
        )
        for case in violations:
            print(
                f"  {case.shape} noise={100.0 * case.noise_level:.1f}% "
                f"seed={case.seed_index} "
                f"ratio={float(case.metrics['error_to_bound_ratio']):.6f}",
                flush=True,
            )
    else:
        maximum_ratio = max(
            float(case.metrics["error_to_bound_ratio"]) for case in noisy_cases
        )
        print(
            "All finite-noise trials satisfy the audited local estimate; "
            f"maximum error/bound ratio={maximum_ratio:.6f}.",
            flush=True,
        )
    print(f"Wrote convex-realization prototype outputs to {output_dir}", flush=True)


if __name__ == "__main__":
    main()
