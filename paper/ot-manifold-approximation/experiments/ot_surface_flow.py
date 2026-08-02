#!/usr/bin/env python3
# /// script
# requires-python = ">=3.12,<3.13"
# dependencies = [
#   "numpy==2.5.1",
#   "torch==2.13.0",
# ]
# ///
"""Topology-preserving neural realization of a two-dimensional OT metric.

The input is a triangular chart ``u_i in R^2`` and three target lengths for
each face.  Those lengths first define a raw piecewise-constant Riemannian
metric.  A weighted face-adjacency quadratic estimator smooths its matrix
logarithm and then exponentiates the result, preserving positive definiteness.
This finite-to-continuous estimation stage is disabled exactly by setting its
strength to zero.  The module then fits a smooth map

    F(u) = Phi(s * normalized(u)_1, s * normalized(u)_2, 0),

where ``Phi: R^3 -> R^3`` is a composition of invertible affine-coupling
blocks.  Consequently, the restriction of ``Phi`` to the lifted chart is an
embedded disk whenever the input chart is an embedded disk.  In particular,
the architecture cannot create a self-intersection by identifying two chart
points.  Unlike a height field, later coupling blocks can change every output
coordinate and can create overhangs.

The primary loss compares the smoothed target metric ``G`` to the pullback metric
``J_F.T @ J_F`` using squared logarithmic generalized stretches.  Optimization
has two explicit stages:

1. find a metric-faithful realization;
2. minimize adjacent-face normal bending while retaining the first-stage
   isometry loss within a declared tolerance.

The three-dimensional shape is therefore a topology-preserving, low-bending
representative of the intrinsic metric.  It is not claimed to be a unique
extrinsic ground-truth shape.

Run the small deterministic controlled check with

    uv run --python 3.12 ot_surface_flow.py --self-test
"""

from __future__ import annotations

import argparse
import math
import random
from dataclasses import dataclass
from typing import Final

import numpy as np
import torch
from torch import Tensor, nn

DEFAULT_DTYPE: Final = torch.float64
NUMERICAL_EPSILON: Final = 1.0e-12


@dataclass(frozen=True)
class SurfaceFlowConfig:
    """Deterministic architecture and optimizer settings."""

    seed: int = 20260802
    hidden_width: int = 40
    coupling_blocks: int = 9
    coupling_scale_limit: float = 0.40
    near_identity_noise: float = 2.0e-3
    metric_smoothing_strength: float = 1.0
    stage_one_steps: int = 1_200
    stage_two_steps: int = 900
    stage_one_learning_rate: float = 5.0e-3
    stage_two_learning_rate: float = 2.0e-3
    metric_relative_slack: float = 0.08
    metric_absolute_slack: float = 1.0e-7
    metric_constraint_weight: float = 100.0
    metric_floor_weight: float = 1.0e-3
    gradient_norm_clip: float = 20.0
    record_every: int = 20

    def __post_init__(self) -> None:
        if self.hidden_width < 2:
            raise ValueError("hidden_width must be at least two")
        if self.coupling_blocks < 3:
            raise ValueError("at least three coupling blocks are required")
        if not 0.0 < self.coupling_scale_limit <= 2.0:
            raise ValueError("coupling_scale_limit must lie in (0, 2]")
        if self.near_identity_noise < 0.0:
            raise ValueError("near_identity_noise must be nonnegative")
        if (
            not math.isfinite(self.metric_smoothing_strength)
            or self.metric_smoothing_strength < 0.0
        ):
            raise ValueError("metric_smoothing_strength must be finite and nonnegative")
        if self.stage_one_steps < 1 or self.stage_two_steps < 0:
            raise ValueError("invalid optimizer step count")
        if self.stage_one_learning_rate <= 0.0:
            raise ValueError("stage_one_learning_rate must be positive")
        if self.stage_two_steps > 0 and self.stage_two_learning_rate <= 0.0:
            raise ValueError("stage_two_learning_rate must be positive")
        if self.metric_relative_slack < 0.0:
            raise ValueError("metric_relative_slack must be nonnegative")
        if self.metric_absolute_slack < 0.0:
            raise ValueError("metric_absolute_slack must be nonnegative")
        if self.metric_constraint_weight <= 0.0:
            raise ValueError("metric_constraint_weight must be positive")
        if self.record_every < 1:
            raise ValueError("record_every must be positive")


@dataclass(frozen=True)
class FaceMetricData:
    """Target first fundamental forms induced by triangle edge lengths."""

    centroids: np.ndarray
    metrics: np.ndarray
    edge_grams: np.ndarray
    target_areas: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class OptimizationRecord:
    """One compact, deterministic optimization diagnostic."""

    stage: str
    step: int
    objective: float
    isometry_loss: float
    bending_loss: float
    metric_feasible: bool


@dataclass(frozen=True)
class DenseSurfaceEvaluation:
    """A per-face subdivision evaluated by a fitted smooth surface map."""

    chart_coordinates: np.ndarray
    coordinates: np.ndarray
    faces: np.ndarray
    parent_faces: np.ndarray


@dataclass(frozen=True)
class SurfaceFlowResult:
    """Fitted map and all values needed to reproduce its selection."""

    model: TopologyPreservingSurface
    coordinates: np.ndarray
    raw_target_face_metrics: np.ndarray
    smoothed_target_face_metrics: np.ndarray
    target_face_metrics: np.ndarray
    face_weights: np.ndarray
    metric_smoothing_strength: float
    raw_metric_log_roughness: float
    smoothed_metric_log_roughness: float
    initial_isometry_loss: float
    stage_one_isometry_loss: float
    stage_two_metric_threshold: float
    final_isometry_loss: float
    final_bending_loss: float
    generalized_stretch_min: float
    generalized_stretch_max: float
    history: tuple[OptimizationRecord, ...]


def seed_deterministically(seed: int) -> None:
    """Configure repeatable CPU execution for this small full-batch problem."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def _as_chart_array(
    chart_coordinates: np.ndarray, minimum_count: int = 3
) -> np.ndarray:
    chart = np.asarray(chart_coordinates, dtype=np.float64)
    if chart.ndim != 2 or chart.shape[1] != 2:
        raise ValueError("chart_coordinates must have shape (vertices, 2)")
    if len(chart) < minimum_count or not np.isfinite(chart).all():
        raise ValueError("chart_coordinates must contain finite coordinates")
    return chart


def _as_face_array(faces: np.ndarray, vertex_count: int) -> np.ndarray:
    result = np.asarray(faces, dtype=np.int64)
    if result.ndim != 2 or result.shape[1] != 3 or len(result) < 1:
        raise ValueError("faces must have shape (face_count, 3)")
    if np.min(result) < 0 or np.max(result) >= vertex_count:
        raise ValueError("face vertex index is out of range")
    if np.any(
        (result[:, 0] == result[:, 1])
        | (result[:, 1] == result[:, 2])
        | (result[:, 2] == result[:, 0])
    ):
        raise ValueError("a triangular face cannot repeat a vertex")
    return result


def _as_face_lengths(
    target_face_edge_lengths: np.ndarray, face_count: int
) -> np.ndarray:
    lengths = np.asarray(target_face_edge_lengths, dtype=np.float64)
    if lengths.shape != (face_count, 3):
        raise ValueError("target_face_edge_lengths must have shape (face_count, 3)")
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("all target edge lengths must be finite and positive")
    return lengths


def target_edge_grams(target_face_edge_lengths: np.ndarray) -> np.ndarray:
    """Return target edge Gram matrices from lengths ``(l01,l12,l20)``.

    The two columns represented by each Gram matrix are the target edges from
    face vertex zero to vertices one and two.
    """

    lengths = np.asarray(target_face_edge_lengths, dtype=np.float64)
    if lengths.ndim != 2 or lengths.shape[1] != 3:
        raise ValueError("target lengths must have shape (face_count, 3)")
    if not np.isfinite(lengths).all() or np.any(lengths <= 0.0):
        raise ValueError("all target edge lengths must be finite and positive")
    first, opposite, second = lengths.T
    if np.any(first + opposite <= second):
        raise ValueError("target lengths violate a strict triangle inequality")
    if np.any(first + second <= opposite):
        raise ValueError("target lengths violate a strict triangle inequality")
    if np.any(opposite + second <= first):
        raise ValueError("target lengths violate a strict triangle inequality")

    cross = 0.5 * (first**2 + second**2 - opposite**2)
    grams = np.empty((len(lengths), 2, 2), dtype=np.float64)
    grams[:, 0, 0] = first**2
    grams[:, 0, 1] = cross
    grams[:, 1, 0] = cross
    grams[:, 1, 1] = second**2
    eigenvalues = np.linalg.eigvalsh(grams)
    if np.any(eigenvalues[:, 0] <= NUMERICAL_EPSILON):
        raise ValueError("a target triangle is numerically degenerate")
    return grams


def target_face_metrics(
    chart_coordinates: np.ndarray,
    faces: np.ndarray,
    target_face_edge_lengths: np.ndarray,
    face_confidence: np.ndarray | None = None,
) -> FaceMetricData:
    """Convert target triangle lengths to metrics in the supplied chart.

    For a face ``f=(i,j,k)``, let ``D=[u_j-u_i,u_k-u_i]`` and let ``H`` be
    the target edge Gram matrix.  The chart-coordinate metric is

        G_f = D^{-T} H D^{-1}.

    ``face_confidence`` can downweight extrapolative or low-density faces.  It
    is multiplied by target triangle area and normalized to sum to one.
    """

    chart = _as_chart_array(chart_coordinates)
    triangles = _as_face_array(faces, len(chart))
    lengths = _as_face_lengths(target_face_edge_lengths, len(triangles))
    grams = target_edge_grams(lengths)

    first = chart[triangles[:, 1]] - chart[triangles[:, 0]]
    second = chart[triangles[:, 2]] - chart[triangles[:, 0]]
    bases = np.stack([first, second], axis=2)
    determinants = np.linalg.det(bases)
    if np.any(np.abs(determinants) <= NUMERICAL_EPSILON):
        raise ValueError("the input chart contains a degenerate face")
    if np.any(determinants < 0.0) and np.any(determinants > 0.0):
        raise ValueError("input faces must have a consistent orientation")
    inverse = np.linalg.inv(bases)
    metrics = np.einsum("fai,fab,fbj->fij", inverse, grams, inverse, optimize=True)
    metrics = 0.5 * (metrics + np.swapaxes(metrics, 1, 2))
    if np.any(np.linalg.eigvalsh(metrics)[:, 0] <= NUMERICAL_EPSILON):
        raise ValueError("the induced chart metric is numerically singular")

    target_areas = 0.5 * np.sqrt(np.linalg.det(grams))
    if face_confidence is None:
        confidence = np.ones(len(triangles), dtype=np.float64)
    else:
        confidence = np.asarray(face_confidence, dtype=np.float64)
        if confidence.shape != (len(triangles),):
            raise ValueError("face_confidence must have shape (face_count,)")
        if np.any(confidence < 0.0) or not np.isfinite(confidence).all():
            raise ValueError("face_confidence must be finite and nonnegative")
    raw_weights = target_areas * confidence
    if float(np.sum(raw_weights)) <= NUMERICAL_EPSILON:
        raise ValueError("face weights have zero total mass")
    weights = raw_weights / np.sum(raw_weights)
    centroids = np.mean(chart[triangles], axis=1)
    return FaceMetricData(
        centroids=centroids,
        metrics=metrics,
        edge_grams=grams,
        target_areas=target_areas,
        weights=weights,
    )


def optimally_scaled_plane(
    normalized_chart: np.ndarray,
    faces: np.ndarray,
    target_face_edge_lengths: np.ndarray,
) -> float:
    """Return the positive plane scale minimizing relative edge stress."""

    chart = _as_chart_array(normalized_chart)
    triangles = _as_face_array(faces, len(chart))
    targets = _as_face_lengths(target_face_edge_lengths, len(triangles))
    base_lengths = np.column_stack(
        [
            np.linalg.norm(chart[triangles[:, 1]] - chart[triangles[:, 0]], axis=1),
            np.linalg.norm(chart[triangles[:, 2]] - chart[triangles[:, 1]], axis=1),
            np.linalg.norm(chart[triangles[:, 0]] - chart[triangles[:, 2]], axis=1),
        ]
    )
    ratios = (base_lengths / targets).ravel()
    denominator = float(np.dot(ratios, ratios))
    if denominator <= NUMERICAL_EPSILON:
        raise ValueError("the chart has no nonzero face edges")
    return float(np.sum(ratios) / denominator)


class CouplingConditioner(nn.Module):
    """Small smooth network producing log-scale and translation."""

    def __init__(self, hidden_width: int, near_identity_noise: float) -> None:
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(2, hidden_width, dtype=DEFAULT_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden_width, hidden_width, dtype=DEFAULT_DTYPE),
            nn.Tanh(),
            nn.Linear(hidden_width, 2, dtype=DEFAULT_DTYPE),
        )
        final = self.network[-1]
        if not isinstance(final, nn.Linear):
            raise TypeError("unexpected conditioner architecture")
        with torch.no_grad():
            final.weight.normal_(mean=0.0, std=near_identity_noise)
            final.bias.zero_()

    def forward(self, inputs: Tensor) -> Tensor:
        return self.network(inputs)


class AffineCouplingBlock(nn.Module):
    """Invert one coordinate conditioned on the other two coordinates."""

    def __init__(
        self,
        transformed_coordinate: int,
        hidden_width: int,
        scale_limit: float,
        near_identity_noise: float,
    ) -> None:
        super().__init__()
        if transformed_coordinate not in (0, 1, 2):
            raise ValueError("transformed_coordinate must be zero, one, or two")
        self.transformed_coordinate = transformed_coordinate
        self.passive_coordinates = tuple(
            index for index in range(3) if index != transformed_coordinate
        )
        self.scale_limit = float(scale_limit)
        self.conditioner = CouplingConditioner(
            hidden_width=hidden_width,
            near_identity_noise=near_identity_noise,
        )

    def _coupling_parameters(self, coordinates: Tensor) -> tuple[Tensor, Tensor]:
        passive = coordinates[..., list(self.passive_coordinates)]
        raw = self.conditioner(passive)
        log_scale = self.scale_limit * torch.tanh(raw[..., 0])
        translation = raw[..., 1]
        return log_scale, translation

    def forward(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != 3:
            raise ValueError("ambient coordinates must end in dimension three")
        log_scale, translation = self._coupling_parameters(coordinates)
        transformed = (
            coordinates[..., self.transformed_coordinate] * torch.exp(log_scale)
            + translation
        )
        components = [coordinates[..., index] for index in range(3)]
        components[self.transformed_coordinate] = transformed
        return torch.stack(components, dim=-1)

    def inverse(self, coordinates: Tensor) -> Tensor:
        if coordinates.shape[-1] != 3:
            raise ValueError("ambient coordinates must end in dimension three")
        # Passive coordinates are unchanged, so the conditioner has the same
        # input in the forward and inverse directions.
        log_scale, translation = self._coupling_parameters(coordinates)
        transformed = (
            coordinates[..., self.transformed_coordinate] - translation
        ) * torch.exp(-log_scale)
        components = [coordinates[..., index] for index in range(3)]
        components[self.transformed_coordinate] = transformed
        return torch.stack(components, dim=-1)


class AmbientCouplingDiffeomorphism(nn.Module):
    """A smooth, analytically invertible map ``Phi: R^3 -> R^3``."""

    def __init__(self, config: SurfaceFlowConfig) -> None:
        super().__init__()
        # First create an out-of-plane displacement; subsequent blocks can
        # change x and y conditional on that displacement, so this is not a
        # graph-valued height-only architecture.
        transformed_order = (2, 0, 1)
        self.blocks = nn.ModuleList(
            [
                AffineCouplingBlock(
                    transformed_coordinate=transformed_order[
                        index % len(transformed_order)
                    ],
                    hidden_width=config.hidden_width,
                    scale_limit=config.coupling_scale_limit,
                    near_identity_noise=config.near_identity_noise,
                )
                for index in range(config.coupling_blocks)
            ]
        )

    def forward(self, coordinates: Tensor) -> Tensor:
        result = coordinates
        for block in self.blocks:
            result = block(result)
        return result

    def inverse(self, coordinates: Tensor) -> Tensor:
        result = coordinates
        for block in reversed(self.blocks):
            result = block.inverse(result)
        return result


class TopologyPreservingSurface(nn.Module):
    """Lift a two-dimensional chart and deform it by an ambient diffeomorphism."""

    def __init__(
        self,
        chart_center: np.ndarray,
        chart_scale: float,
        initial_plane_scale: float,
        config: SurfaceFlowConfig,
    ) -> None:
        super().__init__()
        center = np.asarray(chart_center, dtype=np.float64)
        if center.shape != (2,):
            raise ValueError("chart_center must have shape (2,)")
        if chart_scale <= 0.0 or initial_plane_scale <= 0.0:
            raise ValueError("chart and initial plane scales must be positive")
        self.register_buffer(
            "chart_center", torch.as_tensor(center, dtype=DEFAULT_DTYPE)
        )
        self.register_buffer(
            "chart_scale", torch.tensor(float(chart_scale), dtype=DEFAULT_DTYPE)
        )
        self.register_buffer(
            "initial_plane_scale",
            torch.tensor(float(initial_plane_scale), dtype=DEFAULT_DTYPE),
        )
        self.ambient_map = AmbientCouplingDiffeomorphism(config)

    def lift(self, chart_coordinates: Tensor) -> Tensor:
        if chart_coordinates.shape[-1] != 2:
            raise ValueError("chart coordinates must end in dimension two")
        normalized = (chart_coordinates - self.chart_center) / self.chart_scale
        planar = self.initial_plane_scale * normalized
        zero = torch.zeros_like(planar[..., :1])
        return torch.cat([planar, zero], dim=-1)

    def forward(self, chart_coordinates: Tensor) -> Tensor:
        return self.ambient_map(self.lift(chart_coordinates))


def surface_jacobians(
    model: TopologyPreservingSurface, chart_coordinates: Tensor
) -> tuple[Tensor, Tensor]:
    """Evaluate a pointwise surface map and its ``3 by 2`` Jacobians."""

    if chart_coordinates.ndim != 2 or chart_coordinates.shape[1] != 2:
        raise ValueError("chart_coordinates must have shape (count, 2)")
    points = chart_coordinates
    if not points.requires_grad:
        points = points.detach().clone().requires_grad_(True)
    coordinates = model(points)
    rows: list[Tensor] = []
    for output_coordinate in range(3):
        gradient = torch.autograd.grad(
            coordinates[:, output_coordinate].sum(),
            points,
            create_graph=True,
            retain_graph=True,
        )[0]
        rows.append(gradient)
    return coordinates, torch.stack(rows, dim=1)


def pullback_metrics(jacobians: Tensor) -> Tensor:
    """Return ``J.T @ J`` for a batch of ``3 by 2`` Jacobians."""

    if jacobians.ndim != 3 or jacobians.shape[1:] != (3, 2):
        raise ValueError("jacobians must have shape (count, 3, 2)")
    return torch.einsum("nra,nrb->nab", jacobians, jacobians)


def normalized_metric_eigenvalues(
    predicted_metrics: Tensor, target_metrics: Tensor
) -> Tensor:
    """Return squared generalized length stretches of two SPD metrics."""

    if predicted_metrics.shape != target_metrics.shape:
        raise ValueError("predicted and target metric batches must have equal shape")
    cholesky = torch.linalg.cholesky(target_metrics)
    left = torch.linalg.solve_triangular(cholesky, predicted_metrics, upper=False)
    relative = torch.linalg.solve_triangular(
        cholesky, left.transpose(-1, -2), upper=False
    ).transpose(-1, -2)
    relative = 0.5 * (relative + relative.transpose(-1, -2))
    return torch.linalg.eigvalsh(relative).clamp_min(NUMERICAL_EPSILON)


def metric_isometry_loss(
    model: TopologyPreservingSurface,
    chart_samples: Tensor,
    target_metrics: Tensor,
    weights: Tensor,
) -> tuple[Tensor, Tensor]:
    """Compare ``F*e`` and the target metric by log generalized stretches."""

    _, jacobians = surface_jacobians(model, chart_samples)
    predicted = pullback_metrics(jacobians)
    eigenvalues = normalized_metric_eigenvalues(predicted, target_metrics)
    # Eigenvalues are squared length stretches.  Multiplication by one quarter
    # converts squared log eigenvalues to squared log length stretches.
    per_sample = 0.25 * torch.sum(torch.log(eigenvalues).square(), dim=1)
    normalized_weights = weights / weights.sum().clamp_min(NUMERICAL_EPSILON)
    return torch.sum(normalized_weights * per_sample), eigenvalues


def adjacent_face_pairs(faces: np.ndarray) -> np.ndarray:
    """Return all face pairs sharing an interior edge."""

    triangles = np.asarray(faces, dtype=np.int64)
    incident: dict[tuple[int, int], list[int]] = {}
    for face_index, (first, second, third) in enumerate(triangles.tolist()):
        for left, right in ((first, second), (second, third), (third, first)):
            edge = tuple(sorted((int(left), int(right))))
            incident.setdefault(edge, []).append(face_index)
    pairs = [indices for indices in incident.values() if len(indices) == 2]
    if not pairs:
        return np.empty((0, 2), dtype=np.int64)
    return np.asarray(pairs, dtype=np.int64)


def _symmetric_matrix_function(
    matrices: np.ndarray, function, require_positive: bool
) -> np.ndarray:
    """Apply a scalar spectral function to symmetric ``2 by 2`` matrices."""

    values = np.asarray(matrices, dtype=np.float64)
    if values.ndim != 3 or values.shape[1:] != (2, 2):
        raise ValueError("metric matrices must have shape (count, 2, 2)")
    if not np.isfinite(values).all():
        raise ValueError("metric matrices must be finite")
    symmetric = 0.5 * (values + np.swapaxes(values, 1, 2))
    eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
    if require_positive and np.any(eigenvalues <= NUMERICAL_EPSILON):
        raise ValueError("metric matrices must be numerically positive definite")
    transformed = function(eigenvalues)
    result = np.einsum(
        "fiq,fq,fjq->fij",
        eigenvectors,
        transformed,
        eigenvectors,
        optimize=True,
    )
    return 0.5 * (result + np.swapaxes(result, 1, 2))


def metric_matrix_logs(metrics: np.ndarray) -> np.ndarray:
    """Return the principal matrix logarithm of an SPD metric batch."""

    return _symmetric_matrix_function(metrics, np.log, require_positive=True)


def metric_matrix_exponentials(log_metrics: np.ndarray) -> np.ndarray:
    """Exponentiate a batch of finite symmetric log-metrics."""

    return _symmetric_matrix_function(log_metrics, np.exp, require_positive=False)


def metric_log_roughness(metrics: np.ndarray, faces: np.ndarray) -> float:
    """Return mean adjacent-face squared Frobenius log-metric variation."""

    log_metrics = metric_matrix_logs(metrics)
    adjacent = adjacent_face_pairs(faces)
    if len(adjacent) == 0:
        return 0.0
    differences = log_metrics[adjacent[:, 0]] - log_metrics[adjacent[:, 1]]
    return float(np.mean(np.sum(differences * differences, axis=(1, 2))))


def smooth_face_metrics_log_euclidean(
    raw_metrics: np.ndarray,
    faces: np.ndarray,
    face_weights: np.ndarray,
    strength: float,
) -> np.ndarray:
    r"""Estimate a continuous metric field by weighted log-SPD smoothing.

    Let ``S_f=log(G_f)`` be the raw finite-OT metric on face ``f``.  For each
    independent symmetric component, this function solves the strictly convex
    quadratic problem

    .. math::

       \min_X \sum_f w_f\|X_f-S_f\|_F^2
       + \lambda\sum_{f\sim g}w_{fg}\|X_f-X_g\|_F^2,

    with harmonic adjacency weight
    ``w_fg=2*w_f*w_g/(w_f+w_g)``.  The output is ``exp(X_f)`` and is therefore
    SPD by construction.  ``strength=0`` returns an exact copy of the raw
    metrics, making the finite-to-continuous estimator explicitly optional.
    """

    if not math.isfinite(strength) or strength < 0.0:
        raise ValueError("metric smoothing strength must be finite and nonnegative")
    metrics = np.asarray(raw_metrics, dtype=np.float64)
    logs = metric_matrix_logs(metrics)
    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape != (len(metrics), 3):
        raise ValueError("faces must have shape (metric_count, 3)")
    weights = np.asarray(face_weights, dtype=np.float64)
    if weights.shape != (len(metrics),):
        raise ValueError("face_weights must have shape (metric_count,)")
    if not np.isfinite(weights).all() or np.any(weights < 0.0):
        raise ValueError("face_weights must be finite and nonnegative")
    total_weight = float(np.sum(weights))
    if total_weight <= NUMERICAL_EPSILON:
        raise ValueError("face_weights must have positive total mass")
    if strength == 0.0:
        return metrics.copy()

    normalized_weights = weights / total_weight
    positive = normalized_weights[normalized_weights > 0.0]
    weight_floor = max(float(np.mean(positive)) * 1.0e-10, NUMERICAL_EPSILON)
    stable_weights = np.maximum(normalized_weights, weight_floor)
    system = np.diag(stable_weights)
    adjacent = adjacent_face_pairs(triangles)
    for first, second in adjacent:
        first_weight = stable_weights[int(first)]
        second_weight = stable_weights[int(second)]
        adjacency_weight = (
            2.0
            * first_weight
            * second_weight
            / max(first_weight + second_weight, NUMERICAL_EPSILON)
        )
        coefficient = strength * adjacency_weight
        system[first, first] += coefficient
        system[second, second] += coefficient
        system[first, second] -= coefficient
        system[second, first] -= coefficient

    # The Frobenius factor of two on the symmetric off-diagonal component is
    # present in both quadratic terms and cancels from its normal equations.
    components = np.column_stack([logs[:, 0, 0], logs[:, 0, 1], logs[:, 1, 1]])
    right_hand_side = stable_weights[:, None] * components
    smoothed_components = np.linalg.solve(system, right_hand_side)
    smoothed_logs = np.empty_like(logs)
    smoothed_logs[:, 0, 0] = smoothed_components[:, 0]
    smoothed_logs[:, 0, 1] = smoothed_components[:, 1]
    smoothed_logs[:, 1, 0] = smoothed_components[:, 1]
    smoothed_logs[:, 1, 1] = smoothed_components[:, 2]
    smoothed = metric_matrix_exponentials(smoothed_logs)
    if not np.isfinite(smoothed).all():
        raise FloatingPointError("metric smoothing produced non-finite values")
    if np.any(np.linalg.eigvalsh(smoothed)[:, 0] <= NUMERICAL_EPSILON):
        raise FloatingPointError("metric smoothing failed to preserve SPD metrics")
    return smoothed


def mesh_bending_loss(coordinates: Tensor, faces: Tensor, adjacent: Tensor) -> Tensor:
    """Return mean adjacent-face normal bending ``1-cos(theta)``."""

    if len(adjacent) == 0:
        return coordinates.new_zeros(())
    first = coordinates[faces[:, 1]] - coordinates[faces[:, 0]]
    second = coordinates[faces[:, 2]] - coordinates[faces[:, 0]]
    normals = torch.linalg.cross(first, second, dim=1)
    norms = torch.linalg.vector_norm(normals, dim=1, keepdim=True)
    normals = normals / norms.clamp_min(NUMERICAL_EPSILON)
    dots = torch.sum(normals[adjacent[:, 0]] * normals[adjacent[:, 1]], dim=1)
    return torch.mean(1.0 - dots.clamp(-1.0, 1.0))


def _clone_state(model: nn.Module) -> dict[str, Tensor]:
    return {name: value.detach().clone() for name, value in model.state_dict().items()}


def _evaluate_losses(
    model: TopologyPreservingSurface,
    chart_vertices: Tensor,
    face_centroids: Tensor,
    target_metrics: Tensor,
    face_weights: Tensor,
    faces: Tensor,
    adjacent: Tensor,
) -> tuple[Tensor, Tensor, Tensor]:
    isometry, eigenvalues = metric_isometry_loss(
        model, face_centroids, target_metrics, face_weights
    )
    coordinates = model(chart_vertices)
    bending = mesh_bending_loss(coordinates, faces, adjacent)
    return isometry, bending, eigenvalues


def _record(
    history: list[OptimizationRecord],
    stage: str,
    step: int,
    objective: Tensor,
    isometry: Tensor,
    bending: Tensor,
    threshold: float,
) -> None:
    history.append(
        OptimizationRecord(
            stage=stage,
            step=step,
            objective=float(objective.detach()),
            isometry_loss=float(isometry.detach()),
            bending_loss=float(bending.detach()),
            metric_feasible=float(isometry.detach()) <= threshold,
        )
    )


def fit_surface_flow(
    chart_coordinates: np.ndarray,
    faces: np.ndarray,
    target_face_edge_lengths: np.ndarray,
    config: SurfaceFlowConfig | None = None,
    face_confidence: np.ndarray | None = None,
) -> SurfaceFlowResult:
    """Fit a topology-preserving low-bending realization of an OT metric.

    Args:
        chart_coordinates: Vertex coordinates of an oriented triangular chart.
        faces: Vertex indices with shape ``(face_count, 3)``.
        target_face_edge_lengths: Lengths ``(l01,l12,l20)`` for every face.
        config: Reproducible architecture and optimization settings.
        face_confidence: Optional nonnegative reliability per target face.

    Returns:
        A frozen result record.  Its ``model`` maps arbitrary new chart points,
        while ``coordinates`` contains the fitted input vertices.
    """

    settings = SurfaceFlowConfig() if config is None else config
    seed_deterministically(settings.seed)
    chart = _as_chart_array(chart_coordinates)
    triangles = _as_face_array(faces, len(chart))
    lengths = _as_face_lengths(target_face_edge_lengths, len(triangles))
    metric_data = target_face_metrics(
        chart, triangles, lengths, face_confidence=face_confidence
    )
    raw_metrics = metric_data.metrics
    smoothed_metrics = smooth_face_metrics_log_euclidean(
        raw_metrics,
        triangles,
        metric_data.weights,
        strength=settings.metric_smoothing_strength,
    )
    raw_roughness = metric_log_roughness(raw_metrics, triangles)
    smoothed_roughness = metric_log_roughness(smoothed_metrics, triangles)

    chart_center = np.mean(chart, axis=0)
    centered = chart - chart_center
    chart_scale = math.sqrt(float(np.mean(np.sum(centered * centered, axis=1))) / 2.0)
    if chart_scale <= NUMERICAL_EPSILON:
        raise ValueError("the chart has zero numerical scale")
    normalized = centered / chart_scale
    initial_plane_scale = optimally_scaled_plane(normalized, triangles, lengths)
    model = TopologyPreservingSurface(
        chart_center=chart_center,
        chart_scale=chart_scale,
        initial_plane_scale=initial_plane_scale,
        config=settings,
    ).to(dtype=DEFAULT_DTYPE, device="cpu")

    chart_tensor = torch.as_tensor(chart, dtype=DEFAULT_DTYPE)
    centroid_tensor = torch.as_tensor(metric_data.centroids, dtype=DEFAULT_DTYPE)
    target_tensor = torch.as_tensor(smoothed_metrics, dtype=DEFAULT_DTYPE)
    weight_tensor = torch.as_tensor(metric_data.weights, dtype=DEFAULT_DTYPE)
    face_tensor = torch.as_tensor(triangles, dtype=torch.long)
    adjacent_tensor = torch.as_tensor(adjacent_face_pairs(triangles), dtype=torch.long)

    history: list[OptimizationRecord] = []
    initial_isometry, _initial_bending, _ = _evaluate_losses(
        model,
        chart_tensor,
        centroid_tensor,
        target_tensor,
        weight_tensor,
        face_tensor,
        adjacent_tensor,
    )
    initial_value = float(initial_isometry.detach())

    # Stage one: metric fidelity only.  The invertible architecture supplies
    # the topology constraint without a collision penalty.
    stage_one_optimizer = torch.optim.Adam(
        model.parameters(), lr=settings.stage_one_learning_rate
    )
    best_stage_one_loss = initial_value
    best_stage_one_state = _clone_state(model)
    for step in range(settings.stage_one_steps):
        stage_one_optimizer.zero_grad(set_to_none=True)
        isometry, bending, _ = _evaluate_losses(
            model,
            chart_tensor,
            centroid_tensor,
            target_tensor,
            weight_tensor,
            face_tensor,
            adjacent_tensor,
        )
        objective = isometry
        scalar = float(isometry.detach())
        if math.isfinite(scalar) and scalar < best_stage_one_loss:
            best_stage_one_loss = scalar
            best_stage_one_state = _clone_state(model)
        objective.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), settings.gradient_norm_clip)
        stage_one_optimizer.step()
        if step % settings.record_every == 0 or step + 1 == settings.stage_one_steps:
            _record(
                history,
                "isometry",
                step,
                objective,
                isometry,
                bending,
                math.inf,
            )

    # Include the post-update model in the first-stage selection.
    stage_one_isometry, _, _ = _evaluate_losses(
        model,
        chart_tensor,
        centroid_tensor,
        target_tensor,
        weight_tensor,
        face_tensor,
        adjacent_tensor,
    )
    if float(stage_one_isometry.detach()) < best_stage_one_loss:
        best_stage_one_loss = float(stage_one_isometry.detach())
        best_stage_one_state = _clone_state(model)
    model.load_state_dict(best_stage_one_state)

    metric_threshold = max(
        best_stage_one_loss * (1.0 + settings.metric_relative_slack),
        best_stage_one_loss + settings.metric_absolute_slack,
    )
    constraint_scale = max(metric_threshold, 1.0e-8)

    # Stage two: choose the least-bending state among metric-feasible states.
    # Checkpoint selection, rather than the weighted objective alone, enforces
    # the declared lexicographic convention.
    stage_two_isometry, stage_two_bending, _ = _evaluate_losses(
        model,
        chart_tensor,
        centroid_tensor,
        target_tensor,
        weight_tensor,
        face_tensor,
        adjacent_tensor,
    )
    best_feasible_bending = float(stage_two_bending.detach())
    best_feasible_isometry = float(stage_two_isometry.detach())
    best_feasible_state = _clone_state(model)

    if settings.stage_two_steps > 0:
        stage_two_optimizer = torch.optim.Adam(
            model.parameters(), lr=settings.stage_two_learning_rate
        )
        for step in range(settings.stage_two_steps):
            stage_two_optimizer.zero_grad(set_to_none=True)
            isometry, bending, _ = _evaluate_losses(
                model,
                chart_tensor,
                centroid_tensor,
                target_tensor,
                weight_tensor,
                face_tensor,
                adjacent_tensor,
            )
            normalized_violation = torch.relu(isometry - metric_threshold) / (
                constraint_scale
            )
            objective = (
                bending
                + settings.metric_constraint_weight * normalized_violation.square()
                + settings.metric_floor_weight * isometry / constraint_scale
            )
            scalar_isometry = float(isometry.detach())
            scalar_bending = float(bending.detach())
            if (
                math.isfinite(scalar_isometry)
                and math.isfinite(scalar_bending)
                and scalar_isometry <= metric_threshold
                and (
                    scalar_bending < best_feasible_bending
                    or (
                        scalar_bending == best_feasible_bending
                        and scalar_isometry < best_feasible_isometry
                    )
                )
            ):
                best_feasible_bending = scalar_bending
                best_feasible_isometry = scalar_isometry
                best_feasible_state = _clone_state(model)
            objective.backward()
            torch.nn.utils.clip_grad_norm_(
                model.parameters(), settings.gradient_norm_clip
            )
            stage_two_optimizer.step()
            if (
                step % settings.record_every == 0
                or step + 1 == settings.stage_two_steps
            ):
                _record(
                    history,
                    "bending",
                    step,
                    objective,
                    isometry,
                    bending,
                    metric_threshold,
                )

        # The last optimizer update has not yet appeared at the beginning of
        # another iteration, so include it explicitly in feasible selection.
        post_isometry, post_bending, _ = _evaluate_losses(
            model,
            chart_tensor,
            centroid_tensor,
            target_tensor,
            weight_tensor,
            face_tensor,
            adjacent_tensor,
        )
        scalar_post_isometry = float(post_isometry.detach())
        scalar_post_bending = float(post_bending.detach())
        if (
            math.isfinite(scalar_post_isometry)
            and math.isfinite(scalar_post_bending)
            and scalar_post_isometry <= metric_threshold
            and (
                scalar_post_bending < best_feasible_bending
                or (
                    scalar_post_bending == best_feasible_bending
                    and scalar_post_isometry < best_feasible_isometry
                )
            )
        ):
            best_feasible_state = _clone_state(model)

    model.load_state_dict(best_feasible_state)
    model.eval()
    final_isometry, final_bending, final_eigenvalues = _evaluate_losses(
        model,
        chart_tensor,
        centroid_tensor,
        target_tensor,
        weight_tensor,
        face_tensor,
        adjacent_tensor,
    )
    with torch.no_grad():
        fitted_coordinates = model(chart_tensor).numpy()
    stretches = torch.sqrt(final_eigenvalues.detach()).numpy()
    return SurfaceFlowResult(
        model=model,
        coordinates=fitted_coordinates,
        raw_target_face_metrics=raw_metrics.copy(),
        smoothed_target_face_metrics=smoothed_metrics.copy(),
        target_face_metrics=smoothed_metrics.copy(),
        face_weights=metric_data.weights.copy(),
        metric_smoothing_strength=settings.metric_smoothing_strength,
        raw_metric_log_roughness=raw_roughness,
        smoothed_metric_log_roughness=smoothed_roughness,
        initial_isometry_loss=initial_value,
        stage_one_isometry_loss=best_stage_one_loss,
        stage_two_metric_threshold=metric_threshold,
        final_isometry_loss=float(final_isometry.detach()),
        final_bending_loss=float(final_bending.detach()),
        generalized_stretch_min=float(np.min(stretches)),
        generalized_stretch_max=float(np.max(stretches)),
        history=tuple(history),
    )


def subdivide_parameter_mesh(
    chart_coordinates: np.ndarray, faces: np.ndarray, subdivisions: int = 4
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Subdivide each parameter triangle for smooth dense rendering.

    Points on shared input edges are intentionally duplicated.  Evaluating the
    same continuous map gives identical positions on both copies, while this
    simple representation also preserves a parent-face index for colors.
    """

    if subdivisions < 1:
        raise ValueError("subdivisions must be positive")
    chart = _as_chart_array(chart_coordinates)
    triangles = _as_face_array(faces, len(chart))
    dense_points: list[np.ndarray] = []
    dense_faces: list[tuple[int, int, int]] = []
    parent_faces: list[int] = []
    for parent_index, face in enumerate(triangles):
        vertices = chart[face]
        local: dict[tuple[int, int], int] = {}
        for first_index in range(subdivisions + 1):
            for second_index in range(subdivisions + 1 - first_index):
                first_weight = first_index / subdivisions
                second_weight = second_index / subdivisions
                point = (
                    (1.0 - first_weight - second_weight) * vertices[0]
                    + first_weight * vertices[1]
                    + second_weight * vertices[2]
                )
                local[(first_index, second_index)] = len(dense_points)
                dense_points.append(point)
        for first_index in range(subdivisions):
            for second_index in range(subdivisions - first_index):
                lower = local[(first_index, second_index)]
                right = local[(first_index + 1, second_index)]
                upper = local[(first_index, second_index + 1)]
                dense_faces.append((lower, right, upper))
                parent_faces.append(parent_index)
                if first_index + second_index < subdivisions - 1:
                    diagonal = local[(first_index + 1, second_index + 1)]
                    dense_faces.append((right, diagonal, upper))
                    parent_faces.append(parent_index)
    return (
        np.asarray(dense_points, dtype=np.float64),
        np.asarray(dense_faces, dtype=np.int64),
        np.asarray(parent_faces, dtype=np.int64),
    )


@torch.no_grad()
def evaluate_surface(
    model: TopologyPreservingSurface,
    chart_coordinates: np.ndarray,
    batch_size: int = 4_096,
) -> np.ndarray:
    """Evaluate a fitted surface map on arbitrary chart coordinates."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    chart = _as_chart_array(chart_coordinates, minimum_count=1)
    batches: list[np.ndarray] = []
    for start in range(0, len(chart), batch_size):
        points = torch.as_tensor(chart[start : start + batch_size], dtype=DEFAULT_DTYPE)
        batches.append(model(points).cpu().numpy())
    return np.concatenate(batches, axis=0)


def evaluate_dense_surface(
    model: TopologyPreservingSurface,
    chart_coordinates: np.ndarray,
    faces: np.ndarray,
    subdivisions: int = 4,
    batch_size: int = 4_096,
) -> DenseSurfaceEvaluation:
    """Subdivide a chart and evaluate the smooth surface for visualization."""

    points, dense_faces, parent_faces = subdivide_parameter_mesh(
        chart_coordinates, faces, subdivisions=subdivisions
    )
    return DenseSurfaceEvaluation(
        chart_coordinates=points,
        coordinates=evaluate_surface(model, points, batch_size=batch_size),
        faces=dense_faces,
        parent_faces=parent_faces,
    )


def _checkerboard_mesh(side: int) -> tuple[np.ndarray, np.ndarray]:
    axis = np.linspace(-1.0, 1.0, side, dtype=np.float64)
    horizontal, vertical = np.meshgrid(axis, axis, indexing="xy")
    chart = np.column_stack([horizontal.ravel(), vertical.ravel()])
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
    return chart, np.asarray(faces, dtype=np.int64)


def _controlled_coordinates(chart: np.ndarray) -> np.ndarray:
    """A smooth target that moves x, y, and z, with variable curvature."""

    horizontal = chart[:, 0]
    vertical = chart[:, 1]
    return np.column_stack(
        [
            horizontal + 0.16 * np.sin(1.3 * vertical),
            vertical + 0.12 * np.sin(1.1 * horizontal),
            0.30 * np.sin(1.2 * horizontal) * np.cos(0.9 * vertical)
            + 0.10 * horizontal * vertical,
        ]
    )


def _face_edge_lengths(coordinates: np.ndarray, faces: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.linalg.norm(coordinates[faces[:, 1]] - coordinates[faces[:, 0]], axis=1),
            np.linalg.norm(coordinates[faces[:, 2]] - coordinates[faces[:, 1]], axis=1),
            np.linalg.norm(coordinates[faces[:, 0]] - coordinates[faces[:, 2]], axis=1),
        ]
    )


def controlled_self_test() -> SurfaceFlowResult:
    """Run a tiny deterministic end-to-end check without external data."""

    chart, faces = _checkerboard_mesh(side=6)
    truth = _controlled_coordinates(chart)
    target_lengths = _face_edge_lengths(truth, faces)
    config = SurfaceFlowConfig(
        seed=1701,
        hidden_width=24,
        coupling_blocks=6,
        metric_smoothing_strength=0.75,
        stage_one_steps=260,
        stage_two_steps=160,
        stage_one_learning_rate=7.0e-3,
        stage_two_learning_rate=2.5e-3,
        metric_relative_slack=0.15,
        record_every=40,
    )
    result = fit_surface_flow(chart, faces, target_lengths, config=config)
    repeated = fit_surface_flow(chart, faces, target_lengths, config=config)

    if not np.array_equal(result.coordinates, repeated.coordinates):
        raise AssertionError(
            "repeated deterministic fits produced different coordinates"
        )
    if result.history != repeated.history:
        raise AssertionError("repeated deterministic fits produced different histories")
    if not np.array_equal(
        result.target_face_metrics, result.smoothed_target_face_metrics
    ):
        raise AssertionError("effective target metric is not the smoothed metric")
    if np.any(
        np.linalg.eigvalsh(result.raw_target_face_metrics)[:, 0] <= NUMERICAL_EPSILON
    ) or np.any(
        np.linalg.eigvalsh(result.smoothed_target_face_metrics)[:, 0]
        <= NUMERICAL_EPSILON
    ):
        raise AssertionError("raw or smoothed target metric lost positive definiteness")
    disabled_smoothing = smooth_face_metrics_log_euclidean(
        result.raw_target_face_metrics,
        faces,
        result.face_weights,
        strength=0.0,
    )
    if not np.array_equal(disabled_smoothing, result.raw_target_face_metrics):
        raise AssertionError("zero metric smoothing did not preserve the raw field")
    repeated_smoothing = smooth_face_metrics_log_euclidean(
        result.raw_target_face_metrics,
        faces,
        result.face_weights,
        strength=config.metric_smoothing_strength,
    )
    if not np.array_equal(repeated_smoothing, result.smoothed_target_face_metrics):
        raise AssertionError("log-Euclidean metric smoothing is not deterministic")
    if result.smoothed_metric_log_roughness > result.raw_metric_log_roughness + 1.0e-12:
        raise AssertionError("metric smoothing increased adjacent-face roughness")

    ambient_samples = torch.tensor(
        [
            [-0.7, 0.2, 0.4],
            [0.1, -0.5, -0.3],
            [0.8, 0.6, 0.2],
        ],
        dtype=DEFAULT_DTYPE,
    )
    with torch.no_grad():
        transported = result.model.ambient_map(ambient_samples)
        round_trip = result.model.ambient_map.inverse(transported)
    round_trip_error = float(torch.max(torch.abs(round_trip - ambient_samples)))
    if round_trip_error > 2.0e-10:
        raise AssertionError(f"ambient inverse check failed: {round_trip_error:.3e}")
    if not result.final_isometry_loss < 0.80 * result.initial_isometry_loss:
        raise AssertionError(
            "controlled fit did not materially improve the pullback metric: "
            f"{result.initial_isometry_loss:.4e} -> "
            f"{result.final_isometry_loss:.4e}"
        )
    if result.final_isometry_loss > result.stage_two_metric_threshold * (1.0 + 1e-8):
        raise AssertionError("stage-two selection violated its metric threshold")
    dense = evaluate_dense_surface(result.model, chart, faces, subdivisions=2)
    expected_faces = len(faces) * 4
    if len(dense.faces) != expected_faces:
        raise AssertionError("dense triangle subdivision has the wrong face count")
    if not np.isfinite(dense.coordinates).all():
        raise AssertionError("dense surface contains a non-finite coordinate")

    print(
        "ot_surface_flow self-test passed: "
        f"isometry {result.initial_isometry_loss:.4e} -> "
        f"{result.final_isometry_loss:.4e}, "
        f"bending={result.final_bending_loss:.4e}, "
        f"stretch=[{result.generalized_stretch_min:.3f},"
        f"{result.generalized_stretch_max:.3f}], "
        f"metric_roughness={result.raw_metric_log_roughness:.3e}->"
        f"{result.smoothed_metric_log_roughness:.3e}, "
        f"inverse_error={round_trip_error:.2e}"
    )
    return result


def parse_arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="run the small deterministic controlled check",
    )
    return parser.parse_args()


def main() -> None:
    arguments = parse_arguments()
    if arguments.self_test:
        controlled_self_test()
    else:
        print("Use --self-test to run the controlled topology/metric check.")


if __name__ == "__main__":
    main()
