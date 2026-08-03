"""Reusable explanatory figure for Wasserstein latent-surface correction.

The module deliberately contains no dataset or model loading.  It consumes
arrays produced by an experiment and draws one consistent visual story:

1. the actual posterior support and the chart mesh selected from it;
2. equal-radius local Wasserstein balls and an OT shortest path in the raw
   two-dimensional chart; and
3. the same distance bands and path on corrected three-dimensional vertices.

The image strip decodes landmarks along that exact shortest path.  Thus the
colors, path, point numbers, and images have the same meaning in every panel.

``w2_edge_lengths`` must correspond either to ``edges`` or, when ``edges`` is
omitted, to :func:`unique_edges_from_faces` in its lexicographic order.
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager
from matplotlib.axes import Axes
from matplotlib.collections import PolyCollection
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.figure import Figure
from matplotlib.lines import Line2D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

EPSILON = 1.0e-12


@dataclass(frozen=True)
class OTSurfaceFigureData:
    """Arrays required by :func:`plot_ot_surface_figure`.

    ``posterior_points`` and ``chart_points`` must use the same raw latent
    coordinate system.  Vertex ``i`` of ``chart_points``, ``corrected_points``,
    ``decoded_images``, and optional ``source_images`` must refer to the same
    decoded distribution and its corresponding observed input.  An optional
    ``surface_evaluator`` maps arbitrary raw-chart points of shape ``(N, 2)``
    into the same corrected coordinate system as ``corrected_points``.
    """

    posterior_points: np.ndarray
    posterior_labels: np.ndarray | None
    chart_points: np.ndarray
    faces: np.ndarray
    w2_edge_lengths: np.ndarray
    corrected_points: np.ndarray
    decoded_images: np.ndarray
    edges: np.ndarray | None = None
    focus_label: Any | None = None
    source_images: np.ndarray | None = None
    surface_evaluator: Callable[[np.ndarray], np.ndarray] | None = None


@dataclass(frozen=True)
class LocalMetricEstimate:
    """A local SPD metric and its incident-edge fit diagnostics."""

    matrices: np.ndarray
    relative_rms: np.ndarray
    incident_counts: np.ndarray


@dataclass(frozen=True)
class OTSurfaceFigureResult:
    """Figure plus the geometric objects used to construct it."""

    figure: Figure
    axes: dict[str, Axes]
    local_metrics: LocalMetricEstimate
    graph_distances: np.ndarray
    anchor_index: int
    endpoint_index: int
    path_indices: np.ndarray
    landmark_indices: np.ndarray


@dataclass(frozen=True)
class _ValidatedData:
    posterior_points: np.ndarray
    posterior_labels: np.ndarray | None
    chart_points: np.ndarray
    faces: np.ndarray
    edges: np.ndarray
    edge_lengths: np.ndarray
    corrected_points: np.ndarray
    decoded_images: np.ndarray
    focus_label: Any | None
    source_images: np.ndarray | None
    surface_evaluator: Callable[[np.ndarray], np.ndarray] | None


def unique_edges_from_faces(faces: np.ndarray) -> np.ndarray:
    """Return sorted undirected face edges in lexicographic order."""

    triangles = np.asarray(faces, dtype=np.int64)
    if triangles.ndim != 2 or triangles.shape[1] != 3:
        raise ValueError("faces must have shape (face_count, 3)")
    raw = np.concatenate(
        [triangles[:, [0, 1]], triangles[:, [1, 2]], triangles[:, [2, 0]]],
        axis=0,
    )
    raw.sort(axis=1)
    return np.unique(raw, axis=0)


def _validated(data: OTSurfaceFigureData) -> _ValidatedData:
    posterior = np.asarray(data.posterior_points, dtype=np.float64)
    chart = np.asarray(data.chart_points, dtype=np.float64)
    faces = np.asarray(data.faces, dtype=np.int64)
    corrected = np.asarray(data.corrected_points, dtype=np.float64)
    images = np.asarray(data.decoded_images)
    source_images = (
        None if data.source_images is None else np.asarray(data.source_images)
    )
    surface_evaluator = data.surface_evaluator
    lengths = np.asarray(data.w2_edge_lengths, dtype=np.float64).reshape(-1)
    labels = (
        None if data.posterior_labels is None else np.asarray(data.posterior_labels)
    )

    if posterior.ndim != 2 or posterior.shape[1] != 2 or len(posterior) == 0:
        raise ValueError("posterior_points must have nonzero shape (sample_count, 2)")
    if chart.ndim != 2 or chart.shape[1] != 2 or len(chart) < 3:
        raise ValueError("chart_points must have shape (vertex_count, 2)")
    if corrected.shape != (len(chart), 3):
        raise ValueError("corrected_points must have shape (vertex_count, 3)")
    if faces.ndim != 2 or faces.shape[1] != 3 or len(faces) == 0:
        raise ValueError("faces must have nonzero shape (face_count, 3)")
    if np.min(faces) < 0 or np.max(faces) >= len(chart):
        raise ValueError("faces contain an out-of-range vertex index")
    if np.any(
        (faces[:, 0] == faces[:, 1])
        | (faces[:, 1] == faces[:, 2])
        | (faces[:, 2] == faces[:, 0])
    ):
        raise ValueError("faces must not contain repeated vertices")
    if images.ndim < 3 or len(images) != len(chart):
        raise ValueError("decoded_images must start with the vertex dimension")
    if source_images is not None and (
        source_images.ndim < 3 or len(source_images) != len(chart)
    ):
        raise ValueError("source_images must start with the vertex dimension")
    if labels is not None and (labels.ndim != 1 or len(labels) != len(posterior)):
        raise ValueError("posterior_labels must have one value per posterior point")

    edges = (
        unique_edges_from_faces(faces)
        if data.edges is None
        else np.asarray(data.edges, dtype=np.int64)
    )
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if len(edges) == 0:
        raise ValueError("edges must be nonempty")
    edges = np.sort(edges, axis=1)
    if len(np.unique(edges, axis=0)) != len(edges):
        raise ValueError("edges must be unique")
    if np.min(edges) < 0 or np.max(edges) >= len(chart):
        raise ValueError("edges contain an out-of-range vertex index")
    if len(lengths) != len(edges):
        raise ValueError("w2_edge_lengths must have one value per edge")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("w2_edge_lengths must be finite and strictly positive")
    face_edges = unique_edges_from_faces(faces)
    edge_set = {tuple(edge) for edge in edges.tolist()}
    if any(tuple(edge) not in edge_set for edge in face_edges.tolist()):
        raise ValueError("edges must include every edge of faces")

    for name, array in (
        ("posterior_points", posterior),
        ("chart_points", chart),
        ("corrected_points", corrected),
    ):
        if not np.all(np.isfinite(array)):
            raise ValueError(f"{name} must be finite")
    if surface_evaluator is not None:
        if not callable(surface_evaluator):
            raise ValueError("surface_evaluator must be callable")
        _evaluate_surface(surface_evaluator, chart[: min(3, len(chart))])

    return _ValidatedData(
        posterior_points=posterior,
        posterior_labels=labels,
        chart_points=chart,
        faces=faces,
        edges=edges,
        edge_lengths=lengths,
        corrected_points=corrected,
        decoded_images=images,
        focus_label=data.focus_label,
        source_images=source_images,
        surface_evaluator=surface_evaluator,
    )


def estimate_local_spd_metrics(
    chart_points: np.ndarray,
    edges: np.ndarray,
    w2_edge_lengths: np.ndarray,
    *,
    ridge: float = 1.0e-3,
    eigenvalue_floor: float = 1.0e-4,
) -> LocalMetricEstimate:
    r"""Fit a local metric ``G_i`` from incident Wasserstein edge lengths.

    At vertex ``i`` the fit uses

    .. math::

       \ell_{ij}^2 \simeq (z_j-z_i)^\top G_i (z_j-z_i).

    The least-squares equations are normalized by ``ell_ij**2``, mildly
    shrunk toward the local isotropic scale, and projected onto the SPD cone.
    The returned RMS is the relative error in squared edge length.
    """

    points = np.asarray(chart_points, dtype=np.float64)
    edge_array = np.asarray(edges, dtype=np.int64)
    lengths = np.asarray(w2_edge_lengths, dtype=np.float64).reshape(-1)
    if points.ndim != 2 or points.shape[1] != 2:
        raise ValueError("chart_points must have shape (vertex_count, 2)")
    if not np.all(np.isfinite(points)):
        raise ValueError("chart_points must be finite")
    if edge_array.ndim != 2 or edge_array.shape[1] != 2:
        raise ValueError("edges must have shape (edge_count, 2)")
    if len(edge_array) == 0:
        raise ValueError("edges must be nonempty")
    if len(edge_array) != len(lengths):
        raise ValueError("edges and w2_edge_lengths must have equal length")
    if np.min(edge_array) < 0 or np.max(edge_array) >= len(points):
        raise ValueError("edges contain an out-of-range vertex index")
    if not np.all(np.isfinite(lengths)) or np.any(lengths <= 0.0):
        raise ValueError("w2_edge_lengths must be finite and strictly positive")
    if ridge < 0.0 or eigenvalue_floor <= 0.0:
        raise ValueError("ridge must be nonnegative and eigenvalue_floor positive")

    incident: list[list[tuple[int, float]]] = [[] for _ in range(len(points))]
    ratios: list[float] = []
    for (left, right), length in zip(edge_array, lengths):
        left_index = int(left)
        right_index = int(right)
        delta = points[right_index] - points[left_index]
        latent_squared = float(delta @ delta)
        if latent_squared <= EPSILON:
            continue
        incident[left_index].append((right_index, float(length)))
        incident[right_index].append((left_index, float(length)))
        ratios.append(float(length * length / latent_squared))

    global_scale_squared = float(np.median(ratios)) if ratios else 1.0
    matrices = np.empty((len(points), 2, 2), dtype=np.float64)
    relative_rms = np.full(len(points), np.nan, dtype=np.float64)
    incident_counts = np.zeros(len(points), dtype=np.int64)
    shrink_target = np.asarray([1.0, 0.0, 1.0], dtype=np.float64)

    for vertex, neighbours in enumerate(incident):
        design_rows: list[list[float]] = []
        squared_lengths: list[float] = []
        local_ratios: list[float] = []
        for neighbour, length in neighbours:
            delta = points[neighbour] - points[vertex]
            latent_squared = float(delta @ delta)
            if latent_squared <= EPSILON:
                continue
            dx, dy = float(delta[0]), float(delta[1])
            design_rows.append([dx * dx, 2.0 * dx * dy, dy * dy])
            squared_lengths.append(length * length)
            local_ratios.append(length * length / latent_squared)

        incident_counts[vertex] = len(design_rows)
        if not design_rows:
            matrices[vertex] = global_scale_squared * np.eye(2)
            continue

        design = np.asarray(design_rows, dtype=np.float64)
        target_squared = np.asarray(squared_lengths, dtype=np.float64)
        local_scale_squared = float(np.median(local_ratios))
        normalized_design = local_scale_squared * design / target_squared[:, None]
        if ridge > 0.0:
            root_ridge = math.sqrt(ridge)
            system = np.vstack([normalized_design, root_ridge * np.eye(3)])
            target = np.concatenate([np.ones(len(design)), root_ridge * shrink_target])
        else:
            system = normalized_design
            target = np.ones(len(design))
        normalized_coefficients, *_ = np.linalg.lstsq(system, target, rcond=None)
        coefficients = local_scale_squared * normalized_coefficients
        raw_metric = np.asarray(
            [
                [coefficients[0], coefficients[1]],
                [coefficients[1], coefficients[2]],
            ],
            dtype=np.float64,
        )
        eigenvalues, eigenvectors = np.linalg.eigh(raw_metric)
        floor = max(eigenvalue_floor * local_scale_squared, EPSILON)
        eigenvalues = np.maximum(eigenvalues, floor)
        metric = (eigenvectors * eigenvalues) @ eigenvectors.T
        matrices[vertex] = metric

        displacements = np.asarray(
            [points[neighbour] - points[vertex] for neighbour, _ in neighbours],
            dtype=np.float64,
        )
        prediction = np.einsum("ni,ij,nj->n", displacements, metric, displacements)
        relative = prediction / target_squared - 1.0
        relative_rms[vertex] = float(np.sqrt(np.mean(relative * relative)))

    return LocalMetricEstimate(
        matrices=matrices,
        relative_rms=relative_rms,
        incident_counts=incident_counts,
    )


def _weighted_graph(vertex_count: int, edges: np.ndarray, edge_lengths: np.ndarray):
    rows = np.concatenate([edges[:, 0], edges[:, 1]])
    columns = np.concatenate([edges[:, 1], edges[:, 0]])
    values = np.concatenate([edge_lengths, edge_lengths])
    return coo_matrix(
        (values, (rows, columns)), shape=(vertex_count, vertex_count)
    ).tocsr()


def shortest_path_from_anchor(
    vertex_count: int,
    edges: np.ndarray,
    edge_lengths: np.ndarray,
    anchor_index: int,
    endpoint_index: int | None = None,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return mesh distances, endpoint, and its shortest vertex path."""

    if not 0 <= anchor_index < vertex_count:
        raise ValueError("anchor_index is out of range")
    graph = _weighted_graph(vertex_count, edges, edge_lengths)
    distances, predecessors = dijkstra(
        graph,
        directed=False,
        indices=anchor_index,
        return_predecessors=True,
    )
    distances = np.asarray(distances, dtype=np.float64)
    predecessors = np.asarray(predecessors, dtype=np.int64)
    finite = np.flatnonzero(np.isfinite(distances))
    if len(finite) <= 1:
        raise ValueError(
            "the anchor belongs to a component with fewer than two vertices"
        )

    if endpoint_index is None:
        endpoint = int(finite[np.argmax(distances[finite])])
    else:
        endpoint = int(endpoint_index)
        if not 0 <= endpoint < vertex_count:
            raise ValueError("endpoint_index is out of range")
        if not np.isfinite(distances[endpoint]):
            raise ValueError("endpoint_index is not connected to anchor_index")
        if endpoint == anchor_index:
            raise ValueError("endpoint_index must differ from anchor_index")

    reversed_path = [endpoint]
    current = endpoint
    while current != anchor_index:
        current = int(predecessors[current])
        if current < 0:
            raise RuntimeError("failed to reconstruct the shortest path")
        reversed_path.append(current)
        if len(reversed_path) > vertex_count:
            raise RuntimeError("shortest-path predecessor cycle detected")
    path = np.asarray(reversed_path[::-1], dtype=np.int64)
    return distances, endpoint, path


def _automatic_anchor(data: _ValidatedData) -> int:
    if data.posterior_labels is not None and data.focus_label is not None:
        focus = data.posterior_labels == data.focus_label
        target = (
            np.median(data.posterior_points[focus], axis=0)
            if np.any(focus)
            else np.median(data.chart_points, axis=0)
        )
    else:
        target = np.median(data.chart_points, axis=0)
    return int(np.argmin(np.sum((data.chart_points - target) ** 2, axis=1)))


def _farthest_point_indices(
    points: np.ndarray,
    count: int,
    *,
    first_index: int,
    candidates: np.ndarray,
) -> np.ndarray:
    candidate_indices = np.flatnonzero(candidates)
    if count <= 0 or len(candidate_indices) == 0:
        return np.empty(0, dtype=np.int64)
    chosen = [
        first_index
        if candidates[first_index]
        else int(
            candidate_indices[
                np.argmin(
                    np.sum(
                        (points[candidate_indices] - points[first_index]) ** 2,
                        axis=1,
                    )
                )
            ]
        )
    ]
    minimum_squared = np.sum((points - points[chosen[0]]) ** 2, axis=1)
    while len(chosen) < min(count, len(candidate_indices)):
        scores = minimum_squared.copy()
        scores[~candidates] = -np.inf
        scores[np.asarray(chosen, dtype=np.int64)] = -np.inf
        candidate = int(np.argmax(scores))
        if not np.isfinite(scores[candidate]):
            break
        chosen.append(candidate)
        squared = np.sum((points - points[candidate]) ** 2, axis=1)
        minimum_squared = np.minimum(minimum_squared, squared)
    return np.asarray(chosen, dtype=np.int64)


def _path_landmarks(
    path: np.ndarray,
    graph_distances: np.ndarray,
    count: int,
) -> np.ndarray:
    if len(path) <= count:
        return path.copy()
    along = graph_distances[path]
    targets = np.linspace(float(along[0]), float(along[-1]), count)
    selected = []
    for target in targets:
        index = int(np.argmin(np.abs(along - target)))
        vertex = int(path[index])
        if not selected or vertex != selected[-1]:
            selected.append(vertex)
    if selected[-1] != int(path[-1]):
        selected[-1] = int(path[-1])
    return np.asarray(selected, dtype=np.int64)


def _metric_ellipse_curve(
    center: np.ndarray,
    metric: np.ndarray,
    radius: float,
    *,
    sample_count: int = 96,
) -> np.ndarray:
    if sample_count < 8:
        raise ValueError("ellipse sample_count must be at least eight")
    eigenvalues, eigenvectors = np.linalg.eigh(metric)
    eigenvalues = np.maximum(eigenvalues, EPSILON)
    order = np.argsort(eigenvalues)
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]
    semiaxes = radius / np.sqrt(eigenvalues)
    angles = np.linspace(0.0, 2.0 * np.pi, sample_count, endpoint=True)
    unit_circle = np.column_stack([np.cos(angles), np.sin(angles)])
    return center + (unit_circle * semiaxes) @ eigenvectors.T


def _evaluate_surface(
    evaluator: Callable[[np.ndarray], np.ndarray], points: np.ndarray
) -> np.ndarray:
    query = np.asarray(points, dtype=np.float64)
    if query.ndim != 2 or query.shape[1] != 2:
        raise ValueError("surface_evaluator input must have shape (N, 2)")
    if not np.all(np.isfinite(query)):
        raise ValueError("surface_evaluator input must be finite")
    try:
        mapped = np.asarray(evaluator(query), dtype=np.float64)
    except Exception as error:
        raise ValueError("surface_evaluator failed on an (N, 2) query") from error
    if mapped.shape != (len(query), 3):
        raise ValueError("surface_evaluator must return shape (N, 3)")
    if not np.all(np.isfinite(mapped)):
        raise ValueError("surface_evaluator output must be finite")
    return mapped


def _japanese_font_family() -> str:
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


def _posterior_scatter(axis: Axes, data: _ValidatedData, japanese: bool) -> None:
    labels = data.posterior_labels
    if labels is None:
        axis.scatter(
            data.posterior_points[:, 0],
            data.posterior_points[:, 1],
            s=3.0,
            color="#8b929a",
            alpha=0.25,
            linewidths=0.0,
            label="実posterior" if japanese else "observed posterior",
        )
        return

    if data.focus_label is not None:
        focus = labels == data.focus_label
        if np.any(~focus):
            axis.scatter(
                data.posterior_points[~focus, 0],
                data.posterior_points[~focus, 1],
                s=2.5,
                color="#a9adb2",
                alpha=0.15,
                linewidths=0.0,
                label="その他" if japanese else "other labels",
            )
        if np.any(focus):
            axis.scatter(
                data.posterior_points[focus, 0],
                data.posterior_points[focus, 1],
                s=4.0,
                color="#1769aa",
                alpha=0.35,
                linewidths=0.0,
                label=(
                    f"label={data.focus_label} の実posterior"
                    if japanese
                    else f"observed posterior, label={data.focus_label}"
                ),
            )
        return

    unique_labels = np.unique(labels)
    if len(unique_labels) > 12:
        axis.scatter(
            data.posterior_points[:, 0],
            data.posterior_points[:, 1],
            s=3.0,
            color="#8b929a",
            alpha=0.25,
            linewidths=0.0,
            label="実posterior" if japanese else "observed posterior",
        )
        return
    colors = mpl.colormaps["tab10"](np.linspace(0.0, 1.0, max(len(unique_labels), 2)))
    for label, color in zip(unique_labels, colors):
        selected = labels == label
        axis.scatter(
            data.posterior_points[selected, 0],
            data.posterior_points[selected, 1],
            s=3.0,
            color=color,
            alpha=0.28,
            linewidths=0.0,
            label=str(label),
        )


def _face_band_colors(
    distances: np.ndarray,
    faces: np.ndarray,
    norm: BoundaryNorm,
    cmap: ListedColormap,
) -> np.ndarray:
    face_distances = np.mean(distances[faces], axis=1)
    finite = np.isfinite(face_distances)
    colors = np.tile(np.asarray([0.82, 0.82, 0.82, 0.42]), (len(faces), 1))
    if np.any(finite):
        colors[finite] = cmap(norm(face_distances[finite]))
    return colors


def _boundary_vertices(vertex_count: int, faces: np.ndarray) -> np.ndarray:
    raw = np.concatenate([faces[:, [0, 1]], faces[:, [1, 2]], faces[:, [2, 0]]], axis=0)
    raw.sort(axis=1)
    face_edges, counts = np.unique(raw, axis=0, return_counts=True)
    boundary = np.zeros(vertex_count, dtype=bool)
    boundary[np.unique(face_edges[counts == 1])] = True
    return boundary


def _set_equal_2d_limits(axis: Axes, points: np.ndarray, padding: float = 0.05) -> None:
    minimum = np.min(points, axis=0)
    maximum = np.max(points, axis=0)
    span = np.maximum(maximum - minimum, EPSILON)
    margin = padding * max(float(span[0]), float(span[1]))
    axis.set_xlim(minimum[0] - margin, maximum[0] + margin)
    axis.set_ylim(minimum[1] - margin, maximum[1] + margin)
    axis.set_aspect("equal", adjustable="box")


def _style_surface_axis(axis: Axes, coordinates: np.ndarray) -> None:
    minimum = np.min(coordinates, axis=0)
    maximum = np.max(coordinates, axis=0)
    raw_span = maximum - minimum
    reference_span = max(float(np.max(raw_span)), 1.0)
    span = np.maximum(raw_span, 0.05 * reference_span)
    center = 0.5 * (minimum + maximum)
    axis.set_xlim(center[0] - 0.5 * span[0], center[0] + 0.5 * span[0])
    axis.set_ylim(center[1] - 0.5 * span[1], center[1] + 0.5 * span[1])
    axis.set_zlim(center[2] - 0.5 * span[2], center[2] + 0.5 * span[2])
    axis.set_box_aspect(span)
    axis.set_xticks([])
    axis.set_yticks([])
    axis.set_zticks([])
    axis.grid(False)
    axis.view_init(elev=27, azim=-62)


def _display_image(image: np.ndarray) -> tuple[np.ndarray, str | None]:
    value = np.asarray(image)
    if value.ndim == 2:
        return value, "gray"
    if value.ndim != 3:
        raise ValueError("each decoded image must be 2D grayscale or 3D color")
    if value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        value = np.moveaxis(value, 0, -1)
    if value.shape[-1] == 1:
        return value[..., 0], "gray"
    if value.shape[-1] not in (3, 4):
        raise ValueError("color decoded images must have 3 or 4 channels")
    return value, None


def _draw_image_axis(
    axis: Axes,
    image: np.ndarray,
    *,
    border_color: str,
    title: str | None = None,
) -> None:
    image_value, image_cmap = _display_image(image)
    image_options: dict[str, Any] = {}
    if image_cmap is not None:
        image_options["cmap"] = image_cmap
        finite_image = image_value[np.isfinite(image_value)]
        if (
            len(finite_image) > 0
            and np.min(finite_image) >= 0.0
            and np.max(finite_image) <= 1.0
        ):
            image_options.update(vmin=0.0, vmax=1.0)
    axis.imshow(image_value, **image_options)
    axis.set_xticks([])
    axis.set_yticks([])
    if title is not None:
        axis.set_title(title, fontsize=7.5)
    for spine in axis.spines.values():
        spine.set_edgecolor(border_color)
        spine.set_linewidth(0.8)


def plot_ot_surface_figure(
    data: OTSurfaceFigureData,
    *,
    japanese: bool = False,
    anchor_index: int | None = None,
    endpoint_index: int | None = None,
    ellipse_radius: float | None = None,
    ellipse_count: int = 8,
    distance_band_count: int = 7,
    image_count: int = 7,
    figure_size: tuple[float, float] = (14.2, 7.7),
) -> OTSurfaceFigureResult:
    """Draw raw support, chart distortion, corrected R3, and path images."""

    arrays = _validated(data)
    if distance_band_count < 2:
        raise ValueError("distance_band_count must be at least two")
    if image_count < 2:
        raise ValueError("image_count must be at least two")
    anchor = _automatic_anchor(arrays) if anchor_index is None else int(anchor_index)
    graph_distances, endpoint, path = shortest_path_from_anchor(
        len(arrays.chart_points),
        arrays.edges,
        arrays.edge_lengths,
        anchor,
        endpoint_index,
    )
    landmarks = _path_landmarks(path, graph_distances, image_count)
    local_metrics = estimate_local_spd_metrics(
        arrays.chart_points, arrays.edges, arrays.edge_lengths
    )

    if ellipse_radius is None:
        ellipse_radius = 0.75 * float(np.median(arrays.edge_lengths))
    if ellipse_radius <= 0.0:
        raise ValueError("ellipse_radius must be positive")

    degree = np.bincount(arrays.edges.ravel(), minlength=len(arrays.chart_points))
    boundary = _boundary_vertices(len(arrays.chart_points), arrays.faces)
    glyph_candidates = (~boundary) & (degree >= min(4, int(np.max(degree))))
    if not np.any(glyph_candidates):
        glyph_candidates = degree > 0
    glyph_indices = _farthest_point_indices(
        arrays.chart_points,
        ellipse_count,
        first_index=anchor,
        candidates=glyph_candidates,
    )
    glyph_curves: list[tuple[np.ndarray, np.ndarray | None]] = []
    for glyph_index in glyph_indices:
        raw_curve = _metric_ellipse_curve(
            arrays.chart_points[glyph_index],
            local_metrics.matrices[glyph_index],
            float(ellipse_radius),
        )
        mapped_curve = (
            None
            if arrays.surface_evaluator is None
            else _evaluate_surface(arrays.surface_evaluator, raw_curve)
        )
        glyph_curves.append((raw_curve, mapped_curve))

    finite_distances = graph_distances[np.isfinite(graph_distances)]
    distance_max = float(np.max(finite_distances))
    if distance_max <= EPSILON:
        distance_max = 1.0
    boundaries = np.linspace(0.0, distance_max, distance_band_count + 1)
    cmap = ListedColormap(
        mpl.colormaps["viridis"](np.linspace(0.08, 0.94, distance_band_count))
    )
    norm = BoundaryNorm(boundaries, cmap.N, clip=True)
    face_colors = _face_band_colors(graph_distances, arrays.faces, norm, cmap)

    font_context = {"font.family": _japanese_font_family()} if japanese else {}
    with mpl.rc_context(font_context):
        figure = plt.figure(figsize=figure_size)
        grid = figure.add_gridspec(
            2,
            3,
            height_ratios=[
                3.10 if arrays.source_images is not None else 3.25,
                1.55 if arrays.source_images is not None else 1.20,
            ],
            hspace=0.18,
            wspace=0.08,
        )
        support_axis = figure.add_subplot(grid[0, 0])
        metric_axis = figure.add_subplot(grid[0, 1])
        surface_axis = figure.add_subplot(grid[0, 2], projection="3d")
        strip_axis = figure.add_subplot(grid[1, :])

        _posterior_scatter(support_axis, arrays, japanese)
        support_axis.triplot(
            arrays.chart_points[:, 0],
            arrays.chart_points[:, 1],
            arrays.faces,
            color=(0.08, 0.08, 0.08, 0.30),
            linewidth=0.25,
            label="補正対象メッシュ" if japanese else "mesh to be corrected",
        )
        support_axis.scatter(
            arrays.chart_points[:, 0],
            arrays.chart_points[:, 1],
            s=5.0,
            color="#e36a2e",
            alpha=0.55,
            linewidths=0.0,
            zorder=3,
        )
        combined_2d = np.concatenate(
            [arrays.posterior_points, arrays.chart_points], axis=0
        )
        _set_equal_2d_limits(support_axis, combined_2d)
        support_axis.set_xticks([])
        support_axis.set_yticks([])
        support_axis.set_title(
            "(a) 元のVAE潜在座標\n点＝実posterior、線＝補正対象"
            if japanese
            else "(a) Original VAE coordinates\npoints = posterior support; lines = corrected domain",
            fontsize=10.5,
        )
        support_axis.legend(
            loc="best",
            fontsize=7.5,
            frameon=True,
            framealpha=0.82,
            markerscale=1.8,
            ncol=1 if arrays.focus_label is not None else 2,
        )

        metric_collection = PolyCollection(
            arrays.chart_points[arrays.faces],
            facecolors=face_colors,
            edgecolors=(0.08, 0.08, 0.08, 0.20),
            linewidths=0.22,
        )
        metric_axis.add_collection(metric_collection)
        for raw_curve, _ in glyph_curves:
            metric_axis.plot(
                raw_curve[:, 0],
                raw_curve[:, 1],
                color="white",
                linewidth=3.2,
                solid_capstyle="round",
                zorder=4,
            )
            metric_axis.plot(
                raw_curve[:, 0],
                raw_curve[:, 1],
                color=(0.02, 0.02, 0.02, 0.78),
                linewidth=1.2,
                solid_capstyle="round",
                zorder=5,
            )
        raw_path = arrays.chart_points[path]
        metric_axis.plot(
            raw_path[:, 0],
            raw_path[:, 1],
            color="white",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=6,
        )
        metric_axis.plot(
            raw_path[:, 0],
            raw_path[:, 1],
            color="#ee3e80",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=7,
        )
        metric_axis.scatter(
            arrays.chart_points[anchor, 0],
            arrays.chart_points[anchor, 1],
            marker="*",
            s=90,
            color="white",
            edgecolor="#111111",
            linewidth=0.8,
            zorder=8,
        )
        metric_axis.scatter(
            arrays.chart_points[endpoint, 0],
            arrays.chart_points[endpoint, 1],
            marker="X",
            s=55,
            color="#ee3e80",
            edgecolor="white",
            linewidth=0.7,
            zorder=8,
        )
        for number, vertex in enumerate(landmarks, start=1):
            point = arrays.chart_points[vertex]
            metric_axis.annotate(
                str(number),
                xy=point,
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=7.5,
                weight="bold",
                color="#111111",
                zorder=9,
            )
        _set_equal_2d_limits(metric_axis, arrays.chart_points)
        metric_axis.set_xticks([])
        metric_axis.set_yticks([])
        metric_axis.set_title(
            "(b) 元座標で見える歪み\n楕円＝同じW2半径、色＝OT測地距離"
            if japanese
            else "(b) Distortion in the raw chart\nellipses = equal W2 radius; color = OT geodesic distance",
            fontsize=10.5,
        )
        metric_axis.legend(
            handles=[
                Line2D(
                    [0],
                    [0],
                    color="white",
                    marker="o",
                    markerfacecolor="none",
                    markeredgecolor="#222222",
                    markersize=8,
                    linewidth=0,
                    label=(
                        f"同じW2半径 r={ellipse_radius:.3g}"
                        + (
                            "（右にも写像）"
                            if arrays.surface_evaluator is not None
                            else ""
                        )
                        if japanese
                        else f"equal W2 radius r={ellipse_radius:.3g}"
                        + (
                            " (mapped at right)"
                            if arrays.surface_evaluator is not None
                            else ""
                        )
                    ),
                ),
                Line2D(
                    [0],
                    [0],
                    color="#ee3e80",
                    linewidth=2.2,
                    label="OT最短経路" if japanese else "OT shortest path",
                ),
            ],
            loc="best",
            fontsize=7.5,
            frameon=True,
            framealpha=0.82,
        )

        surface_collection = Poly3DCollection(
            arrays.corrected_points[arrays.faces],
            facecolors=face_colors,
            edgecolors=(0.08, 0.08, 0.08, 0.18),
            linewidths=0.22,
            alpha=0.98,
        )
        surface_axis.add_collection3d(surface_collection)
        for _, mapped_curve in glyph_curves:
            if mapped_curve is None:
                continue
            surface_axis.plot(
                mapped_curve[:, 0],
                mapped_curve[:, 1],
                mapped_curve[:, 2],
                color="white",
                linewidth=3.2,
                solid_capstyle="round",
                zorder=5,
            )
            surface_axis.plot(
                mapped_curve[:, 0],
                mapped_curve[:, 1],
                mapped_curve[:, 2],
                color=(0.02, 0.02, 0.02, 0.78),
                linewidth=1.2,
                solid_capstyle="round",
                zorder=6,
            )
        corrected_path = arrays.corrected_points[path]
        surface_axis.plot(
            corrected_path[:, 0],
            corrected_path[:, 1],
            corrected_path[:, 2],
            color="white",
            linewidth=4.0,
            solid_capstyle="round",
            zorder=7,
        )
        surface_axis.plot(
            corrected_path[:, 0],
            corrected_path[:, 1],
            corrected_path[:, 2],
            color="#ee3e80",
            linewidth=2.2,
            solid_capstyle="round",
            zorder=8,
        )
        surface_axis.scatter(
            *arrays.corrected_points[anchor],
            marker="*",
            s=70,
            color="white",
            edgecolor="#111111",
            linewidth=0.8,
            depthshade=False,
            zorder=9,
        )
        surface_axis.scatter(
            *arrays.corrected_points[endpoint],
            marker="X",
            s=48,
            color="#ee3e80",
            edgecolor="white",
            linewidth=0.7,
            depthshade=False,
            zorder=9,
        )
        for number, vertex in enumerate(landmarks, start=1):
            point = arrays.corrected_points[vertex]
            surface_axis.text(
                point[0],
                point[1],
                point[2],
                str(number),
                fontsize=7.5,
                weight="bold",
                color="#111111",
                zorder=10,
            )
        mapped_curves = [
            mapped_curve for _, mapped_curve in glyph_curves if mapped_curve is not None
        ]
        surface_display_coordinates = (
            arrays.corrected_points
            if not mapped_curves
            else np.concatenate([arrays.corrected_points, *mapped_curves], axis=0)
        )
        _style_surface_axis(surface_axis, surface_display_coordinates)
        if arrays.surface_evaluator is None:
            surface_title = (
                "(c) OT距離に合わせた3次元表示\n色・経路・番号は中央と同一"
                if japanese
                else "(c) Corrected 3D representation\nsame distances, path, and landmarks as the center panel"
            )
        else:
            surface_title = (
                "(c) OT距離に合わせた3次元表示\n同じ色・経路・W2円を写像"
                if japanese
                else "(c) Corrected 3D representation\nsame bands, path, and mapped W2 circles"
            )
        surface_axis.set_title(surface_title, fontsize=10.5)

        scalar = mpl.cm.ScalarMappable(norm=norm, cmap=cmap)
        scalar.set_array([])
        colorbar_axis = figure.add_axes([0.947, 0.425, 0.012, 0.34])
        colorbar = figure.colorbar(scalar, cax=colorbar_axis, ticks=boundaries)
        colorbar.set_label(
            "anchorからのOTメッシュ測地距離"
            if japanese
            else "OT mesh-geodesic distance from anchor",
            fontsize=8.5,
        )
        colorbar.ax.tick_params(labelsize=7)

        strip_axis.set_axis_off()
        if arrays.source_images is None:
            strip_title = (
                "同じOT最短経路上でデコードされた画像"
                if japanese
                else "Decoded images along the same OT shortest path"
            )
        else:
            strip_title = (
                "同じOT最短経路：実入力（上）とデコード（下）"
                if japanese
                else "Same OT shortest path: real inputs (top), decoded images (bottom)"
            )
        strip_axis.set_title(strip_title, fontsize=10, pad=2)
        left_margin = 0.075
        right_margin = 0.025
        gap = 0.018
        count = len(landmarks)
        width = (1.0 - left_margin - right_margin - gap * (count - 1)) / count
        for column, vertex in enumerate(landmarks):
            left = left_margin + column * (width + gap)
            title_prefix = (
                "★ " if arrays.source_images is not None and column == 0 else ""
            )
            image_title = (
                f"{title_prefix}#{column + 1}   d={graph_distances[vertex]:.3g}"
            )
            if arrays.source_images is None:
                decoded_axis = strip_axis.inset_axes([left, 0.03, width, 0.84])
                _draw_image_axis(
                    decoded_axis,
                    arrays.decoded_images[vertex],
                    border_color="#ee3e80",
                    title=image_title,
                )
            else:
                source_axis = strip_axis.inset_axes([left, 0.51, width, 0.36])
                decoded_axis = strip_axis.inset_axes([left, 0.04, width, 0.36])
                _draw_image_axis(
                    source_axis,
                    arrays.source_images[vertex],
                    border_color="#4f7ea8",
                    title=image_title,
                )
                _draw_image_axis(
                    decoded_axis,
                    arrays.decoded_images[vertex],
                    border_color="#ee3e80",
                )
            if column + 1 < count:
                arrow_x = left + width + 0.5 * gap
                strip_axis.annotate(
                    "",
                    xy=(arrow_x + 0.35 * gap, 0.44),
                    xytext=(arrow_x - 0.35 * gap, 0.44),
                    xycoords=strip_axis.transAxes,
                    arrowprops={"arrowstyle": "->", "color": "#555555", "lw": 0.8},
                )
        if arrays.source_images is None:
            strip_axis.text(
                0.015,
                0.44,
                "開始\n★" if japanese else "start\n★",
                transform=strip_axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
            )
        else:
            strip_axis.text(
                0.025,
                0.68,
                "実入力" if japanese else "real input",
                transform=strip_axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#355f85",
            )
            strip_axis.text(
                0.025,
                0.21,
                "デコード" if japanese else "decoded",
                transform=strip_axis.transAxes,
                ha="center",
                va="center",
                fontsize=8,
                color="#b5285e",
            )

        figure.suptitle(
            "VAE潜在座標の歪みをWasserstein幾何で補正する"
            if japanese
            else "Correcting VAE latent distortion with Wasserstein geometry",
            fontsize=13.5,
            y=0.985,
        )
        figure.subplots_adjust(left=0.025, right=0.925, bottom=0.025, top=0.90)

    return OTSurfaceFigureResult(
        figure=figure,
        axes={
            "support": support_axis,
            "metric": metric_axis,
            "surface": surface_axis,
            "images": strip_axis,
            "colorbar": colorbar_axis,
        },
        local_metrics=local_metrics,
        graph_distances=graph_distances,
        anchor_index=anchor,
        endpoint_index=endpoint,
        path_indices=path,
        landmark_indices=landmarks,
    )


def save_bilingual_ot_surface_figures(
    data: OTSurfaceFigureData,
    output_stem: str | Path,
    *,
    dpi: int = 220,
    close: bool = True,
    **plot_options: Any,
) -> dict[str, tuple[Path, Path]]:
    """Save English and Japanese PNG/PDF figures.

    For ``output_stem="figure"``, this writes ``figure.png``, ``figure.pdf``,
    ``figure_ja.png``, and ``figure_ja.pdf``.
    """

    stem = Path(output_stem)
    if stem.suffix:
        stem = stem.with_suffix("")
    stem.parent.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, tuple[Path, Path]] = {}
    for language, japanese, suffix in (
        ("en", False, ""),
        ("ja", True, "_ja"),
    ):
        options = dict(plot_options)
        options["japanese"] = japanese
        result = plot_ot_surface_figure(data, **options)
        base = stem.with_name(stem.name + suffix)
        png = base.with_suffix(".png")
        pdf = base.with_suffix(".pdf")
        result.figure.savefig(
            png,
            dpi=dpi,
            bbox_inches="tight",
            metadata={"Software": "ot_surface_figure.py"},
        )
        result.figure.savefig(
            pdf,
            bbox_inches="tight",
            metadata={
                "Creator": "ot_surface_figure.py",
                "CreationDate": None,
                "ModDate": None,
            },
        )
        if close:
            plt.close(result.figure)
        outputs[language] = (png, pdf)
    return outputs


__all__ = [
    "LocalMetricEstimate",
    "OTSurfaceFigureData",
    "OTSurfaceFigureResult",
    "estimate_local_spd_metrics",
    "plot_ot_surface_figure",
    "save_bilingual_ot_surface_figures",
    "shortest_path_from_anchor",
    "unique_edges_from_faces",
]
