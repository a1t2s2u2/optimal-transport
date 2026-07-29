#!/usr/bin/env python3
"""Finite-K checks for tangent-atlas Wasserstein approximation rates.

The script intentionally uses only the Python standard library.  It performs no
gradient-based learning: prescribed grids or farthest-point sampling choose the
atlas centers, and Monte Carlo/quasi-Monte Carlo integration evaluates the
distance to the resulting union of affine tangent spaces.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Sequence, TypeVar


Point = TypeVar("Point")
Vector = tuple[float, ...]
Frame = tuple[Vector, ...]  # orthonormal columns

OUT_DIR = Path(__file__).resolve().parent
SEED = 20260729


@dataclass(frozen=True)
class Result:
    manifold: str
    q: int
    construction: str
    k: int
    error: float
    exact_error: float | None = None


def dot(x: Sequence[float], y: Sequence[float]) -> float:
    return sum(a * b for a, b in zip(x, y))


def normalize(x: Sequence[float]) -> Vector:
    length = math.sqrt(dot(x, x))
    if length < 1e-14:
        raise ValueError("cannot normalize a zero vector")
    return tuple(value / length for value in x)


def gaussian_unit_vector(dim: int, rng: random.Random) -> Vector:
    return normalize(tuple(rng.gauss(0.0, 1.0) for _ in range(dim)))


def haar_frame(dim: int, rank: int, rng: random.Random) -> Frame:
    """Gaussian QR with a fixed sign convention; returns orthonormal columns."""
    columns: list[Vector] = []
    while len(columns) < rank:
        vector = [rng.gauss(0.0, 1.0) for _ in range(dim)]
        for column in columns:
            coefficient = dot(vector, column)
            vector = [value - coefficient * basis for value, basis in zip(vector, column)]
        length = math.sqrt(dot(vector, vector))
        if length > 1e-12:
            columns.append(tuple(value / length for value in vector))
    return tuple(columns)


def matrix_cross_gram(x: Frame, y: Frame) -> tuple[tuple[float, ...], ...]:
    return tuple(tuple(dot(x_col, y_col) for y_col in y) for x_col in x)


def frame_chordal_sq(x: Frame, y: Frame) -> float:
    return sum((a - b) ** 2 for x_col, y_col in zip(x, y) for a, b in zip(x_col, y_col))


def stiefel_tangent_sq(x: Frame, y: Frame) -> float:
    gram = matrix_cross_gram(x, y)
    rank = len(x)
    total = 0.0
    for i in range(rank):
        for j in range(rank):
            symmetric = 0.5 * (gram[i][j] + gram[j][i])
            total += ((1.0 if i == j else 0.0) - symmetric) ** 2
    return total


def grassmann_chordal_sq(x: Frame, y: Frame) -> float:
    gram = matrix_cross_gram(x, y)
    cosine_sq_sum = sum(value * value for row in gram for value in row)
    return max(0.0, 2.0 * (len(x) - cosine_sq_sum))


def grassmann_tangent_sq(x: Frame, y: Frame) -> float:
    """2 sum_i sin(theta_i)^4 in the orthogonal-projector embedding."""
    gram = matrix_cross_gram(x, y)
    rank = len(x)
    cosine_sq_sum = sum(value * value for row in gram for value in row)
    # trace((C^T C)^2) = ||C^T C||_F^2.
    ctc = [
        [sum(gram[k][i] * gram[k][j] for k in range(rank)) for j in range(rank)]
        for i in range(rank)
    ]
    cosine_fourth_sum = sum(value * value for row in ctc for value in row)
    return max(0.0, 2.0 * (rank - 2.0 * cosine_sq_sum + cosine_fourth_sum))


def farthest_point_order(
    candidates: Sequence[Point],
    count: int,
    distance_sq: Callable[[Point, Point], float],
) -> list[Point]:
    """Greedy maximin ordering over a fixed candidate cloud."""
    if not candidates or count < 1 or count > len(candidates):
        raise ValueError("invalid farthest-point request")
    selected = [candidates[0]]
    nearest = [distance_sq(point, selected[0]) for point in candidates]
    used = {0}
    while len(selected) < count:
        index = max((i for i in range(len(candidates)) if i not in used), key=nearest.__getitem__)
        center = candidates[index]
        selected.append(center)
        used.add(index)
        for i, point in enumerate(candidates):
            candidate_distance = distance_sq(point, center)
            if candidate_distance < nearest[i]:
                nearest[i] = candidate_distance
    return selected


def errors_at_checkpoints(
    samples: Sequence[Point],
    centers: Sequence[Point],
    checkpoints: Iterable[int],
    tangent_distance_sq: Callable[[Point, Point], float],
) -> dict[int, float]:
    wanted = set(checkpoints)
    nearest = [math.inf] * len(samples)
    errors: dict[int, float] = {}
    for k, center in enumerate(centers, start=1):
        for i, sample in enumerate(samples):
            distance = tangent_distance_sq(center, sample)
            if distance < nearest[i]:
                nearest[i] = distance
        if k in wanted:
            errors[k] = math.sqrt(math.fsum(nearest) / len(nearest))
    return errors


def circle_exact_error(k: int) -> float:
    value = (k / math.pi) * (
        3.0 * math.pi / (2.0 * k)
        - 2.0 * math.sin(math.pi / k)
        + 0.25 * math.sin(2.0 * math.pi / k)
    )
    return math.sqrt(max(0.0, value))


def circle_experiment() -> list[Result]:
    checkpoints = [4, 8, 16, 32, 64, 128, 256]
    sample_count = 131_072
    # Midpoints of a deterministic quadrature grid avoid evaluating at centers.
    samples = [2.0 * math.pi * (i + 0.5) / sample_count for i in range(sample_count)]
    rows = []
    for k in checkpoints:
        spacing = 2.0 * math.pi / k
        total = 0.0
        for theta in samples:
            delta = (theta + 0.5 * spacing) % spacing - 0.5 * spacing
            distance = 1.0 - math.cos(delta)
            total += distance * distance
        rows.append(Result("$S^1$", 1, "等間隔", k, math.sqrt(total / sample_count), circle_exact_error(k)))
    return rows


def fibonacci_sphere(count: int, offset: float = 0.5) -> list[Vector]:
    golden_ratio = (1.0 + math.sqrt(5.0)) / 2.0
    points = []
    for i in range(count):
        z = 1.0 - 2.0 * (i + offset) / count
        radius = math.sqrt(max(0.0, 1.0 - z * z))
        angle = 2.0 * math.pi * i / golden_ratio
        points.append((radius * math.cos(angle), radius * math.sin(angle), z))
    return points


def sphere_tangent_sq(center: Vector, sample: Vector) -> float:
    distance = max(0.0, 1.0 - dot(center, sample))
    return distance * distance


def sphere_experiment() -> list[Result]:
    checkpoints = [8, 16, 32, 64, 128, 256]
    samples = fibonacci_sphere(16_387, offset=0.137)
    rows = []
    for k in checkpoints:
        centers = fibonacci_sphere(k)
        error = errors_at_checkpoints(samples, centers, [k], sphere_tangent_sq)[k]
        rows.append(Result("$S^2$", 2, "Fibonacci", k, error))
    return rows


def torus_tangent_sq(center: tuple[float, float], sample: tuple[float, float]) -> float:
    first = 1.0 - math.cos(center[0] - sample[0])
    second = 1.0 - math.cos(center[1] - sample[1])
    return first * first + second * second


def torus_experiment() -> list[Result]:
    side_lengths = [3, 4, 6, 8, 12, 16, 24]
    # Irrational rank-1 lattice rule on [0, 2pi)^2.
    sample_count = 32_771
    alpha = math.sqrt(2.0)
    samples = [
        (2.0 * math.pi * (i + 0.5) / sample_count, 2.0 * math.pi * ((i * alpha) % 1.0))
        for i in range(sample_count)
    ]
    rows = []
    for side in side_lengths:
        spacing = 2.0 * math.pi / side
        k = side * side
        total = 0.0
        for theta, phi in samples:
            first_delta = (theta + 0.5 * spacing) % spacing - 0.5 * spacing
            second_delta = (phi + 0.5 * spacing) % spacing - 0.5 * spacing
            first = 1.0 - math.cos(first_delta)
            second = 1.0 - math.cos(second_delta)
            total += first * first + second * second
        error = math.sqrt(total / sample_count)
        rows.append(Result(r"$\mathbb T^2$", 2, r"$m\times m$ 格子", k, error))
    return rows


def stiefel_experiment() -> list[Result]:
    checkpoints = [8, 16, 32, 64, 128, 256]
    rng_candidates = random.Random(SEED + 31)
    rng_samples = random.Random(SEED + 32)
    candidates = [haar_frame(3, 2, rng_candidates) for _ in range(6_000)]
    samples = [haar_frame(3, 2, rng_samples) for _ in range(24_000)]
    centers = farthest_point_order(candidates, checkpoints[-1], frame_chordal_sq)
    errors = errors_at_checkpoints(samples, centers, checkpoints, stiefel_tangent_sq)
    return [Result("$\\St(3,2)$", 3, "FPS", k, errors[k]) for k in checkpoints]


def grassmann_experiment() -> list[Result]:
    checkpoints = [8, 16, 32, 64, 128, 256]
    rng_candidates = random.Random(SEED + 41)
    rng_samples = random.Random(SEED + 42)
    candidates = [haar_frame(4, 2, rng_candidates) for _ in range(6_000)]
    samples = [haar_frame(4, 2, rng_samples) for _ in range(24_000)]
    centers = farthest_point_order(candidates, checkpoints[-1], grassmann_chordal_sq)
    errors = errors_at_checkpoints(samples, centers, checkpoints, grassmann_tangent_sq)
    return [Result("$\\Gr(4,2)$", 4, "FPS", k, errors[k]) for k in checkpoints]


def regression(rows: Sequence[Result], tail: int = 4) -> tuple[float, float]:
    points = rows[-tail:]
    xs = [math.log(row.k) for row in points]
    ys = [math.log(row.error) for row in points]
    x_mean = math.fsum(xs) / len(xs)
    y_mean = math.fsum(ys) / len(ys)
    denominator = math.fsum((x - x_mean) ** 2 for x in xs)
    slope = math.fsum((x - x_mean) * (y - y_mean) for x, y in zip(xs, ys)) / denominator
    fitted = [y_mean + slope * (x - x_mean) for x in xs]
    residual = math.fsum((y - prediction) ** 2 for y, prediction in zip(ys, fitted))
    total = math.fsum((y - y_mean) ** 2 for y in ys)
    r_squared = 1.0 - residual / total if total > 0.0 else 1.0
    return slope, r_squared


def write_results(groups: Sequence[Sequence[Result]]) -> None:
    flat = [row for group in groups for row in group]
    with (OUT_DIR / "results.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["manifold", "intrinsic_dimension", "construction", "K", "W2_upper", "exact_W2"])
        for row in flat:
            writer.writerow(
                [
                    row.manifold.replace("$", ""),
                    row.q,
                    row.construction.replace("$", ""),
                    row.k,
                    f"{row.error:.12g}",
                    "" if row.exact_error is None else f"{row.exact_error:.12g}",
                ]
            )

    summaries = []
    for group in groups:
        slope, r_squared = regression(group)
        circle_relative = None
        if group[0].exact_error is not None:
            circle_relative = max(
                abs(row.error - row.exact_error) / row.exact_error
                for row in group
                if row.exact_error is not None
            )
        summaries.append((group[0], group[0].k, group[-1].k, slope, r_squared, circle_relative))

    with (OUT_DIR / "slopes.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["manifold", "intrinsic_dimension", "construction", "K_min", "K_max", "predicted_slope", "observed_slope", "R_squared", "max_circle_relative_error"]
        )
        for first, k_min, k_max, slope, r_squared, circle_relative in summaries:
            writer.writerow(
                [
                    first.manifold.replace("$", ""),
                    first.q,
                    first.construction.replace("$", ""),
                    k_min,
                    k_max,
                    f"{-2.0 / first.q:.6f}",
                    f"{slope:.6f}",
                    f"{r_squared:.6f}",
                    "" if circle_relative is None else f"{circle_relative:.6g}",
                ]
            )

    with (OUT_DIR / "results_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Generated by experiments/run.py; do not edit by hand.\n")
        handle.write("\\begin{tabular}{lrrrrr}\n")
        handle.write("\\hline\n")
        handle.write("多様体 & $q$ & $K$ の範囲 & 理論傾き & 観測傾き & $R^2$ \\\\\n")
        handle.write("\\hline\n")
        for first, k_min, k_max, slope, r_squared, _ in summaries:
            handle.write(
                f"{first.manifold} & {first.q} & {k_min}--{k_max} & {-2.0 / first.q:.3f} & {slope:.3f} & {r_squared:.4f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")

    print("manifold       predicted   observed       R^2")
    for first, _, _, slope, r_squared, _ in summaries:
        print(f"{first.manifold:14s} {-2.0 / first.q:9.3f} {slope:10.3f} {r_squared:10.4f}")


def main() -> None:
    groups = [
        circle_experiment(),
        sphere_experiment(),
        torus_experiment(),
        stiefel_experiment(),
        grassmann_experiment(),
    ]
    write_results(groups)


if __name__ == "__main__":
    main()
