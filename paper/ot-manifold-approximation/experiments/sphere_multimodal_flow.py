#!/usr/bin/env python3
"""Conditional Flow Matching on a multimodal distribution over S^2.

Six tangent-plane decoders are anchored at the coordinate axes.  Every decoder
is assigned a three-mode distribution in its two-dimensional tangent
coordinates, yielding an 18-mode target distribution on the sphere.  A shared
chart-conditional MLP learns the six local velocity fields.

The implementation intentionally uses only the Python standard library.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


Vector2 = tuple[float, float]
Vector3 = tuple[float, float, float]
Example = tuple[float, Vector2, Vector2, int]

OUT_DIR = Path(__file__).resolve().parent
SEED = 20260730
CHARTS = 6
MODES_PER_CHART = 3
WIDTHS = [4, 16, 64]
TRAIN_STEPS = 6_000
BATCH_SIZE = 96
HISTORY_INTERVAL = 100
VALIDATION_SAMPLES = 2_048
EVAL_PER_CHART = 384
PLOT_PER_CHART = 128
ODE_STEPS = 40
SLICE_DIRECTIONS = 48
MODE_RADIUS = 0.62
MODE_NOISE = 0.07


def dot(first: Sequence[float], second: Sequence[float]) -> float:
    return math.fsum(a * b for a, b in zip(first, second))


def norm(vector: Sequence[float]) -> float:
    return math.sqrt(dot(vector, vector))


def normalize(vector: Sequence[float]) -> Vector3:
    length = norm(vector)
    return tuple(value / length for value in vector)  # type: ignore[return-value]


def cross(first: Vector3, second: Vector3) -> Vector3:
    return (
        first[1] * second[2] - first[2] * second[1],
        first[2] * second[0] - first[0] * second[2],
        first[0] * second[1] - first[1] * second[0],
    )


ANCHORS: list[Vector3] = [
    (1.0, 0.0, 0.0),
    (-1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, -1.0, 0.0),
    (0.0, 0.0, 1.0),
    (0.0, 0.0, -1.0),
]


def tangent_basis(anchor: Vector3) -> tuple[Vector3, Vector3]:
    helper = (0.0, 0.0, 1.0) if abs(anchor[2]) < 0.9 else (1.0, 0.0, 0.0)
    first = normalize(cross(helper, anchor))
    second = normalize(cross(anchor, first))
    return first, second


BASES = [tangent_basis(anchor) for anchor in ANCHORS]


def add_scaled(
    anchor: Vector3,
    first: Vector3,
    first_scale: float,
    second: Vector3,
    second_scale: float,
) -> Vector3:
    return tuple(
        center + first_scale * first_value + second_scale * second_value
        for center, first_value, second_value in zip(anchor, first, second)
    )  # type: ignore[return-value]


def mode_center(chart: int, mode: int) -> Vector2:
    phase = 0.31 * chart + 2.0 * math.pi * mode / MODES_PER_CHART
    return MODE_RADIUS * math.cos(phase), MODE_RADIUS * math.sin(phase)


def sample_target_for_chart(
    chart: int,
    rng: random.Random,
) -> tuple[Vector2, Vector3]:
    center = mode_center(chart, rng.randrange(MODES_PER_CHART))
    local = (
        center[0] + MODE_NOISE * rng.gauss(0.0, 1.0),
        center[1] + MODE_NOISE * rng.gauss(0.0, 1.0),
    )
    first, second = BASES[chart]
    sphere_point = normalize(add_scaled(ANCHORS[chart], first, local[0], second, local[1]))
    projected_coordinate = (
        dot(tuple(value - center_value for value, center_value in zip(sphere_point, ANCHORS[chart])), first),
        dot(tuple(value - center_value for value, center_value in zip(sphere_point, ANCHORS[chart])), second),
    )
    return projected_coordinate, sphere_point


def flow_example(rng: random.Random) -> Example:
    chart = rng.randrange(CHARTS)
    target, _ = sample_target_for_chart(chart, rng)
    source = (rng.gauss(0.0, 1.0), rng.gauss(0.0, 1.0))
    time = rng.random()
    state = (
        (1.0 - time) * source[0] + time * target[0],
        (1.0 - time) * source[1] + time * target[1],
    )
    velocity = target[0] - source[0], target[1] - source[1]
    return time, state, velocity, chart


def fixed_validation_examples() -> list[Example]:
    rng = random.Random(SEED + 999)
    return [flow_example(rng) for _ in range(VALIDATION_SAMPLES)]


@dataclass
class HistoryPoint:
    step: int
    training_loss_ema: float | None
    validation_loss: float


@dataclass
class Metrics:
    width: int
    validation_loss: float | None
    local_sliced_w2: float
    geometric_rms: float
    generated_radial_rmse: float


class ConditionalVelocityMLP:
    """One-hidden-layer (t, z, one-hot chart) -> R^2 velocity network."""

    def __init__(self, width: int, seed: int) -> None:
        self.width = width
        self.input_dim = 3 + CHARTS
        rng = random.Random(seed)
        self.w = [
            [rng.gauss(0.0, 0.55) for _ in range(self.input_dim)]
            for _ in range(width)
        ]
        self.b = [rng.gauss(0.0, 0.04) for _ in range(width)]
        self.a = [
            [rng.gauss(0.0, 0.22 / math.sqrt(width)) for _ in range(width)]
            for _ in range(2)
        ]
        self.c = [0.0, 0.0]

        self.m_w = [[0.0] * self.input_dim for _ in range(width)]
        self.v_w = [[0.0] * self.input_dim for _ in range(width)]
        self.m_b = [0.0] * width
        self.v_b = [0.0] * width
        self.m_a = [[0.0] * width for _ in range(2)]
        self.v_a = [[0.0] * width for _ in range(2)]
        self.m_c = [0.0, 0.0]
        self.v_c = [0.0, 0.0]
        self.adam_step = 0

    def inputs(self, time: float, state: Vector2, chart: int) -> list[float]:
        values = [time, state[0], state[1]] + [0.0] * CHARTS
        values[3 + chart] = 1.0
        return values

    def predict(self, time: float, state: Vector2, chart: int) -> Vector2:
        inputs = self.inputs(time, state, chart)
        hidden = [
            math.tanh(math.fsum(weight * value for weight, value in zip(row, inputs)) + bias)
            for row, bias in zip(self.w, self.b)
        ]
        return (
            self.c[0] + math.fsum(weight * value for weight, value in zip(self.a[0], hidden)),
            self.c[1] + math.fsum(weight * value for weight, value in zip(self.a[1], hidden)),
        )

    @staticmethod
    def adam_update(
        parameter: float,
        first_moment: float,
        second_moment: float,
        gradient: float,
        learning_rate: float,
        correction1: float,
        correction2: float,
    ) -> tuple[float, float, float]:
        gradient = max(-20.0, min(20.0, gradient))
        first_moment = 0.9 * first_moment + 0.1 * gradient
        second_moment = 0.999 * second_moment + 0.001 * gradient * gradient
        first = first_moment / correction1
        second = second_moment / correction2
        parameter -= learning_rate * first / (math.sqrt(second) + 1e-8)
        return parameter, first_moment, second_moment

    def train_batch(self, examples: list[Example], learning_rate: float) -> float:
        grad_w = [[0.0] * self.input_dim for _ in range(self.width)]
        grad_b = [0.0] * self.width
        grad_a = [[0.0] * self.width for _ in range(2)]
        grad_c = [0.0, 0.0]
        total_loss = 0.0

        for time, state, target, chart in examples:
            inputs = self.inputs(time, state, chart)
            hidden = [
                math.tanh(math.fsum(weight * value for weight, value in zip(row, inputs)) + bias)
                for row, bias in zip(self.w, self.b)
            ]
            prediction = [
                self.c[output]
                + math.fsum(
                    weight * value for weight, value in zip(self.a[output], hidden)
                )
                for output in range(2)
            ]
            errors = [prediction[index] - target[index] for index in range(2)]
            total_loss += errors[0] ** 2 + errors[1] ** 2

            for output in range(2):
                grad_c[output] += errors[output]
                for unit in range(self.width):
                    grad_a[output][unit] += errors[output] * hidden[unit]

            for unit in range(self.width):
                back = (
                    errors[0] * self.a[0][unit] + errors[1] * self.a[1][unit]
                ) * (1.0 - hidden[unit] ** 2)
                grad_b[unit] += back
                for index in range(self.input_dim):
                    grad_w[unit][index] += back * inputs[index]

        scale = 1.0 / len(examples)
        self.adam_step += 1
        correction1 = 1.0 - 0.9**self.adam_step
        correction2 = 1.0 - 0.999**self.adam_step

        for unit in range(self.width):
            for index in range(self.input_dim):
                (
                    self.w[unit][index],
                    self.m_w[unit][index],
                    self.v_w[unit][index],
                ) = self.adam_update(
                    self.w[unit][index],
                    self.m_w[unit][index],
                    self.v_w[unit][index],
                    grad_w[unit][index] * scale,
                    learning_rate,
                    correction1,
                    correction2,
                )
            self.b[unit], self.m_b[unit], self.v_b[unit] = self.adam_update(
                self.b[unit],
                self.m_b[unit],
                self.v_b[unit],
                grad_b[unit] * scale,
                learning_rate,
                correction1,
                correction2,
            )

        for output in range(2):
            for unit in range(self.width):
                (
                    self.a[output][unit],
                    self.m_a[output][unit],
                    self.v_a[output][unit],
                ) = self.adam_update(
                    self.a[output][unit],
                    self.m_a[output][unit],
                    self.v_a[output][unit],
                    grad_a[output][unit] * scale,
                    learning_rate,
                    correction1,
                    correction2,
                )
            self.c[output], self.m_c[output], self.v_c[output] = self.adam_update(
                self.c[output],
                self.m_c[output],
                self.v_c[output],
                grad_c[output] * scale,
                learning_rate,
                correction1,
                correction2,
            )
        return total_loss * scale


def validation_loss(model: ConditionalVelocityMLP, examples: list[Example]) -> float:
    total = 0.0
    for time, state, target, chart in examples:
        prediction = model.predict(time, state, chart)
        total += (prediction[0] - target[0]) ** 2 + (prediction[1] - target[1]) ** 2
    return total / len(examples)


def train_model(width: int) -> tuple[ConditionalVelocityMLP, list[HistoryPoint]]:
    model = ConditionalVelocityMLP(width, SEED + 100 * width)
    rng = random.Random(SEED + 100 * width + 1)
    validation = fixed_validation_examples()
    history = [HistoryPoint(0, None, validation_loss(model, validation))]
    loss_ema: float | None = None

    for step_index in range(TRAIN_STEPS):
        examples = [flow_example(rng) for _ in range(BATCH_SIZE)]
        warmup = min(1.0, (step_index + 1) / 100.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * step_index / TRAIN_STEPS))
        learning_rate = 0.003 * warmup * (0.15 + 0.85 * decay)
        batch_loss = model.train_batch(examples, learning_rate)
        loss_ema = (
            batch_loss if loss_ema is None else 0.98 * loss_ema + 0.02 * batch_loss
        )

        step = step_index + 1
        if step % HISTORY_INTERVAL == 0:
            point = HistoryPoint(step, loss_ema, validation_loss(model, validation))
            history.append(point)
            if step % 1_000 == 0:
                print(
                    f"S2 width={width:2d} step={step:4d} "
                    f"train_ema={loss_ema:.5f} val={point.validation_loss:.5f}"
                )
    return model, history


def integrate(
    model: ConditionalVelocityMLP,
    source: Vector2,
    chart: int,
) -> Vector2:
    state = source
    step_size = 1.0 / ODE_STEPS
    for step in range(ODE_STEPS):
        time = step * step_size
        first = model.predict(time, state, chart)
        predictor = (
            state[0] + step_size * first[0],
            state[1] + step_size * first[1],
        )
        second = model.predict(time + step_size, predictor, chart)
        state = (
            state[0] + 0.5 * step_size * (first[0] + second[0]),
            state[1] + 0.5 * step_size * (first[1] + second[1]),
        )
    return state


def fixed_evaluation_data() -> tuple[list[list[Vector2]], list[list[Vector2]], list[list[Vector3]]]:
    sources: list[list[Vector2]] = []
    targets: list[list[Vector2]] = []
    sphere_points: list[list[Vector3]] = []
    for chart in range(CHARTS):
        source_rng = random.Random(SEED + 5_000 + chart)
        target_rng = random.Random(SEED + 6_000 + chart)
        sources.append(
            [
                (source_rng.gauss(0.0, 1.0), source_rng.gauss(0.0, 1.0))
                for _ in range(EVAL_PER_CHART)
            ]
        )
        chart_targets = []
        chart_points = []
        for _ in range(EVAL_PER_CHART):
            target, point = sample_target_for_chart(chart, target_rng)
            chart_targets.append(target)
            chart_points.append(point)
        targets.append(chart_targets)
        sphere_points.append(chart_points)
    return sources, targets, sphere_points


def sliced_w2(
    targets: list[list[Vector2]],
    generated: list[list[Vector2]],
) -> float:
    total = 0.0
    for direction in range(SLICE_DIRECTIONS):
        angle = math.pi * (direction + 0.5) / SLICE_DIRECTIONS
        axis = math.cos(angle), math.sin(angle)
        for chart in range(CHARTS):
            target_projection = sorted(dot(point, axis) for point in targets[chart])
            generated_projection = sorted(dot(point, axis) for point in generated[chart])
            total += math.fsum(
                (first - second) ** 2
                for first, second in zip(target_projection, generated_projection)
            ) / EVAL_PER_CHART
    return math.sqrt(total / (SLICE_DIRECTIONS * CHARTS))


def decode(chart: int, coordinate: Vector2) -> Vector3:
    first, second = BASES[chart]
    return add_scaled(ANCHORS[chart], first, coordinate[0], second, coordinate[1])


def evaluate(
    model: ConditionalVelocityMLP | None,
    sources: list[list[Vector2]],
    targets: list[list[Vector2]],
    sphere_points: list[list[Vector3]],
) -> tuple[Metrics, list[list[Vector2]]]:
    if model is None:
        generated = sources
        width = 0
        loss = None
    else:
        generated = [
            [integrate(model, source, chart) for source in sources[chart]]
            for chart in range(CHARTS)
        ]
        width = model.width
        loss = validation_loss(model, fixed_validation_examples())

    geometry_total = 0.0
    radial_total = 0.0
    count = CHARTS * EVAL_PER_CHART
    for chart in range(CHARTS):
        for point in sphere_points[chart]:
            distance = 1.0 - dot(ANCHORS[chart], point)
            geometry_total += distance * distance
        for coordinate in generated[chart]:
            radial = norm(decode(chart, coordinate)) - 1.0
            radial_total += radial * radial

    return (
        Metrics(
            width=width,
            validation_loss=loss,
            local_sliced_w2=sliced_w2(targets, generated),
            geometric_rms=math.sqrt(geometry_total / count),
            generated_radial_rmse=math.sqrt(radial_total / count),
        ),
        generated,
    )


def write_outputs(
    rows: list[Metrics],
    histories: dict[int, list[HistoryPoint]],
    generated_by_width: dict[int, list[list[Vector2]]],
    sphere_points: list[list[Vector3]],
) -> None:
    with (OUT_DIR / "sphere_flow_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "hidden_width",
                "training_steps",
                "fm_validation_loss",
                "local_sliced_W2",
                "geometric_RMS",
                "generated_radial_RMSE",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    row.width,
                    0 if row.width == 0 else TRAIN_STEPS,
                    "" if row.validation_loss is None else f"{row.validation_loss:.10g}",
                    f"{row.local_sliced_w2:.10g}",
                    f"{row.geometric_rms:.10g}",
                    f"{row.generated_radial_rmse:.10g}",
                ]
            )

    with (OUT_DIR / "sphere_flow_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Generated by sphere_multimodal_flow.py; do not edit.\n")
        handle.write("\\begin{tabular}{rrrrr}\n")
        handle.write("\\hline\n")
        handle.write("hidden幅 & 学習step & FM loss & 局所SW$_2$ & 半径RMSE \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            width = "未学習" if row.width == 0 else str(row.width)
            loss = "--" if row.validation_loss is None else f"{row.validation_loss:.4f}"
            handle.write(
                f"{width} & {0 if row.width == 0 else TRAIN_STEPS} & {loss} & "
                f"{row.local_sliced_w2:.4f} & {row.generated_radial_rmse:.4f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")

    for width, history in histories.items():
        with (OUT_DIR / f"sphere_flow_history_width{width}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle, lineterminator="\n")
            writer.writerow(["step", "training_loss_ema", "validation_loss"])
            for point in history:
                writer.writerow(
                    [
                        point.step,
                        (
                            ""
                            if point.training_loss_ema is None
                            else f"{point.training_loss_ema:.10g}"
                        ),
                        f"{point.validation_loss:.10g}",
                    ]
                )

    with (OUT_DIR / "sphere_flow_outputs.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(
            [
                "target_x",
                "target_y",
                "target_z",
                *[
                    f"width_{width}_{axis}"
                    for width in WIDTHS
                    for axis in ("x", "y", "z")
                ],
            ]
        )
        for chart in range(CHARTS):
            for index in range(PLOT_PER_CHART):
                row: list[float] = list(sphere_points[chart][index])
                for width in WIDTHS:
                    row.extend(decode(chart, generated_by_width[width][chart][index]))
                writer.writerow(f"{value:.10g}" for value in row)

    print(f"S2 local geometric RMS={rows[0].geometric_rms:.6f}")
    print("width    FM loss  local SW2  radial RMSE")
    for row in rows:
        loss = "    n/a" if row.validation_loss is None else f"{row.validation_loss:7.4f}"
        print(
            f"{row.width:5d} {loss} {row.local_sliced_w2:10.5f} "
            f"{row.generated_radial_rmse:12.5f}"
        )


def main() -> None:
    sources, targets, sphere_points = fixed_evaluation_data()
    baseline, baseline_generated = evaluate(None, sources, targets, sphere_points)
    rows = [baseline]
    histories: dict[int, list[HistoryPoint]] = {}
    generated_by_width = {0: baseline_generated}

    for width in WIDTHS:
        model, history = train_model(width)
        metrics, generated = evaluate(model, sources, targets, sphere_points)
        rows.append(metrics)
        histories[width] = history
        generated_by_width[width] = generated

    write_outputs(rows, histories, generated_by_width, sphere_points)


if __name__ == "__main__":
    main()
