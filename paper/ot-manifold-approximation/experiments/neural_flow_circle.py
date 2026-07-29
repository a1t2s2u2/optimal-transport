#!/usr/bin/env python3
"""Train a small neural Flow Matching model inside tangent charts of S^1.

The experiment is intentionally dependency-free.  A one-hidden-layer tanh MLP
models the time-dependent velocity v(t, z), Adam minimizes the conditional Flow
Matching objective, and Heun integration generates terminal latent samples.
It isolates the two terms in Proposition 2.7:

  * the exact geometric error of K tangent-line decoders; and
  * the learned local-flow distribution error in one-dimensional coordinates.
"""

from __future__ import annotations

import csv
import math
import random
from dataclasses import dataclass
from pathlib import Path
from statistics import NormalDist


OUT_DIR = Path(__file__).resolve().parent
SEED = 20260729
CHARTS = 8
TRAIN_STEPS = 6_000
BATCH_SIZE = 96
EVAL_SAMPLES = 2_048
ODE_STEPS = 40
WIDTHS = [4, 16, 64]


def circle_exact_error(k: int) -> float:
    value = (k / math.pi) * (
        3.0 * math.pi / (2.0 * k)
        - 2.0 * math.sin(math.pi / k)
        + 0.25 * math.sin(2.0 * math.pi / k)
    )
    return math.sqrt(max(0.0, value))


def target_coordinate(k: int, rng: random.Random) -> float:
    delta = rng.uniform(-math.pi / k, math.pi / k)
    return math.sin(delta)


@dataclass
class Metrics:
    width: int
    training_steps: int
    fm_validation_loss: float | None
    latent_w2: float
    geometric_w2: float
    orthogonal_coupling_upper: float
    triangle_upper: float


class VelocityMLP:
    """Scalar (t, z) -> velocity MLP with manual reverse-mode derivatives."""

    def __init__(self, width: int, seed: int) -> None:
        self.width = width
        rng = random.Random(seed)
        input_scale = 0.7
        output_scale = 0.25 / math.sqrt(width)
        self.wt = [rng.gauss(0.0, input_scale) for _ in range(width)]
        self.wz = [rng.gauss(0.0, input_scale) for _ in range(width)]
        self.b = [rng.gauss(0.0, 0.05) for _ in range(width)]
        self.a = [rng.gauss(0.0, output_scale) for _ in range(width)]
        self.c = 0.0

        self.m = {name: [0.0] * width for name in ("wt", "wz", "b", "a")}
        self.v = {name: [0.0] * width for name in ("wt", "wz", "b", "a")}
        self.m_c = 0.0
        self.v_c = 0.0
        self.adam_step = 0

    def predict(self, t: float, z: float) -> float:
        return self.c + math.fsum(
            output * math.tanh(time_weight * t + state_weight * z + bias)
            for time_weight, state_weight, bias, output in zip(
                self.wt, self.wz, self.b, self.a
            )
        )

    def train_batch(
        self,
        examples: list[tuple[float, float, float]],
        learning_rate: float,
    ) -> float:
        grad = {name: [0.0] * self.width for name in ("wt", "wz", "b", "a")}
        grad_c = 0.0
        total_loss = 0.0

        for t, z, target_velocity in examples:
            hidden = [
                math.tanh(self.wt[i] * t + self.wz[i] * z + self.b[i])
                for i in range(self.width)
            ]
            prediction = self.c + math.fsum(
                self.a[i] * hidden[i] for i in range(self.width)
            )
            error = prediction - target_velocity
            total_loss += error * error
            grad_c += error
            for i in range(self.width):
                grad["a"][i] += error * hidden[i]
                back = error * self.a[i] * (1.0 - hidden[i] * hidden[i])
                grad["wt"][i] += back * t
                grad["wz"][i] += back * z
                grad["b"][i] += back

        scale = 1.0 / len(examples)
        grad_c *= scale
        for values in grad.values():
            for i in range(self.width):
                values[i] *= scale

        self.adam_step += 1
        beta1 = 0.9
        beta2 = 0.999
        correction1 = 1.0 - beta1**self.adam_step
        correction2 = 1.0 - beta2**self.adam_step
        epsilon = 1e-8

        for name in ("wt", "wz", "b", "a"):
            parameter = getattr(self, name)
            for i in range(self.width):
                value = max(-20.0, min(20.0, grad[name][i]))
                self.m[name][i] = beta1 * self.m[name][i] + (1.0 - beta1) * value
                self.v[name][i] = beta2 * self.v[name][i] + (1.0 - beta2) * value * value
                first = self.m[name][i] / correction1
                second = self.v[name][i] / correction2
                parameter[i] -= learning_rate * first / (math.sqrt(second) + epsilon)

        clipped_c = max(-20.0, min(20.0, grad_c))
        self.m_c = beta1 * self.m_c + (1.0 - beta1) * clipped_c
        self.v_c = beta2 * self.v_c + (1.0 - beta2) * clipped_c * clipped_c
        first_c = self.m_c / correction1
        second_c = self.v_c / correction2
        self.c -= learning_rate * first_c / (math.sqrt(second_c) + epsilon)

        return total_loss * scale


def flow_matching_example(k: int, rng: random.Random) -> tuple[float, float, float]:
    source = rng.gauss(0.0, 1.0)
    target = target_coordinate(k, rng)
    time = rng.random()
    state = (1.0 - time) * source + time * target
    velocity = target - source
    return time, state, velocity


def train_model(width: int) -> VelocityMLP:
    model = VelocityMLP(width, SEED + 100 * width)
    rng = random.Random(SEED + 100 * width + 1)
    for step in range(TRAIN_STEPS):
        examples = [flow_matching_example(CHARTS, rng) for _ in range(BATCH_SIZE)]
        # A short warmup prevents large early Adam steps; cosine decay improves tails.
        warmup = min(1.0, (step + 1) / 100.0)
        decay = 0.5 * (1.0 + math.cos(math.pi * step / TRAIN_STEPS))
        learning_rate = 0.003 * warmup * (0.15 + 0.85 * decay)
        model.train_batch(examples, learning_rate)
    return model


def integrate(model: VelocityMLP, source: float) -> float:
    state = source
    step_size = 1.0 / ODE_STEPS
    for step in range(ODE_STEPS):
        time = step * step_size
        first = model.predict(time, state)
        predictor = state + step_size * first
        second = model.predict(time + step_size, predictor)
        state += 0.5 * step_size * (first + second)
    return state


def gaussian_quantiles(count: int) -> list[float]:
    normal = NormalDist()
    return [normal.inv_cdf((i + 0.5) / count) for i in range(count)]


def target_quantiles(k: int, count: int) -> list[float]:
    half_width = math.pi / k
    return [
        math.sin(-half_width + 2.0 * half_width * (i + 0.5) / count)
        for i in range(count)
    ]


def empirical_w2(sorted_first: list[float], sorted_second: list[float]) -> float:
    return math.sqrt(
        math.fsum((first - second) ** 2 for first, second in zip(sorted_first, sorted_second))
        / len(sorted_first)
    )


def validation_loss(model: VelocityMLP, count: int = 4_096) -> float:
    rng = random.Random(SEED + 999 + model.width)
    total = 0.0
    for _ in range(count):
        time, state, target = flow_matching_example(CHARTS, rng)
        error = model.predict(time, state) - target
        total += error * error
    return total / count


def evaluate(model: VelocityMLP | None) -> Metrics:
    source = gaussian_quantiles(EVAL_SAMPLES)
    target = target_quantiles(CHARTS, EVAL_SAMPLES)
    if model is None:
        generated = source
        width = 0
        steps = 0
        loss = None
    else:
        generated = sorted(integrate(model, value) for value in source)
        width = model.width
        steps = TRAIN_STEPS
        loss = validation_loss(model)

    latent_error = empirical_w2(target, generated)
    geometric_error = circle_exact_error(CHARTS)
    return Metrics(
        width=width,
        training_steps=steps,
        fm_validation_loss=loss,
        latent_w2=latent_error,
        geometric_w2=geometric_error,
        orthogonal_coupling_upper=math.hypot(geometric_error, latent_error),
        triangle_upper=geometric_error + latent_error,
    )


def write_outputs(rows: list[Metrics]) -> None:
    with (OUT_DIR / "neural_flow_results.csv").open(
        "w", newline="", encoding="utf-8"
    ) as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "charts",
                "hidden_width",
                "training_steps",
                "fm_validation_loss",
                "latent_W2",
                "geometric_W2_floor",
                "orthogonal_coupling_upper",
                "triangle_upper",
            ]
        )
        for row in rows:
            writer.writerow(
                [
                    CHARTS,
                    row.width,
                    row.training_steps,
                    "" if row.fm_validation_loss is None else f"{row.fm_validation_loss:.10g}",
                    f"{row.latent_w2:.10g}",
                    f"{row.geometric_w2:.10g}",
                    f"{row.orthogonal_coupling_upper:.10g}",
                    f"{row.triangle_upper:.10g}",
                ]
            )

    with (OUT_DIR / "neural_flow_table.tex").open("w", encoding="utf-8") as handle:
        handle.write("% Generated by experiments/neural_flow_circle.py; do not edit.\n")
        handle.write("\\begin{tabular}{rrrrr}\n")
        handle.write("\\hline\n")
        handle.write("hidden幅 & 学習step & FM loss & 潜在$W_2$ & 結合上界 \\\\\n")
        handle.write("\\hline\n")
        for row in rows:
            width = "未学習" if row.width == 0 else str(row.width)
            loss = "--" if row.fm_validation_loss is None else f"{row.fm_validation_loss:.4f}"
            handle.write(
                f"{width} & {row.training_steps} & {loss} & "
                f"{row.latent_w2:.4f} & {row.orthogonal_coupling_upper:.4f} \\\\\n"
            )
        handle.write("\\hline\n")
        handle.write("\\end{tabular}\n")

    print(f"K={CHARTS}, exact geometric W2 floor={rows[0].geometric_w2:.6f}")
    print("width    FM loss  latent W2  coupled upper")
    for row in rows:
        loss = "    n/a" if row.fm_validation_loss is None else f"{row.fm_validation_loss:7.4f}"
        print(f"{row.width:5d} {loss} {row.latent_w2:10.5f} {row.orthogonal_coupling_upper:14.5f}")


def main() -> None:
    rows = [evaluate(None)]
    for width in WIDTHS:
        rows.append(evaluate(train_model(width)))
    write_outputs(rows)


if __name__ == "__main__":
    main()
