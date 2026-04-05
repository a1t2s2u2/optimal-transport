"""Cuturi (2013) の動機を可視化するための小さな比較実験"""

import time

import numpy as np

from .sinkhorn import sinkhorn_loss_batch, sinkhorn_value_log


def build_line_cost(size):
    """1 次元格子上の |x-y| コスト行列"""
    x = np.linspace(0.0, 1.0, size)
    return np.abs(x[:, np.newaxis] - x[np.newaxis, :])


def exact_w1_line(a, b):
    """1 次元離散分布の厳密 W1 距離"""
    if len(a) == 1:
        return 0.0
    step = 1.0 / (len(a) - 1)
    return step * np.sum(np.abs(np.cumsum(a - b)[:-1]))


def benchmark_sinkhorn_scaling(
    sizes=(256, 512, 768, 1024, 1536, 2048),
    eps=0.05,
    max_iter=100,
    repeats=3,
    calls_per_repeat=2,
    seed=0,
):
    """
    Sinkhorn の実測時間を測る。

    厳密 LP はこの実装に含めず、理論参照として O(n^3 log n) を同時に返す。
    """
    rng = np.random.default_rng(seed)
    sizes = np.asarray(sizes, dtype=np.int64)
    times = []

    for size in sizes:
        C = build_line_cost(size)
        K = np.exp(-C / eps)

        # ウォームアップ
        a = rng.random((1, size))
        b = rng.random((1, size))
        a /= a.sum(axis=1, keepdims=True)
        b /= b.sum(axis=1, keepdims=True)
        sinkhorn_loss_batch(a, b, K, eps, max_iter=max_iter)

        runs = []
        for _ in range(repeats):
            start = time.perf_counter()
            for _ in range(calls_per_repeat):
                sinkhorn_loss_batch(a, b, K, eps, max_iter=max_iter)
            runs.append((time.perf_counter() - start) / calls_per_repeat)

        times.append(np.median(runs))

    times = np.asarray(times)
    slope = np.polyfit(np.log(sizes), np.log(times), 1)[0]

    quad_ref = times[0] * (sizes / sizes[0]) ** 2
    lp_ref_raw = sizes**3 * np.log(sizes)
    lp_ref = times[0] * lp_ref_raw / lp_ref_raw[0]

    return {
        "sizes": sizes,
        "times": times,
        "quad_ref": quad_ref,
        "lp_ref": lp_ref,
        "slope": slope,
    }


def smoothness_profile(
    size=41,
    eps_values=(0.20, 0.05, 0.01),
    n_points=201,
    max_iter=400,
):
    """
    1 次元の経路上で厳密 W1 と entropic OT の損失曲面を比較する。

    exact W1 は区分線形、entropic OT は滑らかになることを可視化する。
    """
    C = build_line_cost(size)
    x = np.linspace(0.0, 1.0, size)

    a = np.exp(-0.5 * ((x - 0.5) / 0.12) ** 2)
    a += 0.35 * np.exp(-0.5 * ((x - 0.7) / 0.07) ** 2)
    a /= a.sum()

    left = np.zeros(size)
    right = np.zeros(size)
    left[size // 8] = 1.0
    right[-size // 8 - 1] = 1.0

    ts = np.linspace(0.0, 1.0, n_points)
    exact = np.zeros(n_points)
    entropic = {eps: np.zeros(n_points) for eps in eps_values}

    for idx, t in enumerate(ts):
        b = (1.0 - t) * left + t * right
        b = (b + 1e-12) / (b.sum() + size * 1e-12)
        exact[idx] = exact_w1_line(a, b)
        for eps in eps_values:
            value, _, _ = sinkhorn_value_log(a, b, C, eps, max_iter=max_iter)
            entropic[eps][idx] = value

    exact_grad = np.gradient(exact, ts)
    entropic_grad = {eps: np.gradient(values, ts) for eps, values in entropic.items()}

    return {
        "ts": ts,
        "exact": exact,
        "entropic": entropic,
        "exact_grad": exact_grad,
        "entropic_grad": entropic_grad,
        "source": a,
        "left": left,
        "right": right,
    }
