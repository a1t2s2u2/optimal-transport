"""Sinkhorn ア��ゴリズムの実装（対数領域版 + バッチ標準版）"""

import numpy as np


def logsumexp(x, axis):
    """数値安定な log-sum-exp"""
    x_max = np.max(x, axis=axis, keepdims=True)
    return np.log(np.sum(np.exp(x - x_max), axis=axis)) + np.squeeze(
        x_max, axis=axis
    )


def sinkhorn_log(a, b, C, eps, max_iter=100, tol=1e-6):
    """
    対数領域 Sinkhorn（単一ペア用）

    min_{P in U(a,b)} <C, P> - eps * H(P)

    Returns: P (輸送計画), f, g (Kantorovich ポテンシャル)
    """
    n, m = len(a), len(b)
    f, g = np.zeros(n), np.zeros(m)
    log_a = np.log(a + 1e-300)
    log_b = np.log(b + 1e-300)

    for _ in range(max_iter):
        f_prev = f.copy()
        f = eps * log_a - eps * logsumexp((g[np.newaxis, :] - C) / eps, axis=1)
        g = eps * log_b - eps * logsumexp((f[:, np.newaxis] - C) / eps, axis=0)
        if np.max(np.abs(f - f_prev)) < tol:
            break

    P = np.exp((f[:, np.newaxis] + g[np.newaxis, :] - C) / eps)
    return P, f, g


def sinkhorn_value_log(a, b, C, eps, max_iter=100, tol=1e-6):
    """単一ペアの正則化 OT 値とポテンシャルを返す"""
    _, f, g = sinkhorn_log(a, b, C, eps, max_iter=max_iter, tol=tol)
    value = np.dot(a, f) + np.dot(b, g)
    return value, f, g


def sinkhorn_batch(a, b, K, CK, eps, max_iter=50):
    """
    バッチ標準 Sinkhorn（Gibbs kernel K = exp(-C/eps) を事前計算）

    a, b: (B, N) 分布のバッチ
    K: (N, N) Gibbs kernel
    CK: (N, N) C * K

    Returns: cost (B,), g (B, N) Kantorovich ポテンシャル
    """
    B, N = a.shape
    v = np.ones((B, N))

    for _ in range(max_iter):
        u = a / (v @ K.T + 1e-300)
        u = np.clip(u, 0, 1e10)
        v = b / (u @ K + 1e-300)
        v = np.clip(v, 0, 1e10)

    g = eps * np.log(v + 1e-300)
    cost = np.sum(u * (v @ CK.T), axis=1)
    return cost, g


def sinkhorn_loss_batch(a, b, K, eps, max_iter=50):
    """
    バッチ版の正則化 OT 値と、その第2引数 b に関する勾配を返す。

    Returns: value (B,), g (B, N)
      value は双対目的 a·f + b·g による正則化 OT 値
      g は b に関する Kantorovich ポテンシャル
    """
    B, N = a.shape
    v = np.ones((B, N))

    for _ in range(max_iter):
        u = a / (v @ K.T + 1e-300)
        u = np.clip(u, 0, 1e10)
        v = b / (u @ K + 1e-300)
        v = np.clip(v, 0, 1e10)

    f = eps * np.log(u + 1e-300)
    g = eps * np.log(v + 1e-300)
    value = np.sum(a * f, axis=1) + np.sum(b * g, axis=1)
    return value, g
