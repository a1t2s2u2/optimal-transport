"""Sinkhorn 損失による Autoencoder"""

import numpy as np


class SinkhornAutoencoder:
    """
    入出力ともに確率分布。損失は分布間の正則化 Kantorovich コスト。
    勾配は Kantorovich ポテンシャル g* = nabla_q L^eps で与えられる。
    """

    def __init__(self, sizes, lr=0.01):
        self.lr = lr
        self.n_layers = len(sizes) - 1
        self.n_encoder = self.n_layers // 2
        self.W, self.b = [], []
        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        """入力 → ReLU 層 → ReLU+正規化（確率分布を出力）"""
        self.act = [X]
        self.pre = []
        h = X
        for i in range(self.n_layers - 1):
            z = h @ self.W[i] + self.b[i]
            self.pre.append(z)
            h = np.maximum(z, 0)
            self.act.append(h)
        # 出力層: ReLU + 正規化 → 確率分布（スパースな出力が可能）
        z = h @ self.W[-1] + self.b[-1]
        self.pre.append(z)
        r = np.maximum(z, 0) + 1e-8
        q = r / r.sum(axis=1, keepdims=True)
        self.act.append(q)
        return q

    def backward(self, grad_q):
        """逆伝播: grad_q = g* (Kantorovich ポテンシャル)"""
        q = self.act[-1]
        z_out = self.pre[-1]
        # ReLU+正規化の勾配: delta_k = (z_k > 0) / S * (g_k - <g, q>)
        mask = (z_out > 0).astype(np.float64)
        S = (np.maximum(z_out, 0) + 1e-8).sum(axis=1, keepdims=True)
        gq = (grad_q * q).sum(axis=1, keepdims=True)
        delta = mask / S * (grad_q - gq)

        for i in range(self.n_layers - 1, -1, -1):
            B = delta.shape[0]
            dW = self.act[i].T @ delta / B
            db = delta.mean(axis=0)
            if i > 0:
                delta = (delta @ self.W[i].T) * (self.pre[i - 1] > 0)
            self.W[i] -= self.lr * dW
            self.b[i] -= self.lr * db

    def encode(self, X):
        """Encoder 部分のみ実行して潜在表現を返す"""
        h = X
        for i in range(self.n_encoder):
            h = np.maximum(h @ self.W[i] + self.b[i], 0)
        return h
