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
        self.W = []
        self.b = []
        for i in range(self.n_layers):
            scale = np.sqrt(2.0 / sizes[i])
            self.W.append(np.random.randn(sizes[i], sizes[i + 1]) * scale)
            self.b.append(np.zeros(sizes[i + 1]))

    def forward(self, X):
        """入力 → ReLU 層 → Softmax（確率分布を出力）"""
        self.act = [X]
        self.pre = []
        h = X
        for i in range(self.n_layers - 1):
            z = h @ self.W[i] + self.b[i]
            self.pre.append(z)
            h = np.maximum(z, 0)
            self.act.append(h)
        z = h @ self.W[-1] + self.b[-1]
        self.pre.append(z)
        e = np.exp(z - z.max(axis=1, keepdims=True))
        q = e / e.sum(axis=1, keepdims=True)
        self.act.append(q)
        return q

    def backward(self, grad_q):
        """逆伝播: grad_q = g* (Kantorovich ポテンシャル)"""
        q = self.act[-1]
        N = q.shape[1]
        delta = N * q * (grad_q - (grad_q * q).sum(axis=1, keepdims=True))

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
