"""
Sinkhorn Autoencoder for MNIST — エントリポイント

理論的背景は chapters/11_application.tex を参照
実行: uv run python main.py
"""

import numpy as np
from src import load_mnist, image_to_dist, build_pixel_cost
from src import sinkhorn_log, sinkhorn_batch
from src import SinkhornAutoencoder
from src.visualize import (
    plot_distributions,
    plot_transport,
    plot_interpolation,
    plot_reconstructions,
    plot_latent_space,
    plot_training_curve,
)


def demo_image_ot(X_train, y_train, C_pixel, pos):
    """第1部: 画像間の最適輸送の可視化"""
    print("\n=== Part 1: OT between images ===")
    idx_3 = np.where(y_train == 3)[0][0]
    idx_8 = np.where(y_train == 8)[0][0]
    a = image_to_dist(X_train[idx_3])
    b = image_to_dist(X_train[idx_8])

    print("Computing Sinkhorn (784x784)...")
    P, _, _ = sinkhorn_log(a, b, C_pixel, eps=0.05, max_iter=200)
    print(f"Transport cost <C, P*> = {np.sum(C_pixel * P):.6f}")
    print(
        f"Marginal errors: {np.linalg.norm(P.sum(1) - a):.1e}, "
        f"{np.linalg.norm(P.sum(0) - b):.1e}"
    )

    plot_transport(a, b, P, pos, "source (3)", "target (8)")
    plot_interpolation(P, pos, title="McCann interpolation: 3 -> 8")


def train_autoencoder(X_tr, C_pixel, eps, n_train, batch_size, epochs, lr):
    """第2部: Sinkhorn Autoencoder の訓練"""
    print("\n=== Part 2: Sinkhorn Autoencoder ===")

    K = np.exp(-C_pixel / eps)
    CK = C_pixel * K

    model = SinkhornAutoencoder([784, 256, 64, 256, 784], lr=lr)
    history = []

    print(f"Training: {n_train} samples, batch={batch_size}, eps={eps}")
    for epoch in range(epochs):
        perm = np.random.permutation(n_train)
        epoch_loss, n_batch = 0, 0

        for start in range(0, n_train, batch_size):
            end = min(start + batch_size, n_train)
            a_batch = X_tr[perm[start:end]]

            q_batch = model.forward(a_batch)
            cost, g = sinkhorn_batch(a_batch, q_batch, K, CK, eps)
            model.backward(g)

            epoch_loss += cost.mean()
            n_batch += 1

        avg_loss = epoch_loss / n_batch
        history.append(avg_loss)
        model.lr *= 0.95
        print(f"  Epoch {epoch + 1:2d}/{epochs} | loss = {avg_loss:.6f}")

    return model, history


def main():
    np.random.seed(42)

    # --- パラメータ ---
    N_TRAIN = 5000
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.01
    EPS = 0.5

    # --- データ ---
    X_train, y_train, X_test, y_test = load_mnist()
    C_pixel, pos = build_pixel_cost(28)
    print(f"Train: {X_train.shape[0]}, Test: {X_test.shape[0]}")

    plot_distributions(X_train, y_train)

    # --- 画像間 OT ---
    demo_image_ot(X_train, y_train, C_pixel, pos)

    # --- Autoencoder ---
    X_tr = np.array([image_to_dist(x) for x in X_train[:N_TRAIN]])
    X_te = np.array([image_to_dist(x) for x in X_test[:1000]])

    model, history = train_autoencoder(
        X_tr, C_pixel, EPS, N_TRAIN, BATCH_SIZE, EPOCHS, LR
    )

    # --- 結果の可視化 ---
    plot_training_curve(history, EPOCHS)
    plot_reconstructions(model, X_te, y_test, n_show=8)
    plot_latent_space(model, X_te, y_test)

    # --- 再構成の輸送計画 ---
    print("\n=== Reconstruction transport plan ===")
    a_s = X_te[0]
    q_s = model.forward(a_s.reshape(1, -1))[0]
    P_r, _, _ = sinkhorn_log(a_s, q_s, C_pixel, eps=0.05, max_iter=200)
    print(f"Reconstruction cost: {np.sum(C_pixel * P_r):.6f}")
    plot_interpolation(P_r, pos, title=f"reconstruction path (digit {y_test[0]})")


if __name__ == "__main__":
    main()
