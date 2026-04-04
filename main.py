"""
Sinkhorn Autoencoder for Fashion-MNIST — エントリポイント

理論的背景は chapters/11_application.tex を参照
実行: uv run python main.py
"""

import numpy as np
from tqdm import tqdm, trange
from src import load_dataset, image_to_dist, build_pixel_cost
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

DATASET = "fashion"  # "mnist" or "fashion"


def demo_image_ot(X_train, y_train, C_pixel, pos, label_names, cls_a=3, cls_b=7):
    """画像間の最適輸送を計算・可視化"""
    idx_a = np.where(y_train == cls_a)[0][0]
    idx_b = np.where(y_train == cls_b)[0][0]
    a = image_to_dist(X_train[idx_a])
    b = image_to_dist(X_train[idx_b])
    name_a, name_b = label_names[cls_a], label_names[cls_b]

    P, _, _ = sinkhorn_log(a, b, C_pixel, eps=0.05, max_iter=200)
    cost = np.sum(C_pixel * P)
    err_a = np.linalg.norm(P.sum(1) - a)
    err_b = np.linalg.norm(P.sum(0) - b)
    print(f"  輸送コスト: {cost:.6f}  周辺誤差: {err_a:.1e}, {err_b:.1e}")

    plot_transport(a, b, P, pos, name_a, name_b)
    plot_interpolation(P, pos, title=f"McCann interpolation: {name_a} -> {name_b}")


def train_autoencoder(X_tr, C_pixel, eps, n_train, batch_size, epochs, lr):
    """Sinkhorn Autoencoder を訓練"""
    K = np.exp(-C_pixel / eps)
    CK = C_pixel * K

    model = SinkhornAutoencoder([784, 256, 64, 256, 784], lr=lr)
    history = []

    for epoch in trange(epochs, desc="Training"):
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
        tqdm.write(f"  Epoch {epoch+1:2d} | loss = {avg_loss:.6f}")

    return model, history


def main():
    np.random.seed(42)

    N_TRAIN = 5000
    BATCH_SIZE = 32
    EPOCHS = 10
    LR = 0.01
    EPS = 0.5

    # ── データ読み込み ─────────────────────────
    print(f"[1/4] データ読み込み ({DATASET})")
    X_train, y_train, X_test, y_test, label_names = load_dataset(DATASET)
    C_pixel, pos = build_pixel_cost(28)
    print(f"  訓練 {X_train.shape[0]} 枚 / テスト {X_test.shape[0]} 枚")
    print(f"  クラス: {', '.join(label_names)}")

    plot_distributions(X_train, y_train, label_names)

    # ── 画像間の最適輸送 ───────────────────────
    cls_a, cls_b = 0, 7  # T-shirt -> Sneaker
    print(f"\n[2/4] 画像間の最適輸送 ({label_names[cls_a]} → {label_names[cls_b]})")
    demo_image_ot(X_train, y_train, C_pixel, pos, label_names, cls_a, cls_b)

    # ── Autoencoder 訓練 ───────────────────────
    print(f"\n[3/4] Autoencoder 訓練 ({N_TRAIN}枚, batch={BATCH_SIZE}, eps={EPS})")
    X_tr = np.array([image_to_dist(x) for x in tqdm(X_train[:N_TRAIN], desc="Preparing")])
    X_te = np.array([image_to_dist(x) for x in X_test[:1000]])

    model, history = train_autoencoder(
        X_tr, C_pixel, EPS, N_TRAIN, BATCH_SIZE, EPOCHS, LR
    )

    plot_training_curve(history, EPOCHS)
    plot_reconstructions(model, X_te, y_test, label_names, n_show=8)
    plot_latent_space(model, X_te, y_test)

    # ── 再構成の輸送経路 ───────────────────────
    label = label_names[y_test[0]]
    print(f"\n[4/4] 再構成の輸送経路 ({label})")
    a_s = X_te[0]
    q_s = model.forward(a_s.reshape(1, -1))[0]
    P_r, _, _ = sinkhorn_log(a_s, q_s, C_pixel, eps=0.05, max_iter=200)
    print(f"  再構成コスト: {np.sum(C_pixel * P_r):.6f}")
    plot_interpolation(P_r, pos, title=f"reconstruction path ({label})",
                       filename="reconstruction_path.png")

    print("\n完了")


if __name__ == "__main__":
    main()
