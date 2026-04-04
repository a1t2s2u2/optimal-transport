"""
Sinkhorn Autoencoder — エントリポイント

エントロピー正則化により Kantorovich 問題が滑らかになり、
単純な勾配降下法で安定に学習できることを検証する。

理論的背景は chapters/11_application.tex を参照
実行: uv run python main.py
"""

import numpy as np
from tqdm import tqdm, trange
from src import load_dataset, image_to_dist, build_pixel_cost, downsample
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

DATASET = "mnist"
IMG_SIZE = 7  # 28x28 → 7x7 にダウンサンプル


def demo_image_ot(X, y, C_pixel, pos, label_names, cls_a=3, cls_b=8):
    """画像間の最適輸送を計算・可視化"""
    idx_a = np.where(y == cls_a)[0][0]
    idx_b = np.where(y == cls_b)[0][0]
    a = image_to_dist(X[idx_a])
    b = image_to_dist(X[idx_b])
    name_a, name_b = label_names[cls_a], label_names[cls_b]

    P, _, _ = sinkhorn_log(a, b, C_pixel, eps=0.05, max_iter=200)
    cost = np.sum(C_pixel * P)
    print(f"  輸送コスト: {cost:.6f}")

    plot_transport(a, b, P, pos, name_a, name_b)
    plot_interpolation(P, pos, title=f"McCann interpolation: {name_a} -> {name_b}",
                       img_size=IMG_SIZE)


def train_autoencoder(X_tr, C_pixel, eps_schedule, n_train, batch_size, epochs, lr):
    """
    Sinkhorn Autoencoder を単純な SGD で訓練

    eps_schedule: [(eps, n_epochs), ...] ε-スケーリング
      大きい ε → 強い勾配で粗く学習 → 小さい ε → 精密化
    """
    n_pixels = X_tr.shape[1]
    model = SinkhornAutoencoder([n_pixels, 64, 16, 64, n_pixels], lr=lr)
    history = []
    epoch_count = 0

    for eps, n_ep in eps_schedule:
        K = np.exp(-C_pixel / eps)
        CK = C_pixel * K
        tqdm.write(f"  --- eps = {eps} ({n_ep} epochs) ---")

        for _ in trange(n_ep, desc=f"eps={eps}", leave=False):
            epoch_count += 1
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
            tqdm.write(f"  Epoch {epoch_count:2d} | loss = {avg_loss:.6f} (eps={eps})")

    return model, history


def main():
    np.random.seed(42)

    N_TRAIN = 10000
    BATCH_SIZE = 64
    LR = 0.5
    # ε-スケーリング: 大→小で勾配を安定化
    EPS_SCHEDULE = [(1.0, 15), (0.3, 15), (0.1, 15)]

    # ── データ読み込み ─────────────────────────
    print(f"[1/4] データ読み込み ({DATASET}, {IMG_SIZE}x{IMG_SIZE})")
    X_train, y_train, X_test, y_test, label_names = load_dataset(DATASET)
    X_train = downsample(X_train, 28 // IMG_SIZE)
    X_test = downsample(X_test, 28 // IMG_SIZE)
    C_pixel, pos = build_pixel_cost(IMG_SIZE)
    print(f"  訓練 {X_train.shape[0]} 枚 / テスト {X_test.shape[0]} 枚")
    print(f"  画素数: {X_train.shape[1]}")

    plot_distributions(X_train, y_train, label_names)

    # ── 画像間の最適輸送 ───────────────────────
    cls_a, cls_b = 3, 8
    print(f"\n[2/4] 画像間の最適輸送 ({label_names[cls_a]} → {label_names[cls_b]})")
    demo_image_ot(X_train, y_train, C_pixel, pos, label_names, cls_a, cls_b)

    # ── Autoencoder 訓練 ───────────────────────
    eps_str = " -> ".join(f"{e}" for e, _ in EPS_SCHEDULE)
    total_epochs = sum(n for _, n in EPS_SCHEDULE)
    print(f"\n[3/4] Autoencoder 訓練 (SGD, lr={LR}, eps: {eps_str})")
    X_tr = np.array([image_to_dist(x) for x in tqdm(X_train[:N_TRAIN], desc="Preparing")])
    X_te = np.array([image_to_dist(x) for x in X_test[:1000]])

    model, history = train_autoencoder(
        X_tr, C_pixel, EPS_SCHEDULE, N_TRAIN, BATCH_SIZE, total_epochs, LR
    )

    plot_training_curve(history, total_epochs)
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
                       filename="reconstruction_path.png", img_size=IMG_SIZE)

    print("\n完了")


if __name__ == "__main__":
    main()
