"""可視化関数（すべてファイルに保存）"""

import os
import numpy as np
import matplotlib.pyplot as plt
from .data import image_to_dist

OUTPUT_DIR = "figures"


def _save(fig, name):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    path = os.path.join(OUTPUT_DIR, name)
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  -> {path}")


def plot_distributions(X, y, label_names=None, title="distributions"):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for cls in range(10):
        ax = axes[cls // 5, cls % 5]
        idx = np.where(y == cls)[0][0]
        dist = image_to_dist(X[idx])
        s = int(np.sqrt(X.shape[1]))
        ax.imshow(dist.reshape(s, s), cmap="hot")
        name = label_names[cls] if label_names else str(cls)
        ax.set_title(name, fontsize=10)
        ax.axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, "distributions.png")


def plot_transport(a, b, P, pos, label_a="source", label_b="target"):
    s = int(np.sqrt(len(a)))
    mx = s - 1
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.5))

    axes[0].imshow(a.reshape(s, s), cmap="Blues")
    axes[0].set_title(label_a, fontsize=13)
    axes[0].axis("off")

    axes[1].imshow(b.reshape(s, s), cmap="Reds")
    axes[1].set_title(label_b, fontsize=13)
    axes[1].axis("off")

    # 輸送ベクトル場: 背景に両画像を重ね、移動距離で色分け
    ax = axes[2]
    blend = np.stack([
        b.reshape(s, s),
        np.zeros((s, s)),
        a.reshape(s, s),
    ], axis=-1)
    blend = blend / (blend.max() + 1e-30) * 0.4
    ax.imshow(blend, extent=(-0.5, mx + 0.5, mx + 0.5, -0.5))

    # 各画素の輸送先と移動距離を計算
    step = max(1, s // 14)
    srcs, dxs, dys, dists, masses = [], [], [], [], []
    for i in range(0, s, step):
        for j in range(0, s, step):
            k = i * s + j
            if a[k] > 0.001:
                weights = P[k, :] / (a[k] + 1e-30)
                tgt = weights @ pos * mx
                src = pos[k] * mx
                dx, dy = tgt[1] - src[1], tgt[0] - src[0]
                d = np.sqrt(dx**2 + dy**2)
                if d > 0.3:
                    srcs.append(src)
                    dxs.append(dx)
                    dys.append(dy)
                    dists.append(d)
                    masses.append(a[k])

    if dists:
        dists = np.array(dists)
        masses = np.array(masses)
        norm_d = dists / (dists.max() + 1e-30)
        cmap = plt.cm.plasma

        # 距離が短い順に描画（長いものが上に来る）
        order = np.argsort(dists)
        for idx in order:
            color = cmap(norm_d[idx])
            lw = 1.0 + 2.0 * masses[idx] / (masses.max() + 1e-30)
            ax.annotate("",
                xy=(srcs[idx][1] + dxs[idx], srcs[idx][0] + dys[idx]),
                xytext=(srcs[idx][1], srcs[idx][0]),
                arrowprops=dict(
                    arrowstyle="-|>", color=color,
                    lw=lw, mutation_scale=10,
                ),
            )

    ax.set_xlim(-0.5, mx + 0.5)
    ax.set_ylim(mx + 0.5, -0.5)
    ax.set_title("transport field", fontsize=13)
    ax.axis("off")

    # カラーバー
    sm = plt.cm.ScalarMappable(cmap=plt.cm.plasma,
                                norm=plt.Normalize(0, dists.max() if len(dists) else 1))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("distance (px)", fontsize=10)

    plt.tight_layout()
    _save(fig, "transport.png")


def plot_interpolation(P, pos, n_steps=7, title="displacement interpolation",
                       filename="interpolation.png", img_size=28):
    I, J = np.where(P > 1e-10)
    masses = P[I, J]
    mx = img_size - 1

    ts = np.linspace(0, 1, n_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3))
    for idx, t in enumerate(ts):
        interp = np.zeros((img_size, img_size))
        p_t = (1 - t) * pos[I] + t * pos[J]
        r = np.clip(np.round(p_t[:, 0] * mx).astype(int), 0, mx)
        c = np.clip(np.round(p_t[:, 1] * mx).astype(int), 0, mx)
        np.add.at(interp, (r, c), masses)
        axes[idx].imshow(interp, cmap="hot")
        axes[idx].set_title(f"t={t:.2f}", fontsize=11)
        axes[idx].axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    _save(fig, filename)


def plot_reconstructions(model, X_te, y_te, label_names=None, n_show=8):
    s = int(np.sqrt(X_te.shape[1]))
    test_input = X_te[:n_show]
    test_output = model.forward(test_input)

    fig, axes = plt.subplots(3, n_show, figsize=(2.2 * n_show, 7))
    for i in range(n_show):
        axes[0, i].imshow(test_input[i].reshape(s, s), cmap="hot")
        axes[0, i].axis("off")
        name = label_names[y_te[i]] if label_names else str(y_te[i])
        axes[0, i].set_title(name, fontsize=10)

        axes[1, i].imshow(test_output[i].reshape(s, s), cmap="hot")
        axes[1, i].axis("off")

        diff_img = (test_output[i] - test_input[i]).reshape(s, s)
        v = max(abs(diff_img.min()), abs(diff_img.max())) + 1e-10
        axes[2, i].imshow(diff_img, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("input a", fontsize=11, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("output q", fontsize=11, rotation=0, labelpad=50)
    axes[2, 0].set_ylabel("q - a", fontsize=11, rotation=0, labelpad=50)
    plt.suptitle("Sinkhorn Autoencoder reconstruction", fontsize=13)
    plt.tight_layout()
    _save(fig, "reconstructions.png")


def plot_latent_space(model, X_te, y_te, n_vis=1000):
    latent = model.encode(X_te[:n_vis])
    labels = y_te[:n_vis]

    mean = latent.mean(axis=0)
    centered = latent - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_2d = centered @ Vt[:2].T

    fig = plt.figure(figsize=(8, 6))
    for digit in range(10):
        mask = labels == digit
        plt.scatter(
            pca_2d[mask, 0], pca_2d[mask, 1], s=8, label=str(digit), alpha=0.7
        )
    plt.legend(title="digit", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.title("latent space PCA (64D -> 2D)")
    plt.xlabel("PC1")
    plt.ylabel("PC2")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "latent_space.png")


def plot_training_curve(history, epochs):
    fig = plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), history, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("Sinkhorn divergence")
    plt.title("Training loss (Sinkhorn divergence)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    _save(fig, "training_curve.png")
