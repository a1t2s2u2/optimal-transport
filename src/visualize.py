"""可視化��数"""

import numpy as np
import matplotlib.pyplot as plt
from .data import image_to_dist


def plot_distributions(X, y, title="MNIST distributions"):
    fig, axes = plt.subplots(2, 5, figsize=(14, 6))
    for digit in range(10):
        ax = axes[digit // 5, digit % 5]
        idx = np.where(y == digit)[0][0]
        dist = image_to_dist(X[idx])
        ax.imshow(dist.reshape(28, 28), cmap="hot")
        ax.set_title(f"digit {digit}", fontsize=10)
        ax.axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_transport(a, b, P, pos, label_a="source", label_b="target"):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    axes[0].imshow(a.reshape(28, 28), cmap="Blues")
    axes[0].set_title(label_a, fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(b.reshape(28, 28), cmap="Reds")
    axes[1].set_title(label_b, fontsize=12)
    axes[1].axis("off")

    ax = axes[2]
    ax.imshow(np.zeros((28, 28)), cmap="gray", alpha=0.1)
    step = 3
    for i in range(0, 28, step):
        for j in range(0, 28, step):
            k = i * 28 + j
            if a[k] > 0.002:
                weights = P[k, :] / (a[k] + 1e-30)
                tgt = weights @ pos * 27
                src = pos[k] * 27
                ax.arrow(
                    src[1], src[0], tgt[1] - src[1], tgt[0] - src[0],
                    head_width=0.3, fc="blue", ec="blue", alpha=0.5,
                )
    ax.set_xlim(-1, 28)
    ax.set_ylim(28, -1)
    ax.set_title("transport field", fontsize=12)
    ax.axis("off")
    plt.tight_layout()
    plt.show()


def plot_interpolation(P, pos, n_steps=7, title="displacement interpolation"):
    I, J = np.where(P > 1e-10)
    masses = P[I, J]

    ts = np.linspace(0, 1, n_steps)
    fig, axes = plt.subplots(1, n_steps, figsize=(3 * n_steps, 3))
    for idx, t in enumerate(ts):
        interp = np.zeros((28, 28))
        p_t = (1 - t) * pos[I] + t * pos[J]
        r = np.clip(np.round(p_t[:, 0] * 27).astype(int), 0, 27)
        c = np.clip(np.round(p_t[:, 1] * 27).astype(int), 0, 27)
        np.add.at(interp, (r, c), masses)
        axes[idx].imshow(interp, cmap="hot")
        axes[idx].set_title(f"t={t:.2f}", fontsize=11)
        axes[idx].axis("off")
    plt.suptitle(title, fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_reconstructions(model, X_te, y_te, n_show=8):
    test_input = X_te[:n_show]
    test_output = model.forward(test_input)

    fig, axes = plt.subplots(3, n_show, figsize=(2.2 * n_show, 7))
    for i in range(n_show):
        axes[0, i].imshow(test_input[i].reshape(28, 28), cmap="hot")
        axes[0, i].axis("off")
        axes[0, i].set_title(f"digit {y_te[i]}", fontsize=10)

        axes[1, i].imshow(test_output[i].reshape(28, 28), cmap="hot")
        axes[1, i].axis("off")

        diff_img = (test_output[i] - test_input[i]).reshape(28, 28)
        v = max(abs(diff_img.min()), abs(diff_img.max())) + 1e-10
        axes[2, i].imshow(diff_img, cmap="RdBu_r", vmin=-v, vmax=v)
        axes[2, i].axis("off")

    axes[0, 0].set_ylabel("input a", fontsize=11, rotation=0, labelpad=50)
    axes[1, 0].set_ylabel("output q", fontsize=11, rotation=0, labelpad=50)
    axes[2, 0].set_ylabel("q - a", fontsize=11, rotation=0, labelpad=50)
    plt.suptitle("Sinkhorn Autoencoder reconstruction", fontsize=13)
    plt.tight_layout()
    plt.show()


def plot_latent_space(model, X_te, y_te, n_vis=1000):
    latent = model.encode(X_te[:n_vis])
    labels = y_te[:n_vis]

    mean = latent.mean(axis=0)
    centered = latent - mean
    _, _, Vt = np.linalg.svd(centered, full_matrices=False)
    pca_2d = centered @ Vt[:2].T

    plt.figure(figsize=(8, 6))
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
    plt.show()


def plot_training_curve(history, epochs):
    plt.figure(figsize=(8, 4))
    plt.plot(range(1, epochs + 1), history, "b-o")
    plt.xlabel("Epoch")
    plt.ylabel("Sinkhorn loss")
    plt.title("Training loss (regularized Kantorovich cost)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
