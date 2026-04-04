"""MNIST データ読み込みとユーティリティ"""

import numpy as np
import struct
import gzip
import os
import urllib.request


def download_mnist(path="./mnist_data"):
    os.makedirs(path, exist_ok=True)
    base_url = "https://ossci-datasets.s3.amazonaws.com/mnist/"
    files = [
        "train-images-idx3-ubyte.gz",
        "train-labels-idx1-ubyte.gz",
        "t10k-images-idx3-ubyte.gz",
        "t10k-labels-idx1-ubyte.gz",
    ]
    for fname in files:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
    return path


def load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols).astype(np.float64) / 255.0


def load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_mnist(path="./mnist_data"):
    """MNIST を読み込んで (X_train, y_train, X_test, y_test) を返す"""
    data_path = download_mnist(path)
    X_train = load_images(os.path.join(data_path, "train-images-idx3-ubyte.gz"))
    y_train = load_labels(os.path.join(data_path, "train-labels-idx1-ubyte.gz"))
    X_test = load_images(os.path.join(data_path, "t10k-images-idx3-ubyte.gz"))
    y_test = load_labels(os.path.join(data_path, "t10k-labels-idx1-ubyte.gz"))
    return X_train, y_train, X_test, y_test


def image_to_dist(img, reg=1e-6):
    """画像を確率分布に変換"""
    p = img + reg
    return p / p.sum()


def build_pixel_cost(size=28):
    """画素位置間の二乗ユークリッドコスト行列を構築"""
    pos = (
        np.array(
            [(i, j) for i in range(size) for j in range(size)], dtype=np.float64
        )
        / (size - 1)
    )
    diff = pos[:, np.newaxis, :] - pos[np.newaxis, :, :]
    C = np.sum(diff**2, axis=2)
    return C, pos
