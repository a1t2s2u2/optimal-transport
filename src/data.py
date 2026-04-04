"""データ読み込みとユーティリティ"""

import numpy as np
import struct
import gzip
import os
import urllib.request

DATASETS = {
    "mnist": {
        "url": "https://ossci-datasets.s3.amazonaws.com/mnist/",
        "labels": [str(i) for i in range(10)],
    },
    "fashion": {
        "url": "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/",
        "labels": [
            "T-shirt", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot",
        ],
    },
}

_FILES = [
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
]


def _download(base_url, path):
    os.makedirs(path, exist_ok=True)
    for fname in _FILES:
        fpath = os.path.join(path, fname)
        if not os.path.exists(fpath):
            print(f"  Downloading {fname}...")
            urllib.request.urlretrieve(base_url + fname, fpath)
    return path


def _load_images(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num, rows, cols = struct.unpack(">IIII", f.read(16))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data.reshape(num, rows * cols).astype(np.float64) / 255.0


def _load_labels(filepath):
    with gzip.open(filepath, "rb") as f:
        _, num = struct.unpack(">II", f.read(8))
        data = np.frombuffer(f.read(), dtype=np.uint8)
    return data


def load_dataset(name="fashion"):
    """
    データセットを読み込む

    name: "mnist" or "fashion"
    Returns: X_train, y_train, X_test, y_test, label_names
    """
    ds = DATASETS[name]
    path = f"./{name}_data"
    _download(ds["url"], path)
    X_train = _load_images(os.path.join(path, _FILES[0]))
    y_train = _load_labels(os.path.join(path, _FILES[1]))
    X_test = _load_images(os.path.join(path, _FILES[2]))
    y_test = _load_labels(os.path.join(path, _FILES[3]))
    return X_train, y_train, X_test, y_test, ds["labels"]


def downsample(images, factor=2):
    """画像を factor x factor の平均プーリングでダウンサンプル"""
    n = images.shape[0]
    h = int(np.sqrt(images.shape[1]))
    new_h = h // factor
    imgs = images.reshape(n, h, h)
    imgs = imgs.reshape(n, new_h, factor, new_h, factor).mean(axis=(2, 4))
    return imgs.reshape(n, new_h * new_h)


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
