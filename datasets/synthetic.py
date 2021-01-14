import numpy as np

def create_dataset_sin_cos(k=100):
    coords = np.arange(k, dtype=np.float32) / k * np.pi * 2
    X_11 = np.tile(np.sin(coords), [1, k, 1])
    X_12 = np.tile(np.cos(coords), [1, k, 1])
    X_21 = np.tile(np.sin(2*coords), [1, k, 1])
    X_22 = np.tile(np.cos(2*coords), [1, k, 1])
    Xs = np.concatenate([[X_11], [X_12], [X_21], [X_22]], 0)
    ys = np.hstack([np.zeros(2), np.ones(2)])
    return Xs, ys


def create_dataset_dots(k=100):
    X_1 = np.zeros([1, 1, k, k])
    X_1[:, :, k//2, k//2] = 1
    X_2 = np.zeros([1, 1, k, k])
    X_2[:, :, k//2, k//2] = -1
    Xs = np.concatenate([X_1, X_2], 0)
    ys = np.hstack([np.zeros(1), np.ones(1)])
    return Xs, ys