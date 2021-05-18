import numpy as np
import torch


def normalize_adjacent_matrix(A, mode='AD'):
    """Normalize adjacent matrix for GCN. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        A (ndarray): The adjacent matrix.
        mode (string): The normalize mode.

    returns:
        G (ndarray): The normalized adjacent matrix.
    """
    assert A.ndim == 2
    assert A.shape[0] == A.shape[1]

    if mode == 'DAD':
        A = A + np.eye(A.shape[0])
        d = np.sum(A, axis=0)
        d_inv = np.power(d, -0.5).flatten()
        d_inv[np.isinf(d_inv)] = 0.0
        d_inv = np.diag(d_inv)
        G = A.dot(d_inv).transpose().dot(d_inv)
        G = torch.from_numpy(G)
    elif mode == 'AD':
        A = A + np.eye(A.shape[0])
        A = torch.from_numpy(A)
        D = A.sum(1, keepdim=True)
        G = A.div(D)
    else:
        raise NotImplementedError
    return G


def euclidean_distance_matrix(A, B):
    """Calculate the Euclidean distance matrix.

    Args:
        A (ndarray): The point sequence.
        B (ndarray): The point sequence with the same dimensions as A.

    returns:
        D (ndarray): The Euclidean distance matrix.
    """
    assert A.ndim == 2
    assert B.ndim == 2
    assert A.shape[1] == B.shape[1]

    m = A.shape[0]
    n = B.shape[0]

    A_dots = (A * A).sum(axis=1).reshape((m, 1)) * np.ones(shape=(1, n))
    B_dots = (B * B).sum(axis=1) * np.ones(shape=(m, 1))
    D_squared = A_dots + B_dots - 2 * A.dot(B.T)

    zero_mask = np.less(D_squared, 0.0)
    D_squared[zero_mask] = 0.0
    D = np.sqrt(D_squared)
    return D


def feature_embedding(input_feats, out_feat_len):
    """Embed features. This code was partially adapted from
    https://github.com/GXYM/DRRG licensed under the MIT license.

    Args:
        input_feats (ndarray): The input features of shape (N, d), where N is
            the number of nodes in graph, d is the input feature vector length.
        out_feat_len (int): The length of output feature vector.

    Returns:
        embedded_feats (ndarray): The embedded features.
    """
    assert input_feats.ndim == 2
    assert isinstance(out_feat_len, int)
    assert out_feat_len >= input_feats.shape[1]

    num_nodes = input_feats.shape[0]
    feat_dim = input_feats.shape[1]
    feat_repeat_times = out_feat_len // feat_dim
    residue_dim = out_feat_len % feat_dim

    if residue_dim > 0:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times + 1)
            for j in range(feat_repeat_times + 1)
        ]).reshape((feat_repeat_times + 1, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0)
        residue_feats = np.hstack([
            input_feats[:, 0:residue_dim],
            np.zeros((num_nodes, feat_dim - residue_dim))
        ])
        residue_feats = np.expand_dims(residue_feats, axis=0)
        repeat_feats = np.concatenate([repeat_feats, residue_feats], axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1))[:, 0:out_feat_len]
    else:
        embed_wave = np.array([
            np.power(1000, 2.0 * (j // 2) / feat_repeat_times)
            for j in range(feat_repeat_times)
        ]).reshape((feat_repeat_times, 1, 1))
        repeat_feats = np.repeat(
            np.expand_dims(input_feats, axis=0), feat_repeat_times, axis=0)
        embedded_feats = repeat_feats / embed_wave
        embedded_feats[:, 0::2] = np.sin(embedded_feats[:, 0::2])
        embedded_feats[:, 1::2] = np.cos(embedded_feats[:, 1::2])
        embedded_feats = np.transpose(embedded_feats, (1, 0, 2)).reshape(
            (num_nodes, -1)).astype(np.float32)

    return embedded_feats
