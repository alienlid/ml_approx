import math
import numpy as np
from sklearn.random_projection import GaussianRandomProjection, SparseRandomProjection
from scipy.linalg import hadamard

def exact_sparse(A, c):
    x, y = np.unravel_index(np.argsort(np.abs(A), axis=None), A.shape)
    num_zero = math.ceil(A.size * c)
    new_A = np.array(A)
    new_A[x[:num_zero], y[:num_zero]] = 0
    return new_A

def weighted_rand_sparse(A, c, p):
    new_A = np.array(A)
    comp = np.power(np.abs(A), p) / np.random.exponential(1, A.shape)
    x, y = np.unravel_index(np.argsort(comp, axis=None), A.shape)
    num_zero = math.ceil(A.size * c)
    new_A[x[:num_zero], y[:num_zero]] = 0
    return new_A

weighted_rand_sparse_0 = lambda A, c: weighted_rand_sparse(A, c, 0)
weighted_rand_sparse_1 = lambda A, c: weighted_rand_sparse(A, c, 1)
weighted_rand_sparse_2 = lambda A, c: weighted_rand_sparse(A, c, 2)

def low_rank(A, c):
    U, S, Vh = np.linalg.svd(A, full_matrices=False)
    rank = math.floor((1 - c) * S.shape[0])
    S[rank:] = 0
    return U @ np.diag(S) @ Vh

def gaussian_rand_proj(A, c):
    if c == 1:
        return np.zeros_like(A)
    d = math.floor(A.shape[0] * (1 - c))
    grp = GaussianRandomProjection(n_components=d, compute_inverse_components=True)
    grp.fit(A.T)
    return grp.inverse_components_ @ grp.components_ @ A

def sparse_rand_proj(A, c):
    if c == 1:
        return np.zeros_like(A)
    d = math.floor(A.shape[0] * (1 - c))
    srp = SparseRandomProjection(n_components=d, density=1/3, compute_inverse_components=True)
    srp.fit(A.T)
    return srp.inverse_components_ @ srp.components_ @ A

def fast_jlt(A, c):
    if c == 1:
        return np.zeros_like(A)
    m, n = A.shape
    d = math.floor(m * (1 - c))
    srp = SparseRandomProjection(n_components=d, density=1/3, compute_inverse_components=True)
    srp.fit(A.T)
    T = srp.components_ @ hadamard(m) @ np.diag(np.random.choice([1.0, -1.0], m))
    return (np.linalg.pinv(T) @ T @ A).astype(A.dtype)

def quantize(A, c):
    eps = (A.max() - A.min()) * np.float_power(2, -32 * (1 - c))
    return (np.round(A / eps) * eps).astype(A.dtype)
    # mx = np.ceil(np.log2(abs(A).max()))
    # eps = 2 ** (mx - (mx + 53) * (1 - c))
    # return (np.round(A / eps) * eps).astype(A.dtype)

def rand_svd(A, c, q):
    d = math.floor(A.shape[0] * (1 - c))
    X = np.random.randn(A.shape[1], d)
    Z = (A @ A.T)
    Y = np.linalg.matrix_power(Z, q) @ A @ X
    return (Y @ np.linalg.pinv(Y) @ A).astype(A.dtype)

rand_svd_0 = lambda A, c: rand_svd(A, c, 0)
rand_svd_1 = lambda A, c: rand_svd(A, c, 1)
rand_svd_2 = lambda A, c: rand_svd(A, c, 2)

