import mindspore
import numpy as np
from sklearn.cluster import KMeans


def fuzzy_c_means(data, k, max_iter=100, epsilon=1e-6):
    n_examples, d = data.shape
    u = mindspore.ops.rand(k, n_examples)
    u = u / mindspore.ops.sum(u, dim=0)

    for _ in range(max_iter):
        v = update_centroids(data, u, k)
        u_new = update_membership(data, v, k)
        if mindspore.ops.max(mindspore.ops.abs(u_new - u)) < epsilon:
            break
        u = u_new

    return v, u, _

def update_centroids(data, u, k):
    u_power = u ** 2
    v = mindspore.ops.zeros(k, data.shape[1])
    for i in range(k):
        v[i] = mindspore.ops.sum(data * u_power[i].unsqueeze(1), axis=0) / mindspore.ops.sum(u_power[i])
    return v

def update_membership(data, v, k):
    distances = mindspore.ops.cdist(data, v)
    distances = distances ** 2
    u = distances / mindspore.ops.sum(distances, axis=1, keepdim=True)
    return u

def gene_ante_fcm(data, K):
    k = K
    n_examples, d = data.shape

    data_tensor = mindspore.Tensor.from_numpy(data).float()
    v, U, _ = fuzzy_c_means(data_tensor, k)

    b = mindspore.ops.zeros((k, d))
    for i in range(k):
        v1 = v[i].unsqueeze(0).expand(n_examples, d)
        u = U[i]
        uu = u.unsqueeze(1).expand(n_examples, d)
        b[i] = mindspore.ops.sum((data_tensor - v1) ** 2 * uu, dim=0) / mindspore.ops.sum(uu, axisis=0) / 1

    b = b + mindspore.ops.finfo(mindspore.float32).eps
    return v.numpy(), b.numpy()