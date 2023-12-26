import mindspore as ms
import numpy as np

def fromXtoZ(xt, v, b):
    Nt = xt.shape[0]
    xt1 = np.hstack((xt, np.ones((Nt, 1))))
    M, d0 = v.shape

    wt = np.zeros((Nt, M))
    for i in range(M):
        v1 = np.tile(v[i, :], (Nt, 1))
        bb = np.tile(b[i, :], (Nt, 1))
        wt[:, i] = np.exp(-np.sum((xt - v1) ** 2 / bb, axis=1))

    wt2 = np.sum(wt, axis=1)
    wt = wt / np.tile(wt2.reshape(-1, 1), (1, M))

    zt = np.empty((Nt, 0))
    for i in range(M):
        wt1 = wt[:, i]
        wt2 = np.tile(wt1.reshape(-1, 1), (1, d0 + 1))
        zt = np.hstack((zt, xt1 * wt2))

    mask = np.isnan(zt)
    zt[mask] = 1e-5

    return ms.Tensor(zt, dtype=ms.float32)