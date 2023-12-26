import numpy as np
from mindspore import Tensor
from mindspore.nn.metric import Accuracy
from mindspore.common import dtype as mstype

def clustering_measure(y, pred_y):
    pred_y = best_map(y, pred_y)

    if y.shape[1] != 1:
        y = y.transpose()
    if pred_y.shape[1] != 1:
        pred_y = pred_y.transpose()

    n = len(y)

    u_y = np.unique(y)
    nclass = len(u_y)
    y_0 = np.zeros(n)
    if nclass != np.max(y):
        for i in range(nclass):
            y_0[np.where(y == u_y[i])] = i + 1
        y = y_0.astype(np.int32)

    u_y = np.unique(pred_y)
    nclass = len(u_y)
    pred_y_0 = np.zeros(n)
    if nclass != np.max(pred_y):
        for i in range(nclass):
            pred_y_0[np.where(pred_y == u_y[i])] = i + 1
        pred_y = pred_y_0.astype(np.int32)

    lidx = np.unique(y)
    class_num = len(lidx)
    pred_lidx = np.unique(pred_y)
    pred_class_num = len(pred_lidx)

    # Purity
    correct_num = 0
    for ci in range(pred_class_num):
        incluster = y[np.where(pred_y == pred_lidx[ci])]
        inclunub = np.histogram(incluster, bins=np.arange(1, np.max(incluster) + 2))[0]
        if len(inclunub) == 0:
            inclunub = 0
        correct_num += np.max(inclunub)
    purity = correct_num / len(pred_y)

    # Accuracy
    accuracy = len(np.where(y == pred_y)[0]) / len(y)

    # Normalized Mutual Information
    mi_hat = mutual_info(y, pred_y)

    result = [accuracy, mi_hat, purity]
    return result

def best_map(l1, l2):
    l1 = l1.flatten()
    l2 = l2.flatten()
    if l1.shape != l2.shape:
        raise ValueError("Shape of l1 and l2 must be the same.")

    l1 = l1 - np.min(l1) + 1
    l2 = l2 - np.min(l2) + 1

    n_class = max(np.max(l1), np.max(l2))
    g = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            g[i, j] = np.sum(np.logical_and(l1 == i + 1, l2 == j + 1))

    c, _ = hungarian(-g)
    new_l2 = np.zeros(n_class)
    for i in range(n_class):
        new_l2[l2 == i + 1] = c[i]

    return new_l2

def mutual_info(l1, l2):
    l1 = l1.flatten()
    l2 = l2.flatten()
    if l1.shape != l2.shape:
        raise ValueError("Shape of l1 and l2 must be the same.")

    l1 = l1 - np.min(l1) + 1
    l2 = l2 - np.min(l2) + 1

    n_class = max(np.max(l1), np.max(l2))
    g = np.zeros((n_class, n_class))
    for i in range(n_class):
        for j in range(n_class):
            g[i, j] = np.sum(np.logical_and(l1 == i + 1, l2 == j + 1))

    sum_g = np.sum(g)

    p1 = np.sum(g, axis=1) / sum_g
    p2 = np.sum(g, axis=0) / sum_g
    h1 = np.sum(-p1 * np.log2(p1))
    h2 = np.sum(-p2 * np.log2(p2))

    p12 = g / sum_g
    ppp = p12 / np.outer(p1, p2)
    ppp[np.abs(ppp) < 1e-12] = 1
    mi = np.sum(p12 * np.log2(ppp))

    mi_hat = mi / max(h1, h2)

    return mi_hat

def hungarian(a):
    m, n = a.shape
    if m != n:
        raise ValueError("Cost matrix must be square.")

    orig = a.copy()

    a, c, u = hminired(a)

    while u[n]:
        lr = np.zeros(n, dtype=npint32)
        t = np.zeros(n, dtype=np.int32)
        s = np.zeros(n, dtype=np.int32)

        q = 0
        p = 0
        s[p] = n
        done = False

        while not done:
            if q == n:
                q = s[p]
                p -= 1
                t[q - 1] = s[p]
                q += 1
            while q < n and not done:
                if a[p, q] == 0 and not u[q]:
                    done = True
                else:
                    if a[p, q] == 0 and u[q]:
                        break
                    q += 1
            if not done:
                if q < n:
                    u[q] = True
                    p += 1
                    s[p] = q + 1
                    q = 0
                else:
                    p -= 1
                    q = s[p]
                    q += 1

        for p in range(n):
            a[p, :] -= a[p, t[p] - 1]

        c, a, u = hminired(a)

    return c, orig[:, np.argsort(t)]

def hminired(a):
    n = a.shape[0]
    c = 0
    u = np.zeros(n, dtype=bool)
    v = np.zeros(n, dtype=bool)
    d = np.zeros(n, dtype=a.dtype)

    for i in range(n):
        d[i] = np.min(a[i, :])

    for i in range(n):
        for j in range(n):
            if a[i, j] == d[i]:
                c += 1
                if not u[i] and not v[j]:
                    a[i, :] -= d[i]
                    a[:, j] -= d[i]
                    u[i] = True
                    v[j] = True

    u.fill(False)
    v.fill(False)
    return c, a, u

# 将numpy数组转换为MindSpore张量
def numpy_to_mindspore(array):
    return Tensor(array, dtype=mstype.float32)

# 测试代码
y = np.array([1, 2, 1, 2, 3])
pred_y = np.array([2, 1, 1, 3, 2])

y_tensor = numpy_to_mindspore(y)
pred_y_tensor = numpy_to_mindspore(pred_y)

measure_result = clustering_measure(y_tensor, pred_y_tensor)
print(measure_result)