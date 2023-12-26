import mindspore

def fsvd(A, k, i=1, usePowerMethod=False):
    if len(A.size()) < 2:
        raise ValueError("Input matrix A must have at least 2 dimensions.")
    
    if A.size(0) < A.size(1):
        A = A.t()
        isTransposed = True
    else:
        isTransposed = False

    n = A.size(1)
    l = k + 2

    G = mindspore.ops.randn(n, l)

    if usePowerMethod:
        H = A.mm(G)
        for j in range(2, i + 2):
            H = A.mm(A.t().mm(H))
    else:
        H = [None] * (i + 1)
        H[0] = A.mm(G)
        for j in range(2, i + 2):
            H[j - 1] = A.mm(A.t().mm(H[j - 2]))

        H = mindspore.ops.cat(H, dim=1)

    T = A.t().mm(H)

    Vt, St, W = mindspore.ops.svd(T, some=True)

    Ut = H.mm(W)

    if isTransposed:
        V = Ut[:, :k]
        U = Vt[:, :k]
    else:
        U = Ut[:, :k]
        V = Vt[:, :k]

    S = St[:k, :k]

    return U, S, V