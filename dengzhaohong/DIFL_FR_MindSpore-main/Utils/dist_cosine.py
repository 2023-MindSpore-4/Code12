import mindspore

def dist_cosine(A, B):
    # A, B are matrices of example data vectors, one per column
    # The distance is sum_i (u_i-v_i)^2/(u_i+v_i+epsilon)
    # The output distance matrix is (#examples in A) x (#examples in B)

    d, m = A.size()
    d1, n = B.size()
    if d != d1:
        raise ValueError("column length of A ({}) != column length of B ({})".format(d, d1))

    A = A * mindspore.ops.sqrt(mindspore.ops.sum(A ** 2, axis=0))
    B = B * mindspore.ops.sqrt(mindspore.ops.sum(B ** 2, axis=0))

    D = mindspore.ops.zeros(m, n)
    for i in range(m):  # m is number of samples of A
        if i % 1000 == 0:
            print(".", end="")
        D[i, :] = 1 - mindspore.ops.dot(A[:, i], B)

    return D