import mindspore

def ML_TSKFS(X, Y, optmParameter):
    alpha = optmParameter['alpha']
    beta = optmParameter['beta']
    gamma = optmParameter['gamma']
    maxIter = optmParameter['maxIter']
    miniLossMargin = optmParameter['minimumLossMargin']

    num_dim = X.size(1)
    XTX = mindspore.ops.matmul(X.t(), X)
    XTY = mindspore.ops.matmul(X.t(), Y)
    W_s = mindspore.ops.matmul(mindspore.ops.inverse(XTX + gamma * mindspore.ops.eye(num_dim)), XTY)
    W_s_1 = W_s.clone()
    R = 1 - mindspore.ops.Tensor.corr(Y, rowvar=False)
    iter = 1
    oldloss = 0

    Lip = mindspore.ops.sqrt(2 * (mindspore.ops.norm(XTX) ** 2 + mindspore.ops.norm(alpha * R) ** 2))

    bk = 1
    bk_1 = 1

    while iter <= maxIter:
        W_s_k = W_s + (bk_1 - 1) / bk * (W_s - W_s_1)
        Gw_s_k = W_s_k - 1 / Lip * ((mindspore.ops.matmul(XTX, W_s_k) - XTY) + alpha * mindspore.ops.matmul(W_s_k, R))
        bk_1 = bk
        bk = (1 + mindspore.ops.sqrt(4 * bk ** 2 + 1)) / 2
        W_s_1 = W_s.clone()
        W_s = softthres(Gw_s_k, beta / Lip)

        predictionLoss = mindspore.ops.trace(mindspore.ops.matmul((X.matmul(W_s) - Y).t(), (X.matmul(W_s) - Y)))
        correlation = mindspore.ops.trace(mindspore.ops.matmul(R, W_s.t().matmul(W_s)))
        sparsity = mindspore.ops.sum(W_s != 0)
        totalloss = predictionLoss + alpha * correlation + beta * sparsity

        if mindspore.ops.abs(oldloss - totalloss) <= miniLossMargin:
            break
        elif totalloss <= 0:
            break
        else:
            oldloss = totalloss

        iter += 1

    model_W = W_s
    return model_W

def softthres(W_t, lambda_):
    return F.relu(W_t - lambda_) - F.relu(-W_t - lambda_)
