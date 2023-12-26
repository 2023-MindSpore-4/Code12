import numpy as np
import mindspore
from mindspore import Tensor
from mindspore.ops import operations as P


def MVFCM_JRLCP(X, option):
    view_nums = len(X)  # 输入数据的不同视图的个数。
    # X（（100，256），（100，123），（100，562）） 这里视图个数3 每个视图都是100张图片，后面256代表的是特征维度。
    lambda1 = option['lambda1']  # 控制模型复杂度和防止过拟合。
    lambda2 = option['lambda2']
    Maxitems = option['Maxitems']  # 最大迭代次数
    com_dim = option['com_dim']  # 共享空间维度

    m = option['m']  # 模糊程度参数 m的值较小，那么每个数据点主要会属于一个聚类 m的值较大，那么每个数据点可能会同时属于多个聚类
    c = option['clusters']  # 形成的聚类数量

    tolerance = option['tolerance']  # 容忍度参数

    # 初始化
    N, _ = X[0].shape  # 每个数据的第一个试图的长度，也就是图片数
    # 每个视图通常都有相同数量的样本，但是特征维度不一定一样
    np.random.seed(10)
    Hc = np.random.rand(N, com_dim)  # 矩阵的形状是(N, com_dim)，其中N是图片数，com_dim是共享空间的维度

    B = {}
    V = {}
    Hs = {}
    H = {}
    w = {}
    d = {}  # 初始化 d

    for v in range(view_nums):
        _, d[v] = X[v].shape  # 先获得当前试图的特征维度存储在d[v]中
        w[v] = 1 / (view_nums + 1)  # 初始化权重
        np.random.seed(v * 10)

        # 初始化下面随机随机矩阵
        B[v, 1] = np.random.rand(com_dim, d[v])  # 矩阵的形状是(com_dim, d[v])，其中com_dim是共享空间的维度，d[v]是第v个视图的特征维度。
        # 存储在字典 B 中，对应的键是 (v, 1)

        V[v, 1] = np.random.rand(c, com_dim)    # 矩阵的形状是(c, com_dim)，其中c是聚类数量，com_dim是共享空间的维度
        # 它将第 v 个视图的原始特征空间映射到了一个共享的低维空间。
        np.random.seed(v * 10)
        Hs[v, 1] = np.random.rand(N, com_dim)   # N是数据点的数量，com_dim是共享空间的维度
        H[v, 1] = Hc + Hs[v, 1]  # 共享特征矩阵（Hc）和特定于视图的特征矩阵（Hs）的和。
        # 找到一个共享的低维空间（由 Hc 表示），在这个空间中，所有视图的数据都可以得到良好的表示。
        # 同时，每个视图还有自己特定的特征（由 Hs[v, 1] 表示），这些特征捕获了该视图独有的信息。
        # 通过这种方式，算法可以同时利用所有视图之间的共享信息和每个视图的独特信息，从而获得更好的聚类结果

    Hs[view_nums + 1] = Hc
    # 将共享特征矩阵Hc赋值给字典Hs的键(view_nums + 1)。这意味着在所有视图之后，我们有一个额外的"视图"，它只包含共享特征。
    w[view_nums + 1] = 1 / (view_nums + 1)
    # 为这个额外"视图"设置了权重
    V[view_nums + 1, 1] = np.random.rand(c, com_dim)

    U = np.random.rand(c, N)
    # 随机矩阵U，它表示每个数据点属于每个聚类的隶属度。 N是图片数
    temp = np.sum(U)
    U /= temp * np.ones((c, 1))
    # 算了U中所有元素的且 将U归一化，使得每列（即每个数据点对应的隶属度）之和为1。
    Um = U ** m
    # 算了U的m次方，其中m是模糊C-均值聚类中的模糊参数。
    final_obj = np.zeros(Maxitems)
    # 这行代码初始化了一个向量，用于存储每次迭代的目标函数值。

    # 迭代更新
    for iter in range(Maxitems):
        sumHs = np.zeros((N, com_dim))  # 特点视图矩阵
        sumH = np.zeros((N, com_dim))  # Hc共享特征矩阵 特征矩阵（Hs）
        temp_w = np.zeros(view_nums + 1)
        temp_U = np.zeros((c, N))  # U 计算每个数据点属于每个聚类的隶属度。

        for v in range(view_nums + 1):
            for j in range(c):  # c是形成聚类的数量
                for i in range(N):  # N是图片数
                    temp_w[v] += np.exp(-(Um[j, i]*np.linalg.norm(Hs[v, 1][i, :]-V[v][j, :]))*lambda2)

        for v in range(view_nums):
            # 更新w
            w[v] = temp_w[v] / sum(temp_w)
            # 更新B
            tempB = np.linalg.pinv(H[v].T @ H[v]) @ (H[v].T @ X[v])
            B[v, 1] = tempB

            # 更新H
            tempH = (X[v] @ B[v].T + lambda1 * Hc + lambda1 * Hs[v]) @ \
                     np.linalg.pinv(B[v] @ B[v].T + lambda1 * np.eye(com_dim))
            H[v] = tempH

            # 更新Hs
            sum1Um = np.sum(Um, axis=0)
            sumV = np.sum(V[v], axis=0)
            tempHs = np.zeros((N, com_dim))

            for i in range(N):
                tempHs1 = lambda1 + w[v]*sum1Um[i]
                tempHs2 = lambda1*H[v][i, :] - lambda1*Hc[i, :] + w[v]*sum1Um[i]*sumV
                tempHs[i, :] = (tempHs2 / max(tempHs1, 1e-30))

            Hs[v, 1] = Normalization(tempHs)

            # 更新Vs
            V[v, 1] =(Um @ Hs[v]) / (np.ones((com_dim, 1)) * sum(Um).reshape(1,-1))
            sumHs += Hs[v]
            sumH += H[v]
            dist1_v_1= distfcm(V[v], Hs[v])

            temp_U += w[view_nums]*dist1_v_1

        # 更新w v+1
        w[view_nums+1] =(temp_w[view_nums+1]) / sum(temp_w)

        # 更新Hc
        tempHc = np.zeros((N, com_dim))
        sumV = np.sum(V[view_nums+1], axis=0)

        for i in range(N):
           tempHc[i, :] =(lambda1*sumH[i, :] - lambda1*sumHs[i, :] + w[view_nums+1]*sum1Um[i]*sumV) / \
                         (lambda1 + w[view_nums+1]*sum1Um[i])

        Hc = Normalization(tempHc)
        Hs[view_nums+1] = Hc

        # 更新Vc
        V[view_nums+1, 1] =(Um @ Hc) / (np.ones((com_dim, 1)) * sum(Um).reshape(1, -1))

        # 更新U
        dist2 = distfcm(V[view_nums+1], Hc)
        temp_U += w[view_nums+1]*dist2
        temp_Um = temp_U ** (1 / (1 - m))
        temp_Um[np.isnan(temp_Um)] = 0
        temp_Um[np.isinf(temp_Um)] = 1e5
        temp_Um = np.real(temp_Um)
        U = temp_Um / (np.ones((c, 1)) * sum(temp_Um))
        Um = U ** m

        cost = [0] * view_nums

        for v in range(view_nums):
            cost[v] = np.linalg.norm(X[v]-H[v]@B[v], 'fro') + lambda1*np.linalg.norm(H[v]-Hc-Hs[v], 'fro') + \
                       w[v]*np.sum(np.sum(dist1_v_1 * Um)) + lambda2*w[v]*np.log(w[v])

        cost.append(w[view_nums+1]*np.sum(np.sum(dist2 * Um)) + lambda2*w[view_nums+1]*np.log(w[view_nums+1]))

        final_obj[iter] = sum(cost)

        if iter > 0:
           if abs(final_obj[iter]-final_obj[iter-1]) < tolerance:
               break

    return U, V, w, Hs, Hc, final_obj


def Normalization(X):
    X = Tensor(X, dtype=mindspore.float32)
    X = P.ZerosLike()(X)  # 将X中的NaN值替换为0
    X = P.Fill()(X.shape, 1e5)  # 将X中的inf值替换为1e5
    norm_mat = P.Sqrt()(P.ReduceSum()(P.Square()(X), 1)).reshape(-1, 1) * P.OnesLike()(X)
    for i in range(norm_mat.shape[0]):
        if norm_mat[i, 0] == 0:
            norm_mat[i, :] = 1
    X /= norm_mat

    return X


def distfcm(center, data):  # 返回一个矩阵，其中每个元素是对应数据点到聚类中心的欧氏距离。
    out = P.ZerosLike()(center)  # 创建一个与center形状相同的零矩阵
    for k in range(P.Shape()(center)[0]):
        center_k = P.Reshape()(center[k, :], (-1, P.Shape()(center)[1]))
        out[k, :] = P.Sqrt()(P.ReduceSum()(P.Square()(P.Sub()(center_k, data)), -1))
    return out