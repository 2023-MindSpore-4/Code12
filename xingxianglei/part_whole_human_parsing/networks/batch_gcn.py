import mindspore as ms
import mindspore.nn as nn
from mindspore import ops, Tensor
import numpy as np


def unique(edges):
    edges_ndarray = np.array(edges, dtype=np.int64)
    _, degree = np.unique(edges_ndarray[:, 0], return_counts=True)
    degree = Tensor(degree)
    return degree


class BatchGCN(nn.Cell):
    def __init__(self, edges=None, node_num=6, feat_dim=64, stat_dim=64, T=3):
        super().__init__()
        self.edges = edges
        self.node_num = node_num
        self.embed_dim = feat_dim
        self.stat_dim = stat_dim
        self.T = T
        self.layer_norm = nn.LayerNorm([feat_dim])
        edges_tensor = Tensor(self.edges, dtype=ms.int64)
        self.X_Node, self.X_Neis = edges_tensor[:, 0], edges_tensor[:, 1]
        degree = unique(edges)
        self.deg = ops.gather_elements(degree, dim=0, index=self.X_Node)

        self.Hw = Hw(feat_dim, stat_dim)
        # 实现H的分组求和
        self.Aggr = AggrSum(node_num)
        self.out_layer = nn.Dense(feat_dim, feat_dim)
        # self.out_layer = nn.Sequential(
        #     nn.Linear(feat_dim, feat_dim),
        #     nn.LayerNorm(feat_dim),
        # )

    def construct(self, feat):
        # feat [batch_size, num_parts, feat_dim]
        batch_size, num_parts, feat_dim = feat.shape
        node_embeds = ops.index_select(input=feat, axis=1, index=self.X_Node)
        neis_embeds = ops.index_select(input=feat, axis=1, index=self.X_Neis)
        X = ops.cat([node_embeds, neis_embeds], axis=-1)
        H = ops.zeros([batch_size, num_parts, feat_dim])
        for t in range(self.T):
            H = ops.index_select(input=H, axis=1, index=self.X_Neis)
            H = self.Hw(X, H, self.deg)
            H = self.Aggr(H, self.X_Node)
        output = self.out_layer(H)
        return output


class AggrSum(nn.Cell):
    def __init__(self, node_num):
        super(AggrSum, self).__init__()
        self.num_nodes = node_num

    def construct(self, H, X_node):
        # H [batch_size, num_edges, feat_dim]  X_node [num_edges]
        batch_size = H.shape[0]
        mask = ops.stack([X_node] * self.num_nodes, 0)  # [num_nodes, num_edges]
        mask = mask.float() - ops.unsqueeze(ops.arange(0, self.num_nodes).float(), 1)
        mask = (mask == 0).float().unsqueeze(0)
        mask = mask.repeat(batch_size, axis=0)
        # (1, num_nodes, num_edges) * (batch_size, num_edges, feat_dim) -> (batch_size, num_nodes, feat_dim)
        return ops.bmm(mask, H)


class Hw(nn.Cell):
    def __init__(self, feat_dim=64, mu=0.9):
        super(Hw, self).__init__()
        self.feat_dim = feat_dim
        self.mu = mu

        # 初始化网络层
        self.Xi = Xi(feat_dim, feat_dim)
        self.Rou = Rou(feat_dim, feat_dim)

    def construct(self, X, H, dg_list):
        # X [batch_size, num_edges, feat_dim*2]  H [batch_size, num_edges, feat_dim]
        # dg_list [num_edges]
        A = (self.Xi(X) * self.mu / self.feat_dim) / dg_list.view(1, -1, 1,
                                                                  1)  # batch_size, num_edges, feat_dim, feat_dim
        b = self.Rou(ops.chunk(X, chunks=2, axis=-1)[0])  # batch_size, num_edges, feat_dim
        out = ops.squeeze(A @ ops.unsqueeze(H, -1), -1) + b
        return out  # batch_size, num_edges, feat_dim


class Xi(nn.Cell):
    def __init__(self, ln, s):
        super(Xi, self).__init__()
        self.ln = ln  # 节点特征向量的维度
        self.s = s  # 节点的个数

        # 线性网络层
        self.linear = nn.Dense(in_channels=2 * ln,
                               out_channels=s ** 2,
                               has_bias=True)
        # 激活函数
        self.tanh = nn.Tanh()

    def construct(self, X):
        batch_size, num_edges, _ = X.shape
        out = self.linear(X)
        out = self.tanh(out)
        return out.view(batch_size, num_edges, self.s, self.s)


class Rou(nn.Cell):
    def __init__(self, ln, s):
        super(Rou, self).__init__()
        self.linear = nn.Dense(in_channels=ln,
                               out_channels=s,
                               has_bias=True)
        self.tanh = nn.Tanh()

    def construct(self, X):
        return self.tanh(self.linear(X))


if __name__ == "__main__":
    part_features = ops.randn([10, 6, 64])
    edges = [[5, 0], [5, 1], [5, 2], [5, 3], [5, 4], [0, 5], [1, 5], [2, 5], [3, 5], [4, 5]]
    gcn = BatchGCN(edges=edges)
    out_features = gcn(part_features)
    print(out_features.shape)
