import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn
import x2ms_adapter.nn_functional


class BilinearAttention(nn.Cell):
    def __init__(self, query_size, key_size, hidden_size):
        super().__init__()
        self.linear_key = x2ms_nn.Linear(key_size, hidden_size, bias=False)
        self.linear_query = x2ms_nn.Linear(query_size, hidden_size, bias=True)
        self.v = x2ms_nn.Linear(hidden_size, 1, bias=False)
        self.hidden_size = hidden_size

    def score(self, query, key, softmax_dim=-1, mask=None):
        attn = self.matching(query, key, mask)

        attn = x2ms_adapter.nn_functional.softmax(attn, dim=softmax_dim)

        return attn

    def matching(self, query, key, mask=None):
        wq = self.linear_query(query)
        wq = x2ms_adapter.tensor_api.unsqueeze(wq, -2)

        uh = self.linear_key(key)
        uh = x2ms_adapter.tensor_api.unsqueeze(uh, -3)

        wuc = wq + uh

        wquh = x2ms_adapter.tanh(wuc)

        attn = x2ms_adapter.tensor_api.squeeze(self.v(wquh), -1)

        if mask is not None:
            attn = x2ms_adapter.tensor_api.masked_fill(attn, mask, -float('inf'))

        return attn

    def construct(self, query, key, value, mask=None):
        attn = self.score(query, key, mask=mask)
        h = x2ms_adapter.bmm(x2ms_adapter.tensor_api.view(attn, -1, x2ms_adapter.tensor_api.x2ms_size(attn, -2), x2ms_adapter.tensor_api.x2ms_size(attn, -1)), x2ms_adapter.tensor_api.view(value, -1, x2ms_adapter.tensor_api.x2ms_size(value, -2), x2ms_adapter.tensor_api.x2ms_size(value, -1)))

        return x2ms_adapter.tensor_api.view(h, list(x2ms_adapter.tensor_api.x2ms_size(attn))[:-2] + [x2ms_adapter.tensor_api.x2ms_size(attn, -2), -1]), attn
