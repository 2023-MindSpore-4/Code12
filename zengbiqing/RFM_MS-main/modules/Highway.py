import mindspore
import mindspore.nn as nn
import x2ms_adapter
import x2ms_adapter.nn as x2ms_nn

class Highway(nn.Cell):
    def __init__(self, input_size, output_size, num_layers=1, f=x2ms_adapter.tanh):

        super(Highway, self).__init__()

        self.num_layers = num_layers

        self.nonlinear = x2ms_nn.ModuleList([x2ms_nn.Linear(input_size, output_size) for _ in range(num_layers)])

        self.linear = x2ms_nn.ModuleList([x2ms_nn.Linear(input_size, output_size) for _ in range(num_layers)])

        self.gate = x2ms_nn.ModuleList([x2ms_nn.Linear(input_size, output_size) for _ in range(num_layers)])

        self.f = f

    def construct(self, x):
        """
            :param x: tensor with shape of [batch_size, size]
            :return: tensor with shape of [batch_size, size]
            applies σ(x) ⨀ (f(G(x))) + (1 - σ(x)) ⨀ (Q(x)) transformation | G and Q is affine transformation,
            f is non-linear transformation, σ(x) is affine transformation with sigmoid non-linearition
            and ⨀ is element-wise multiplication
            """

        for layer in range(self.num_layers):
            gate = x2ms_adapter.sigmoid(self.gate[layer](x))

            nonlinear = self.f(self.nonlinear[layer](x))
            linear = self.linear[layer](x)

            x = gate * nonlinear + (1 - gate) * linear

        return x