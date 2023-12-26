from mindspore import Tensor,ops,nn,Parameter

class Conv2d_Spetral_Norm(nn.Cell):

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size,
                 stride=1,
                 pad_mode='pad',
                 padding=0,
                 dilation=1,
                 group=1,
                 has_bias=False,
                 weight_init='normal',
                 bias_init='zeros',
                 data_format='NCHW'):
        super().__init__()
        
        self.conv2d=ops.Conv2D(out_channels,kernel_size,1,pad_mode,padding,stride,dilation,group,data_format)
        weight_shape=(out_channels,in_channels,kernel_size,kernel_size)
        self.u=Parameter(ops.standard_normal((1,weight_shape[-1])))
        self.weight=Parameter(ops.standard_normal(weight_shape))
        self.l2_normalize=ops.L2Normalize(epsilon=1e-12)

    def spectral_norm(self,weight:Tensor,iterarion = 1)->Tensor:
        weight_shape = weight.shape

        w = weight.reshape(-1,weight_shape[-1])

        u_hat = Tensor(self.u)
        v_hat = None

        for _ in range(iterarion):
            v_hat=self.l2_normalize(ops.matmul(u_hat,w.T))
            u_hat=self.l2_normalize(ops.matmul(v_hat,w))

        # u_hat=ops.stop_gradient(u_hat)
        # v_hat=ops.stop_gradient(v_hat)

        norm_value = ops.matmul(ops.matmul(v_hat,w),u_hat.T)

        self.u=u_hat

        w_norm=w/norm_value
        w_norm=w_norm.reshape(weight_shape)

        return w_norm

    def construct(self, x:Tensor):
        weight=Tensor(self.weight)
        weight=self.spectral_norm(weight)
        x=self.conv2d(x,weight)

        return x