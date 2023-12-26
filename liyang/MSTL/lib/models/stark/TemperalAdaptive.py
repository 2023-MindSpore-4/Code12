import math
import torch
from mindspore import nn
import torch.nn.functional as F
from torch.nn.modules.utils import _triple
from einops import rearrange,repeat
from plot_features.visual712 import plot_map,PLT_MAP

class Tada(nn.Cell):
    """
    The  function for generating the calibration weights.
    1.Tada will not change the dimension
    2.calibration for the input dimension (in iclr2022)
    """
    def __init__(self,dim ,Temporal_dim = 3,use_bias =False ,use_weight =False , adaptive_feature = True ,
                 resdual_connection = False, draw = True):
        """
        Args:
            c_in (int): number of input channels.
            ratio (int): reduction ratio for the routing function.
        kernels (list): temporal kernel size of the stacked 1D convolutions
        """
        super(Tada, self).__init__()
        self.dim = dim
        self.Temporal_dim = Temporal_dim

        """(B,C,T,H,W)->(B,C,T1=1,H1=1,T1=1)"""
        self.globalPool = nn.AdaptiveAvgPool2d(output_size=(1,1))

        self.use_bias = use_bias
        if use_bias:
            self.linear_bias = nn.Dense(dim, dim)
            self.bias = nn.Parameter(ops.randn(1, dim)) # B C_OUT

        self.use_weight = use_weight
        if use_weight:
            self.linear_weight = nn.Linear(in_features=dim,out_features=dim)
            self.weight = nn.Parameter(ops.randn(1,dim, dim, 3, 3)  )# (B C_OUT C_IN KH KW)

        self.adaptive_feature =  adaptive_feature
        print('use adaptive feature:', adaptive_feature)
        if  adaptive_feature:
            self.feature_weight = nn.Linear(in_features=dim, out_features=dim)
            self.conv = nn.Conv2d(in_channels=dim,out_channels= dim,kernel_size=(3,3),stride=1
                                  ,padding=1)
        else:
            self.conv = nn.Conv2d(in_channels=dim,out_channels= dim,kernel_size=(3,3),stride=1
                                  ,padding=1)

        self.resdual = resdual_connection
        if self.resdual:
            self.b_f1 = nn.SequentialCell(
                nn.BatchNorm2d(dim),
                nn.ReLU(inplace=True)
            )
        if draw:
            self.plot  =PLT_MAP(num_col=2,num_row=2)





    def init_temporal_context(self,input_feature,pos_enc):
        """mask the padding part of the template,mask is of size B,128,128 ,type bool"""
        """reshape the pos_enc"""
        #print(pos_enc.shape)
        descriptor = pos_enc.unsqueeze(1) * input_feature
        descriptor = self.globalPool(descriptor)
        self.temporal_context_descriptor = [ descriptor for i in range(self.Temporal_dim)]


    def update_temporal_context(self,input_feature,pos_enc,remain_template = False,track = False,pop_id=-1):
        """mask the padding part of the template,mask is of size B,128,128 ,typE BOOL"""
        #print(pos_enc.shape)
        HW,B = pos_enc.shape
        H = W = int( math.sqrt(HW) )
        pos_enc = rearrange(pos_enc,'(H W) (C B) -> B C H W',B=B,C=1,H=W,W=W )
        # self.plot.plot(window_id=0,tensor=pos_enc ,_type='(T B) C H W',B=B,C=1,H=W,W=W,T=1 ,save=True,path= '/home/suixin/MyPaper/imgs/OnlineTada/input_feat/' )
        # self.plot.plot(window_id=1,tensor= input_feature , _type='(T B) C H W', B=B, C=192, H=W, W=W,T=1 ,save=True,path= '/home/suixin/MyPaper/imgs/OnlineTada/input_feat/')
        descriptor = (pos_enc.reshape(B,1,H,W ) ) * input_feature # 1 B H W -> B 1 H W
        # self.plot.plot(window_id=2,tensor=descriptor, _type='(T B) C H W', B=B, C=192, H=W, W=W, T=1)
        descriptor = self.globalPool(descriptor)
        if remain_template and not track:
            self.temporal_context_descriptor.pop(-1) # train with -1,eval with -2
        elif track:
            self.temporal_context_descriptor.pop(pop_id)
        else:
            self.temporal_context_descriptor.pop(0)
        self.temporal_context_descriptor.append( descriptor )








    def forward(self,x):
        B,C, H, W= x.shape
        if self.adaptive_feature:
            descriptor = ops.cat(self.temporal_context_descriptor, dim=-1)
            descriptor = self.globalPool(descriptor)
            feature_cal = self.feature_weight( descriptor.reshape( descriptor.shape[0] , -1 ) )
            feature_cal = repeat(feature_cal , 'B C_IN -> B C_IN H W', C_IN=192, B=x.shape[0], H=1, W=1)
            # self.plot.plot(window_id=0, tensor=x, _type= '(B T) C H W',B=B,C= C,H=H,W=W,T=1 )
            #plot_map(tensor=x, _type= '(B T) C H W',B=B,C= C,H=H,W=W,T=1 )
            x = x *(1+ feature_cal)
            # self.plot.plot(window_id=1, tensor=x, _type='(B T) C H W', B=B, C=C, H=H, W=W, T=1)
            #plot_map(tensor=x, _type='(B T) C H W', B=B, C=C, H=H, W=W, T=1)
            x = self.conv(x)
            # self.plot.plot(window_id=2, tensor=x, _type='(B T) C H W', B=B, C=C, H=H, W=W, T=1)
            #plot_map(tensor=x, _type='(B T) C H W', B=B, C=C, H=H, W=W, T=1)
        else:
            x = self.conv(x)

        return x;



# Model = Tada(256,256)
# z=ops.randn(2,256,8,8)
# x = ops.randn(2,256,20,20)
# mask = torch.full( (2,128,128),1)
# previous = ops.randn(2,3,256,20,20)
# mask[1:,:,:] = False
# Model(z,'template',mask = mask)
# out = Model(x,previous = previous ,mode = 'scr')


# Model = WeightGenerating(c_in=256,c_out=128, ratio=2,T_in=3,type='1D')
# Model2 = TAdaConv2d(in_channels=256,out_channels=128,kernel_size=[1, 3, 3],  # usually the temporal kernel size is fixed to be 1
#             stride=[1, 1, 1],  # usually the temporal stride is fixed to be 1
#             padding=[0, 1, 1],  # usually the temporal padding is fixed to be 0
#             cal_dim="cout")
# z=ops.randn(2,256,8,8)
# x = ops.randn(2,3,256,20,20)
# Model.init_template(z)
# out = Model(x)
# out = Model2(x,out)
# print(out.shape)