import math
from histogram_layer import Stat_fea
import mindspore
from mindspore import nn, ops
EPSILON = 1e-15

########################   Centeral-difference (second order, with 9 parameters and a const theta for 3x3 kernel) 2D Convolution   ##############################
## | a1 a2 a3 |   | w1 w2 w3 |
## | a4 a5 a6 | * | w4 w5 w6 | --> output = \sum_{i=1}^{9}(ai * wi) - \sum_{i=1}^{9}wi * a5 --> Conv2d (k=3) - Conv2d (k=1)
## | a7 a8 a9 |   | w7 w8 w9 |
##
##   --> output = 
## | a1 a2 a3 |   |  w1  w2  w3 |     
## | a4 a5 a6 | * |  w4  w5  w6 |  -  | a | * | w\_sum |     (kernel_size=1x1, padding=0)
## | a7 a8 a9 |   |  w7  w8  w9 |     

class Conv2d_cd(nn.Cell):
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1,
                 padding=1, dilation=1, group=1, has_bias=False, theta=0.7):
        super(Conv2d_cd, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, pad_mode='pad',dilation=dilation, group=group, has_bias=has_bias)
        self.theta = theta

    def construct(self, x):
        out_normal = self.conv(x)
        if math.fabs(self.theta - 0.0) < 1e-8:
            return out_normal 
        else:
            #pdb.set_trace()
            [C_out,C_in, kernel_size,kernel_size] = self.conv.weight.shape
            kernel_diff = self.conv.weight.sum(2).sum(2)
            kernel_diff = kernel_diff[:, :, None, None]
            out_diff = ops.conv2d(input=x, weight=kernel_diff, has_bias=self.conv.has_bias, stride=self.conv.stride, padding=0, groups=self.conv.groups)

            return out_normal - self.theta * out_diff


class SpatialAttention(nn.Cell):
    def __init__(self, kernel = 3):
        super(SpatialAttention, self).__init__()


        self.conv1 = nn.Conv2d(2, 1, kernel_size=kernel, padding=kernel//2, has_bias=False,pad_mode='pad')
        self.sigmoid = nn.Sigmoid()

    def construct(self, x):
        avg_out = ops.mean(x, axis=1, keep_dims=True)
        max_out, _ = ops.max(x, axis=1, keepdims=True)
        x = ops.cat([avg_out, max_out], axis=1)
        x = self.conv1(x)
        
        return self.sigmoid(x)



class AttentionModule(nn.Cell):
    def __init__(self, in_channels, out_channels, **kwargs):
        super(AttentionModule, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, has_bias=False, **kwargs)
        self.bn = nn.BatchNorm2d(out_channels, eps=0.001)

    def construct(self, x):
        x = self.conv(x)
        x = self.bn(x)
        # 本身就是0-1 所以不用sigmoid 可以用attention
        return ops.relu(x, inplace=True)
        # return F.sigmoid(x)

class CCAModule(nn.Cell):
    def __init__(self, pool='GAP'):
        super(CCAModule, self).__init__()
        assert pool in ['GAP', 'GMP']
        if pool == 'GAP':
            self.pool = None
        else:
            self.pool = nn.AdaptiveMaxPool2d(1)

    def construct(self, features, attentions):
        B, C, H, W = features.size()
        _, M, AH, AW = attentions.size()

        # match size
        if AH != H or AW != W:
            attentions = F.upsample_bilinear(attentions, size=(H, W))

        # imjk, injk->imn: i*m*j*k i*n*j*k -> i*m*n
        if self.pool is None:
            feature_matrix = (ops.einsum('imjk,injk->imn', (attentions, features)) / float(H * W)).view(B, -1)
        else:
            feature_matrix = []
            # 每个attention map与 F
            for i in range(M):
                AiF = self.pool(features * attentions[:, i:i + 1, ...]).view(B, -1)
                feature_matrix.append(AiF)
            # feature_mat 0 ~ 13
            feature_matrix = ops.Concat(feature_matrix, dim=1)

        feature_matrix = ops.sign(feature_matrix) * ops.sqrt(ops.abs(feature_matrix)+ EPSILON)

        fake_att = ops.zeros_like(attentions).uniform_(0, 2)

        counterfactual_feature = (ops.einsum('imjk,injk->imn', (fake_att, features)) / float(H * W)).view(B, -1)

        # 0 ~ 2
        counterfactual_feature = ops.sign(counterfactual_feature) * ops.sqrt(ops.abs(counterfactual_feature)+ EPSILON)

        return feature_matrix, counterfactual_feature

class HFMNetwork(nn.Cell):

    def __init__(self, basic_conv=Conv2d_cd, theta=0.0):
        super(HFMNetwork, self).__init__()
        
        
        self.conv1 = nn.SequentialCell(
            basic_conv(3, 64, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(64),
            nn.ReLU(),    
        )
        
        self.Block1 = nn.SequentialCell(
            basic_conv(64, 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),

            basic_conv(128, int(128*1.6), kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.6)),
            nn.ReLU(),  
            basic_conv(int(128*1.6), 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad'),
            
        )
        
        self.Block2 = nn.SequentialCell(
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            basic_conv(128, int(128*1.4), kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.4)),
            nn.ReLU(),  
            basic_conv(int(128*1.4), 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),  
            
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad'),
        )
        
        self.Block3 = nn.SequentialCell(
            basic_conv(128, 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            basic_conv(128, int(128*1.2), kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(int(128*1.2)),
            nn.ReLU(),  
            basic_conv(int(128*1.2), 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1,pad_mode='pad'),
        )
        
        # Original
        
        self.lastconv1 = nn.SequentialCell(
            basic_conv(128*3, 128, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            basic_conv(128, 1, kernel_size=3, stride=1, padding=1, has_bias=False, theta= theta),
            nn.ReLU(),    
        )

        self.sa1 = SpatialAttention(kernel = 7)
        self.sa2 = SpatialAttention(kernel = 5)
        self.sa3 = SpatialAttention(kernel = 3)
        self.downsample32x32 = nn.Upsample(size=(32, 32), mode='bilinear')

        # 0425 add
        self.linear = nn.Dense(32 * 32 + 128 + 128, 2)
        self.linear_1024_2 = nn.Dense(32 * 32, 2)

        # 0425 add 128为量化level
        self.stat_fea = Stat_fea(128, 64)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_384_1024 = nn.Dense(384, 1024)
        self.fc_768_1024 = nn.Dense(768, 1024)

        self.attentions = AttentionModule(384, 8, kernel_size=1)
        self.cca = CCAModule(pool='GMP')
        self.fc_1 = nn.Dense(384 * 8, 32 * 32)

    # 11层conv_l 每层conv-bn-relu :conv conv_l-conv_l-conv_l (block_1) attention_l (conv-sigmoid) conv_l-conv_l-conv_l-conv_l attention_l conv_l-conv_l-conv_l-conv_l (block_2) attention_l conv_l-conv_l-conv_l (block_3) attention_l
    def construct(self, x):
        x_input = x
        x = self.conv1(x)
        x_Block1 = self.Block1(x)
        attention1 = self.sa1(x_Block1)
        x_Block1_SA = attention1 * x_Block1
        x_Block1_32x32 = self.downsample32x32(x_Block1_SA)   
        
        x_Block2 = self.Block2(x_Block1)	    
        attention2 = self.sa2(x_Block2)  
        x_Block2_SA = attention2 * x_Block2
        x_Block2_32x32 = self.downsample32x32(x_Block2_SA)  
        
        x_Block3 = self.Block3(x_Block2)	    
        attention3 = self.sa3(x_Block3)  
        x_Block3_SA = attention3 * x_Block3	
        x_Block3_32x32 = self.downsample32x32(x_Block3_SA)

        # n, 384, 32, 32
        x_concat = ops.cat((x_Block1_32x32,x_Block2_32x32,x_Block3_32x32), axis=1)

        fea = self.avgpool(x_concat)
        # print(fea)
        # feature_tensor = fea  # Use a different variable name
        #
        # feature_tensor = feature_tensor.view(feature_tensor.size(0), -1)
        # fea = fea.view((fea.size(0), -1))
        size = fea.shape[0]
        a2 = fea.view((size,-1))
        x_cf = self.fc_384_1024(a2)

        stat_output, stat_feas = self.stat_fea(x)
        cat_output = ops.cat((x_cf, stat_output, stat_feas), axis=1)
        # cat_output_env = torch.cat((x_cf_env, quant_output, quant_level_feas), dim=1)
        output = self.linear(cat_output)
        # output_env = self.linear(cat_output_env)

        # # EM causal intervention
        # fea_ = fea.cpu().detach().numpy()
        # gm = GaussianMixture(n_components=3, random_state=0).fit(fea_)
        # em_feature = torch.from_numpy(gm.means_)
        # # env_feature = torch.cat([em_feature[1], em_feature[2]], 0)
        # env_feature = (em_feature[1] + em_feature[2]) * 0.5
        # env_feature = torch.unsqueeze(env_feature, 0)
        # env_feature = env_feature.expand(fea.size(0), env_feature.size(1)).cuda()
        # fea_cat_env = torch.cat([fea, env_feature.float()], 1)
        # x_cf_env = self.fc_768_1024(fea_cat_env)

        # output_env = self.linear_1024_2(x_cf_env)
        # output = self.linear_1024_2(x_cf)
        output_env = output

        map_x = self.lastconv1(x_concat)
        map_x = map_x.squeeze(1)

        return map_x, x_concat, attention1, attention2, attention3, x_input, output, output_env, x_concat
		



