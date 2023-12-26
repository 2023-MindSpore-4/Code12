# %%
# %matplotlib inline
import random
import mindspore as ms
import mindspore.ops as ops
from mindspore.ops import operations as P
from mindspore import Tensor, Parameter, context

from mindspore.common.initializer import initializer, Zero, Normal, Uniform
import mindspore.nn as nn

import numpy as np


# %%
def synthetic_data(w, b, num_examples):  #@save
    """生成y=Xw+b+噪声"""
    
    X = ops.normal((num_examples, len(w)), Tensor(0, ms.int32), Tensor(1, ms.int32))
    y = ops.matmul(X, w) + b
    y += ops.normal(y.shape, Tensor(0, ms.int32), Tensor(0.01, ms.float32))
    return X, y.reshape((-1, 1))


# %%
def data_iter(batch_size, feature, labels):
    num_examples = len(feature)
    indices = list(range(num_examples))

    # random.shuffle(indices)
    for i in range(0, num_examples, batch_size):
        batch_indices = Tensor(
            indices[i: min(i + batch_size, num_examples)])
        yield feature[batch_indices], labels[batch_indices]

# %%


    
    
import pandas as pd

 
# 读取Excel文件
excel_file = pd.ExcelFile('data.xls')

# 获取所有表格名称
sheet_names = excel_file.sheet_names

# 导入指定名称的表格数据
if 'Andan' in sheet_names:
    data_andan = excel_file.parse('Andan')
    print("Andan表格数据：")
    print(data_andan)
else:
    print("未找到名称为Andan的表格")

# %%
## 提取数据
features_andan = data_andan[['进水TP', '厌氧末端ORP','好氧前段DO', '好氧末端TSS', '出水pH','温度']]
labels = data_andan[['出水NH4-N']]

features_andan= features_andan.to_numpy()
labels = labels.to_numpy()

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import check_random_state

# 假设 features_andan 和 labels 是已有的 numpy 数组

# 划分训练数据和测试数据
random_state = check_random_state(0)
# X_train, X_test, y_train, y_test = train_test_split(features_andan, labels, test_size=0.2, random_state=random_state)
X_train = features_andan[1:200,:]
y_train = labels[1:200,:]
X_test = features_andan[200:250,:]
y_test = labels[200:250,:]

# 归一化处理
scaler = MinMaxScaler()
yscaler = MinMaxScaler()
X_train_normalized = scaler.fit_transform(X_train)
y_train_normalized = yscaler.fit_transform(y_train)
X_test_normalized = scaler.transform(X_test)
y_test_normalized = yscaler.transform(y_test)

# 添加噪声到训练数据
noise_std = 0.1  # 噪声标准差
noise = random_state.normal(scale=noise_std, size=X_train_normalized.shape)
y_noise = random_state.normal(scale=noise_std, size=y_train_normalized.shape)
X_train_noisy = X_train_normalized + noise
y_train_nosiy = y_train_normalized + y_noise


## 转换格式
X_train = ms.Tensor(X_train_noisy, ms.float32)
y_train = ms.Tensor(y_train_nosiy, ms.float32)

X_test = ms.Tensor(X_test_normalized, ms.float32)
y_test = ms.Tensor(y_test_normalized, ms.float32)



# %%
from mindspore import dataset

class MyDataset(dataset.Dataset):
    def __init__(self, data, labels):
        super(MyDataset, self).__init__()
        self.data = data
        self.labels = labels

    def __getitem__(self, index):
        return self.data[index], self.labels[index]

    def __len__(self):
        return len(self.data)

# %% [markdown]
# ## 定义网络

# %%
import mindspore.numpy as mnp

context.set_context(mode=context.GRAPH_MODE, device_target="CPU")       
## 定义FNN        
class FNN(nn.Cell):
    def __init__(self, InDim, RuleNum, OutDim):
        super(FNN, self).__init__()
        self.exp = ops.Exp()
        self.prod = ops.ReduceProd(keep_dims=True)
        self.sum = ops.ReduceSum(keep_dims=True)
        self.matmul = P.MatMul()
        self.sub = ops.Sub()
        self.square = ops.Square()
        self.div = ops.RealDiv()
        self.mul = ops.Mul()
        self.expand_dims = P.ExpandDims()
        self.transpose = P.Transpose()
        self.InDim = InDim
        self.RuleNum = RuleNum
        self.OutDim = OutDim
        self.Center1 = Parameter(initializer('normal', [InDim, RuleNum]), name='center1')
        self.Center2 = Parameter(initializer('normal', [InDim, RuleNum]), name='center2')
        self.Width = Parameter(initializer('ones', [InDim, RuleNum]), name='width')
        self.W = Parameter(initializer('normal', [InDim, RuleNum]), name='W')
        self.q = 0.55
        self.b =Parameter(initializer('zero', [1, RuleNum]), name='b') 
        
    
    def predict(self, SamIn):
        MemFunUnitOut1 = Tensor(np.random.randn(self.InDim, self.RuleNum), ms.float32)
        MemFunUnitOut2 = Tensor(np.random.randn(self.InDim, self.RuleNum), ms.float32)
        for i in range(0, SamIn.shape[1]):
            for j in range(0, self.RuleNum):
                if SamIn[0, i] > self.Center1[i, j] :
                    MemFunUnitOut1[i,j]=self.exp(-self.div(self.square(SamIn[0, i]-self.Center1[i,j]), 2*self.square(self.Width[i,j])))
                elif SamIn[0, i] <= self.Center1[i, j] and  SamIn[0, i] > self.Center2[i,j]:
                    MemFunUnitOut1[i,j]=1
                else:
                    MemFunUnitOut1[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center2[i,j]), 2*self.square(self.Width[i,j])))
                
                if SamIn[0, i] >  self.square(self.Center1[i,j] + self.Center2[i,j]):
                    MemFunUnitOut2[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center2[i,j]), 2*self.square(self.Width[i,j])))
                else:  
                    MemFunUnitOut2[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center1[i,j]), 2*self.square(self.Width[i,j])))
        ReluUnitOut1 = self.prod(MemFunUnitOut1, 0)
        ReluUnitOut2 = self.prod(MemFunUnitOut2, 0)
        
        ReluUnitOutSum1 = self.sum(ReluUnitOut1)
        ReluUnitOutSum2 = self.sum(ReluUnitOut2)
        Norm1 = ReluUnitOut1/ReluUnitOutSum1
        Norm2 = ReluUnitOut2/ReluUnitOutSum2
        
        # quanzhong = []
        quanzhong = Tensor(np.random.randn(1, self.RuleNum), ms.float32)
        for j in range(0, self.RuleNum):
            #quanzhong[:, j] = Tensor((self.sum(self.mul(self.W[:, j], self.transpose(SamIn, (1,0)))) + self.b[0, j]).asnumpy(), ms.float32)
            quanzhong[:, j] = self.sum(self.W[:, j])+ self.b[0, j]
            # quanzhong[:, j] = Tensor((self.sum(self.mul(self.W[:, j], self.transpose(SamIn, (1,0)))) + self.b[0, j]).asnumpy(), ms.float32).squeeze(axis=1)
        
        NetOut = self.q * self.sum(self.mul(Norm2, quanzhong)) + (1-self.q)*self.sum(self.mul(Norm1,quanzhong))
        return NetOut
    
    def construct(self, SamIn):
        MemFunUnitOut1 = Tensor(np.random.randn(self.InDim, self.RuleNum), ms.float32)
        MemFunUnitOut2 = Tensor(np.random.randn(self.InDim, self.RuleNum), ms.float32)
        for i in range(0, SamIn.shape[1]):
            for j in range(0, self.RuleNum):
                if SamIn[0, i] > self.Center1[i, j] :
                    MemFunUnitOut1[i,j]=self.exp(-self.div(self.square(SamIn[0, i]-self.Center1[i,j]), 2*self.square(self.Width[i,j])))
                elif SamIn[0, i] <= self.Center1[i, j] and  SamIn[0, i] > self.Center2[i,j]:
                    MemFunUnitOut1[i,j]=1
                else:
                    MemFunUnitOut1[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center2[i,j]), 2*self.square(self.Width[i,j])))
                
                if SamIn[0, i] >  self.square(self.Center1[i,j] + self.Center2[i,j]):
                    MemFunUnitOut2[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center2[i,j]), 2*self.square(self.Width[i,j])))
                else:  
                    MemFunUnitOut2[i,j] = self.exp(-self.div(self.square(SamIn[0, i]-self.Center1[i,j]), 2*self.square(self.Width[i,j])))
        ReluUnitOut1 = self.prod(MemFunUnitOut1, 0)
        ReluUnitOut2 = self.prod(MemFunUnitOut2, 0)
        
        ReluUnitOutSum1 = self.sum(ReluUnitOut1)
        ReluUnitOutSum2 = self.sum(ReluUnitOut2)
        Norm1 = ReluUnitOut1/ReluUnitOutSum1
        Norm2 = ReluUnitOut2/ReluUnitOutSum2
        # quanzhong= []
        quanzhong = Tensor(np.random.randn(1, self.RuleNum), ms.float32)
        for j in range(0, self.RuleNum):
            #quanzhong[:, j] = Tensor((self.sum(self.mul(self.W[:, j], self.transpose(SamIn, (1,0)))) + self.b[0, j]).asnumpy(), ms.float32)
            quanzhong[:, j] = self.sum(self.W[:, j])+ self.b[0, j]
            # quanzhong[:, j] = Tensor((self.sum(self.mul(self.W[:, j], self.transpose(SamIn, (1,0)))) + self.b[0, j]), ms.float32).squeeze()
        
        NetOut = self.q * self.sum(self.mul(Norm2, quanzhong)) + (1-self.q)*self.sum(self.mul(Norm1,quanzhong))
        
        return NetOut

class WithLossCell(nn.Cell):
    def __init__(self, network, loss):
        super(WithLossCell, self).__init__(auto_prefix=False)
        self.network = network
        self.loss = loss

    def construct(self, data, label):
        predict = self.network(data)
        return self.loss(predict, label)
    

class MyWithEvalCell(nn.Cell):
    """定义验证流程"""

    def __init__(self, network):
        super(MyWithEvalCell, self).__init__(auto_prefix=False)
        self.network = network

    def construct(self, data, label):
        outputs = self.network(data)
        return outputs, label

class MyMAE(nn.Metric):
    """定义metric"""

    def __init__(self):
        super(MyMAE, self).__init__()
        self.clear()

    def clear(self):
        """初始化变量abs_error_sum和samples_num"""
        self.abs_error_sum = 0
        self.samples_num = 0

    def update(self, *inputs):
        """更新abs_error_sum和samples_num"""
        y_pred = inputs[0].asnumpy()
        y = inputs[1].asnumpy()

        # 计算预测值与真实值的绝对误差
        # print(y_pred.shape)
        # print(y_pred)
        
        error_abs = np.abs(y.reshape(y_pred.shape) - y_pred)
        self.abs_error_sum += error_abs.sum()
        self.samples_num += y.shape[0]  # 样本的总数

    def eval(self):
        """计算最终评估结果"""
        return self.abs_error_sum / self.samples_num



class FNNTrain:
    def __init__(self, InDim, RuleNum, OutDim, lr, MaxEpoch, E0):
        self.network = FNN(InDim, RuleNum, OutDim)
        self.loss = nn.MSELoss()
        self.optimizer = nn.SGD(self.network.trainable_params(), learning_rate=lr)
        self.net_with_loss = WithLossCell(self.network, self.loss)
        self.train_network = nn.TrainOneStepCell(self.net_with_loss, self.optimizer)
        self.train_network.set_train()
        self.MaxEpoch = MaxEpoch
        self.E0 = E0
        self.losslist = []
        self.mae = MyMAE()
        
    
    
    def test(self, fe, out):
        eval_net = MyWithEvalCell(self.network)
        eval_net.set_train(False)
        
        testloader = data_iter(1, fe, out)

        all_loss = []
        
        # for x, y in testloader:
        #     output, eval_y = eval_net(x, y)
        #     print(f"output:{output}") 
        #     self.mae.update(output, eval_y)
        # mae_result = self.mae.eval()
        # print("MAE: ", mae_result)
        
        for i, (x, y) in enumerate(testloader):
            print(f'x.shpae:{x.shape}; y.shape:{y.shape}')
            self.network.set_train(False)
            pred = self.network(x)
            print(f'pred:{pred}')
            loss = self.loss(pred, y) 
            all_loss.append(loss)
            print(f"loss:{loss}\n")
        print(f'average loss:{sum(all_loss)/len(all_loss)}')
     
    def run(self, fe, out):
        for epoch in range(0,self.MaxEpoch):
            print(epoch)
            a=0
            train_dataset = data_iter(1, fe, out)
            avg_loss = []
            for i, (data, label) in enumerate(train_dataset):
                a=a+1
                loss = self.train_network(data, label)
                avg_loss.append(loss)    
            
            print(f"epoch: {epoch}/{self.MaxEpoch}, loss: {sum(avg_loss)/len(avg_loss)}")
            self.losslist.append(loss.asnumpy())
            
        return self.network

# %% [markdown]
# ## 训练

# %%





network = FNN(InDim=6, RuleNum=15, OutDim=1)
loss_fn = nn.MSELoss()
opt = nn.SGD(network.trainable_params(), 0.05)


def forward_fn(data, label):
    logits = network.construct(data)
    loss = loss_fn(logits, label)
    return loss, logits

grad_fn = ms.value_and_grad(forward_fn, None, opt.parameters, has_aux=True)

def train_step(data, label):
    (loss, _), grads = grad_fn(data, label)
    opt(grads)
    return loss

def train(fe, out, model,MaxEpoch):
    losslist = []
    model.set_train()
    for epoch in range(0,MaxEpoch):
        print(epoch)
        a=0
        train_dataset = data_iter(1, fe, out)
        avg_loss = []
        for i, (data, label) in enumerate(train_dataset):
            a=a+1
            loss = train_step(data, label)
            avg_loss.append(loss)    
        
        print(f"epoch: {epoch}/{MaxEpoch}, loss: {sum(avg_loss)/len(avg_loss)}")
        losslist.append(loss.asnumpy())
    return losslist

def test(fe, out, model,loss_fn):
    
    model.set_train(False)
    testloader = data_iter(1, fe, out)

    all_loss = []
    
    for i, (x, y) in enumerate(testloader):
        print(f'x.shpae:{x}; y.shape:{y.shape}')
        
        pred = network.predict(x)
        print(f'pred:{pred}')
        loss = loss_fn(pred, y) 
        all_loss.append(loss)
        print(f"loss:{loss}\n")
    print(f'average loss:{sum(all_loss)/len(all_loss)}')



TrainSamInN = Tensor(X_train, ms.float32)  
TrainSamOutN = Tensor(y_train, ms.float32)
TestSamInN = Tensor(X_test, ms.float32)  
TestSamOutN = Tensor(y_test, ms.float32)


allloss = train(TrainSamInN, TrainSamOutN, network, MaxEpoch=100)

test(TestSamInN, TestSamOutN, network, loss_fn)


# fnn_train = FNNTrain(InDim=6, RuleNum=15, OutDim=1, lr=0.05, MaxEpoch=100, E0=0.001)
# network = fnn_train.run(TrainSamInN, TrainSamOutN)


