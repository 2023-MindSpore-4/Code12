import mindspore
import mindspore.nn as nn
import mindspore.ops as F
import mindspore.numpy as mnp
from utils.config import cfg
import mindspore.common.initializer as init
from mindspore import load_checkpoint, load_param_into_net

__all__ = ['ResNet', 'resnet18', 'resnet34', 'resnet50', 'resnet101',
           'resnet152', 'resnext50_32x4d', 'resnext101_32x8d',
           'wide_resnet50_2', 'wide_resnet101_2']


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     pad_mode='pad', padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Cell):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU()
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out

class Bottleneck(nn.Cell):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU()
        self.downsample = downsample
        self.stride = stride

    def construct(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out = out + identity
        out = self.relu(out)

        return out


class MLP(nn.Cell):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(MLP, self).__init__()
        self.fc1 = nn.Dense(num_features, hidden_size, weight_init='normal')
        self.fc2 = nn.Dense(hidden_size, num_classes, weight_init='normal')

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)

    def forward(self, data):
        x = data
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.fc1(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, axis=1)


def _one_hot(idx, num_class):
    return mindspore.ops.zeros(len(idx), num_class).to(idx.device).scatter_(
        1, idx.unsqueeze(1), 1.)

class GCN(nn.Cell):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 dropout=0.5,
                 activation="relu"):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(num_features, hidden_size)
        self.conv2 = GCNConv(hidden_size, num_classes)
        self.dropout = nn.Dropout(keep_prob=1 - dropout)
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)()

    def forward(self, data):
        x, edge_index, edge_weights = data.x, data.edge_index, data.edge_attr   # x[2708,1433] edge_index[2,10556]
        # x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.activation(self.conv1(x, edge_index))  # graph convolution x->[2708,64]
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv2(x, edge_index)  # x[2708,7]
        return F.log_softmax(x, axis=1)  # 先计算softmax 再计算log

class LSM(nn.Cell):
    def __init__(self,
                 num_features,
                 num_classes,
                 hidden_size,
                 hidden_x,
                 dropout=0.5,
                 activation="relu",
                 neg_ratio=1.0):
        super(LSM, self).__init__()
        self.p_y_x = MLP(num_features, num_classes, hidden_size, dropout,
                         activation)
        self.x_enc = nn.Dense(num_features, hidden_x)
        self.p_e_xy = nn.Dense(2 * (hidden_x + num_classes), 1)

        self.dropout = dropout
        assert activation in ["relu", "elu"]
        self.activation = getattr(F, activation)
        self.neg_ratio = neg_ratio

    def forward(self, data):
        y_log_prob = self.p_y_x(data.x)
        y_prob = mnp.exp(y_log_prob)
        # 训练数据保留标签，其他数据保留预测。 可以看做是无标签数据的后验概率
        y_prob = F.where(
            data.train_mask.unsqueeze(1),
            self.one_hot(data.y, y_prob.shape[1]),
            y_prob)
        x = F.Dropout(self.dropout)(data.x, self.training)
        x = self.activation(self.x_enc(x))

        # Positive edges.
        y_query = self.embedding(data.edge_index[0], y_prob)
        y_key = self.embedding(data.edge_index[1], y_prob)
        x_query = self.embedding(data.edge_index[0], x)
        x_key = self.embedding(data.edge_index[1], x)
        xy = F.cat([x_query, x_key, y_query, y_key], axis=1)
        e_pred_pos = self.p_e_xy(xy)

        # Negative edges.
        e_pred_neg = None
        if self.neg_ratio > 0:
            num_edges_pos = data.edge_index.shape[1]
            num_nodes = data.x.shape[0]
            num_edges_neg = int(self.neg_ratio * num_edges_pos)
            edge_index_neg = mnp.randint(num_nodes,
                                                (2, num_edges_neg)).astype(mnp.int32).to(x.device)
            y_query = self.embedding(edge_index_neg[0], y_prob)
            y_key = self.embedding(edge_index_neg[1], y_prob)
            x_query = self.embedding(edge_index_neg[0], x)
            x_key = self.embedding(edge_index_neg[1], x)
            xy = F.cat([x_query, x_key, y_query, y_key], axis=1)
            e_pred_neg = self.p_e_xy(xy)

        return e_pred_pos, e_pred_neg, y_log_prob

    def nll_generative(self, data, post_y_log_prob):
        e_pred_pos, e_pred_neg, y_log_prob = self.forward(data)
        # unlabel_mask = data.val_mask + data.test_mask
        unlabel_mask = mindspore.ops.ones_like(data.train_mask) - data.train_mask

        # nll of p_g_xy
        nll_p_g_xy = -mindspore.ops.mean(F.logsigmoid(e_pred_pos))  # 生成的边均值
        if e_pred_neg is not None:
            nll_p_g_xy += -mindspore.ops.mean(F.logsigmoid(-e_pred_neg))  # 非生成边均值

        # nll of p_y_x
        nll_p_y_x = F.nll_loss(y_log_prob[data.train_mask],
                               data.y[data.train_mask])
        nll_p_y_x += -mindspore.ops.mean(
            mindspore.ops.exp(post_y_log_prob[unlabel_mask]) *
            y_log_prob[unlabel_mask])

        # nll of q_y_xg
        nll_q_y_xg = -mindspore.ops.mean(
            mindspore.ops.exp(post_y_log_prob[unlabel_mask]) *
            post_y_log_prob[unlabel_mask])

        return nll_p_g_xy + nll_p_y_x + nll_q_y_xg  # 1是生成损失，2和3都针对有标签数据


class ResNet(nn.Cell):

    def __init__(self, block, layers, num_classes=1000, zero_init_residual=False,
                 groups=1, width_per_group=64, replace_stride_with_dilation=None,
                 norm_layer=None):
        super(ResNet, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        self._norm_layer = norm_layer
        self.dropout = cfg.train.dropout
        self.inplanes = 64
        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError("replace_stride_with_dilation should be None "
                             "or a 3-element tuple, got {}".format(replace_stride_with_dilation))
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(3, self.inplanes, kernel_size=7, stride=2, padding=3,
                               bias=False)
        self.bn1 = norm_layer(self.inplanes)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2,
                                       dilate=replace_stride_with_dilation[0])
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2,
                                       dilate=replace_stride_with_dilation[1])
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2,
                                       dilate=replace_stride_with_dilation[2])
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Dense(512 * block.expansion, num_classes)

        # LSM definition
        self.p_y_x = MLP(512 * block.expansion, 101, hidden_size=2, dropout=self.dropout,
                         activation='relu')
        self.x_enc = nn.Dense(512 * block.expansion, 101)  # y = xA^T + b
        self.p_e_xy = nn.Dense(2 * (2 + 101), 1)
        # self.p_e_xy = MLP(2 * (2 + 101), 1, hidden_size=2, dropout=0, activation='relu')

        self.activation = getattr(F, 'relu')

        # GCN definition
        self.conv1_g = GCNConv(in_channels=512 * block.expansion, out_channels=256)
        self.conv2_g = GCNConv(in_channels=256, out_channels=101)

        self.conv1_g1 = GCNConv(in_channels=512 * block.expansion, out_channels=256)
        self.conv2_g1 = GCNConv(in_channels=256, out_channels=101)
        # supCL definition Projection
        self.head = nn.SequentialCell(
            nn.Dense(256, 256),
            nn.ReLU(),
            nn.Dense(256, 128)
        )
        # weights-initialize
        for m in self.trainable_params():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Dense):
                init.initializer(m.weight)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                init.initializer(init.One(), m.gamma)
                init.initializer(init.Zero(), m.beta)
        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        if zero_init_residual:
            for m in self.trainable_params():
                if isinstance(m, Bottleneck):
                    init.initializer(init.Zero(), m.bn3.gamma)
                elif isinstance(m, BasicBlock):
                    init.initializer(init.Zero(), m.bn2.gamma)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.SequentialCell(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.SequentialCell(*layers)

    def _forward_impl(self, x, labels, train_mask):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = F.flatten(x, 1)
        y_log_prob = self.p_y_x(x)
        y_prob = F.exp(y_log_prob)
        # 训练数据保留标签，其他数据保留预测。 可以看做是无标签数据的后验概率, 半监督的实现
        y_prob = F.where(train_mask.unsqueeze(1), labels, y_prob)
        # x_encoding = self.activation(self.x_enc(x))  # x[bsz,2] relu 后可能出现编码值为全0的输出 导致NAN
        x_encoding = self.x_enc(x)

        # labels_graph
        num_lab_nodes = F.sum(train_mask)
        if num_lab_nodes != 0:
            num_lab_edges = num_lab_nodes * 10
            edge_lab_index = F.randint(0, num_lab_nodes, (2, num_lab_edges)).to(x.device)  # 初始化入点和出点两个列表

        # Whole graph.
        num_nodes = F.shape(x)[0]
        num_edges = num_nodes * 10
        edge_index = F.randint(0, num_nodes, (2, num_edges)).to(x.device)  # 初始化入点和出点两个列表

        # 将子图并入完全图中
        if num_lab_nodes:
            edge_index = F.stack(
                (F.cat((edge_lab_index[0], edge_index[0])), F.cat((edge_lab_index[1], edge_index[1]))))
        y_query = F.embedding(edge_index[0], y_prob)  # 出点 [num_edges,101] ops.embedding(a,b) a是节点索引 b是嵌入矩阵
        y_key = F.embedding(edge_index[1], y_prob)  # 入点 [num_edges,101]
        x_query = F.embedding(edge_index[0], x_encoding)  # [num_edges,2]
        x_key = F.embedding(edge_index[1], x_encoding)  # [num_edges,2]
        xy_query = F.div(F.unsqueeze(F.concat((x_query, y_query)), 1),
                         F.norm(F.concat((x_query, y_query)), 'euclidean', 2, True))
        xy_key = F.div(F.unsqueeze(F.concat((x_key, y_key)), 2), F.norm(F.concat((x_key, y_key)), 'euclidean', 1, True))
        e_pred = F.squeeze(F.matmul(xy_query, xy_key))
        e_pred = self.activation(e_pred)

        # Graph Construction
        if num_lab_nodes:
            x_lab = x[0:num_lab_nodes]
            edge_lab_id = F.stack((edge_index[0][0:num_lab_edges], edge_index[1][:num_lab_edges]))
            edge_lab_wg = e_pred[0:num_lab_edges]
            x_lab = self.activation(self.conv1_g1(x_lab, edge_lab_id, edge_lab_wg))  # graph convolution x->[bsz,64]

            # supcl
            feat_lab = F.normalize(self.head(x_lab), dim=1)
            x_lab = F.dropout(x_lab, self.dropout)
            x_lab = self.conv2_g1(x_lab, edge_lab_id, edge_lab_wg)  # x[bsz,7]

        x, edge_id, edge_wg = x, edge_index, e_pred
        x = self.activation(self.conv1_g(x, edge_id, edge_wg))  # graph convolution x->[2708,64]

        if num_lab_nodes:
            # supcl
            feat = F.normalize(self.head(x[0:num_lab_nodes]), dim=1)
            feat = F.concat((F.unsqueeze(feat, 1), F.unsqueeze(feat_lab, 1)), 1)  # [bsz, 2, proj_dim]
        else:
            feat = 0

        x = F.dropout(x, self.dropout)
        x = self.conv2_g(x, edge_id, edge_wg)
        if num_lab_nodes:
            # x_whole = x[0:num_lab_nodes][:]
            x_whole = x
            x = F.concat([x_whole, x_lab], 0)
        x = F.softmax(x, 1)
        return x, feat

    def forward(self, x, labels, train_mask):
        return self._forward_impl(x, labels, train_mask)


def _resnet(arch, block, layers, pretrained,progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        checkpoint_path = model_urls[arch]
        param_dict = load_checkpoint(checkpoint_path)
        load_param_into_net(model, param_dict)
    return model

def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet18', BasicBlock, [2, 2, 2, 2], pretrained, progress,
                   **kwargs)


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet34', BasicBlock, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet50', Bottleneck, [3, 4, 6, 3], pretrained, progress,
                   **kwargs)


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet101', Bottleneck, [3, 4, 23, 3], pretrained, progress,
                   **kwargs)


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet('resnet152', Bottleneck, [3, 8, 36, 3], pretrained, progress,
                   **kwargs)


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 4
    return _resnet('resnext50_32x4d', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['groups'] = 32
    kwargs['width_per_group'] = 8
    return _resnet('resnext101_32x8d', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet50_2', Bottleneck, [3, 4, 6, 3],
                   pretrained, progress, **kwargs)


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_

    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs['width_per_group'] = 64 * 2
    return _resnet('wide_resnet101_2', Bottleneck, [3, 4, 23, 3],
                   pretrained, progress, **kwargs)
