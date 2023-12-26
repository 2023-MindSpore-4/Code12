# Mindhub's VGG is currently unusable


from typing import List, Union, Dict
import logging
from mindspore import nn, Tensor
import mindspore.common.initializer as init
from mindspore import load_checkpoint, load_param_into_net
import math

def _search_param_name(params_names: List,
                       param_name: str
                       ) -> str:
    for pi in params_names:
        if param_name in pi:
            return pi
    return ""


def load_pretrained(model, default_cfg, path='./', num_classes=1000, in_channels=3, filter_fn=None):
    '''load pretrained model depending on cfgs of model'''

    param_dict = load_checkpoint("./model/vgg19_224.ckpt")

    if in_channels == 1:
        conv1_name = default_cfg['first_conv']
        logging.info(
            'Converting first conv (%s) from 3 to 1 channel', conv1_name)
        con1_weight = param_dict[conv1_name + '.weight']
        con1_weight.set_data(con1_weight.sum(
            axis=1, keepdims=True), slice_shape=True)
    elif in_channels != 3:
        raise ValueError('Invalid in_channels for pretrained weights')

    classifier_name = default_cfg['classifier']
    if num_classes == 1000 and default_cfg['num_classes'] == 1001:
        classifier_weight = param_dict['classifier_name' + '.weight']
        classifier_weight.set_data(classifier_weight[:1000], slice_shape=True)
        classifier_bias = param_dict[classifier_name + '.bias']
        classifier_bias.set_data(classifier_bias[:1000], slice_shape=True)
    elif num_classes != default_cfg['num_classes']:
        params_names = list(param_dict.keys())
        param_dict.pop(_search_param_name(params_names, classifier_name+'.weight'),
                       "No Parameter {} in ParamDict".format(classifier_name+'.weight'))
        param_dict.pop(_search_param_name(params_names, classifier_name+'.bias'),
                       "No Parameter {} in ParamDict".format(classifier_name+'.bias'))

    if filter_fn is not None:
        param_dict = filter_fn(param_dict)

    load_param_into_net(model, param_dict)


cfgs: Dict[str, List[Union[str, int]]] = {
    "vgg11": [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg13": [64, 64, "M", 128, 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"],
    "vgg16": [64, 64, "M", 128, 128, "M", 256, 256, 256, "M", 512, 512, 512, "M", 512, 512, 512, "M"],
    "vgg19": [64, 64, "M", 128, 128, "M", 256, 256, 256, 256, "M", 512, 512, 512, 512, "M", 512, 512, 512, 512, "M"],
}


def _make_layers(cfg: List[Union[str, int]],
                 batch_norm: bool = False,
                 in_channels: int = 3) -> nn.SequentialCell:
    """define the basic block of VGG"""
    layers = []
    for v in cfg:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3,
                               pad_mode="pad", padding=1)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = v

    return nn.SequentialCell(layers)


class VGG(nn.Cell):
    r"""VGGNet model class, based on
    `"Very Deep Convolutional Networks for Large-Scale Image Recognition" <https://arxiv.org/abs/1409.1556>`_

    Args:
        model_name: name of the architecture. 'vgg11', 'vgg13', 'vgg16' or 'vgg19'.
        batch_norm: use batch normalization or not. Default: False.
        num_classes: number of classification classes. Default: 1000.
        in_channels: number the channels of the input. Default: 3.
        drop_rate: dropout rate of the classifier. Default: 0.5.
    """

    def __init__(self,
                 model_name: str,
                 batch_norm: bool = False,
                 num_classes: int = 1000,
                 in_channels: int = 3,
                 drop_rate: float = 0.5) -> None:
        super().__init__()
        cfg = cfgs[model_name]
        self.features = _make_layers(
            cfg, batch_norm=batch_norm, in_channels=in_channels)
        self.flatten = nn.Flatten()
        self.classifier = nn.SequentialCell([
            nn.Dense(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=1 - drop_rate),
            nn.Dense(4096, 4096),
            nn.ReLU(),
            nn.Dropout(keep_prob=1 - drop_rate),
            nn.Dense(4096, num_classes),
        ])

        self.resize_image=nn.ResizeBilinear()

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """Initialize weights for cells."""
        for _, cell in self.cells_and_names():
            if isinstance(cell, nn.Conv2d):
                cell.weight.set_data(
                    init.initializer(init.HeNormal(math.sqrt(5), mode='fan_out', nonlinearity='relu'),
                                     cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(
                        init.initializer('zeros', cell.bias.shape, cell.bias.dtype))
            elif isinstance(cell, nn.Dense):
                cell.weight.set_data(
                    init.initializer(init.Normal(0.01), cell.weight.shape, cell.weight.dtype))
                if cell.bias is not None:
                    cell.bias.set_data(init.initializer(
                        'zeros', cell.bias.shape, cell.bias.dtype))

    def forward_features(self, x: Tensor) -> Tensor:
        x = self.features(x)
        return x

    def forward_head(self, x: Tensor) -> Tensor:
        x = self.flatten(x)
        x = self.classifier(x)
        return x

    def construct(self, x: Tensor) -> Tensor:
        x=self.resize_image(x,[224,224])
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,
        'first_conv': 'features.0', 'classifier': 'classifier.6',
        **kwargs
    }


def vgg19(pretrained: bool = False, num_classes: int = 1000, in_channels=3, **kwargs) -> VGG:
    """Get 19 layers VGG model.
     Refer to the base class `models.VGG` for more details.
     """
    default_cfg = _cfg(
        url='https://download.mindspore.cn/toolkits/mindcv/vgg/vgg19_224.ckpt')
    model = VGG(model_name='vgg19', num_classes=num_classes,
                in_channels=in_channels, **kwargs)

    if pretrained:
        load_pretrained(model, default_cfg,
                        num_classes=num_classes, in_channels=in_channels)

    return model
