import pytest
import numpy as np
import mindspore.ops as ops
import mindspore as ms
import mindspore.nn as nn
from mindspore import context

context.set_context(mode=context.GRAPH_MODE, device_target='CPU')


class TestHorizontalPoolingPyramid:
    from opengait.modeling.modules import HorizontalPoolingPyramid
    test_model = HorizontalPoolingPyramid()

    # def __init__(self):
    #
    #     from opengait.modeling.modules import HorizontalPoolingPyramid
    #     self.test_model = HorizontalPoolingPyramid()

    def test_forward(self):
        n = 2  # Batch size
        c = 3  # Number of channels
        h = 16  # Height
        w = 16  # Width
        input_data = ops.randn([n, c, h, w])
        output_data = self.test_model(input_data)
        assert np.array_equal(output_data.shape, (2, 3, 31))


class TestSetBlockWrapper:
    from opengait.modeling.modules import SetBlockWrapper
    test_model = SetBlockWrapper(nn.MaxPool2d(kernel_size=2, stride=2))

    # def __init__(self):
    #
    #     from opengait.modeling.modules import HorizontalPoolingPyramid
    #     self.test_model = HorizontalPoolingPyramid()

    def test_forward(self):
        n = 2  # Batch size
        c = 3  # Number of channels
        s = 15  # Number of sequence
        h = 16  # Height
        w = 16  # Width
        input_data = ops.randn([n, c, s, h, w])
        output_data = self.test_model(input_data)
        print(output_data.shape)
        # assert np.array_equal(output_data.shape, (2, 3, 31))


class TestPackSequenceWrapper:
    from opengait.modeling.modules import PackSequenceWrapper
    test_model = PackSequenceWrapper(ops.max)

    def test_forward(self):
        n = 2  # Batch size
        c = 3  # Number of channels
        s = 15  # Number of sequence
        p = 100
        input_data = ops.randn([n, c, s, p])
        output_data = self.test_model(input_data, seqL=None, options={"axis": 3})[0]

        print(output_data.shape)


class TestBasicConv2d:

    def test_forward(self):
        # 定义输入数据
        in_channels = 3
        out_channels = 64
        kernel_size = 3
        stride = 1
        padding = 1
        input_data = ms.Tensor(np.random.randn(1, in_channels, 32, 32).astype(np.float32))
        from opengait.modeling.modules import BasicConv2d
        # 初始化 BasicConv2d 模型
        model = BasicConv2d(in_channels, out_channels, kernel_size, stride, padding)

        # 执行前向传播
        output = model(input_data)

        # 检查输出的形状是否正确
        expected_shape = (1, out_channels, 32, 32)
        assert output.shape == expected_shape


class TestSeparateFCs:
    def test_forward(self):
        parts_num = 3
        in_channels = 4
        out_channels = 5
        norm = True  # You can set this to True or False depending on your test
        from opengait.modeling.modules import SeparateFCs
        separate_fcs = SeparateFCs(parts_num, in_channels, out_channels, norm)
        n = 2
        c_in = 4
        p = 3
        input_data = ms.Tensor(np.random.rand(n, c_in, p).astype(np.float32))
        # 执行前向传播
        output = separate_fcs(input_data)
        # 检查输出的形状是否正确
        expected_shape = (2, out_channels, p)
        assert output.shape == expected_shape


class TestSeparateBNNecks:
    def test_forward(self):
        parts_num = 31
        in_channels = 256
        out_channels = 128
        norm = True  # You can set this to True or False depending on your test
        from opengait.modeling.modules import SeparateBNNecks
        separate_fcs = SeparateBNNecks(parts_num, in_channels, out_channels, norm, True)
        B = 2
        c_in = 256
        p = 31
        input_data = ms.Tensor(np.random.rand(B, c_in, p).astype(np.float32))
        # 执行前向传播
        output = separate_fcs(input_data)
        # 检查输出的形状是否正确
        expected_shape = (2, out_channels, p)
        # assert output.shape == expected_shape


class TestFocalConv2d:
    def test_forward(self):
        from opengait.modeling.modules import FocalConv2d
        in_channels = 3
        out_channels = 16
        kernel_size = 3
        halving = 1

        focal_conv = FocalConv2d(in_channels, out_channels, kernel_size, halving)
        batch_size = 2
        height = 32
        width = 32
        input_data = ms.Tensor(np.random.rand(batch_size, in_channels, height, width).astype(np.float32))
        # 执行前向传播
        output = focal_conv(input_data)
        # 检查输出的形状是否正确
        expected_shape = (batch_size, out_channels, 1)
        # assert output.shape == expected_shape


class TestBasicConv3d:
    def test_forward(self):
        from opengait.modeling.modules import BasicConv3d
        in_channels = 3
        out_channels = 16

        basic_conv3d = BasicConv3d(in_channels, out_channels)
        batch_size = 2
        s = 15
        height = 32
        width = 32
        input_data = ms.Tensor(np.random.rand(batch_size, in_channels, s, height, width).astype(np.float32))
        # 执行前向传播
        output = basic_conv3d(input_data)
        # 检查输出的形状是否正确
        expected_shape = (batch_size, out_channels, s, height, width)
        assert output.shape == expected_shape


# class TestGaitAlign:
#     def test_forward(self):
#         from opengait.modeling.modules import GaitAlign
#         gait_align = GaitAlign()
#         batch_size = 4
#         channels = 3
#         height = 64
#         width = 44
#         feature_map = ms.Tensor(np.random.rand(batch_size, channels, height, width).astype(np.float32))
#         binary_mask = ms.Tensor(np.ones(batch_size, channels, ))