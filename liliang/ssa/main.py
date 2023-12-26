import mindspore
from mindspore import context
from model import train

context.set_context(mode=context.PYNATIVE_MODE,
                    device_target="Ascend")

train()
