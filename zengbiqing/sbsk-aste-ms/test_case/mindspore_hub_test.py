import mindspore_hub as mshub
import mindspore as ms
from mindspore import context

def test_install_bert():
    context.set_context(
        mode = context.GRAPH_MODE,
        device_target = "GPU",
        device_id = 0
        )
    
    model = 'mindspore/1.9/bertfinetune_squad_squad'
    network = mshub.load(model)
    network.set_train(False)
    print(network(ms.Tensor([1,2,3], dtype=ms.int64)))