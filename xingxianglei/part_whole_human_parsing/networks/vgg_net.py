# import mindspore_hub as mshub
import mindcv


# def vgg19():
#     model = "mindspore/1.9/vgg19_imagenet2012"
#     network = mshub.load(model)
#     network.set_train(False)
#     return network


def vgg19():
    network = mindcv.create_model("vgg19", pretrained=True)
    for param in network.trainable_params():
        param.requires_grad = False
    # network.set_train(False)
    return network
