import mindspore
from mindspore import Tensor, ops, nn, dataset
from util.network import Generator, Discriminator
from conf import conf
import os
from util.loss import *
from util.vgg import vgg19
from util.guided_image_filter import guided_filter
from util.utility import gray_ish
from typing import Tuple
from util.dataset import MyDatasetLoader
from matplotlib import pyplot as plt


def train():

    generator = Generator()
    surface_discriminator = Discriminator(3)
    texture_discriminator = Discriminator(1)

    VGG = vgg19(True)

    if conf["continue_training"]:
        if os.path.exists("./model/generator.ckpt"):
            mindspore.load_checkpoint("./model/generator.ckpt", generator)
        if os.path.exists("./model/surface_discriminator.ckpt"):
            mindspore.load_checkpoint("./model/surface_discriminator.ckpt", surface_discriminator)
        if os.path.exists("./model/surface_discriminator.ckpt"):
            mindspore.load_checkpoint("./model/texture_discriminator.ckpt", texture_discriminator)

    learning_rate = conf["lr"]
    optimizer_generator = nn.Adam(
        generator.trainable_params(), learning_rate=learning_rate)
    
    optimizer_surface_discriminator = nn.Adam(
        surface_discriminator.trainable_params(), learning_rate=learning_rate)
    
    optimizer_texture_discriminator = nn.Adam(
        texture_discriminator.trainable_params(), learning_rate=learning_rate)
    
    optimizer_generator.update_parameters_name('optim_g.')
    optimizer_surface_discriminator.update_parameters_name('optim_s_d.')
    optimizer_texture_discriminator.update_parameters_name('optim_t_d.')

    loss_tv = Loss_tv()
    loss_lsgan_d = Lsgan_loss_d()
    loss_lsgan_g = Lsgan_loss_g()
    loss_gan_d = Gan_loss_d()
    loss_gan_g = Gan_loss_g()
    adversarial_loss = nn.BCELoss(reduction='mean')
    valid_lab = ops.ones((conf["batch"], 1), mindspore.float32)
    fake_lab = ops.zeros((conf["batch"], 1), mindspore.float32)
    def generator_forward(ori: Tensor, last_ori: Tensor = None, last_pred: Tensor = None) -> Tuple[Tensor]:
        
        pred = generator(ori)

        ori_vgg = VGG(ori)
        pred_vgg = VGG(pred)
        pred_gray = gray_ish(pred)
        pred_guided = guided_filter(pred, pred, 8, 0.05)
        
        s_d = surface_discriminator(pred_guided)
        t_d = texture_discriminator(pred_gray)

        # # surface_loss = conf["W_surface"] * \
        # #     lsgan_loss_g(s_d)
        surface_loss = conf["W_surface"] * loss_lsgan_g(s_d)
        # surface_loss = adversarial_loss(s_d, valid_lab)

        # # texture_loss = conf["W_texture"] * \
        # #     lsgan_loss_g(t_d)
        texture_loss = conf["W_texture"] * loss_lsgan_g(t_d)
        # texture_loss = adversarial_loss(t_d, valid_lab)

        structure_loss = conf["W_structure"]*L_structure(pred, pred_vgg, VGG)
        content_loss = conf["W_content"] * \
            L_content(pred, ori, pred_vgg, ori_vgg, VGG)
        # tv_loss = conf["W_tv"]*L_tv(pred)
        tv_loss = conf["W_tv"]*loss_tv(pred)

        motion_loss = 0
        if last_ori is not None and last_pred is not None:
            motion_loss = conf["W_motion"] * \
                L_motion(last_pred, pred, last_ori, ori, pred_vgg, ori_vgg, VGG)

        loss = surface_loss+texture_loss+structure_loss+content_loss+tv_loss+motion_loss
        

        return loss, pred

    def surface_discriminator_forward(real: Tensor, fake: Tensor) -> Tuple[Tensor]:
        real_guided = guided_filter(real, real, 8, 0.05)
        fake_guided = guided_filter(fake, fake, 8, 0.05)
        
        # fake_guided = ops.stop_gradient(fake_guided)
        # real_guided = ops.stop_gradient(real_guided)
        
        real = surface_discriminator(real_guided)
        fake = surface_discriminator(fake_guided)

        # real_loss = adversarial_loss(real, valid_lab)
        # fake_loss = adversarial_loss(fake, fake_lab)
        # loss = (real_loss + fake_loss) / 2
            
        # loss = lsgan_loss_d(real, fake)
        loss = loss_lsgan_d(real,fake)
        
        return loss

    def texture_discriminator_forward(real: Tensor, fake: Tensor) -> Tuple[Tensor]:
        real_gray = gray_ish(real)
        fake_gray = gray_ish(fake)
        
        # fake_gray = ops.stop_gradient(fake_gray)
        # real_gray = ops.stop_gradient(real_gray)
        
        real = texture_discriminator(real_gray)
        fake = texture_discriminator(fake_gray)

        # real_loss = adversarial_loss(real, valid_lab)
        # fake_loss = adversarial_loss(fake, fake_lab)
        # loss = (real_loss + fake_loss) / 2

        # loss = lsgan_loss_d(real, fake)
        loss = loss_lsgan_d(real,fake)

        return loss

    grad_generator_fn = ops.value_and_grad(
        generator_forward, None, optimizer_generator.parameters, has_aux=True)
    grad_surface_discriminator_fn = ops.value_and_grad(
        surface_discriminator_forward, None, optimizer_surface_discriminator.parameters, has_aux=False
    )
    grad_texture_discriminator_fn = ops.value_and_grad(
        texture_discriminator_forward, None, optimizer_texture_discriminator.parameters, has_aux=False
    )

    real_train_loader = MyDatasetLoader(
        conf["real_train_dataset"])
    real_train = dataset.GeneratorDataset(
        real_train_loader, ["real_imgs"], shuffle=True)

    real_test_loader = MyDatasetLoader(
        conf["real_test_dataset"])
    real_test = dataset.GeneratorDataset(
        real_test_loader, ["real_imgs"], shuffle=True)

    cartoon_train_loader = MyDatasetLoader(
        conf["cartoon_train_dataset"])
    cartoon_train = dataset.GeneratorDataset(
        cartoon_train_loader, ["cartoon_imgs"], shuffle=True)

    cartoon_test_loader = MyDatasetLoader(
        conf["cartoon_test_dataset"])
    cartoon_test = dataset.GeneratorDataset(
        cartoon_test_loader, ["cartoon_imgs"], shuffle=True)

    real_train=real_train.batch(conf["batch"])
    real_test=real_test.batch(conf["batch"])
    cartoon_train=cartoon_train.batch(conf["batch"])
    cartoon_test=cartoon_test.batch(conf["batch"])

    # @mindspore.ms_function
    def train_step(real: Tensor, cartoon: Tensor) -> Tuple[Tensor]:
        
        generator.set_train()
        surface_discriminator.set_train()
        texture_discriminator.set_train()
        
        gen_img_i = generator(real)
        
        surface_d_loss, surface_d_grads = grad_surface_discriminator_fn(
            cartoon, gen_img_i)
        texture_d_loss, texture_d_grads = grad_texture_discriminator_fn(
            cartoon, gen_img_i)
        optimizer_surface_discriminator(surface_d_grads)
        optimizer_texture_discriminator(texture_d_grads)
        
        (g_loss, gen_img), g_grads = grad_generator_fn(real)
        optimizer_generator(g_grads)


        return g_loss, surface_d_loss+texture_d_loss, gen_img

    G_losses = []
    D_losses = []
    
    print("Starting Training Loop...")
    print(len(real_train_loader),len(real_test_loader))

    total = max(len(real_train_loader), len(real_test_loader))

    for epoch in range(conf["epoch"]):
        generator.set_train()
        surface_discriminator.set_train()
        texture_discriminator.set_train()

        for i, (real, cartoon) in enumerate(zip(real_train.create_tuple_iterator(), cartoon_train.create_tuple_iterator())):

            real=real[0]
            cartoon=cartoon[0]

            g_loss, d_loss, gen_imgs = train_step(real, cartoon)

            if i % 5 == 0 or i == total-1:
                print("[%d/%d][%d/%d] Loss_D:%7.4f Loss_G:%7.4f" % (
                    epoch+1, conf["epoch"], (i+1)*conf["batch"], total, d_loss.asnumpy(), g_loss.asnumpy()))

            D_losses.append(d_loss.asnumpy())
            G_losses.append(g_loss.asnumpy())
        
        print("Saving...")

        mindspore.save_checkpoint(generator, "./model/generator.ckpt")
        mindspore.save_checkpoint(surface_discriminator,
                                  "./model/surface_discriminator.ckpt")
        mindspore.save_checkpoint(texture_discriminator,
                                  "./model/texture_discriminator.ckpt")
        print("Saved...")
        
    # plt.figure(figsize=(10, 5))
    # plt.title("Losses")
    # plt.plot(G_losses, label="G", color="blue")
    # plt.plot(D_losses, label="D", color="orange")
    # plt.xlabel("iterations")
    # plt.ylabel("loss")
    # plt.legend()
    # plt.show()
