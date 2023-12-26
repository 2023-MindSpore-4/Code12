from mindspore import ops, Tensor, value_and_grad
import mindspore.nn as nn
from mindspore import ops
import numpy as np
from tqdm import tqdm
from base_gan import GANModel


class WGAN_GP(GANModel):
    def __init__(self, generator, discriminator, train_loader, valid_loader=None, logger=None, max_epoch=10):
        super().__init__(generator=generator, discriminator=discriminator, train_loader=train_loader,
                         valid_loader=valid_loader, logger=logger, max_epoch=max_epoch)

    def generator_forward(self, noise):
        fake_data = self.generator(noise)
        fake_out = self.discriminator(fake_data)


    def discriminator_forward(self, real_data, noise):
        pass

    def generator_grad(self):
        return value_and_grad(self.generator_forward,
                              None, self.generator.trainable_params())

    def discriminator_grad(self):
        return value_and_grad(self.discriminator_forward,
                              None, self.discriminator.trainable_params())

    def compute_gradient_penalty(self, real_samples, fake_samples):
        # Random weight term for interpolation between real and fake samples
        alpha = Tensor(np.random.random((real_samples.size(0), 1, 1, 1)))
        # Get random interpolation between real and fake samples
        interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
        d_grad = value_and_grad(self.discriminator,
                                None, self.discriminator.trainable_params())
        d_interpolates, gradients = d_grad(interpolates)
        gradients = gradients.reshape(gradients.size(0), -1)
        gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
        return gradient_penalty

    def forward(self, mask):
        batch_size, num_parts = mask.shape[0:2]
        z = ops.randn([batch_size, num_parts, 64])
        outputs, _ = self.generator(mask, z)
        return outputs

    def training_step(self, batch, batch_idx):
        pass

    def validation_step(self, batch, batch_idx):
        pass

    def train(self):
        for i in range(self.max_epoch):
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Epoch {}:'.format(i))):
                self.training_step(batch, batch_idx)

            if self.valid_loader is not None:
                for test_batch_idx, test_batch in enumerate(
                        tqdm(self.train_loader, desc='Validation Dataloader {}:'.format(i))):
                    self.validation_step(test_batch, test_batch_idx)

    def configure_schedulers(self):
        pass

    def configure_optimizers(self):
        generator_optimizer = nn.Adam(self.generator.parameters(), lr=1e-3)
        discriminator_optimizer = nn.Adam(self.discriminator.parameters(), lr=1e-3)
        # generator_scheduler = optim.lr_scheduler.MultiStepLR(
        #     generator_optimizer, [100, 300, 600], 0.5, last_epoch=-1)
        # discriminator_scheduler = optim.lr_scheduler.MultiStepLR(
        #     discriminator_optimizer, [100, 300, 600], 0.5, last_epoch=-1)
        self.optimizers = (
            generator_optimizer,
            discriminator_optimizer
        )
