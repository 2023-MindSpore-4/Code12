from mindspore import ops, Tensor, value_and_grad
import mindspore.nn as nn
from mindspore import ops
import numpy as np
from tqdm import tqdm


class GANModel:
    def __init__(self, encoder, decoder, train_loader, valid_loader=None, logger=None, max_epoch=10):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.max_epoch = max_epoch
        self.logger = logger
        self.mse = nn.MSE()
        self.global_step = 0
        self.epoch = 0
        self.optimizers = None
        self.schedulers = None
        self.configure_optimizers()
        self.configure_schedulers()

    def _forward(self, noise):
        pass

    def forward(self, z_app, z_deform, pose):
        outputs, _, _, _ = self.decoder(z_app, z_deform, pose)
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
