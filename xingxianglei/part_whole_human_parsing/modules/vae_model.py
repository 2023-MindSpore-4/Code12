from mindspore import value_and_grad
import mindspore.nn as nn
from mindspore import ops, save_checkpoint, load_checkpoint, load_param_into_net
from mindspore import ops
from tqdm import tqdm
import os
from networks.vgg_net import vgg19
from utils.util import reparameterize, compute_kl_divergence
from infer_files.test import visualize_map


class VAEModel:
    def __init__(self, encoder, decoder, train_loader, valid_loader=None, optimizer=None, logger=None
                 , max_epoch=None, data_len=0):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.max_epoch = max_epoch
        self.data_len = data_len
        self.logger = logger
        self.mse = nn.MSELoss()
        self.vgg = vgg19()
        self.global_step = 0
        self.epoch = 0
        self.optimizer = optimizer
        self.grad_fn = value_and_grad(self.grad_forward, None,
                                      list(self.encoder.trainable_params()) + list(self.decoder.trainable_params()),
                                      has_aux=True)

    def forward(self, z_app_sample, z_deform_sample, z_pose_sample):
        outputs, transformed_mask, transformed_deformed_mask, res_deform_field, pose = self.decoder(
                z_app_sample, z_deform_sample, z_pose_sample)
        return outputs

    def grad_forward(self, imgs, mask):
        mask = mask[:, :, None]
        z_app_mu, z_app_logvar, z_deform_mu, z_deform_logvar, z_pose_mu, z_pose_logvar, attn = self.encoder(
            imgs)
        z_app_sample = reparameterize(z_app_mu, z_app_logvar)
        z_pose_sample = reparameterize(z_pose_mu, z_pose_logvar)
        z_deform_sample = reparameterize(z_deform_mu, z_deform_logvar)
        outputs, transformed_mask, transformed_deformed_mask, res_deform_field, pose = self.decoder(
            z_app_sample, z_deform_sample, z_pose_sample)
        grid_reg = self.mse(res_deform_field, ops.zeros_like(res_deform_field))
        # pose_reg = ops.std(pose, axis=0)
        # pose_reg = ops.mean(ops.std(pose, axis=0))
        # deformed_mask_mse = self.mse(transformed_deformed_mask, mask)
        # transformed_mask_mse = self.mse(transformed_mask, mask)
        scaled_imgs = ops.interpolate(imgs, size=(224, 224), mode='bilinear')
        scaled_outputs = ops.interpolate(outputs, size=(224, 224), mode='bilinear')
        real_feature = self.vgg((scaled_imgs + 1.) / 2.)
        fake_feature = self.vgg((scaled_outputs + 1.) / 2.)
        lpips = self.mse(real_feature, fake_feature)
        human_mse = self.mse(outputs, imgs)
        # z_app_kl = compute_kl_divergence(z_app_mu, z_app_logvar)
        # z_deform_kl = compute_kl_divergence(z_deform_mu, z_deform_logvar)
        # z_pose_kl = compute_kl_divergence(z_pose_mu, z_pose_logvar)
        weighted_loss = {
            'mse': human_mse * 1,
            # 'lpips': lpips * 5.,
            # 'deform_field_reg': grid_reg * 2,
            # # 'pose_reg': pose_reg * 0.01,
            # 'deformed_mask_mse': deformed_mask_mse * 100,
            # 'transformed_mask_mse': transformed_mask_mse * 200,
            # 'app_kl': z_app_kl * 0.01,
            # 'pose_kl': z_pose_kl * 0.01,
            # 'deform_kl': z_deform_kl * 0.001,
        }
        visual_outputs = {
            'rec_imgs': outputs,
            'transformed_deformed_mask': transformed_deformed_mask,
            'transformed_mask': transformed_mask
        }
        loss = 0
        for key, value in weighted_loss.items():
            loss = loss + value
        return loss, visual_outputs, weighted_loss

    def training_step(self, batch, batch_idx):
        imgs, mask = batch
        (_, visual_outputs, weighted_loss), input_gradients = self.grad_fn(imgs, mask)
        optimizer = self.optimizer
        optimizer(input_gradients)
        if (batch_idx + 1) % 100 == 0:
            mask = ops.sum(mask, dim=1, keepdim=True)
            for key, value in weighted_loss.items():
                self.logger.add_value('scalar', key, value)

            rec = visual_outputs['rec_imgs']
            visualize_map(ops.cat([imgs, rec], axis=-1))
            self.logger.add_value('image', 'rec_imgs', ops.cat([(imgs + 1.) / 2., (rec + 1.) / 2.], axis=-1))
            transformed_mask = visual_outputs['transformed_mask']
            transformed_mask = ops.sum(transformed_mask, dim=1)
            transformed_mask[transformed_mask > 1.] = 1.
            transformed_deformed_mask = visual_outputs['transformed_deformed_mask']
            transformed_deformed_mask = ops.sum(transformed_deformed_mask, dim=1)
            transformed_deformed_mask[transformed_deformed_mask > 1.] = 1.
            self.logger.add_value('image', 'rec_mask',
                                  ops.cat([mask, transformed_mask, transformed_deformed_mask], axis=-1))

            self.logger.record(self.global_step)

    def validation_step(self, batch, batch_idx):
        pass

    def train(self, ckpt_path=None, save_freq=100):
        for i in range(self.max_epoch):
            for batch_idx, batch in enumerate(tqdm(self.train_loader, desc='Epoch {}'.format(i), total=self.data_len)):
                self.training_step(batch, batch_idx)
                self.global_step += 1

            if self.valid_loader is not None:
                for test_batch_idx, test_batch in enumerate(
                        tqdm(self.train_loader, desc='Validation Dataloader {}:'.format(i))):
                    self.validation_step(test_batch, test_batch_idx)

            if ckpt_path is not None:
                if (self.epoch+1) % save_freq == 0:
                    # self.save(ckpt_path+"_epoch{}.ckpt".format(self.epoch))
                    self.save(os.path.join(ckpt_path, "epoch_{}.ckpt".format(self.epoch)))

            self.epoch += 1
        self.logger.close()

    def save(self, ckpt_path):
        save_checkpoint(save_obj=list(self.encoder.trainable_params())+list(self.decoder.trainable_params()),
                        ckpt_file_name=ckpt_path)

    def load(self, encoder_ckpt, decoder_ckpt):
        encoder_params = load_checkpoint(encoder_ckpt)
        decoder_params = load_checkpoint(decoder_ckpt)
        encoder_params_not_load, _ = load_param_into_net(self.encoder, encoder_params)
        decoder_params_not_load, _ = load_param_into_net(self.decoder, decoder_params)
