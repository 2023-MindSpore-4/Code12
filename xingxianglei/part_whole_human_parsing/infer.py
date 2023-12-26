from mindspore import ops, load_checkpoint, load_param_into_net
import matplotlib.pyplot as plt
from infer_files1.human_part_encoder import HumanPartEncoder
from infer_files1.human_part_decoder import HumanPartDecoder

g_edges = [[0, 13], [11, 13], [9, 11], [8, 9], [5, 13], [3, 5], [2, 3], [12, 13], [10, 12], [7, 10],
                   [6, 13], [4, 6], [1, 4], [13, 0], [13, 11], [11, 9], [9, 8], [13, 5], [5, 3], [3, 2], [13, 12],
                   [12, 10], [10, 7], [13, 6], [6, 4], [4, 1]]
g_relation = [
    [[13, 0]],
    [[13, 11], [11, 9], [9, 8]],
    [[13, 5], [5, 3], [3, 2]],
    [[13, 12], [12, 10], [10, 7]],
    [[13, 6], [6, 4], [4, 1]]
]
encoder = HumanPartEncoder(edges=g_edges)
decoder = HumanPartDecoder(relation_list=g_relation)
encoder_params = load_checkpoint("infer_files1/ms_encoder_weights.ckpt")
decoder_params = load_checkpoint("infer_files1/ms_decoder_weights.ckpt")
encoder_params_not_load, _ = load_param_into_net(encoder, encoder_params)
decoder_params_not_load, _ = load_param_into_net(decoder, decoder_params)

z_app = ops.randn([1, 14, 32])
z_deform = ops.randn([1, 14, 16])
z_pose = ops.randn([1, 14, 16])
o1, o2, o3, o4, o5 = decoder(z_app, z_deform, z_pose)

# sample appearance latent code
sample_app_list = []
for _ in range(8):
    z_app_sample = ops.randn_like(z_app)
    gen_img, _, _, _, _ = decoder(z_app_sample, z_deform, z_pose)
    sample_app_list.append(gen_img)
app_imgs = ops.cat(sample_app_list, axis=-1)
app_imgs = (app_imgs[0].value() + 1.) / 2.
app_imgs = ops.permute(app_imgs, axis=(1, 2, 0)).numpy()
plt.imsave('sample_app.png', app_imgs)
print("sample appearance latent code result saved")

# sample head appearance
sample_app_list = []
for _ in range(8):
    z_head_app_sample = ops.randn_like(z_app[:, 0:1, :])
    z_app_sample = ops.cat([z_head_app_sample, z_app[:, 1:, :]], axis=1)
    gen_img, _, _, _, _ = decoder(z_app_sample, z_deform, z_pose)
    sample_app_list.append(gen_img)
app_imgs = ops.cat(sample_app_list, axis=-1)
app_imgs = (app_imgs[0].value() + 1.) / 2.
app_imgs = ops.permute(app_imgs, axis=(1, 2, 0)).numpy()
plt.imsave('sample_head_app.png', app_imgs)
print("sample head appearance latent code result saved")

# sample pose latent code
sample_pose_imgs_list = []
sample_pose_masks_list = []
for _ in range(8):
    z_pose_sample = ops.randn_like(z_pose)
    gen_img, _, gen_mask, _, _ = decoder(z_app, z_deform, z_pose_sample)
    sample_pose_imgs_list.append(gen_img)
    sample_pose_masks_list.append(gen_mask)
pose_imgs = ops.cat(sample_pose_imgs_list, axis=-1)
pose_masks = ops.sum(ops.cat(sample_pose_masks_list, axis=-1), dim=1)
pose_masks[pose_masks > 1.] = 1.
pose_masks = pose_masks.repeat(3, 1)
pose_imgs = ops.cat([(pose_imgs+1.) / 2., pose_masks], axis=-2)
pose_imgs = pose_imgs[0].value()
pose_imgs = ops.permute(pose_imgs, axis=(1, 2, 0)).numpy()
plt.imsave('sample_pose.png', pose_imgs)
print("sample pose latent code result saved")

# sample left_upper arm pose latent code
sample_pose_imgs_list = []
sample_pose_masks_list = []
for _ in range(8):
    # z_left_upper_arm_pose_sample = ops.randn_like(z_pose[:, 5:6, :])
    # z_pose_sample = ops.cat([z_pose[:, :5, :], z_left_upper_arm_pose_sample, z_pose[:, 6:, :]], axis=1)
    z_left_upper_arm_pose_sample = ops.randn_like(z_pose[:, 3:4, :])
    z_pose_sample = ops.cat([z_pose[:, :3, :], z_left_upper_arm_pose_sample, z_pose[:, 4:, :]], axis=1)
    gen_img, _, gen_mask, _, _ = decoder(z_app, z_deform, z_pose_sample)
    sample_pose_imgs_list.append(gen_img)
    sample_pose_masks_list.append(gen_mask)
pose_imgs = ops.cat(sample_pose_imgs_list, axis=-1)
pose_masks = ops.sum(ops.cat(sample_pose_masks_list, axis=-1), dim=1)
pose_masks[pose_masks > 1.] = 1.
pose_masks = pose_masks.repeat(3, 1)
pose_imgs = ops.cat([(pose_imgs+1.) / 2., pose_masks], axis=-2)
pose_imgs = pose_imgs[0].value()
pose_imgs = ops.permute(pose_imgs, axis=(1, 2, 0)).numpy()
plt.imsave('sample_left_lower_arm_pose.png', pose_imgs)
print("sample left lower arm pose latent code result saved")


sample_pose_imgs_list = []
sample_pose_masks_list = []
for _ in range(8):
    z_left_upper_arm_pose_sample = ops.randn_like(z_pose[:, 5:6, :])
    z_pose_sample = ops.cat([z_pose[:, :5, :], z_left_upper_arm_pose_sample, z_pose[:, 6:, :]], axis=1)
    # z_left_upper_arm_pose_sample = ops.randn_like(z_pose[:, 3:4, :])
    # z_pose_sample = ops.cat([z_pose[:, :3, :], z_left_upper_arm_pose_sample, z_pose[:, 4:, :]], axis=1)
    gen_img, _, gen_mask, _, _ = decoder(z_app, z_deform, z_pose_sample)
    sample_pose_imgs_list.append(gen_img)
    sample_pose_masks_list.append(gen_mask)
pose_imgs = ops.cat(sample_pose_imgs_list, axis=-1)
pose_masks = ops.sum(ops.cat(sample_pose_masks_list, axis=-1), dim=1)
pose_masks[pose_masks > 1.] = 1.
pose_masks = pose_masks.repeat(3, 1)
pose_imgs = ops.cat([(pose_imgs+1.) / 2., pose_masks], axis=-2)
pose_imgs = pose_imgs[0].value()
pose_imgs = ops.permute(pose_imgs, axis=(1, 2, 0)).numpy()
plt.imsave('sample_left_upper_arm_pose.png', pose_imgs)
print("sample left upper arm pose latent code result saved")
