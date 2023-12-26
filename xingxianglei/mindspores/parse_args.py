import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--GPUS', type=bool, default=True, )
    parser.add_argument('--model', type=str, default='diffusion', choices=['diffusion', 'other'])
    parser.add_argument('--save_mode', type=str, default='jpg', choices=['jpg', 'gif'], help='采样结果保存模式')
    parser.add_argument('--name', type=str, default='NEMO', help='项目名称及数据集，具体数据集及路径在datasets中修改')
    parser.add_argument('--load', type=bool, default=False, help='是否载入预训练模型')
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--train', type=bool, default=False, )
    parser.add_argument('--resume', type=bool, default=False, )
    parser.add_argument(
        '--resume_from', type=str,
        default='output/NEMO/2023-04-25-16-05-10/ckpt/model-005000.pth',
    )
    parser.add_argument('--output', type=str, default='output', help='输出路径')
    parser.add_argument('--amp', type=bool, default=True, )
    # test
    parser.add_argument('--test', type=bool, default=True, )
    parser.add_argument('--show_top_layer', default=0, type=int, help='基函数查看层')
    parser.add_argument('--test_batch_size', default=1, type=int)
    parser.add_argument('--mask', default=(None, 1, 3, 3), type=set, help='掩码(c,t,y,x)，None为没有')
    parser.add_argument('--k', default=(1, 1, 1, 1), type=set, help='掩码长度')
    parser.add_argument('--ns', default=1, type=int, help='查看通道数')
    parser.add_argument('--start', default=0, type=int, help='起始通道')
    parser.add_argument('--w', default=1, type=int, help='单次激活通道数')
    # Unet3D
    parser.add_argument('--dim', default=64, type=int)
    parser.add_argument('--attn_dim_head', default=32, type=int)
    parser.add_argument('--dim_mults', default=(1, 2, 4, 8), type=set)
    parser.add_argument('--use_top', default=(True, True, True, True), type=set, help='是否使用top')
    parser.add_argument('--top_rate', default=(1 / 4, 1 / 4, 1 / 4, 1 / 3), type=set)
    parser.add_argument('--with_t', default=(True, True, True, True), type=set, help='op是否带有时间轴top')
    parser.add_argument('--top_mode', default=('op', 'op', 'op', 'op'), type=set, help='ch为单独通道稀疏op为时空共同稀疏')
    parser.add_argument('--top_dim', default=(1, 2, -1, -1), type=set, help='ch通道选择')
    parser.add_argument('--out_model', type=str, default='Tanh', choices=['Tanh', 'NOISE'], help='输出激活函数NOISE为无激活')

    # Diffusion
    parser.add_argument('--image_size', default=64, type=int)
    parser.add_argument('--num_frames', default=64, type=int, help='帧数')
    parser.add_argument('--timesteps', default=256, type=int)
    parser.add_argument('--loss_type', type=str, default='l2', choices=['l1', 'l2'])
    parser.add_argument('--out_mean', type=str, default='START_X', choices=['START_X', 'NOISE'])
    # Trainer
    parser.add_argument('--frameskip', default=3, type=int, help='几帧图片抽取一张')
    parser.add_argument('--flip', default=True, type=bool, help='数据集左右随机翻转')
    parser.add_argument('--flip2', default=True, type=bool, help='数据集时间轴随机翻转')
    parser.add_argument('--use_wscale', default=False, type=bool)
    parser.add_argument('--train_batch_size', default=2, type=int)
    parser.add_argument('--ema_decay', default=0.995, type=float)
    parser.add_argument('--train_lr', default=1e-5, type=float)
    parser.add_argument('--weight_decay', default=0, type=float)
    parser.add_argument('--beta1', default=0.9, type=float)
    parser.add_argument('--beta2', default=0.999, type=float)
    parser.add_argument('--n_plot', type=int, default=2000, help='plot each n epochs')
    parser.add_argument('--n_ckpt', type=int, default=1000, help='save ckpt each n epochs')
    parser.add_argument('--n_stats', type=int, default=10, help='stats each n epochs')
    parser.add_argument('--max_ckpt', default=100, type=int)
    parser.add_argument('--train_num_steps', default=700000, type=int)
    parser.add_argument('--gradient_accumulate_every', default=2, type=int)
    parser.add_argument('--step_start_ema', default=1000, type=int)

    return parser.parse_args()
