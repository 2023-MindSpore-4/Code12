import mindspore

from Train import Trainer
from models import Unet3D as Net
from parse_args import parse_args
from utils.util import to_named_dict
from video_diffusion_model import GaussianDiffusion


if __name__ == "__main__":
    mindspore.set_context(device_target="GPU")
    args = parse_args()
    args = to_named_dict(args)
    model = Net(
        args=args,
        dim=args.dim,
        dim_mults=args.dim_mults,
        attn_dim_head=args.attn_dim_head,
        use_top=args.use_top,
        top_dim=args.top_dim,
        with_t=args.with_t,
        top_rate=args.top_rate,
        top_mode=args.top_mode,
    )
    model = GaussianDiffusion(
        model,
        args=args,
        num_frames=args.num_frames,
        image_size=args.image_size,
        timesteps=args.timesteps,
        loss_type=args.loss_type
    )
    mindspore.load_checkpoint('model.ckpt', model)

    trainer = Trainer(
        args,
        model,
        train_batch_size=args.train_batch_size,
        train_lr=args.train_lr,
        weight_decay=args.weight_decay,
        beta1=args.beta1,
        beta2=args.beta2,
        train_num_steps=args.train_num_steps,
        gradient_accumulate_every=args.gradient_accumulate_every,
        ema_decay=args.ema_decay,
        output_dir='output',
        n_plot=args.n_plot,
        n_ckpt=args.n_ckpt,
        n_stats=args.n_stats,
        rank=0,
        root_pkl='',
        max_ckpt=args.max_ckpt,
        step_start_ema=args.step_start_ema
    )
    trainer.test(args, '1')