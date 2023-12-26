import os
from _csv import writer
import mindspore
from mindspore.dataset import transforms ,vision
from dataset.FASDataset_new_ifas import FASDataset
from utils.transform import RandomGammaCorrection
from utils.utils_new_ifas import read_cfg, get_optimizer, build_network, warmup_scheduler
from train_code.FASTrainer_new_ifas import FASTrainer
from model.loss_ifas import DepthLoss
import mindspore.dataset as ds
from mindspore import nn, ops
import warnings
warnings.filterwarnings("ignore")
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 5"
# ------
# import mindspore as ms
# ms.context.set_context(mode=ms.context.GRAPH_MODE, device_target="GPU")

cfg = read_cfg(cfg_file="/data2/hd/HFM/HFM_218_1/cfg/oulu_ifas.yaml")
network = build_network(cfg)
optimizer = get_optimizer(cfg, network)
criterion = DepthLoss(cfg['model']['w'])
# writer = SummaryWriter(cfg['log_dir'])

train_transform = transforms.Compose([
    RandomGammaCorrection(max_gamma=cfg['dataset']['augmentation']['gamma_correction'][1],
                            min_gamma=cfg['dataset']['augmentation']['gamma_correction'][0]),
    vision.RandomResizedCrop(cfg['model']['input_size'][0]),
    vision.RandomColorAdjust(
        brightness=cfg['dataset']['augmentation']['brightness'],
        contrast=cfg['dataset']['augmentation']['contrast'],
        saturation=cfg['dataset']['augmentation']['saturation'],
        hue=cfg['dataset']['augmentation']['hue']
    ),
    vision.RandomRotation(cfg['dataset']['augmentation']['rotation_range']),
    vision.RandomHorizontalFlip(),
    vision.Resize(cfg['model']['input_size']),
    vision.ToTensor(),
    vision.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'],is_hwc=False)
])

val_transform = transforms.Compose([
    vision.Resize(cfg['model']['input_size']),
    vision.ToTensor(),
    vision.Normalize(cfg['dataset']['mean'], cfg['dataset']['sigma'],is_hwc=False)
])


trainset = FASDataset(
    root_dir=cfg['dataset']['root'],
    depth_map_dir=cfg['dataset']['depth_map_dir'],
    csv_file=cfg['dataset']['train_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=train_transform,
    smoothing=cfg['train']['smoothing']
)

valset = FASDataset(
    root_dir=cfg['dataset']['root'],
    depth_map_dir=cfg['dataset']['depth_map_dir'],
    csv_file=cfg['dataset']['val_set'],
    depth_map_size=cfg['model']['depth_map_size'],
    transform=val_transform,
    smoothing=cfg['train']['smoothing'],
    train=False
)

# trainloader = torch.utils.data.DataLoader(
#     dataset=trainset,
#     batch_size=cfg['train']['batch_size'],
#     shuffle=True,
#     num_workers=4
# )
#
# valloader = torch.utils.data.DataLoader(
#     dataset=valset,
#     batch_size=cfg['val']['batch_size'],
#     # shuffle=True,
#     shuffle=False,
#     num_workers=4
# )

trainsets = ds.GeneratorDataset(trainset, column_names=['img', 'depth_map', 'label', 'video'],shuffle=True,schema=None)
trainsets = trainsets.batch(cfg['train']['batch_size'], drop_remainder=True)
trainloader = trainsets.create_dict_iterator()



valset = ds.GeneratorDataset(valset, column_names=['img', 'depth_map', 'label', 'video'],shuffle=True)
valset = valset.batch(cfg['train']['batch_size'], drop_remainder=True)
valloader = valset.create_dict_iterator()


epoch_iters = len(trainloader.dataset)

num_iters = cfg['train']['num_epochs'] * epoch_iters

learning_rates = cfg['train']['lr']
lr_scheduler = learning_rates

trainer = FASTrainer(
    cfg=cfg,
    network=network,
    optimizer=optimizer,
    criterion=criterion,
    lr_scheduler=lr_scheduler,
    trainloader=trainloader.dataset,
    valloader=valloader.dataset,
    writer=writer,
    pretrained=cfg['model']['pretrained'],
    epoch_iters=epoch_iters,
    num_iters=num_iters
)

trainer.train()

