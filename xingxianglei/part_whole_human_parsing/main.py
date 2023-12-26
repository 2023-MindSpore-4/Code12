from mindspore.dataset import GeneratorDataset
from mindspore.dataset import vision
from mindspore.dataset import transforms
from mindspore import SummaryRecord
import mindspore.nn as nn
from dataset import *
from modules import *
from models import *
from utils.args import parse_args
from utils.util import choose_log_dir


def main():
    args = vars(parse_args())

    # img_transforms = transforms.Compose([
    #     vision.Resize(size=(128, 64)),
    #     vision.ToTensor(),
    #     vision.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
    # ])
    # label_transforms = transforms.Compose([
    #     vision.Resize(size=(128, 64)),
    #     vision.Grayscale(num_output_channels=1),
    #     vision.ToTensor(),
    # ])

    if args['dataset'] == 'PartSHHQ':
        num_caps = 14
        img_size = (128, 64)
        train_dataset = PartSHHQDataset("/media/lz/新加卷1/datasets/PartSSHQ-14", img_size=img_size)
        dataset = GeneratorDataset(train_dataset, column_names=['img', 'label'], shuffle=True)
        # dataset = dataset.map(operations=img_transforms, input_columns="img")
        # dataset = dataset.map(operations=label_transforms, input_columns="label")
        dataset = dataset.batch(batch_size=2)
        data_len = len(dataset)
        train_dataloader = dataset.create_tuple_iterator()
    else:
        num_caps = None
        img_size = None
        train_dataloader = None
        data_len = 0
        print("dataset_choose_error")

    if args['model'] == 'hpg':
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
        ms_human_encoder = HumanPartEncoder(edges=g_edges)
        ms_human_decoder = HumanPartDecoder(relation_list=g_relation)
        ms_scheduler = nn.ExponentialDecayLR(learning_rate=1.e-3, decay_rate=0.9, decay_steps=10)
        ms_optimizer = nn.Adam(params=list(ms_human_encoder.trainable_params()) + list(ms_human_decoder.trainable_params()),
                               learning_rate=ms_scheduler)
        log_dir = choose_log_dir('./log/hpg')
        logger = SummaryRecord(log_dir=log_dir)
        max_epochs = 300
        ms_model = VAEModel(encoder=ms_human_encoder, decoder=ms_human_decoder, train_loader=train_dataloader,
                            optimizer=ms_optimizer, logger=logger, max_epoch=max_epochs, data_len=data_len)
    else:
        log_dir = None
        ms_model = None

    ms_model.train(ckpt_path=log_dir, save_freq=100)


if __name__ == "__main__":
    main()
