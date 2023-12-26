from mindspore.dataset import GeneratorDataset
from mindspore import SummaryRecord
import mindspore.nn as nn
from dataset import *
from modules import *
from infer_files.human_part_encoder import HumanPartEncoder
from infer_files.human_part_decoder import HumanPartDecoder
from utils.util import choose_log_dir


def main():
    num_caps = 14
    img_size = (128, 64)
    train_dataset = PartSHHQDataset("/media/lz/新加卷1/datasets/PartSSHQ-14", img_size=img_size)
    dataset = GeneratorDataset(train_dataset, column_names=['img', 'label'], shuffle=True, num_parallel_workers=4)
    # dataset = dataset.map(operations=img_transforms, input_columns="img")
    # dataset = dataset.map(operations=label_transforms, input_columns="label")
    dataset = dataset.batch(batch_size=8)
    data_len = len(dataset)
    train_dataloader = dataset.create_tuple_iterator()

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

    # freeze params
    for param in ms_human_encoder.trainable_params():
        param.requires_grad = False

    for param in ms_human_decoder.trainable_params():
        if 'to_rgb' not in param.name and 'generator_block2' not in param.name and 'generator_block1' not in param.name:
            param.requires_grad = False
    all_params = list(ms_human_encoder.trainable_params()) + list(ms_human_decoder.trainable_params())
    for param in all_params:
        print(param.name)

    ms_scheduler = nn.ExponentialDecayLR(learning_rate=1.e-3, decay_rate=0.9, decay_steps=10)
    ms_optimizer = nn.Adam(params=list(ms_human_encoder.trainable_params()) + list(ms_human_decoder.trainable_params()),
                           learning_rate=ms_scheduler)
    log_dir = choose_log_dir('./log/freeze')
    logger = SummaryRecord(log_dir=log_dir)
    max_epochs = 50
    ms_model = VAEModel(encoder=ms_human_encoder, decoder=ms_human_decoder, train_loader=train_dataloader,
                        optimizer=ms_optimizer, logger=logger, max_epoch=max_epochs, data_len=data_len)

    ms_model.load("infer_files/ms_encoder_weights.ckpt", "infer_files/ms_decoder_weights.ckpt")
    ms_model.train(ckpt_path=log_dir, save_freq=10)


if __name__ == "__main__":
    main()
