from argparse import ArgumentParser


def parse_args():
    parser = ArgumentParser()
    parser.add_argument('--dataset', default='PartSHHQ', type=str, help='which dataset to use')
    parser.add_argument('--model', default='hpg', type=str,
                        choices=['hpg'],
                        help='which model to use')
    parser.add_argument('--checkpoint', default=None, type=str, help='checkpoint path to load')
    return parser.parse_args()
