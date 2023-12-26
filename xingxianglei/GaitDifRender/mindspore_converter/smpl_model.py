import torch
import torch.nn as nn
from collections import OrderedDict

if __name__ == '__main__':
    checkpoint = torch.load('experiment2-50000.pt')
    checkpoint = checkpoint['model']
    new_dict = OrderedDict()
    for key in checkpoint:
        if 'smpl' in key:
            print(key)
            new_dict[key[5:]] = checkpoint[key]
    torch.save(new_dict, 'smpl.pth')
