from collections import OrderedDict
from typing import List

import torch

folder_path = './ckpt/eff_net_models/'
model_paths = ['model_best.pth', 'model_latest.pth', 'checkpoint-epoch8.pth']


def average_weights(state_dicts: List[dict]):
    average_dict = OrderedDict()
    for k in state_dicts[0].keys():
        average_dict[k] = sum(state_dict[k] for state_dict in state_dicts) / len(state_dicts)
    return average_dict


def main():
    all_weights = [torch.load(folder_path + model_path, map_location='cpu')['state_dict'] for model_path in model_paths]

    average_dict = average_weights(all_weights)

    torch.save({'state_dict': average_dict}, folder_path + 'average.pth')
    print(folder_path + 'average.pth')


if __name__ == '__main__':
    main()
