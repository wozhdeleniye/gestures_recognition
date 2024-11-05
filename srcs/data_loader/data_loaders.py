from pathlib import Path
from typing import List
import os

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
import torchvision.transforms as transforms


class SignDataset(Dataset):
    def __init__(self, paths: List[Path], transform=None):
        self.paths = paths
        self.transform = transforms.Compose([
            transforms.RandomRotation(degrees=5),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        ])

        labels = sorted(set(str(x).split(os.path.sep)[-2] for x in paths))
        self.one_hot_encoding = {label: i for i, label in enumerate(labels)}
    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):
        image = cv2.imread(str(self.paths[idx]))
        label = str(self.paths[idx]).split(os.path.sep)[-2]
        image = cv2.resize(image, (200, 200))
        image = np.transpose(image, (2, 0, 1))
        image = self.transform(torch.tensor(image).float())
        return torch.tensor(image).float(), torch.tensor(self.one_hot_encoding[label])


def get_sign_dataloader(
        path_train, path_val, batch_size, shuffle=True, num_workers=1,
):
    train_dataset = SignDataset(paths=[*Path(path_train).rglob('*.jpg')])
    val_dataset = SignDataset(paths=[*Path(path_val).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': shuffle,
        'num_workers': num_workers
    }
    return DataLoader(train_dataset, **loader_args), DataLoader(val_dataset, **loader_args)


def get_sign_test_dataloader(
        path_test, batch_size, num_workers=1,
):
    test_dataset = SignDataset(paths=[*Path(path_test).rglob('*.jpg')])

    loader_args = {
        'batch_size': batch_size,
        'shuffle': False,
        'num_workers': num_workers
    }
    return DataLoader(test_dataset, **loader_args)
