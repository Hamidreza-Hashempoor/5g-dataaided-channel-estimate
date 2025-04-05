import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader



class CVAEBITS(Dataset):
    def __init__(self, inputs, labels):
        self.inputs = inputs
        self.labels = labels

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        input_data = self.inputs[idx]
        label = self.labels[idx]
        return {'input': torch.as_tensor(input_data), 'label': torch.as_tensor(np.asarray(label),
                                          dtype=torch.int64)}


def get_data(inputs, labels, mode, batch_size):

    # datasets, dataloaders, dataset_sizes = {}, {}, {}

    datasets = CVAEBITS(inputs, labels)
    dataloaders = DataLoader(
        datasets,
        batch_size=batch_size,
        shuffle=mode == 'train',
        num_workers=0
    )
    dataset_sizes = len(datasets)

    return datasets, dataloaders, dataset_sizes


def get_val_images(dataloader):

    batch = next(iter(dataloader))
    inputs = batch['input']
    digits = batch['label']
    return inputs, digits