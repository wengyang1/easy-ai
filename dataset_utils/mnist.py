import torch
from torch.utils.data import Dataset
from common_util import *

DATA_DIR = '../dataset'
MNIST_DIR = 'MNIST/raw'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'


class MyMnist(Dataset):
    def __init__(self, train: bool):
        self.train = train
        self.images = torch.tensor(
            load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_IMAGES if train else TEST_IMAGES)).reshape(-1, 1, 28, 28)
        self.labels = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_LABELS if train else TEST_LABELS)

    def __getitem__(self, idx):
        image = self.images[idx]
        label = self.labels[idx]

        return image, label

    def __len__(self):
        return len(self.labels)
