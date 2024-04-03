import numpy as np
import torch
import torchvision
from torch.utils.data import DataLoader, Dataset
from torch.utils.tensorboard import SummaryWriter
import gzip
import os
import cv2

DATA_DIR = '../dataset'
MNIST_DIR = 'MNIST/raw'
TRAIN_IMAGES = 'train-images-idx3-ubyte.gz'
TRAIN_LABELS = 'train-labels-idx1-ubyte.gz'
TEST_IMAGES = 't10k-images-idx3-ubyte.gz'
TEST_LABELS = 't10k-labels-idx1-ubyte.gz'

my_transform = torchvision.transforms.Compose([
    torchvision.transforms.ToTensor(),
    # torchvision.transforms.Normalize((0.1307,), (0.3081,))
])

torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=my_transform)
torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=my_transform)


def load_gzip(data_folder, data_name):
    assert 'gz' in data_name and ('labels' in data_name or 'images' in data_name)
    # rb : binary, unzip data to binary
    with gzip.open(os.path.join(data_folder, data_name), 'rb') as file:
        # why offset 8 for labels and 16 for images ?
        datas = np.frombuffer(file.read(), np.uint8, offset=8 if 'labels' in data_name else 16)
    return datas


def cv_show(mat):
    cv2.imshow('test', mat=mat)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


class MyMnist(Dataset):
    def __init__(self, train: bool):
        self.train = train
        self.train_images = torch.Tensor(
            load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_IMAGES).reshape(-1, 1, 28, 28))  # NCHW
        self.train_labels = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_LABELS)
        self.val_images = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TEST_IMAGES).reshape(-1, 1, 28, 28)
        self.val_labels = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TEST_LABELS)
        # print(self.train_images, self.train_labels, self.val_images, self.val_labels)
        # print(self.train_images.shape, self.train_labels.shape, self.val_images.shape, self.val_labels.shape)
        # print(self.train_images[0].shape, type(self.train_images[0]))
        # cv_show(np.array(self.train_images[0]).reshape(28, 28))
        # print(self.train_images[0], self.train_labels[0])
        # print(type(self.train_images[0]), self.train_images[0].shape)

    def __getitem__(self, idx):
        if self.train:
            image = self.train_images[idx]
            label = self.train_labels[idx]
        else:
            image = self.val_images[idx]
            label = self.val_labels[idx]
        return image, label

    def __len__(self):
        if self.train:
            return len(self.train_labels)
        return len(self.val_labels)


train_dataset = MyMnist(train=True)
val_dataset = MyMnist(train=False)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=256, shuffle=True, num_workers=0, drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)

writer = SummaryWriter(log_dir='../logs')
epochs = 2
for epoch in range(epochs):
    global_step = 1
    for data in train_dataloader:
        images, labels = data
        writer.add_images(tag='mnist_data_epoch_{}'.format(epoch), img_tensor=images, global_step=global_step)
        global_step += 1
writer.close()
# to watch log events , please run command blow in terminal : tensorboard --logdir=logs --port=6007
