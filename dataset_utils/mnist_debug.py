import torchvision
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from dataset_utils.mnist import MyMnist
from common_util import *

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

'''
使用pytorch封装好的dataset进行加载,debug to see details
'''
train_dataset = torchvision.datasets.MNIST(DATA_DIR, train=True, download=True, transform=my_transform)
val_dataset = torchvision.datasets.MNIST(DATA_DIR, train=False, download=True, transform=my_transform)

'''
解压数据，详细分析数据结构
'''
train_images = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_IMAGES)
train_labels = load_gzip(os.path.join(DATA_DIR, MNIST_DIR), TRAIN_LABELS)

train_images = train_images.reshape(-1, 28, 28)
large_image = np.zeros((28 * 10, 28 * 10))

row_num = 10
col_num = 10
size = 28
for i in range(row_num):
    for j in range(col_num):
        print('i={} j={} j+i*col_num={}'.format(i, j, j + i * col_num))
        large_image[i * size:(i + 1) * size, j * size:(j + 1) * size] = train_images[j + i * col_num]
        j += 1
    i += 1
print('train_labels {}'.format(train_labels[0:100].reshape(10, 10)))

'''
利用numpy数据做可视化
'''
cv_show(mat=large_image)

train_dataset = MyMnist(train=True)
val_dataset = MyMnist(train=False)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
val_dataloader = DataLoader(dataset=val_dataset, batch_size=64, shuffle=True, num_workers=0, drop_last=True)
'''
利用tensorboard做可视化
'''
writer = SummaryWriter(log_dir='../logs')
epochs = 2
for epoch in range(epochs):
    global_step = 1
    for data in tqdm(train_dataloader):
        images, labels = data
        writer.add_images(tag='mnist_data_epoch_{}'.format(epoch), img_tensor=images, global_step=global_step)
        global_step += 1
writer.close()
# to watch log events , please run command blow in terminal : tensorboard --logdir=logs --port=6067
