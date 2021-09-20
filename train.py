from unet import UNet
from vgg16 import VGG16

import torch.nn as nn
import torchvision
import torchvision.transforms.functional as TF
import torch.optim as optim
import numpy as np
import os
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.utils import save_image
from PIL import ImageFile
from tqdm import tqdm
ImageFile.LOAD_TRUNCATED_IMAGES = True

from config import *
from utils import printFixed


class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.vgg = VGG16()
        self.unet = UNet()

    def forward(self, x, y):
        """
        x is the watermarked image, y is the ground truth 
        """
        y_hat = self.unet(x)
        f_y = self.vgg(y)
        f_y_hat = self.vgg(y_hat)
        return y_hat, f_y, f_y_hat


def calc_loss(y_hat, y, f_y_hat, f_y):
    l1_loss = nn.L1Loss()(y_hat, y)
    l2_loss = nn.MSELoss()(f_y_hat, f_y)
    loss = l1_loss + L2_LOSS_WEIGHT * l2_loss
    return loss


def training_loop(n_epochs, optimizer, model, train_loader, val_loader):
    for epoch in range(1, n_epochs + 1):
        printFixed(f"Epoch No. {epoch}", end='\n', color='GREEN', bgColor='YELLOW')
        train_loss = 0.0
        val_loss = 0.0
        train_iter = iter(train_loader)
        val_iter = iter(val_loader)

        model.train()
        for _ in tqdm(range(len(train_iter))):
            imgs, masks = train_iter.next()
            y_hat, f_y, f_y_hat = model(imgs, masks)
            loss = calc_loss(y_hat, imgs, f_y_hat, f_y)
            train_loss += loss.item()

            optimizer.zero_grad() 
            loss.backward() 
            optimizer.step()

        model.eval()
        for _ in range(len(val_iter)):
            imgs, masks = val_iter.next()
            y_hat, f_y, f_y_hat = model(imgs, masks)
            val_loss += calc_loss(y_hat, imgs, f_y_hat, f_y).item()
        
        if epoch == 1 or epoch % 1 == 0:
            print('Epoch {}, Training loss {}, Validation loss {}'.format(
                epoch,
                train_loss,
                val_loss))


class WaterDataSet:
    def __init__(self, img_root_dir, gtruth_root_dir):
        self.img_root_dir = img_root_dir
        self.gtruth_root_dir = gtruth_root_dir
        self.list_img_files = os.listdir(img_root_dir)
        self.list_gtruth_files = os.listdir(gtruth_root_dir)

        assert len(self.list_img_files) == len(self.list_gtruth_files)

    def __len__(self):
        return len(self.list_img_files)

    def __getitem__(self, index):
        def load_file(index, gtruth = False):
            if not gtruth:
                file =  self.list_img_files[index]
                path = os.path.join(self.img_root_dir, file)
            else:
                file =  self.list_gtruth_files[index]
                path = os.path.join(self.gtruth_root_dir, file)

            image = Image.open(path).convert('RGB')
            image = image.resize(INPUT_IMAGE_SIZE)
            image = np.array(image)
            image = torchvision.transforms.ToTensor()(image)

            return image


        img_image = load_file(index)
        gtruth_image = load_file(index, gtruth = True)

        return img_image.to('cuda'), gtruth_image.to('cuda')


model = Net()
model.cuda()

train_loader = DataLoader(WaterDataSet(TRAIN_PATH + 'imgs', TRAIN_PATH + 'masks'), batch_size=TRAIN_BATCH_SIZE)
val_loader = DataLoader(WaterDataSet(TEST_PATH + 'imgs', TEST_PATH + 'masks'), batch_size=TRAIN_BATCH_SIZE)


optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

training_loop(2, optimizer, model, train_loader, val_loader)
