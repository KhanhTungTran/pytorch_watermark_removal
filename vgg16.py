import torch
import torch.nn as nn
from torch.nn.modules.pooling import MaxPool2d

class VGG16(nn.Module):
    def __init__(self, in_channels=3, n_classes=1000):
        super(VGG16, self).__init__()

        # First conv block
        self.block1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            
            nn.MaxPool2d(2, 2)
        )

        # Second conv block
        self.block2 = nn.Sequential(
            nn.Conv2d(64, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),

            nn.Conv2d(128, 128, 3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        #     nn.MaxPool2d(2, 2)
        # )

        # # Third conv block
        # self.block3 = nn.Sequential(
        #     nn.Conv2d(128, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(256, 256, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(256),
        #     nn.ReLU(inplace=True),

        #     nn.MaxPool2d(2, 2)
        # )

        # # Fourth conv block
        # self.block4 = nn.Sequential(
        #     nn.Conv2d(256, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(512, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(512, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.MaxPool2d(2, 2)
        # )

        # # Fifth conv block
        # self.block5 = nn.Sequential(
        #     nn.Conv2d(512, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(512, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.Conv2d(512, 512, 3, padding=1, bias=False),
        #     nn.BatchNorm2d(512),
        #     nn.ReLU(inplace=True),

        #     nn.MaxPool2d(2, 2)
        # )

        # # FC block 1
        # self.fc1 = nn.Sequential(
        #     nn.Flatten(),
        #     nn.Linear(7 * 7 * 512, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(inplace=True)
        # )

        # # FC block 2
        # self.fc2 = nn.Sequential(
        #     nn.Linear(4096, 4096),
        #     nn.BatchNorm1d(4096),
        #     nn.ReLU(inplace=True)
        # )

        # # FC block 3
        # self.fc3 = nn.Sequential(
        #     nn.Linear(4096, n_classes),
        #     nn.BatchNorm1d(n_classes),
        #     nn.ReLU(inplace=True)
        # )

        # # output
        # self.outputs = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        # x = self.block3(x)
        # x = self.block4(x)
        # x = self.block5(x)
        # x = self.fc1(x)
        # x = self.fc2(x)
        # x = self.fc3(x)
        # x = self.outputs(x)
        return x

def test():
    x = torch.randn((3, 3, 224, 224))
    model = VGG16(3)
    preds = model(x)
    print(x.shape)
    print(preds.shape)

if __name__ == "__main__":
    test()