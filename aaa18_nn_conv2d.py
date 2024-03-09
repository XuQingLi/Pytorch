import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

dataset = torchvision.datasets.CIFAR10(
    "./dataset_conv2d",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset, batch_size=64)

class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

test = Test()
writer = SummaryWriter("../logs")
step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    print(imgs.shape)
    print(output.shape)
    writer.add_images("input", imgs, step)
    # torch.Size([64,6,30,38]) ->[xx×,3,30,30]
    output = torch.reshape(output, (-1, 3, 30, 30))
    writer.add_images("output", output, step)
    step = step + 1
writer.close()
