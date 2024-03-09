import torch
import torchvision
from torch import nn
from torch.nn import Conv2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

dataset = torchvision.datasets.CIFAR10(
    "../data", train=False, transform=torchvision.transforms.ToTensor(), download=True
)
dataloader = DataLoader(dataset, batch_size=64)


class Test(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super(Test, self).__init__(*args, **kwargs)
        self.conv1 = Conv2d(
            in_channels=3, out_channels=6, kernel_size=3, stride=1, padding=1
        )

    def forward(self, x):
        x = self.conv1(x)
        return x


test = Test()
# print(test)
writer = SummaryWriter("./log_conv2d")
step = 0
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    # print(output.shape) 
    writer.add_images("input", imgs, step)
    # torch.Size([64, 6, 30, 30])
    output = torch.reshape(output,(-1, 3, output.shape[2], output.shape[3]))
    writer.add_images("output", output, step)
    step = step + 1
    
