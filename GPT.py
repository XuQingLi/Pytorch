import torch
from torch import nn
from torch.utils.tensorboard import SummaryWriter
import torchvision
from torchvision.utils import make_grid
from torch.utils.data import DataLoader

# 加载 CIFAR10 数据集
dataset = torchvision.datasets.CIFAR10(
    "./dataset_conv2d",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
dataloader = DataLoader(dataset, batch_size=64)

# 定义网络结构
class Test(nn.Module):
    def __init__(self):
        super(Test, self).__init__()
        self.conv1 = nn.Conv2d(
            in_channels=3, out_channels=3, kernel_size=3, stride=1, padding=0
        )

    def forward(self, x):
        x = self.conv1(x)
        return x

test = Test()
print(test)

# 初始化 TensorBoard
writer = SummaryWriter("../logs")
step = 0

# 进行数据处理和网络前向传播
for data in dataloader:
    imgs, targets = data
    output = test(imgs)
    
    # 确认输出的形状符合预期
    print("Input shape:", imgs.shape)    # 应为 [64, 3, 32, 32]
    print("Output shape:", output.shape) # 应为 [64, 3, 30, 30]

    # 使用 make_grid 准备 TensorBoard 可视化
    # 注意：这里假设 output 形状已经是 [批量大小, 通道数, 高度, 宽度]
    output_grid = make_grid(output, normalize=True)
    
    # 写入 TensorBoard
    writer.add_images("output", output_grid, step)

    step = step + 1

writer.close()
