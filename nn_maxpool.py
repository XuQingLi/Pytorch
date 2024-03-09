import torch
import torchvision
from torch import nn
from torch.nn import MaxPool2d
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
dataset=torchvision.datasets.CIFAR10("../data",train=False,download=True,transform=torchvision.transforms.ToTensor())
# batch_size训练神经网络时每次处理的数据样本数量
dataloader=DataLoader(dataset,batch_size=64)

# input = torch.tensor(
#     [
#         [1, 2, 0, 3, 1],
#         [0, 1, 2, 3, 1],
#         [1, 2, 1, 0, 0],
#         [5, 2, 3, 1, 1],
#         [2, 1, 0, 1, 1]
#     ],dtype=torch.float32
# )
# # -1:PyTorch自动计算维度大小   1常用于增加一个维度
# input = torch.reshape(input, (-1, 1, 5, 5))
# print(input.shape)


class Test(nn.Module):
    def __init__(self):
        # super() 函数的用法是在继承体系中调用父类的构造函数
        super(Test, self).__init__()
        self.maxpool1 = MaxPool2d(kernel_size=3, ceil_mode=True)

    def forward(self, input):
        output = self.maxpool1(input)
        return output
 
test = Test()
# output = test(input)
# print(output)
writer=SummaryWriter("log_maxpool")
step=0
for data in dataloader:
    imgs,targets=data
    writer.add_images("input",imgs,step)
    output=test(imgs)
    writer.add_images("output",output,step)
    step+=1
writer.close()