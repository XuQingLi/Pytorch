from torch import  nn
import torchvision
from torch.utils.data import DataLoader
from torch.nn import MaxPool2d,Conv2d,Flatten,Linear,Sequential
dataset=torchvision.datasets.CIFAR10("../data",train=False,transform=torchvision.transforms.ToTensor(),download=True)
dataloader=DataLoader(dataset,batch_size=64)
class Test(nn.Module):
    def __init__(self):
        super(Test,self).__init__()
        self.model1=Sequential(
            Conv2d(3,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,32,5,padding=2),
            MaxPool2d(2),
            Conv2d(32,64,5,padding=2),
            MaxPool2d(2),
            Flatten(),
            Linear(1024,64),
            Linear(64,10)
        )
    def forward(self, x):
        x=self.model1(x)
        return x
# 定义损失函数为交叉熵损失,多分类问题中常用
loss=nn.CrossEntropyLoss()
test=Test()

for data in dataloader:
    imgs,targets=data
    outputs=test(imgs)
    # print(outputs)
    # print(targets)
    result_loss=loss(outputs,targets)
    print(result_loss)
    # result_loss.backward()
    print("ok")