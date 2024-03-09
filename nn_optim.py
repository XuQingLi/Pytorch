from torch import  nn
import torchvision
import torch
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
loss=nn.CrossEntropyLoss()
test=Test()
# lr学习速率 随机梯度下降（SGD）算法，学习率设置为0.01
optim=torch.optim.SGD(test.parameters(),lr=0.01)
for epoch in range(20):
    running_loss=0.0    
    for data in dataloader:
        imgs,targets=data
        outputs=test(imgs)
        result_loss=loss(outputs,targets)
        optim.zero_grad()
        result_loss.backward()
        optim.step()