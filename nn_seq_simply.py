from torch import  nn
import torch
from torch.utils.tensorboard import SummaryWriter
from torch.nn import MaxPool2d,Conv2d,Flatten,Linear,Sequential
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
test=Test()
print(test)
input=torch.ones((64,3,32,32))
output=test(input)
print(output.shape)
writer=SummaryWriter("./logs_seq")
writer.add_graph(test,input)
writer.close()
