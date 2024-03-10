import torch
import torchvision
from torch import nn
# from model_save import *
# 方式1->保存方式,加载模型
model=torch.load("vgg16_method1.pth")
print(model)
# 方式2->加载模型
vgg16=torchvision.models.vgg16(weights=None)
vgg16.load_state_dict("vgg16_method2.pth")
print(vgg16)
# 陷阱1
class Test(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.conv1=nn.Conv2d(3,64,kernel_size=3)
    def forward(self,x):
        x=self.conv1(x)
        return x
    
model=torch.load('test_method1.pth')
print(model)