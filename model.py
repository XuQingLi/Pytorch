from torch import nn
import torch
# 搭建神经网络
class Test(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.model=nn.Sequential(
        # Stride 表示在卷积核滑动时的步长或跨度
        # Padding 是在输入数据周围添加额外的像素，以便更好地处理边缘信息。
        # 这里的Padding=2是为了保持in/output输出的大小形状符合实际
        nn.Conv2d(3,32,5,1,2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,32,5,1,2),
        nn.MaxPool2d(2),
        nn.Conv2d(32,64,5,1,2),
        nn.Flatten(),
        nn.Linear(64*4*4,64),
        nn.Linear(64,10) 
        )
    def forward(self,x):
        x=self.model(x)
        return x 
if __name__=='__main__':
    test=Test()
    # 输入的尺寸
    input=torch.ones((64,3,32,32))
    output=test(input)
    print(output.shape)
    
    