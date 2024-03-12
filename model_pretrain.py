import torchvision
from torch import nn

vgg16_false = torchvision.models.vgg16(pretrained=False)
vgg16_true = torchvision.models.vgg16(pretrained=True)
print(vgg16_true)
# train_data=torchvision.datasets.ImageNet("../data",train=True,download=True,transform=torchvision.transforms.ToTensor())
# 在vgg16_true模型的分类器（classifier）部分添加一个新的全连接层
vgg16_true.classifier.add_module("add_linear", nn.Linear(1000, 10))

print(vgg16_true)
print(vgg16_false)
# 修改vgg16_false模型的分类器的最后一层，将输出维度改为10
# 在VGG16的默认分类器中，最后一层的全连接层的索引是6。
vgg16_false.classifier[6] = nn.Linear(4896, 10)
print(vgg16_false)
