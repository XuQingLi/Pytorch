import torchvision
from torch import nn
from model import *
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
# 准备训练集和测试集
train_data = torchvision.datasets.CIFAR10(
    root="../data",
    train=True,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)
test_data = torchvision.datasets.CIFAR10(
    root="../data",
    train=False,
    transform=torchvision.transforms.ToTensor(),
    download=True,
)

train_data_size = len(train_data)
test_data_size = len(test_data)
print("TrainDataSize:{}".format(train_data_size))
print("TestDataSize:{}".format(train_data_size))

# 利用dataloader来加载数据集   batch_size模型训练时每次更新参数的样本数量
train_dataloader = DataLoader(train_data, batch_size=64)
test_dataloader = DataLoader(test_data, batch_size=64)

# 创建网络模型
test = Test()

# 创建损失函数
loss_fn = nn.CrossEntropyLoss()
# 定义优化器,随机梯度下降
# lr是learning_rate
optimizer = torch.optim.SGD(
    test.parameters(),
    lr=0.01,
)

# 设置训练网络的一些参数
# 记录训练的次数
total_train_step = 0
# 记录测试的次数
total_test_step = 0
# 训练的轮数
epoch = 10
writer=SummaryWriter("./logs_model_train")
for i in range(epoch):
    print("第{}轮训练".format(i + 1))
    # 训练步骤开始
    for data in train_dataloader:
        imgs, targets = data
        output = test(imgs)
        loss = loss_fn(output, targets)
        # 优化器调优
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        total_train_step = total_train_step + 1
        if total_train_step%100==0:
            print("训练次数：{},loss:{}".format(total_train_step, loss.item()))
            writer.add_scalar("train_loss",loss.item(),total_train_step)
    # 测试步骤开始
    total_test_loss=0
    with torch.no_grad():
        for data in test_dataloader:
            imgs,targets=data
            outputs=test(imgs)
            loss=loss_fn(outputs,targets)
            total_test_loss=total_test_loss+loss.item()
    print("total_test_loss:{}".format(total_test_step))
    writer.add_scalar("test_loss",total_test_step,total_test_step)
    total_test_step=total_test_step+1 
    torch.save(test,"test_{}.pth".format(i))
writer.close()                    
            