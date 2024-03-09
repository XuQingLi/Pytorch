
from PIL import Image
from torchvision import transforms
from torch.utils.tensorboard import SummaryWriter
writer=SummaryWriter("logs")
img=Image.open("/data/midjourney/jiayihuang/lxq/code/CV/Dataset/hymenoptera_data/train/ants_image/67270775_e9fdf77e9d.jpg")
print(img)

# totensor的使用
trans_totensor=transforms.ToTensor()
img_tensor=trans_totensor(img)

writer.add_image("ToTensor",img_tensor)
# Normalize 归一化
print(img_tensor[0][0][0])
trans_norm=transforms.Normalize([0.5,0.5,0.5],[0.5,0.5,0.5])
img_norm=trans_norm(img_tensor)
print(img_norm[0][0][0])
writer.add_image("Normalize",img_norm)

# Resize
print(img.size)
trans_resize=transforms.Resize((1024,512))
#img PIL->resize ->img resize PIL
img_resize=trans_resize(img)
# img resize PIL ->totensro -> img resize tensor
img_resize=trans_totensor(img_resize)
print(img_resize)

# compose-resize-2
trans_resize_2=transforms.Resize(512)
# Compose()中的参数需要是一个列表;数据需要是 transforms类型
trans_compose=transforms.Compose([trans_resize_2,trans_totensor])
img_resize_2=trans_compose(img)
writer.add_image("Resize",img_resize_2,1)
# RandomCrop
trans_random = transforms.RandomCrop((100,200))
trans_compose_2 = transforms.Compose([trans_random, trans_totensor])
for i in range(10):
    img_crop = trans_compose_2(img)
    writer.add_image("RandomCrop", img_crop, i)

writer.close()