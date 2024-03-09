from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms

# python中的用法 tensor
#绝对路径:/data/midjourney/jiayihuang/lxq/code/CV/Dataset/hymenoptera_data/train/ants_image/1269756697_0bce92cdab.jpg
#相对路径:CV/Dataset/hymenoptera_data/train/ants_image/1269756697_0bce92cdab.jpg

img_path="/data/midjourney/jiayihuang/lxq/code/CV/Dataset/hymenoptera_data/train/ants_image/5650366_e22b7e1065.jpg"
# 使用PIL库打开图片
img = Image.open(img_path)
writer=SummaryWriter("log")
# 使用transforms模块将图片转换为Tensor
tensor_trans = transforms.ToTensor()
tensor_img = tensor_trans(img)

# 打印转换后的Tensor
print(tensor_img)
writer.add_image("Tensor_img",tensor_img)
writer.close()
