from torch.utils.tensorboard import SummaryWriter
from PIL import Image
import numpy as np
# 创建实例
writer=SummaryWriter(log_dir="logs_dir",comment="comment_test")
image_path="/data/midjourney/jiayihuang/lxq/code/CV/Dataset/hymenoptera_data/train/bees_image/2601176055_8464e6aa71.jpg"
img=Image.open(image_path)
img_array=np.array(img)
print(type(img_array))
print(img_array.shape)
img_PIL=Image.open(image_path)
# 调用方法
# img_tensor (torch.Tensor, numpy.ndarray, or string/blobname)
writer.add_image("train",img_array,1,dataformats="HWC")
# writer.add_scalar()
for i in range(100):
    # scalar_value相当于y轴, global_step=None相当于x轴
    writer.add_scalar("y=3*x",3*i,i)
    
writer.close()
