from PIL import Image
import numpy as np

image_path="CV/Dataset/hymenoptera_data/train/ants_image/0013035.jpg"
img=Image.open(image_path)
print(type(img))

img_array=np.array(img)
print(type(img_array))

