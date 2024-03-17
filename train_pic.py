from PIL import Image
import torchvision
from model import *
image_path="/data/midjourney/jiayihuang/lxq/code/imgs/frog.jpg"
image=Image.open(image_path)
print(image)
transform=torchvision.transforms.Compose([torchvision.transforms.Resize((32,32)),torchvision.transforms.ToTensor()])
image=transform(image)
print(image)

model=torch.load("test_0.pth")
print(model)
image=torch.reshape(image,(1,3,32,32))
model.eval()
with torch.no_grad():
    output=model(image)
print(output)
print(output.argmax(1))