import torch
import torchvision
from torchvision import transforms
from dataset.dataset import MyCustomDataset
from PIL import Image
import torchvision.transforms.functional as F
import numpy

txt = './datasets/train\\dog\\dog.9998.jpg'

x = txt.find("whale.")


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')

img = Image.open('../datasets/train/dog/dog.5.jpg')
# img = SquarePad.__call__(SquarePad(), img)
# print(img.show())


img_transform = transforms.Compose([
    transforms.RandomRotation(30, expand=True),  ##
    SquarePad(),
    transforms.Resize((256,256)),
    transforms.RandomHorizontalFlip(),
])
img = img_transform(img)
img.show()