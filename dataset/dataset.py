'''
catdog custom datasets for dataloader
'''
from torch.utils.data.dataset import Dataset
from torchvision import transforms
import torchvision.transforms.functional as F
from PIL import Image, ImageOps
import os
import glob


class MyCustomDataset(Dataset):
    def __init__(self, data_dir, image_size=(256,256), phase='train'):
        self.data_dir = data_dir
        self.image_size = image_size
        self.phase = phase
        if phase == "train":
            self.img_transform = transforms.Compose([
                # SquarePad(),
                transforms.Resize(self.image_size),
                # transforms.RandomRotation(30),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # use imgnet std mean
            ])
        if phase == 'val':
            self.img_transform = transforms.Compose([
                # SquarePad(),
                transforms.Resize(self.image_size),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])  # use imgnet std mean
            ])
        self.sub_dir = os.path.join(data_dir, phase)
        self.img_list = glob.glob(self.sub_dir+'/*/*.jpg')

        # stuff

    def __getitem__(self, index):
        img_path = self.img_list[index]
        img = Image.open(img_path)
        img = self.img_transform(img)
        if img_path.find('dog.') == -1:
            label = 0
        else:
            label = 1
        return img, label

    def __len__(self):
        return len(self.img_list)  # of how many examples(images?) you have


class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


if __name__ == "__main__":
    from torch.utils.data import DataLoader

    data_dir = '../../datasets'
    datasets = {x: MyCustomDataset(data_dir,
                                   (256,256),
                                   x)
                for x in ['train', 'val']}
    data_loader = {x: DataLoader(datasets[x],
                                 batch_size=1,
                                 shuffle=(True if x == 'train' else False),
                                 num_workers=1,
                                 pin_memory=(True if x == 'train' else False), drop_last=True)
                   for x in ['train', 'val']}
    print(len(data_loader['train']))  # 20000, passed
    print(len(data_loader['val']))    # 5000, passed
    # for data in data_loader['val']:
    #     print(data[1])


