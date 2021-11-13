from model.resnet18 import res18
import torch
import os
import numpy
import torchvision.transforms.functional as F
from torchvision import transforms
import shutil
from PIL import Image, ImageOps
saved_model = 'checkpoints/squish_1e-4_50ep_bs64/model_28.pth'
# 'checkpoints/catdog_1e-3_50ep_bs64_dropout0.05/model_14.pth'
# './1e-4_0.9m_bs64_ep28_acc0.9914.pth'

class SquarePad:
    def __call__(self, image):
        max_wh = max(image.size)
        p_left, p_top = [(max_wh - s) // 2 for s in image.size]
        p_right, p_bottom = [max_wh - (s+pad) for s, pad in zip(image.size, [p_left, p_top])]
        padding = (p_left, p_top, p_right, p_bottom)
        return F.pad(image, padding, 0, 'constant')


mean_std_imgnet = ([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
img_transform = transforms.Compose([
    # SquarePad(),
    transforms.Resize((256,256)),
    transforms.ToTensor(),
    transforms.Normalize(*mean_std_imgnet)
])

device = torch.device('cuda')
model = res18()
model.to(device)
checkpoint = torch.load(saved_model, device)
model.load_state_dict(checkpoint)
model.train(False).eval()


def infer():
    gt_dir = './test_gt.txt'
    with open(gt_dir, 'r') as gt:
        lines = gt.readlines()
    count = 0
    for line in lines:
        basename, label, img_path = line.strip().split()
        label = int(label)

        # print(basename)

        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'L':
            img = img.convert('RGB')
        inputs = img_transform(img)[None, :, :, :]
        inputs = inputs.to(device)
        assert inputs.size(0) == 1, 'the batch size should equal to 1'
        with torch.no_grad():
            output = model(inputs)
            # print(output) # [score class 0, score class 1]
        _, predict = torch.max(output.data, 1)
        # print(_, predict) # _ = score of max value class,   predict = class index
        prediction = predict.cpu().numpy()[0]
        if prediction != label:
            print(f'{basename}.jpg x')
            count+=1
    acc = (len(lines)-count)/len(lines)
    print(f'accuracy: {acc}')


if __name__ == '__main__':
    infer()
    # infer_single()
    # print(model)

