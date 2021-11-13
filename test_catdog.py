from model.resnet18 import res18
import torch
import os
import torchvision.transforms.functional as F
from torchvision import transforms
from PIL import Image, ImageOps
import csv
import argparse
saved_model = 'weights/catDog_LR1e-4_0.9m_bs64_ep28_acc0.993.pth'

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
    transforms.Resize((256,256)),  # fixed
    transforms.ToTensor(),
    transforms.Normalize(*mean_std_imgnet)
])

device = torch.device('cuda')  # fixed to device 0 for testing, you may wish to change here.
model = res18()
model.to(device)
checkpoint = torch.load(saved_model, device)
model.load_state_dict(checkpoint)
model.train(False).eval()

# classes = {0:'cat', 1:'dog'}

def infer(args):
    # init csv writer
    csvfile = open(args.csv_dir, 'w', newline='')
    result_writer = csv.writer(csvfile, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
    result_writer.writerow(['id', 'Label'])

    test_dir = f'{args.data_dir_root}/test'
    imgs = os.listdir(test_dir)
    for file_name in imgs:
        basename = file_name.strip('.jpg')
        print(basename)
        img_path = os.path.join(test_dir, file_name)
        img = Image.open(img_path)
        img = ImageOps.exif_transpose(img)
        if img.mode == 'L':
            img = img.convert('RGB')
        inputs = img_transform(img)[None, :, :, :]
        inputs = inputs.to(device)
        with torch.no_grad():
            output = model(inputs)
            # print(output) # [score class 0, score class 1]
        _, predict = torch.max(output.data, 1)
        # print(_, predict) # _ = score of max value class,   predict = class index
        pred = predict.cpu().numpy()[0]
        result_writer.writerow([basename, pred])

    csvfile.close()

if __name__ == '__main__':
    def parse_args():
        parser = argparse.ArgumentParser(description='Train')
        parser.add_argument('--csv_dir', default='./submission_regenerated.csv', help='directory to save models.')
        parser.add_argument('--data_dir_root', default='../datasets', help='training data directory')
        args = parser.parse_args()
        return args

    args = parse_args()
    infer(args)


