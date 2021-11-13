from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import argparse
import os
import time
import torchvision
from torchvision import transforms
# from model.resnet18_cifar10 import res18
from model.resnet18_cifar10_dropout import res18


def cifar10_test(args):
    if torch.cuda.is_available():
        gpu_id = int(args.device)
        print("Use GPU: {} for training".format(gpu_id))
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
    else:
        raise Exception("gpu is not available")

    # dataloaders
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)  # size 10k
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.num_workers)
    test_size = 10000

    # classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
    saved_model = 'weights/cifar10_LR1e-3_0.9mmt_bs64_ep98_acc0.851.pth'
    model = res18(0)
    model.to(device)
    checkpoint = torch.load(saved_model, device)
    model.load_state_dict(checkpoint)
    model.train(False).eval()

    running_loss = 0.0
    running_corrects = 0

    for step, (inputs, labels) in enumerate(testloader):
        right_predictions = 0
        inputs = inputs.to(device)
        labels = labels.to(device)

        with torch.no_grad():
            outputs = model(inputs)

        _, predict = torch.max(outputs.data, 1)  # .data removes the gradients, max(x, 1) returns max score and indices of the max score of each row

        # collect stats
        for i, label in enumerate(labels.data):
            if predict[i] == label:
                right_predictions += 1
        running_corrects += right_predictions
        acc = running_corrects / test_size

    # print epoch val stats
    print(f'Test set accuracy: {acc}')



def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=2, help='the num of training process')
    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = cifar10_test(args)


