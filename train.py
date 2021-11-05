from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import StepLR
import numpy as np
import argparse
import os
import time
from dataset.dataset import MyCustomDataset
from model.resnet18 import res18


def cat_dog_trainer(args):
    since_time = time.time()
    best_acc = 0
    indicator = 0
    log_dir = os.path.join('./runs', args.save_dir.split('/')[-1])
    writer = SummaryWriter(log_dir)

    if torch.cuda.is_available():
        gpu_id = int(args.device)
        print("Use GPU: {} for training".format(gpu_id))
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
    else:
        raise Exception("gpu is not available")

    datasets = {x: MyCustomDataset(args.data_dir,
                                   (args.img_size, args.img_size),
                                   x)
                for x in ['train', 'val']}
    data_loader = {x: DataLoader(datasets[x],
                                 batch_size=32,
                                 shuffle=(True if x == 'train' else False),
                                 num_workers=4,
                                 pin_memory=(True if x == 'train' else False), drop_last=True)
                   for x in ['train', 'val']}

    model = res18()
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    scheduler = StepLR(optimizer, step_size=5, gamma=1.0)
    criterion = nn.CrossEntropyLoss().cuda()

    start_epoch = 0
    if args.resume:
        suf = args.resume.rsplit('.', 1)[-1]
        if suf == 'tar':
            checkpoint = torch.load(args.resume, device)
            model.load_state_dict(checkpoint['model_state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch'] + 1
        elif suf == 'pth':
            model.load_state_dict(torch.load(args.resume, device))

    """Training Process"""
    for epoch in range(start_epoch, args.max_epoch):
        print('Epoch {}/{}-----------------------------------------------'.format(epoch, args.max_epoch - 1))
        start = time.time()

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)  # Set model to training mode
            else:
                model.train(False)

            running_loss = 0.0
            running_corrects = 0

            # sanity check
            # print(len(data_loader['train'])) 20k img = 625 batches for 32 batchsize
            for step, (inputs, labels) in enumerate(data_loader[phase]):
                right_predictions = 0
                inputs = inputs.to(device)
                labels = labels.to(device)

                optimizer.zero_grad()
                if phase == 'val':
                    with torch.no_grad():
                        outputs = model(inputs)
                else:
                    outputs = model(inputs)  # [score class 0, score class 1]
                _, predict = torch.max(outputs.data, 1)  # .data removes the gradients, max(x, 1) returns max score and indices of the max score of each row
                loss = criterion(outputs, labels)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()
                    scheduler.step()

                # collect stats
                running_loss += loss.data
                for i, label in enumerate(labels.data):
                    if predict[i] == label:
                        right_predictions += 1
                running_corrects += right_predictions

            # calculate epoch loss and acc
            epoch_loss = running_loss / datasets[phase].__len__()
            epoch_acc = running_corrects / datasets[phase].__len__()

            # plot to tensorboard for training progress
            writer.add_scalar(phase + '/loss', epoch_loss, epoch)
            writer.add_scalar(phase + '/acc', epoch_acc, epoch)

            # print epoch stats
            epoch_time = time.time() - start
            print(f'{phase} Loss: {epoch_loss} \tAcc: {epoch_acc}')
            print('time elapsed: %.2f sec' % epoch_time)

            # save current model if better than last epoch
            if phase == 'val' and epoch_acc > best_acc and epoch > 0:
                best_acc = epoch_acc
                print('Saving model..')
                torch.save(model.state_dict(), os.path.join(args.save_dir, './model_{}.pth'.format(epoch)))
                indicator = 1
        print('  =>  Check point' + '-' * 40)

    # print overall training stats
    time_elapsed = time.time() - since_time
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # if model did not improve at all.. save last model
    if indicator == 0:
        torch.save(model.state_dict(), './params/finale_model.pth')


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--save_dir', default='./checkpoints/squish_1e-5', help='directory to save models.')
    parser.add_argument('--data_dir', default='C:/Users/crono/Desktop/datasets', help='training data directory')

    parser.add_argument('--img_size', default=256, type=int, help='input image size (square)')

    parser.add_argument('--lr', type=float, default=0.001, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default= 0.9, help='sgd momentum')
    # parser.add_argument('--weight_decay', type=float, default=1e-5, help='the weight decay')
    parser.add_argument('--resume', default='',help='the path of resume training model')
    parser.add_argument('--max_epoch', type=int, default=25, help='max training epoch')
    # parser.add_argument('--val_epoch', type=int, default=1, help='the num of steps to log training information')
    # parser.add_argument('--val_start', type=int, default=0, help='the epoch start to val')
    parser.add_argument('--batch_size', type=int, default=32, help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=4, help='the num of training process')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    # torch.backends.cudnn.benchmark = True
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = cat_dog_trainer(args)


