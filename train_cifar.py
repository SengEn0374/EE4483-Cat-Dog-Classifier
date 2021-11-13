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


def cifar10_trainer(args):
    log_txt = f'./{args.save_dir.split("/")[-1]}.txt'
    f = open(log_txt, 'w')
    f.write(f'learning rate: {args.lr}\nmomentum: {args.momentum}\nbatch size: {args.batch_size}\n')
    since_time = time.time()
    best_acc = 0
    indicator = 0
    log_dir_train = os.path.join('./runs', args.save_dir.split('/')[-1] + "_train")
    log_dir_val = os.path.join('./runs', args.save_dir.split('/')[-1] + "_val")
    writer_train = SummaryWriter(log_dir_train)
    writer_val = SummaryWriter(log_dir_val)

    if torch.cuda.is_available():
        gpu_id = int(args.device)
        print("Use GPU: {} for training".format(gpu_id))
        torch.cuda.set_device(gpu_id)
        device = torch.device("cuda")
        device_count = torch.cuda.device_count()
    else:
        raise Exception("gpu is not available")

    # dataloaders
    transform_train = transforms.Compose(
        [transforms.RandomHorizontalFlip(),
         transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    transform = transforms.Compose(
        [transforms.ToTensor(),
         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    trainset = torchvision.datasets.CIFAR10(root='../data', train=True, download=True, transform=transform_train)  # size 50k
    testset = torchvision.datasets.CIFAR10(root='../data', train=False, download=True, transform=transform)  # size 10k
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=64, shuffle=True, num_workers=args.num_workers)
    testloader = torch.utils.data.DataLoader(testset, batch_size=64, shuffle=False, num_workers=args.num_workers)
    train_size = 50000
    test_size = 10000

    # classes = ('airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    model = res18(args.dropout)
    model.to(device)
    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = StepLR(optimizer, step_size=args.scheduler_step_size, gamma=args.gamma)
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
                running_loss = 0.0
                running_corrects = 0

                for step, (inputs, labels) in enumerate(trainloader):
                    right_predictions = 0
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()

                    outputs = model(inputs)  # [class scores]
                    _, predict = torch.max(outputs.data, 1)  # .data removes the gradients, max(x, 1) returns max score and indices of the max score of each row
                    loss = criterion(outputs, labels)

                    loss.backward()
                    optimizer.step()

                    # collect stats
                    running_loss += loss.data
                    for i, label in enumerate(labels.data):
                        if predict[i] == label:
                            right_predictions += 1
                    running_corrects += right_predictions

                # plot to tensorboard for training progress
                epoch_loss = running_loss / train_size
                epoch_acc = running_corrects / train_size
                writer_train.add_scalar('/loss', epoch_loss, epoch)
                writer_train.add_scalar('/acc', epoch_acc, epoch)

                # print epoch train stats
                epoch_time = time.time() - start
                print(f'{phase} Loss: {epoch_loss} \tAcc: {epoch_acc}')
                print(f'epoch {epoch/args.max_epoch}')
                print(f'{phase} Loss: {epoch_loss} \tAcc: {epoch_acc}', file=f)

                scheduler.step()
                print('time elapsed: %.2f sec' % epoch_time)


            else:
                '''Validation'''
                model.train(False)
                running_loss = 0.0
                running_corrects = 0

                for step, (inputs, labels) in enumerate(testloader):
                    right_predictions = 0
                    inputs = inputs.to(device)
                    labels = labels.to(device)

                    optimizer.zero_grad()
                    with torch.no_grad():
                        outputs = model(inputs)

                    _, predict = torch.max(outputs.data, 1)  # .data removes the gradients, max(x, 1) returns max score and indices of the max score of each row
                    loss = criterion(outputs, labels)

                    # collect stats
                    running_loss += loss.data
                    for i, label in enumerate(labels.data):
                        if predict[i] == label:
                            right_predictions += 1
                    running_corrects += right_predictions

                # plot to tensorboard for training progress
                epoch_loss = running_loss / test_size
                epoch_acc = running_corrects / test_size
                writer_val.add_scalar('/loss', epoch_loss, epoch)
                writer_val.add_scalar('/acc', epoch_acc, epoch)

                # print epoch val stats
                epoch_time = time.time() - start
                print(f'{phase} Loss: {epoch_loss} \tAcc: {epoch_acc}')
                print(f'{phase} Loss: {epoch_loss} \tAcc: {epoch_acc}\n', file=f)
                print('time elapsed: %.2f sec' % epoch_time)

                # save current model if better than last epoch
                if epoch_acc > best_acc and epoch > 0:
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
    print('Best val Acc: {:4f}'.format(best_acc), file=f)

    # if model did not improve at all.. save last model
    if indicator == 0:
        torch.save(model.state_dict(), './params/finale_model.pth')

    f.close()


def parse_args():
    parser = argparse.ArgumentParser(description='Train')
    parser.add_argument('--save_dir', default='./checkpoints/cifar-10_2e-3_drouput0.2_lrHalfEP50', help='directory to save models.')  # cifar-10_1e-4_dropout
    # parser.add_argument('--data_dir', default='C:/Users/crono/Desktop/datasets', help='training data directory')
    # parser.add_argument('--img_size', default=256, type=int, help='input image size (square)')  # cifar 10 is uniform sized 32x32

    parser.add_argument('--lr', type=float, default=0.002, help='the initial learning rate')
    parser.add_argument('--momentum', type=float, default= 0.9, help='sgd momentum')
    parser.add_argument('--weight_decay', type=float, default=0, help='weight decay')

    parser.add_argument('--scheduler_step_size', type=int, default=50, help="lr decay step size")
    parser.add_argument('--gamma', type=float, default=0.5, help="lr decay rate per step")

    parser.add_argument('--dropout', type=float, default=0.20, help='dropout rate')

    parser.add_argument('--resume', default='',help='the path of resume training model')
    parser.add_argument('--max_epoch', type=int, default=100, help='max training epoch')
    parser.add_argument('--batch_size', type=int, default=64, help='train batch size')
    parser.add_argument('--device', default='0', help='assign device')
    parser.add_argument('--num_workers', type=int, default=2, help='the num of training process')

    args = parser.parse_args()
    return args


if __name__ == '__main__':
    args = parse_args()
    os.makedirs(args.save_dir, exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES'] = args.device.strip()  # set vis gpu
    trainer = cifar10_trainer(args)


