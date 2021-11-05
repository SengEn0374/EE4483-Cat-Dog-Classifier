from torchvision import models
from torch import nn



def res18():
    # pretrained feature extractor
    res = models.resnet18(pretrained=True)

    # classifier
    num_features = res.fc.in_features
    # modify ResNet's fc layer to out put 2 classes. ResNet only has one fc layer
    res.fc = nn.Linear(num_features, 2)

    return res