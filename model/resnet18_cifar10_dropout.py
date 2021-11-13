""" model with drop out in between convs"""
from torchvision import models
from torch import nn


def ResNet18():
    # pretrained feature extractor
    res = models.resnet18(pretrained=True)

    # classifier
    num_features = res.fc.in_features
    # modify ResNet's fc layer to out put 2 classes for cat_dog. ResNet only has one fc layer
    res.fc = nn.Linear(num_features, 10)

    return res


def append_dropout(model, rate):
    for name, module in model.named_children():
        if len(list(module.children())) > 0:
            append_dropout(module, rate)  # if single element then do check below
        if isinstance(module, nn.ReLU):
            new = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
            setattr(model, name, new)


def res18(dprate):
    model = ResNet18()
    append_dropout(model, dprate)
    return model



if __name__ == "__main__":
    model_test = res18(0.15)
    print(model_test)

# class Res18(nn.Module):
#     def __init__(self):
#         super(Res18, self).__init__()
#         res = models.resnet18(pretrained=True)
#
#         self.front = nn.Sequential(
#             res.conv1, res.bn1, res.relu, res.maxpool
#         )
#         self.layer1 = res.layer1
#         self.layer2 = res.layer2
#         self.layer3 = res.layer3
#         self.layer4 = res.layer4
#         self.avgpool = res.avgpool
#         num_features = res.fc.in_features
#         self.fc = nn.Linear(num_features, 10, bias=True)
#         self.dropout = nn.Dropout(0.2)
#
#     def forward(self, x):
#         x = self.front(x)
#
#         x = self.layer1(x)
#
#         x = self.layer2(x)
#         x = self.dropout(x)
#
#         x = self.layer3(x)
#         x = self.dropout(x)
#
#         x = self.layer4(x)
#         x = self.dropout(x)
#
#         x = self.avgpool(x)
#         x = self.fc(x)
#
#         return x




