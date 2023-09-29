import torch
from torch import nn
import torchvision.models as models


def get_resnet_model(num_classes):
    """Pretrained Resnet 34 model

    Args:
        num_classes (int): Number of output classes

    Returns:
        pretrained model
    """
    dev =  torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    pretrained_model = models.resnet34(pretrained=True).to(dev)
    IN_FEATURES = pretrained_model.fc.in_features
    pretrained_model.fc = nn.Sequential(
        nn.Linear(IN_FEATURES, 128),
        nn.ReLU(inplace=True),
        nn.Linear(128, num_classes)).to(dev)
    pretrained_model.conv1 = torch.nn.Conv2d(1,64,kernel_size=(7,7),stride=(2,2), padding=(3,3), bias=False).to(dev)
    return pretrained_model