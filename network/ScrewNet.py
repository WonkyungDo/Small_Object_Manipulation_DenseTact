import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
from torch.utils.data import DataLoader, Dataset
from torchvision import models


class ScrewClassifier_resnet(nn.Module):
    def __init__(self, num_classes=9):
        super(ScrewClassifier_resnet, self).__init__()
        base_model = models.resnet18(pretrained=True)
        input_shape = (5, 300, 300)  
        new_conv = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        base_model.conv1 = new_conv

        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        self.base_model = base_model
        for param in base_model.parameters():
            param.requires_grad = False

        for param in base_model.layer4.parameters():
            param.requires_grad = True

        for param in base_model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x

        
class ScrewClassifier_resnet_4ch(nn.Module):
    def __init__(self, num_classes=10
    ):
        super(ScrewClassifier_resnet_4ch, self).__init__()
        base_model = models.resnet18(pretrained=True)
        input_shape = (4, 300, 300)  
        new_conv = nn.Conv2d(input_shape[0], 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        base_model.conv1 = new_conv
        base_model.fc = nn.Sequential(
            nn.Linear(base_model.fc.in_features, 256),
            nn.ReLU(),
            nn.Linear(256, num_classes),
            nn.Softmax(dim=1)
        )
        self.base_model = base_model
        for param in base_model.parameters():
            param.requires_grad = False

        for param in base_model.layer4.parameters():
            param.requires_grad = True

        for param in base_model.fc.parameters():
            param.requires_grad = True

    def forward(self, x):
        x = self.base_model.conv1(x)
        x = self.base_model.bn1(x)
        x = self.base_model.relu(x)
        x = self.base_model.maxpool(x)

        x = self.base_model.layer1(x)
        x = self.base_model.layer2(x)
        x = self.base_model.layer3(x)
        x = self.base_model.layer4(x)

        x = self.base_model.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.base_model.fc(x)
        return x

        

class ScrewClassifier(nn.Module):
    def __init__(self, num_classes=9):
        super(ScrewClassifier, self).__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv2d(1, 8, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.conv2 = nn.Sequential(
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )

        self.conv3 = nn.Sequential(
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2, stride=2)
        )
        
        self.fc1 = nn.Sequential(
            nn.Linear(32 * 10 * 10, 128),
            nn.ReLU()
        )
        
        self.fc2 = nn.Linear(128, num_classes)
        self.softmax = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.conv3(x)
        
        x = x.view(x.size(0), -1)
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.softmax(x)
        
        return x

##############
# model parameter: 3355467145
class ScrewClassifier_toobig(nn.Module):
    def __init__(self, num_classes):
        super(ScrewClassifier, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool2d(2, 2)
        self.fc1 = nn.Linear(64 * 320 * 320, 512)
        self.relu3 = nn.ReLU()
        self.fc2 = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu1(x)
        x = self.conv2(x)
        x = self.relu2(x)
        x = self.pool(x)
        x = x.view(-1, 64 * 320 * 320)
        x = self.fc1(x)
        x = self.relu3(x)
        x = self.fc2(x)
        return x
