import torch
import torch.nn as nn

from torchvision.models import ResNet, VGG
from torchvision.models.resnet import BasicBlock
from torchvision.models.vgg import make_layers


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        n_hidden_layer_1: int = 64,
        n_hidden_layer_2: int = 32,
    ):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, n_hidden_layer_1)
        self.fc2 = nn.Linear(n_hidden_layer_1, n_hidden_layer_2)
        self.fc3 = nn.Linear(n_hidden_layer_2, num_classes)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        y_hat = self.fc3(x)
        return y_hat


class LogRegression(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(LogRegression, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        x = self.linear(x)
        y_pred = torch.sigmoid(x)
        return y_pred


class SVM(nn.Module):
    def __init__(self, input_size: int, num_classes: int):
        super(SVM, self).__init__()
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        y_pred = self.linear(x)
        return y_pred


class ResNet18(ResNet):
    def __init__(self, input_size: int, num_classes: int) -> None:
        super().__init__(block=BasicBlock, layers=[2, 2, 2, 2], num_classes=num_classes)


class VGG11(VGG):
    def __init__(self, input_size: int, num_classes: int) -> None:
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        features = make_layers(cfg, batch_norm=True)
        super().__init__(features=features, num_classes=num_classes)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        #nn.AvgPool2d(kernel_size=1, stride=1)
        self.classifier = nn.Linear(512, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
