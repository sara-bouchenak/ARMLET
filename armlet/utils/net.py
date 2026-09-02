import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import ResNet, VGG
from torchvision.models.resnet import BasicBlock
from torchvision.models.vgg import make_layers


class MLP(nn.Module):
    def __init__(
        self,
        input_size: int,
        num_classes: int,
        hidden_sizes: list[int] = [64, 32],
    ):
        super(MLP, self).__init__()
        assert len(hidden_sizes) > 1

        self.features = nn.Sequential()
        last_hidden_size = input_size
        for hidden_size in hidden_sizes:
            self.features.append(nn.Linear(last_hidden_size, hidden_size))
            self.features.append(nn.ReLU())
            last_hidden_size = hidden_size

        self.classifier = nn.Linear(last_hidden_size, num_classes)

    def forward(self, x):
        x = self.features(x)
        y_hat = self.classifier(x)
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


class PurchaseMLP(nn.Module):
    def __init__(
        self,
        input_size: int = 600,
        num_classes: int = 100,
        hidden_sizes: list[int] = [1024, 512, 256, 128],
        droprate: float = 0.0,
    ):
        super().__init__()
        assert len(hidden_sizes) > 1

        self.features = nn.Sequential()
        last_hidden_size = input_size
        for hidden_size in hidden_sizes:
            self.features.append(nn.Linear(last_hidden_size, hidden_size))
            self.features.append(nn.Tanh())
            last_hidden_size = hidden_size

        if droprate > 0:
            self.classifier = nn.Sequential(
                nn.Dropout(droprate),
                nn.Linear(last_hidden_size, num_classes),
            )
        else:
            self.classifier = nn.Linear(last_hidden_size, num_classes)

    def forward(self, x):
        return self.classifier(self.features(x))


class EuroSATVGG11(nn.Module):
    def __init__(
        self,
        num_classes: int = 10,
        linear_size: int = 2048,
        group_norm_groups: int = 4,
        **kwargs,
    ):
        super().__init__()
        self.linear_size = int(linear_size)
        self.group_norm_groups = int(group_norm_groups)
        cfg = [64, "M", 128, "M", 256, 256, "M", 512, 512, "M", 512, 512, "M"]
        self.features = self._make_layers(cfg)
        self.classifier = nn.Linear(self.linear_size, int(num_classes))

    def forward(self, x):
        out = self.features(x)
        out = out.reshape(out.size(0), -1)
        return self.classifier(out)

    def _make_layers(self, cfg):
        layers = []
        in_channels = 3
        for width in cfg:
            if width == "M":
                layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
            else:
                layers.extend(
                    [
                        nn.Conv2d(in_channels, width, kernel_size=3, padding=1),
                        nn.GroupNorm(self.group_norm_groups, width),
                        nn.ReLU(inplace=False),
                    ]
                )
                in_channels = width
        layers.append(nn.AvgPool2d(kernel_size=1, stride=1))
        return nn.Sequential(*layers)


class EuroSATBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = _conv3x3(inplanes, planes, stride)
        self.gn1 = nn.GroupNorm(_num_groups(planes), planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = _conv3x3(planes, planes)
        self.gn2 = nn.GroupNorm(_num_groups(planes), planes)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.relu(self.gn1(self.conv1(x)))
        out = self.gn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        return self.relu(out)

def _conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


def _num_groups(num_channels):
    return 32 if num_channels % 32 == 0 else 1


class EuroSATResNet20(nn.Module):
    def __init__(self, num_classes: int = 10, fc_size: int = 256, **kwargs):
        super().__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, padding=1, bias=False)
        self.gn1 = nn.GroupNorm(_num_groups(16), 16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(EuroSATBasicBlock, 16, 3)
        self.layer2 = self._make_layer(EuroSATBasicBlock, 32, 3, stride=2)
        self.layer3 = self._make_layer(EuroSATBasicBlock, 64, 3, stride=2)
        self.avgpool = nn.AvgPool2d(8)
        self.fc = nn.Linear(int(fc_size), int(num_classes))
        self._initialize_weights()

    def forward(self, x):
        x = self.relu(self.gn1(self.conv1(x)))
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.GroupNorm(_num_groups(planes * block.expansion), planes * block.expansion),
            )

        layers = [block(self.inplanes, planes, stride, downsample)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                n = module.kernel_size[0] * module.kernel_size[1] * module.out_channels
                module.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(module, nn.GroupNorm):
                module.weight.data.fill_(1)
                module.bias.data.zero_()


class M18(nn.Module):
    """M18 1D CNN for raw Speech Commands waveforms.

    Expected input shape is [batch_size, 1, 16000]. The model returns raw
    logits with shape [batch_size, num_classes] for CrossEntropyLoss.
    """

    def __init__(
        self,
        input_size: int | None,
        num_classes: int,
        n_input: int = 1,
        stride: int = 4,
        n_channel: int = 64,
    ):
        super().__init__()

        self.conv1 = nn.Conv1d(n_input, n_channel, kernel_size=80, stride=stride)
        self.bn1 = nn.BatchNorm1d(n_channel)
        self.pool1 = nn.MaxPool1d(4)

        self.conv2 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(n_channel)
        self.conv3 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn3 = nn.BatchNorm1d(n_channel)
        self.conv4 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn4 = nn.BatchNorm1d(n_channel)
        self.conv5 = nn.Conv1d(n_channel, n_channel, kernel_size=3, padding=1)
        self.bn5 = nn.BatchNorm1d(n_channel)
        self.pool2 = nn.MaxPool1d(4)

        self.conv6 = nn.Conv1d(n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn6 = nn.BatchNorm1d(2 * n_channel)
        self.conv7 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn7 = nn.BatchNorm1d(2 * n_channel)
        self.conv8 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn8 = nn.BatchNorm1d(2 * n_channel)
        self.conv9 = nn.Conv1d(2 * n_channel, 2 * n_channel, kernel_size=3, padding=1)
        self.bn9 = nn.BatchNorm1d(2 * n_channel)
        self.pool3 = nn.MaxPool1d(4)

        self.conv10 = nn.Conv1d(2 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn10 = nn.BatchNorm1d(4 * n_channel)
        self.conv11 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn11 = nn.BatchNorm1d(4 * n_channel)
        self.conv12 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn12 = nn.BatchNorm1d(4 * n_channel)
        self.conv13 = nn.Conv1d(4 * n_channel, 4 * n_channel, kernel_size=3, padding=1)
        self.bn13 = nn.BatchNorm1d(4 * n_channel)
        self.pool4 = nn.MaxPool1d(4)

        self.conv14 = nn.Conv1d(4 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn14 = nn.BatchNorm1d(8 * n_channel)
        self.conv15 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn15 = nn.BatchNorm1d(8 * n_channel)
        self.conv16 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn16 = nn.BatchNorm1d(8 * n_channel)
        self.conv17 = nn.Conv1d(8 * n_channel, 8 * n_channel, kernel_size=3, padding=1)
        self.bn17 = nn.BatchNorm1d(8 * n_channel)

        self.fc1 = nn.Linear(8 * n_channel, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.pool1(x)

        x = F.relu(self.bn2(self.conv2(x)))
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = self.pool2(x)

        x = F.relu(self.bn6(self.conv6(x)))
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = F.relu(self.bn9(self.conv9(x)))
        x = self.pool3(x)

        x = F.relu(self.bn10(self.conv10(x)))
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = F.relu(self.bn13(self.conv13(x)))
        x = self.pool4(x)

        x = F.relu(self.bn14(self.conv14(x)))
        x = F.relu(self.bn15(self.conv15(x)))
        x = F.relu(self.bn16(self.conv16(x)))
        x = F.relu(self.bn17(self.conv17(x)))

        x = F.avg_pool1d(x, x.shape[-1])
        x = x.squeeze(-1)
        return self.fc1(x)
