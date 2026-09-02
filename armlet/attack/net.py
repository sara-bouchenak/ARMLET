import torch.nn as nn


class AttackMLP(nn.Module):
    def __init__(self, input_size, hidden_size=64, out_classes=2):
        super(AttackMLP, self).__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_size, out_classes)
        )

    def forward(self, x):
        out = self.classifier(x)
        return out
