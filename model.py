import torch
import torch.nn as nn
import torch.nn.functional as F


class STN(nn.Module):

    def __init__(self):
        super(STN, self).__init__()
        self.localization = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 8, 7),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            ),
            nn.Sequential(
                nn.Conv2d(8, 10, 5),
                nn.MaxPool2d(2, stride=2),
                nn.ReLU(True)
            )
        )
        self.regressor = nn.Sequential(
            nn.Linear(10 * 12 * 12, 32),
            nn.ReLU(True),
            nn.Linear(32, 3 * 2)
        )
        self.regressor[2].weight.data.zero_()
        self.regressor[2].bias.data.copy_(torch.tensor([1, 0, 0, 0, 1, 0], dtype=torch.float))

    def forward(self, x):
        xs = self.localization(x)
        xs = xs.view(-1, 10 * 12 * 12)
        theta = self.regressor(xs)
        theta = theta.view(-1, 2, 3)
        grid = F.affine_grid(theta, x.size())
        x = F.grid_sample(x, grid)
        return x


class COVID19Net(nn.Module):

    def __init__(self):
        super(COVID19Net, self).__init__()
        self.affine = STN()
        self.features = nn.Sequential(
            nn.Sequential(
                nn.Conv2d(1, 6, 5),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(6),
                nn.ReLU()
            ),
            nn.Sequential(
                nn.Conv2d(6, 16, 5),
                nn.MaxPool2d(2, 2),
                nn.BatchNorm2d(16),
                nn.ReLU()
            ),
        )
        self.classifier = nn.Sequential(
            nn.Sequential(
                nn.Linear(16 * 13 * 13, 120),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(120, 84),
                nn.ReLU(),
                nn.Dropout(0.2)
            ),
            nn.Sequential(
                nn.Linear(84, 1),
                nn.Sigmoid()
            )
        )

    def forward(self, x):
        x = self.affine(x)
        x = self.features(x)
        x = x.view(-1, 16 * 13 * 13)
        x = self.classifier(x)
        return x
