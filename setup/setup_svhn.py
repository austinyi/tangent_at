import torch.nn as nn

class SVHN(nn.Module):
    def __init__(self, features, n_channel, num_classes):
        super(SVHN, self).__init__()
        assert isinstance(features, nn.Sequential), type(features)
        self.features = features
        self.classifier = nn.Sequential(
            nn.Linear(n_channel, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

def make_layers(cfg, batch_norm=False):
    layers = []
    in_channels = 3
    for i, v in enumerate(cfg):
        if v == 'M':
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            padding = v[1] if isinstance(v, tuple) else 1
            out_channels = v[0] if isinstance(v, tuple) else v
            conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=padding)
            if batch_norm:
                layers += [conv2d, nn.BatchNorm2d(out_channels, affine=False), nn.ReLU()]
            else:
                layers += [conv2d, nn.ReLU()]
            in_channels = out_channels
    return nn.Sequential(*layers)

def SVHN32(n_channel=32):
    cfg = [
        n_channel, 'M',
        2*n_channel, 'M',
        4*n_channel, 'M',
        4*n_channel, 'M',
        (8*n_channel, 0), (8*n_channel, 0), 'M'
    ]
    layers = make_layers(cfg, batch_norm=True)
    model = SVHN(layers, n_channel=8*n_channel, num_classes=10)
    return model

class EncCla(nn.Module):
    def __init__(self, n_z, dim_h):
        super(EncCla, self).__init__()
        self.dim_h = dim_h
        self.n_z = n_z
        self.main1 = nn.Sequential( # 3 * 96 * 96
            nn.Conv2d(3, self.dim_h, 4, 2, 1, bias=False),
            nn.ReLU(True), # 48 * 48 
            nn.Conv2d(self.dim_h, self.dim_h * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 2),
            nn.ReLU(True), # 24 * 24
            nn.Conv2d(self.dim_h * 2, self.dim_h * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 4),
            nn.ReLU(True), # 12 * 12
            nn.Conv2d(self.dim_h * 4, self.dim_h * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 8),
            nn.ReLU(True), # 6 * 6
            nn.Conv2d(self.dim_h * 8, self.dim_h * 16, 4, 2, 1, bias=False),
            nn.BatchNorm2d(self.dim_h * 16),
            nn.ReLU(True), # 3 * 3
            nn.Conv2d(self.dim_h * 16, self.dim_h * 16, 3, 2, 0, bias=False),
            nn.BatchNorm2d(self.dim_h * 16),
            nn.ReLU(True), # 1 * 1
        )
        self.fc1 = nn.Linear(self.dim_h * (2 ** 4), self.n_z)
        
        self.classifier = nn.Sequential(
            nn.Linear(self.n_z, 500),
            nn.BatchNorm1d(500),
            nn.ReLU(True),
            nn.Linear(500, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Linear(256, 10),
        ) 

    def forward(self, x):
        x = self.main1(x)
        x = x.squeeze()
        x = self.fc1(x)
        x = self.classifier(x)
        return x    