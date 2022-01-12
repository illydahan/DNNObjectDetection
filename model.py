import torch
import torch.nn as nn

#------------------------------------#
# (type, filter, stride, kernel_size)
#------------------------------------#

model_layout = [
    ("C", 3, 64, 2, 7),
    "M",
    ("C", 64, 192, 1, 3),
    "M",
    ("C", 192, 128, 1, 1),
    ("C", 128, 256, 1, 3),
    ("C", 256, 256, 1, 1),
    ("C", 256, 512, 1, 3),
    "M",
    [("C", 512, 256, 1, 1), ("C", 256, 512, 1, 3), 4],
    ("C", 512, 512, 1, 1),
    ("C", 512, 1024, 1, 3),
    "M",
    [("C", 1024, 512, 1, 1), ("C", 512, 1024, 1, 3), 2],
    ("C", 1024, 1024, 1, 3),
    ("C", 1024, 1024, 2, 3),
    ("C", 1024, 1024, 1, 3),
    ("C", 1024, 1024, 1, 3)
]



class ConvBlock(nn.Module):
    def __init__(self, in_features, out_features, k_size, stride=1):
        super().__init__()

        self.conv = nn.Conv2d(in_channels = in_features, out_channels=out_features, kernel_size=k_size, stride=stride, padding = 1 if k_size == 3 else 0)
        self.bn = nn.BatchNorm2d(out_features)
        self.leaky_relu = nn.LeakyReLU(0.2)
        
    def forward(self, x):
        x = self.leaky_relu(self.bn(self.conv(x)))
        return x
    
class YoloV1(nn.Module):
    
    def __init__(self, in_channels = 3, n_classes = 8, n_grid = 7, n_bbox=2):
        super().__init__()
        
        self.backbone = self._init_feature_extractor(model_layout, in_channels)

        self.fc = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(n_grid*n_grid*1024, 496),
            nn.LeakyReLU(0.1),
            nn.Linear(496, n_grid * n_grid * (n_classes + 5*n_bbox))
        )
        
    def forward(self, x):
        for layer in self.backbone:
            x = layer(x)
        
        out = self.fc(x)
        
        return out
            
    
    def _init_feature_extractor(self, cfg, in_channels = 3):
        
        backbone = nn.ModuleList()
        

        for block in cfg:
            if isinstance(block, tuple):

                _, in_c, out_c, stride, k_size = block
                backbone += [ConvBlock(in_features=in_c, out_features=out_c, k_size=k_size, stride=stride)]
                
            
            elif isinstance(block, list):
                n_repeats = block[-1]
                
                for idx in range(n_repeats):
                    _, in_c, out_c, stride, k_size = block[0]
                    backbone += [ConvBlock(in_features=in_c, out_features=out_c, k_size=k_size, stride=stride)]
                    _, in_c, out_c, stride, k_size = block[1]
                    backbone += [ConvBlock(in_features=in_c, out_features=out_c, k_size=k_size, stride=stride)]

            
            elif isinstance(block, str):
                backbone += [nn.MaxPool2d(kernel_size=2, stride=2)]
                
                
            else:
                raise Exception("Invalid config file entry")
            
        
        return backbone
    


def test_model(device = 'cuda'):
    model = YoloV1().to(device)
    
    sample = torch.rand((16, 3, 448, 448)).to(device)
    
    out = model(sample)
    
    print(out.shape)

#test_model()