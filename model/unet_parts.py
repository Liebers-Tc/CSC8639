import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    """ (3*3 Convolution -> ReLU) * 2 -> 2*2 Maxpooling """
    def __init__(self, in_channels, out_channels, pooling=True):
        super().__init__()

        self.use_pooling = pooling

        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )
        
        if self.use_pooling:
            self.pooling = nn.MaxPool2d(2)

    def forward(self, x):
        x = self.doubleconv(x)
        skip = x
        if self.use_pooling:
            x = self.pooling(x)
        return x, skip



class Decoder(nn.Module):
    """ 2*2 Upsample -> Skip connection -> (3*3 Convolution -> ReLU) * 2 """
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        
        if bilinear:
            self.up = nn.Sequential(
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
                )
        else:
            self.up = nn.ConvTranspose2d(in_channels, out_channels, kernel_size=2, stride=2)
        
        self.doubleconv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.ReLU(inplace=True)
            )

    def forward(self, x, skip):
        #print("Before upsample:", x.shape)
        x = self.up(x)
        #print("After upsample:", x.shape)

        # crop
        # diffY = skip.size(2) - x.size(2)
        # diffX = skip.size(3) - x.size(3)
        # x = F.pad(x, [diffX // 2, diffX - diffX // 2,
        #               diffY // 2, diffY - diffY // 2])

        x = torch.cat([skip, x], dim=1)
        return self.doubleconv(x)


class OutConv(nn.Module):
    """ Output layer """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.outconv = nn.Conv2d(in_channels, out_channels, kernel_size=1)

    def forward(self, x):
        return self.outconv(x)
