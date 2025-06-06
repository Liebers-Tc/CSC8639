from .unet_parts import *


class UNet(nn.Module):
    def __init__(self, in_channels=3, num_classes=104, base_channels=64, bilinear=False):
        super(UNet, self).__init__()

        # Encoder
        self.enc1 = Encoder(in_channels, base_channels)              # 3 -> 64
        self.enc2 = Encoder(base_channels, base_channels * 2)        # 64 -> 128
        self.enc3 = Encoder(base_channels * 2, base_channels * 4)    # 128 -> 256
        self.enc4 = Encoder(base_channels * 4, base_channels * 8)    # 256 -> 512

        # Bottleneck without pooling
        self.bottleneck = Encoder(base_channels * 8, base_channels * 16, pooling=False)  # 512 -> 1024

        # Decoder with skip connection
        self.dec4 = Decoder(base_channels * 16, base_channels * 8, bilinear=bilinear)  # 1024 + 512 -> 512
        self.dec3 = Decoder(base_channels * 8, base_channels * 4, bilinear=bilinear)   # 512 + 256 -> 256
        self.dec2 = Decoder(base_channels * 4, base_channels * 2, bilinear=bilinear)   # 256 + 128 -> 128
        self.dec1 = Decoder(base_channels * 2, base_channels, bilinear=bilinear)       # 128 + 64 -> 64

        # Output
        self.out_conv = OutConv(base_channels, num_classes)  # 64 -> num_classes

    def forward(self, x):
        #print("input.shape:", x.shape)

        # Encoder
        x, skip1 = self.enc1(x)
        #print("x1.shape:", skip1.shape)
        x, skip2 = self.enc2(x)
        #print("x2.shape:", skip2.shape)
        x, skip3 = self.enc3(x)
        #print("x3.shape:", skip3.shape)
        x, skip4 = self.enc4(x)
        #print("x4.shape:", skip4.shape)

        # Bottleneck
        x, _ = self.bottleneck(x)
        #print("x5.shape:", x.shape)

        # Decoder
        x = self.dec4(x, skip4)
        #print("x_dec4.shape:", x.shape)
        x = self.dec3(x, skip3)
        #print("x_dec3.shape:", x.shape)
        x = self.dec2(x, skip2)
        #print("x_dec2.shape:", x.shape)
        x = self.dec1(x, skip1)
        #print("x_dec1.shape:", x.shape)

        x = self.out_conv(x)
        #print("out_conv.shape:", x.shape)

        return x