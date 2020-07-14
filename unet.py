import torch
import torch.nn as nn
import torch.nn.functional as F


# https://github.com/milesial/Pytorch-UNet
# https://github.com/jaxony/unet-pytorch


class DoubleConv(nn.Module):
    '''(conv => BN => ReLU) * 2'''
    def __init__(self, in_channels, out_channels):
        super(DoubleConv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )
        
    def forward(self, x):
        x = self.conv(x)
        return x


class InputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(InputConv, self).__init__()
        self.conv = DoubleConv(in_channels, out_channels)
        
    def forward(self, x):
        x = self.conv(x)
        return x


class DownSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(DownSample, self).__init__()
        self.max_pool = nn.MaxPool2d(2)
        self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x):
        x = self.max_pool(x)
        x = self.conv(x)
        return x


class UpSample(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(UpSample, self).__init__()
        
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, 2, stride=2)
            
        self.conv = DoubleConv(2*out_channels, out_channels)

    def forward(self, x_down, x_up):
        # upsample
        x_up = self.upsample(x_up)
        # adjust downsampled feature map
        offset_x = x_up.size(3) - x_up.size(3)
        offset_y = x_up.size(2) - x_up.size(2)
        x_down = F.pad(x_down, (offset_x//2, int(offset_x/2), offset_y//2, int(offset_y/2)))
        # concat
        x = torch.cat([x_up, x_down], dim=1)
        # conv
        x = self.conv(x)
        return x


class OutputConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(OutputConv, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, 1)
        
    def forward(self, x):
        x = self.conv(x)
        return x
    

class UNet(nn.Module):
    def __init__(self, in_channels, num_classes, init_fs=64, depth=4):
        super(UNet, self).__init__()

        self.in_conv = InputConv(in_channels, init_fs)

        downs = []
        for i in range(depth):
            in_ch = init_fs * (2**i)
            out_ch = init_fs * (2**(i+1))
            down = DownSample(in_ch, out_ch)
            downs.append(down)
        self.downs = nn.ModuleList(downs)
        
        ups = []
        for i in range(depth):
            in_ch = out_ch
            out_ch = in_ch // 2
            up = UpSample(in_ch, out_ch)
            ups.append(up)
        self.ups = nn.ModuleList(ups)
            
        self.out_conv = OutputConv(out_ch, num_classes)

        self.out = nn.Tanh()
        
        self.reset_params()
        
    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_normal_(m.weight)
            nn.init.constant_(m.bias, 0)

    def reset_params(self):
        for m in self.modules():
            self.init_weight(m)
                
    def forward(self, x):
        x = self.in_conv(x)
        
        encoder_outs = []
        for i, down in enumerate(self.downs):
            encoder_outs.append(x)            
            x = down(x)
        
        for i, up in enumerate(self.ups):
            x = up(encoder_outs[-(i+1)], x)

        x = self.out_conv(x)
        x = self.out(x)
        
        return x
