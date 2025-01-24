import torch
import torch.nn as nn
import torch.nn.functional as F


class inconv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(inconv, self).__init__()
        self.conv = nn.Conv2d(in_channel, out_channel,
                              kernel_size=4, stride=2, padding=1, bias=False)

    def forward(self, x):
        x = self.conv(x)
        return x


class lrl_conv_bn(nn.Module):
    '''Leaky ReLU -> Conv -> BN'''

    def __init__(self, in_ch, out_ch):
        super(lrl_conv_bn, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class lrl_conv_bn_triple(nn.Module):
    '''Leaky ReLU -> Conv -> BN'''

    def __init__(self, in_ch, out_ch):
        super(lrl_conv_bn_triple, self).__init__()
        self.conv_bn = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(out_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv_bn(x)
        return x


class lrl_conv(nn.Module):
    '''Leaky ReLU -> Conv'''

    def __init__(self, in_ch, out_ch):
        super(lrl_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.LeakyReLU(negative_slope=0.2, inplace=True),
            nn.Conv2d(in_ch, out_ch, kernel_size=4,
                      stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class rl_convT_bn(nn.Module):
    '''ReLU -> ConvT -> BN'''

    def __init__(self, in_ch, out_ch):
        super(rl_convT_bn, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class rl_convT_bn_triple(nn.Module):
    '''ReLU -> ConvT -> BN'''

    def __init__(self, in_ch, out_ch):
        super(rl_convT_bn_triple, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(out_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
            nn.BatchNorm2d(out_ch)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class rl_convT(nn.Module):
    '''ReLU -> ConvT'''

    def __init__(self, in_ch, out_ch):
        super(rl_convT, self).__init__()
        self.conv = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(in_ch, out_ch, kernel_size=4,
                               stride=2, padding=1, bias=False),
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class ST_CGAN(nn.Module):
    def __init__(self, in_ch=4, out_ch=3):
        super(ST_CGAN, self).__init__()
        self.inc = inconv(in_ch, 64)
        self.conv_1 = lrl_conv_bn(64, 128)
        self.conv_2 = lrl_conv_bn(128, 256)
        self.conv_3 = lrl_conv_bn(256, 512)
        self.conv_4 = lrl_conv_bn_triple(512, 512)
        self.conv_5 = lrl_conv(512, 512)
        self.conv_T6 = rl_convT_bn(512, 512)
        self.conv_T7 = rl_convT_bn_triple(1024, 512)
        self.conv_T8 = rl_convT_bn(1024, 256)
        self.conv_T9 = rl_convT_bn(512, 128)
        self.conv_T10 = rl_convT_bn(256, 64)
        self.conv_T11 = rl_convT(128, out_ch)

    def forward(self, inp, mas):
        cv0 = self.inc(torch.cat((inp, mas), dim=1))
        cv1 = self.conv_1(cv0)
        cv2 = self.conv_2(cv1)
        cv3 = self.conv_3(cv2)
        cv4 = self.conv_4(cv3)
        cv5 = self.conv_5(cv4)
        cvT6 = self.conv_T6(cv5)
        input7 = torch.cat([cvT6, cv4], dim=1)
        cvT7 = self.conv_T7(input7)
        input8 = torch.cat([cvT7, cv3], dim=1)
        cvT8 = self.conv_T8(input8)
        input9 = torch.cat([cvT8, cv2], dim=1)
        cvT9 = self.conv_T9(input9)
        input10 = torch.cat([cvT9, cv1], dim=1)
        cvT10 = self.conv_T10(input10)
        input11 = torch.cat([cvT10, cv0], dim=1)
        cvT11 = self.conv_T11(input11)
        out = torch.tanh(cvT11)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    m = torch.rand(1, 1, 256, 256)
    model = ST_CGAN()
    model.eval()
    with torch.no_grad():
        res = model(x, m)
        print(res.shape)