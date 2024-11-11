import torch
import torch.nn as nn
import torchvision.models as models
from torch.autograd import Variable
import numpy as np
import os
import torch.nn.functional as F


class VGG19(nn.Module):
    def __init__(self):
        super(VGG19, self).__init__()
        # https://pytorch.org/hub/pytorch_vision_vgg/
        mean = np.array(
            [0.485, 0.456, 0.406], dtype=np.float32)
        mean = mean.reshape((1, 3, 1, 1))
        self.mean = Variable(torch.from_numpy(mean))
        std = np.array(
            [0.229, 0.224, 0.225], dtype=np.float32)
        std = std.reshape((1, 3, 1, 1))
        self.std = Variable(torch.from_numpy(std))
        self.initial_model()

    def forward(self, x):
        relu1_1 = self.relu1_1(
            (x-self.mean.to(x.device))/self.std.to(x.device))
        relu1_2 = self.relu1_2(relu1_1)

        relu2_1 = self.relu2_1(relu1_2)
        relu2_2 = self.relu2_2(relu2_1)

        relu3_1 = self.relu3_1(relu2_2)
        relu3_2 = self.relu3_2(relu3_1)
        relu3_3 = self.relu3_3(relu3_2)
        relu3_4 = self.relu3_4(relu3_3)

        relu4_1 = self.relu4_1(relu3_4)
        relu4_2 = self.relu4_2(relu4_1)
        relu4_3 = self.relu4_3(relu4_2)
        relu4_4 = self.relu4_4(relu4_3)

        relu5_1 = self.relu5_1(relu4_4)
        relu5_2 = self.relu5_2(relu5_1)
        relu5_3 = self.relu5_3(relu5_2)
        relu5_4 = self.relu5_4(relu5_3)

        out = {
            'relu1_1': relu1_1,
            'relu1_2': relu1_2,

            'relu2_1': relu2_1,
            'relu2_2': relu2_2,

            'relu3_1': relu3_1,
            'relu3_2': relu3_2,
            'relu3_3': relu3_3,
            'relu3_4': relu3_4,

            'relu4_1': relu4_1,
            'relu4_2': relu4_2,
            'relu4_3': relu4_3,
            'relu4_4': relu4_4,

            'relu5_1': relu5_1,
            'relu5_2': relu5_2,
            'relu5_3': relu5_3,
            'relu5_4': relu5_4,
        }
        return out

    def load_pretrained(self, vgg19_weights_path, gpu):
        if os.path.exists(vgg19_weights_path):
            if torch.cuda.is_available():
                data = torch.load(vgg19_weights_path)
                print("load vgg_pretrained_model:"+vgg19_weights_path)
            else:
                data = torch.load(vgg19_weights_path,
                                  map_location=lambda storage, loc: storage)
            self.initial_model(data)
            self.to(gpu)
        else:
            print("you need download vgg_pretrained_model in the directory of  "+str(self.config.DATA_ROOT) +
                  "\n'vgg19': 'https://download.pytorch.org/models/vgg19-dcbb9e9d.pth'")
            raise Exception("Don't load vgg_pretrained_model")

    def initial_model(self, data=None):
        vgg19 = models.vgg19()
        if data is not None:
            vgg19.load_state_dict(data)
        features = vgg19.features
        self.relu1_1 = torch.nn.Sequential()
        self.relu1_2 = torch.nn.Sequential()

        self.relu2_1 = torch.nn.Sequential()
        self.relu2_2 = torch.nn.Sequential()

        self.relu3_1 = torch.nn.Sequential()
        self.relu3_2 = torch.nn.Sequential()
        self.relu3_3 = torch.nn.Sequential()
        self.relu3_4 = torch.nn.Sequential()

        self.relu4_1 = torch.nn.Sequential()
        self.relu4_2 = torch.nn.Sequential()
        self.relu4_3 = torch.nn.Sequential()
        self.relu4_4 = torch.nn.Sequential()

        self.relu5_1 = torch.nn.Sequential()
        self.relu5_2 = torch.nn.Sequential()
        self.relu5_3 = torch.nn.Sequential()
        self.relu5_4 = torch.nn.Sequential()

        for x in range(2):
            self.relu1_1.add_module(str(x), features[x])

        for x in range(2, 4):
            self.relu1_2.add_module(str(x), features[x])

        for x in range(4, 7):
            self.relu2_1.add_module(str(x), features[x])

        for x in range(7, 9):
            self.relu2_2.add_module(str(x), features[x])

        for x in range(9, 12):
            self.relu3_1.add_module(str(x), features[x])

        for x in range(12, 14):
            self.relu3_2.add_module(str(x), features[x])

        for x in range(14, 16):
            self.relu3_3.add_module(str(x), features[x])

        for x in range(16, 18):
            self.relu3_4.add_module(str(x), features[x])

        for x in range(18, 21):
            self.relu4_1.add_module(str(x), features[x])

        for x in range(21, 23):
            self.relu4_2.add_module(str(x), features[x])

        for x in range(23, 25):
            self.relu4_3.add_module(str(x), features[x])

        for x in range(25, 27):
            self.relu4_4.add_module(str(x), features[x])

        for x in range(27, 30):
            self.relu5_1.add_module(str(x), features[x])

        for x in range(30, 32):
            self.relu5_2.add_module(str(x), features[x])

        for x in range(32, 34):
            self.relu5_3.add_module(str(x), features[x])

        for x in range(34, 36):
            self.relu5_4.add_module(str(x), features[x])


def avgcov2d_layer(pool_kernel_size, pool_stride, in_channels, channels, conv_kernel_size=3, conv_stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU"):
    layer = []
    layer.append(nn.AvgPool2d(pool_kernel_size, pool_stride))
    layer.append(conv2d_layer(in_channels, channels, kernel_size=conv_kernel_size, stride=conv_stride,
                              padding=padding, dilation=dilation, norm=norm, activation_fn=activation_fn))
    return nn.Sequential(*layer)


class FeatureDecouplingModule(nn.Module):
    '''Shadow feature decoupling module'''

    def __init__(self, in_channels=64, channels=3):
        super(FeatureDecouplingModule, self).__init__()
        kernel_size = 1

        w = torch.randn(channels, in_channels, kernel_size, kernel_size)
        self.w0 = torch.nn.Parameter(torch.FloatTensor(
            self.normalize_to_0_1(w)), requires_grad=True)
        w = torch.randn(channels, in_channels, kernel_size, kernel_size)
        self.w1 = torch.nn.Parameter(torch.FloatTensor(
            self.normalize_to_0_1(w)), requires_grad=True)
        w = torch.zeros(channels, in_channels, kernel_size, kernel_size)
        self.w2 = torch.nn.Parameter(torch.FloatTensor(
            w), requires_grad=True)

        self.bias_proportion = torch.nn.Parameter(torch.zeros(
            (channels, 1,  1, 1)), requires_grad=True)

        self.alpha_0 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)
        self.alpha_1 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)
        self.alpha_2 = torch.nn.Parameter(torch.ones(
            (1, channels, 1, 1)), requires_grad=True)

        self.bias_0 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)
        self.bias_1 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)
        self.bias_2 = torch.nn.Parameter(torch.zeros(
            (1, channels, 1, 1)), requires_grad=True)

    def forward(self, x):
        w0 = self.w0
        w1 = self.w1
        o, c, k_w, k_h = w0.shape

        w0_attention = 1+self.w2
        w1_attention = 1-self.w2

        w = w0*w0_attention
        median_w = torch.median(w, dim=1, keepdim=True)
        w0_correct = F.relu(w-median_w.values+self.bias_proportion)

        w0_correct = self.normalize_to_0_1(w0_correct)

        w = w1*w1_attention
        median_w = torch.median(w, dim=1, keepdim=True)
        w1_correct = F.relu(w-median_w.values-self.bias_proportion)
        w1_correct = self.normalize_to_0_1(w1_correct)

        w2_correct = w0_correct+w1_correct

        img = torch.sigmoid(self.alpha_0*F.conv2d(x, w0_correct)+self.bias_0)
        matte = torch.sigmoid(self.alpha_1*F.conv2d(x, w1_correct)+self.bias_1)
        img_free = torch.sigmoid(
            self.alpha_2*F.conv2d(x, w2_correct)+self.bias_2)

        return img, matte, img_free

    def normalize_to_0_1(self, w):
        w = w-w.min()
        w = w/w.max()
        return w
    

class DMTN(nn.Module):
    def __init__(self, in_channels=3, channels=64, norm="batch", stage_num=[6, 4]):
        super(DMTN, self).__init__()
        self.stage_num = stage_num

        # Pre-trained VGG19
        self.add_module('vgg19', VGG19())

        # SE
        cat_channels = in_channels+64+128+256+512+512
        self.se = nn.Sequential(SELayer(cat_channels),
                                conv2d_layer(cat_channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        self.down_sample = conv2d_layer(
            channels, 2*channels, kernel_size=4, stride=2, padding=1, dilation=1, norm=norm)

        # coarse
        coarse_list = []
        for i in range(self.stage_num[0]):
            coarse_list.append(SemiConvModule(
                2*channels, norm, mid_dilation=2**(i % 6)))
        self.coarse_list = nn.Sequential(*coarse_list)

        self.up_conv = conv2d_layer(
            2*channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm=norm)

        # fine
        fine_list = []
        for i in range(self.stage_num[1]):
            fine_list.append(SemiConvModule(
                channels, norm, mid_dilation=2**(i % 6)))
        self.fine_list = nn.Sequential(*fine_list)

        self.se_coarse = nn.Sequential(SELayer(2*channels),
                                       conv2d_layer(2*channels, channels, kernel_size=3,  padding=1, dilation=1, norm=norm))

        # SPP
        self.spp = SPP(channels, norm=norm)

        # Shadow feature decoupling module'
        self.FDM = FeatureDecouplingModule(in_channels=channels, channels=3)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])

        # vgg
        x_vgg = self.vgg19(x)

        # hyper-column features
        x_cat = torch.cat((
            x,
            F.interpolate(x_vgg['relu1_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu2_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu3_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu4_2'], size,
                          mode="bilinear", align_corners=True),
            F.interpolate(x_vgg['relu5_2'], size, mode="bilinear", align_corners=True)), dim=1)

        # SE
        x = self.se(x_cat)

        # coarse
        x_ = x
        x = self.down_sample(x)
        for i in range(self.stage_num[0]):
            x = self.coarse_list[i](x)

        size = (x_.shape[2], x_.shape[3])
        x = F.interpolate(x, size, mode="bilinear", align_corners=True)
        x = self.up_conv(x)

        # fine
        x = self.se_coarse(torch.cat((x_, x), dim=1))
        for i in range(self.stage_num[1]):
            x = self.fine_list[i](x)

        # spp
        x = self.spp(x)

        # output
        _, _, img_free = self.FDM(x)

        return img_free


class BatchNorm_(nn.Module):
    # https://github.com/vinthony/ghost-free-shadow-removal/blob/master/utils.py
    def __init__(self, channels):
        super(BatchNorm_, self).__init__()
        self.w0 = torch.nn.Parameter(
            torch.FloatTensor([1.0]), requires_grad=True)
        self.w1 = torch.nn.Parameter(
            torch.FloatTensor([0.0]), requires_grad=True)
        self.BatchNorm2d = nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False)

    def forward(self, x):
        outputs = self.w0*x+self.w1*self.BatchNorm2d(x)
        return outputs


def spectral_norm(module, mode=True):
    if mode:
        return nn.utils.spectral_norm(module)

    return module


def conv2d_layer(in_channels, channels, kernel_size=3, stride=1, padding=1, dilation=1, norm="batch", activation_fn="LeakyReLU", conv_mode="none", pad_mode="ReflectionPad2d"):
    """
    norm: batch, spectral, instance, spectral_instance, none

    activation_fn: Sigmoid, ReLU, LeakyReLU, none

    conv_mode: transpose, upsample, none

    pad_mode: ReflectionPad2d, ReplicationPad2d, ZeroPad2d
    """
    layer = []
    # padding
    if conv_mode == "transpose":
        pass
    else:
        if pad_mode == "ReflectionPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        elif pad_mode == "ReplicationPad2d":
            layer.append(nn.ReflectionPad2d(padding))
        else:
            layer.append(nn.ZeroPad2d(padding))
        padding = 0

    # conv layer
    if norm == "spectral" or norm == "spectral_instance":
        bias = False
        # conv
        if conv_mode == "transpose":
            conv_ = nn.ConvTranspose2d
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            conv_ = nn.Conv2d
        else:
            conv_ = nn.Conv2d
    else:
        bias = True
        # conv
        if conv_mode == "transpose":
            layer.append(nn.ConvTranspose2d(in_channels, channels, kernel_size,
                                            bias=bias, stride=stride, padding=padding, dilation=dilation))
        elif conv_mode == "upsample":
            layer.append(nn.Upsample(mode='bilinear', scale_factor=stride))
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))
        else:
            layer.append(nn.Conv2d(in_channels, channels, kernel_size,
                                   bias=bias, stride=stride, padding=padding, dilation=dilation))

    # norm
    if norm == "spectral":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
    elif norm == "instance":
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch":
        layer.append(nn.BatchNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "spectral_instance":
        layer.append(spectral_norm(conv_(in_channels, channels, kernel_size,
                                         stride=stride, bias=bias, padding=padding, dilation=dilation), True))
        layer.append(nn.InstanceNorm2d(
            channels, affine=True, track_running_stats=False))
    elif norm == "batch_":
        layer.append(BatchNorm_(channels))
    else:
        pass

    # activation_fn
    if activation_fn == "Sigmoid":
        layer.append(nn.Sigmoid())
    elif activation_fn == "ReLU":
        layer.append(nn.ReLU(True))
    elif activation_fn == "none":
        pass
    else:
        layer.append(nn.LeakyReLU(0.2, inplace=True))

    return nn.Sequential(*layer)


def solve_factor(num):
    # solve factors for a number
    list_factor = []
    i = 1
    if num > 2:
        while i <= num:
            i += 1
            if num % i == 0:
                list_factor.append(i)
            else:
                pass
    else:
        pass

    list_factor = list(set(list_factor))
    list_factor = np.sort(list_factor)
    return list_factor


class SemiConvModule(nn.Module):
    def __init__(self, channels=64, norm="batch", mid_dilation=2):
        super(SemiConvModule, self).__init__()
        list_factor = solve_factor(channels)
        self.group = list_factor[int(len(list_factor)/2)-1]
        self.split_channels = int(channels/2)

        # Conv
        self.conv_dilation = conv2d_layer(
            self.split_channels, self.split_channels, kernel_size=3,  padding=mid_dilation, dilation=mid_dilation, norm=norm)
        self.conv_3x3 = conv2d_layer(
            self.split_channels, self.split_channels, kernel_size=3,  padding=1, dilation=1, norm=norm)

    def forward(self, x):
        SSRD = False
        if SSRD:
            x_conv = x[:, self.split_channels:, :, :]
            x_identity = x[:, 0:self.split_channels, :, :]
        else:
            x_conv = x[:, 0:self.split_channels, :, :]
            x_identity = x[:, self.split_channels:, :, :]

        x_conv = x_conv+self.conv_dilation(x_conv)+self.conv_3x3(x_conv)

        x = torch.cat((x_identity, x_conv), dim=1)
        x = self.channel_shuffle(x)
        return x

    def channel_shuffle(self, x):
        batchsize, num_channels, height, width = x.data.size()
        assert num_channels % self.group == 0
        group_channels = num_channels // self.group

        x = x.reshape(batchsize, group_channels, self.group, height, width)
        x = x.permute(0, 2, 1, 3, 4)
        x = x.reshape(batchsize, num_channels, height, width)

        return x

    def identity(self, x):
        return x


class SPP(nn.Module):
    def __init__(self, channels=64, norm="batch"):
        super(SPP, self).__init__()
        self.net2 = avgcov2d_layer(
            4, 4, channels, channels, 1, padding=0, norm=norm)
        self.net8 = avgcov2d_layer(
            8, 8, channels, channels, 1, padding=0, norm=norm)
        self.net16 = avgcov2d_layer(
            16, 16, channels, channels, 1, padding=0, norm=norm)
        self.net32 = avgcov2d_layer(
            32, 32, channels, channels, 1, padding=0, norm=norm)
        self.output = conv2d_layer(channels*5, channels, 3, norm=norm)

    def forward(self, x):
        size = (x.shape[2], x.shape[3])
        x = torch.cat((
            F.interpolate(self.net2(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net8(x), size, mode="bilinear",
                          align_corners=True),
            F.interpolate(self.net16(x), size,
                          mode="bilinear", align_corners=True),
            F.interpolate(self.net32(x), size,
                          mode="bilinear", align_corners=True),
            x), dim=1)
        x = self.output(x)
        return x


class SELayer(nn.Module):
    def __init__(self, channel, reduction=8):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=True),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y


if __name__ == '__main__':
    model = DMTN()
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)