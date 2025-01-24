import torch
from torch import nn
import torch.nn.functional as F
from torchvision import models
import math
from torch.autograd import Function


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = models.resnext101_32x8d(weights='DEFAULT')
        net = list(net.children())
        self.layer0 = nn.Sequential(
            *net[:3])                   # 200 * 200 * 64
        self.layer1 = nn.Sequential(
            *net[3: 5])                 # 100 * 100 * 256
        self.layer2 = net[5]                                    # 50 * 50 * 512
        # 25 * 25 * 1024
        self.layer3 = net[6]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        return layer3

    def feature_extractor(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)

        return layer0, layer1, layer2, layer3


def weights_init(init_type='gaussian'):
    def init_fun(m):
        classname = m.__class__.__name__
        if (classname.find('Conv') == 0 or classname.find(
                'Linear') == 0) and hasattr(m, 'weight'):
            if init_type == 'gaussian':
                nn.init.normal_(m.weight, 0.0, 0.02)
            elif init_type == 'xavier':
                nn.init.xavier_normal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'kaiming':
                nn.init.kaiming_normal_(m.weight, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                nn.init.orthogonal_(m.weight, gain=math.sqrt(2))
            elif init_type == 'default':
                pass
            else:
                assert 0, "Unsupported initialization: {}".format(init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                nn.init.constant_(m.bias, 0.0)

    return init_fun


class ConvBlock(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation='lrelu', use_bias=True):
        super(ConvBlock, self).__init__()

        self.norm = nn.BatchNorm2d(output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

        self.conv = nn.Conv2d(input_dim, output_dim, kernel_size,
                              stride, padding=padding, bias=use_bias)
        self.conv.apply(weights_init('gaussian'))

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class Deconvolution(nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size, stride, padding=0, activation='lrelu', use_bias=True):
        super(Deconvolution, self).__init__()

        self.conv = nn.ConvTranspose2d(
            input_dim, output_dim, kernel_size, stride, padding, bias=use_bias)
        self.conv.apply(weights_init('gaussian'))

        self.norm = nn.BatchNorm2d(output_dim)

        if activation == 'relu':
            self.activation = nn.ReLU(inplace=True)
        elif activation == 'lrelu':
            self.activation = nn.LeakyReLU(0.2, inplace=True)
        elif activation == 'prelu':
            self.activation = nn.PReLU()
        elif activation == 'selu':
            self.activation = nn.SELU(inplace=True)
        elif activation == 'tanh':
            self.activation = nn.Tanh()
        elif activation == 'sigmoid':
            self.activation = nn.Sigmoid()
        elif activation == 'none':
            self.activation = None

    def forward(self, x):
        x = self.conv(x)
        x = self.norm(x)
        if self.activation:
            x = self.activation(x)
        return x


class ResBlock(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(ResBlock, self).__init__()

        self.conv = ConvBlock(input_dim, output_dim, 3, 1, 1, 'none', False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)

    def forward(self, x):
        y = self.conv(x)
        x = x + y
        x = self.lrelu(x)
        return x


class Resample2dFunction(Function):
    @staticmethod
    def forward(ctx, input1, input2, kernel_size=2, dilation=1):
        """
        input1: (B, C, H, W) – the feature map to be resampled
        input2: (B, 2, H2, W2) – the flow field (or offsets)
        kernel_size: 2  (used here to match bilinear sampling)
        dilation:    1  (not explicitly used here—bilinear sampling is effectively "2 x 2")
        """

        # Save tensors needed for backward
        # (we'll also save the computed grid, so we can compute grad in backward).
        ctx.save_for_backward(input1, input2)
        ctx.kernel_size = kernel_size
        ctx.dilation = dilation

        B, C, H, W = input1.shape
        _, _, H2, W2 = input2.shape

        # input2 is treated as a flow: input2[:,0,:,:] is flow in x-direction,
        # input2[:,1,:,:] is flow in y-direction.
        flow_x = input2[:, 0, :, :]  # shape (B, H2, W2)
        flow_y = input2[:, 1, :, :]  # shape (B, H2, W2)

        # Create a base meshgrid for indices 0..W2-1, 0..H2-1
        # (using indexing='ij').
        device = input1.device
        y_base, x_base = torch.meshgrid(
            torch.arange(H2, device=device),
            torch.arange(W2, device=device),
            indexing='ij'
        )
        # Convert to float and broadcast to (B, H2, W2)
        x_base = x_base.float().unsqueeze(0).expand(B, -1, -1)
        y_base = y_base.float().unsqueeze(0).expand(B, -1, -1)

        # Add flow to get the "warped" sampling coordinates in pixel space
        x_warped = x_base + flow_x
        y_warped = y_base + flow_y

        # Normalize coordinates to the range [-1, 1] for grid_sample
        # The factor (W-1) for x_warped and (H-1) for y_warped
        # is because we’re sampling from input1 which is (H x W).
        x_norm = 2.0 * x_warped / (W - 1.0) - 1.0
        y_norm = 2.0 * y_warped / (H - 1.0) - 1.0

        # Stack into a grid of shape (B, H2, W2, 2)
        grid = torch.stack((x_norm, y_norm), dim=-1)

        # Resample input1 using bilinear sampling
        # align_corners=True here matches traditional optical-flow style warping
        # but you can set it to False depending on your needs.
        output = F.grid_sample(
            input1,
            grid,
            mode='bilinear',
            padding_mode='zeros',
            align_corners=True
        )

        # Save the grid for backward
        ctx.grid = grid

        return output

    @staticmethod
    def backward(ctx, grad_output):
        """
        Manual backward. We replicate the main logic of grid_sample’s backward here
        by re-invoking grid_sample under a grad-enabled context, then letting PyTorch
        compute the derivative. This recovers grad wrt input1. For the original code,
        only grad_input1 is returned. 
        """
        input1, input2 = ctx.saved_tensors
        grid = ctx.grid
        kernel_size = ctx.kernel_size  # not explicitly used here
        dilation = ctx.dilation        # not explicitly used here

        # Ensure grad_output is contiguous
        # (often unnecessary if the training loop ensures it, but we can do it here).
        if not grad_output.is_contiguous():
            grad_output = grad_output.contiguous()

        # We want the gradient with respect to input1 only,
        # so we’ll create a computation graph that can produce d(output)/d(input1).
        with torch.enable_grad():
            # Mark input1 as requiring grad
            input1_detached = input1.detach().clone().requires_grad_(True)
            # Re-run the forward pass with this graph-enabled tensor
            tmp_out = F.grid_sample(
                input1_detached,
                grid,
                mode='bilinear',
                padding_mode='zeros',
                align_corners=True
            )
            # Backprop from the user-provided grad_output
            tmp_out.backward(grad_output)

        # The gradient wrt input2 is not returned in the original code;
        # if you need it, you could similarly call autograd.grad or track input2.
        grad_input1 = input1_detached.grad

        # Return gradients for input1, input2, and the last two are
        # for kernel_size and dilation, which receive no gradient.
        return grad_input1, None, None


class Resample2d(nn.Module):
    def __init__(self, kernel_size=2, dilation=1, sigma=5 ):
        super(Resample2d, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.sigma = torch.tensor(sigma, dtype=torch.float).cuda()

    def forward(self, input1, input2):
        input1_c = input1.contiguous()
        sigma = self.sigma.expand(input2.size(0), 1, input2.size(2), input2.size(3)).type(input2.dtype)
        input2 = torch.cat((input2,sigma), 1)
        return Resample2dFunction.apply(input1_c, input2, self.kernel_size, self.dilation)
    

class CANet(nn.Module):
    def __init__(self):
        super(CANet, self).__init__()

        self.resnet = ResNeXt101()
        self.conv5 = ConvBlock(1024, 1024, 3, 2, 1, 'lrelu', False)
        self.bottle6 = ConvBlock(1024, 512, 1, 1, 0, 'lrelu', False)
        self.CFT = Resample2d(kernel_size=2, dilation=1, sigma=1)
        self.deconv6 = Deconvolution(1024, 1024, 3, 2, 1, 'lrelu', False)

        # self.bottle7 = ConvBlock(1024, 512, 1, 1, 0, 'lrelu', False)
        # self.resample7 = Resample2d(kernel_size=4, dilation=1, sigma=1)
        self.res7 = ResBlock(2048, 2048)
        self.deconv7 = Deconvolution(2048, 512, 4, 2, 1, 'lrelu', False)

        # self.bottle8 = ConvBlock(512, 256, 1, 1, 0, 'lrelu', False)
        # self.resample8 = Resample2d(kernel_size=4, dilation=1, sigma=1)

        self.res8 = ResBlock(1024, 1024)
        self.deconv8 = Deconvolution(1024, 256, 4, 2, 1, 'lrelu', False)

        self.res9 = ResBlock(512, 512)
        self.deconv9 = Deconvolution(512, 64, 4, 2, 1, 'lrelu', False)

        self.res10 = ResBlock(128, 128)
        self.deconv10 = Deconvolution(128, 64, 4, 2, 1, 'lrelu', False)

        self.predict_l_1 = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.lrelu = nn.LeakyReLU(0.2, inplace=True)
        self.predict_l_2 = nn.Conv2d(64, 1, 1, 1, padding=0, bias=False)

        self.predict_ab_1 = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.predict_ab_2 = nn.Conv2d(64, 2, 1, 1, padding=0, bias=False)

        # stage 2

        self.resnet2 = ResNeXt101()
        self.channel_change = ConvBlock(6, 3, 1, 1, 0, 'none', False)

        self.conv_5 = ConvBlock(1024, 1024, 3, 2, 1, 'lrelu', False)
        self.deconv_6 = Deconvolution(1024, 1024, 3, 2, 1, 'lrelu', False)

        self.res_7 = ResBlock(2048, 2048)
        self.deconv_7 = Deconvolution(2048, 512, 4, 2, 1, 'lrelu', False)

        self.res_8 = ResBlock(1024, 1024)
        self.deconv_8 = Deconvolution(1024, 256, 4, 2, 1, 'lrelu', False)

        self.res_9 = ResBlock(512, 512)
        self.deconv_9 = Deconvolution(512, 64, 4, 2, 1, 'lrelu', False)

        self.res_10 = ResBlock(128, 128)
        self.deconv_10 = Deconvolution(128, 64, 4, 2, 1, 'lrelu', False)

        self.predict = nn.Conv2d(64, 64, 1, 1, padding=0, bias=False)
        self.predict_final = nn.Conv2d(64, 3, 1, 1, padding=0, bias=False)

    def forward(self, x, flow_13_1, flow_13_2, flow_13_3):
        f1, f2, f3, f4 = self.resnet.feature_extractor(x)
        f5 = self.conv5(f4)

        f6_temp = self.bottle6(f5)
        f6_temp = torch.cat([f6_temp, self.CFT(
            f6_temp, flow_13_1) + self.CFT(f6_temp, flow_13_2) + self.CFT(f6_temp, flow_13_3)], dim=1)
        f6 = self.deconv6(f6_temp)

        # f7_temp = self.bottle6(f4)
        # f7_temp = torch.cat([f7_temp, self.resample7(f7_temp)], dim=1)
        f7 = self.res7(torch.cat([f6, f4], dim=1))
        f7 = self.deconv7(f7)

        # f8_temp = self.bottle8(f3)
        # f8_temp = torch.cat([f8_temp, self.resample8(f8_temp)], dim=1)
        f8 = self.res8(torch.cat([f7, f3], dim=1))
        f8 = self.deconv8(f8)

        f9 = self.res9(torch.cat([f8, f2], dim=1))
        f9 = self.deconv9(f9)

        f10 = self.res10(torch.cat([f9, f1], dim=1))
        f10 = self.deconv10(f10)

        pre_l = self.lrelu(self.predict_l_1(f10))
        pre_l = self.predict_l_2(pre_l)

        pre_ab = self.lrelu(self.predict_ab_1(f10))
        pre_ab = self.predict_ab_2(pre_ab)

        predict_stage1 = torch.cat([pre_l, pre_ab], dim=1)

        input2 = torch.cat([predict_stage1, x], dim=1)

        f1, f2, f3, f4 = self.resnet2.feature_extractor(
            self.channel_change(input2))
        f5 = self.conv_5(f4)
        # f1:200 200 64     f2:100 100 256      # f3:50 50 512      f4:25 25 1024     f5:12 12 1024

        f6 = self.deconv_6(f5)

        f7 = self.res_7(torch.cat([f6, f4], dim=1))
        f7 = self.deconv_7(f7)

        f8 = self.res_8(torch.cat([f7, f3], dim=1))
        f8 = self.deconv_8(f8)

        f9 = self.res_9(torch.cat([f8, f2], dim=1))
        f9 = self.deconv_9(f9)

        f10 = self.res_10(torch.cat([f9, f1], dim=1))
        f10 = self.deconv_10(f10)

        predict = self.lrelu(self.predict(f10))
        predict_stage2 = self.predict_final(predict)

        return predict_stage1, predict_stage2
