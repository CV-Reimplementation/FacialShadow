import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.transforms.functional import rgb_to_grayscale


class NonLocalBlock(nn.Module):
    def __init__(self, ch=32, out_ch=None, pool=False, norm='batch'):
        super(NonLocalBlock, self).__init__()
        self.out_ch = ch if out_ch is None else out_ch
        self.g     = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.phi   = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.w     = nn.Conv2d(ch // 2, self.out_ch, kernel_size=1, stride=1, padding=0)

        self.norm = norm
        self.bnorm = nn.BatchNorm2d(self.out_ch)

        self.pool = pool
        if pool:
            self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
            self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bsize, ch, h, w = x.size()
        if self.pool:
            h0 = h // 2
            w0 = w // 2
        else:
            h0 = h
            w0 = w
        out_ch = self.out_ch
        # g
        g_x = self.g(x)  # (bsize, ch//2, h, w)
        if self.pool:
            g_x = self.pool1(g_x)  # (bsize, ch//2, h0, w0)
        g_x = g_x.view(bsize, -1, h0 * w0)  # (bsize, ch//2, h0*w0)

        # phi
        phi_x = self.phi(x)  # (bsize, ch//2, h, w)
        if self.pool:
            phi_x = self.pool2(phi_x)  # (bsize, ch//2, h0, w0)
        phi_x = phi_x.view(bsize, -1, h0 * w0)  # (bsize, ch//2, h0*w0)
        phi_x = phi_x.permute(0, 2, 1)  # (bsize, h0*w0, ch//2)

        # theta
        theta_x = self.theta(x)  # (bsize, ch//2, h, w)
        if self.pool:
            theta_x = self.pool3(theta_x)  # (bsize, ch//2, h0, w0)
        theta_x = theta_x.view(bsize, -1, h0 * w0)  # (bsize, ch//2, h0*w0)

        f = torch.matmul(theta_x, phi_x)  # (bsize, h0*w0, h0*w0)
        f_softmax = F.softmax(f, dim=-1)
        y = torch.matmul(f_softmax, g_x)  # (bsize, h0*w0, ch//2)
        y = y.transpose(1, 2).contiguous().view(bsize, -1, h0, w0)  # (bsize, ch//2, h0, w0)

        if h0 != h or w0 != w:
            y = F.interpolate(y, size=(h, w), mode='bilinear', align_corners=False)

        w_y = self.w(y)
        if self.norm:
            w_y = self.bnorm(w_y)
        z = x + w_y

        return z

class Res(nn.Module):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(Res, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=ksize, stride=stride, padding=ksize//2)
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=ksize, stride=stride, padding=ksize//2)
        self.bnorm1 = nn.BatchNorm2d(ch)
        self.bnorm2 = nn.BatchNorm2d(ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.non_local = NonLocalBlock(ch, ch)

    def forward(self, x):
        y = self.conv1(self.bnorm1(x))
        y = self.relu1(y)
        y = self.conv2(self.bnorm2(y))
        y = self.relu2(x + y)
        y = self.non_local(y)
        return y

class ResBottleneck(nn.Module):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(ResBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.conv2 = nn.Conv2d(ch // 2, ch // 2, kernel_size=ksize, stride=stride, padding=ksize // 2)
        self.conv3 = nn.Conv2d(ch // 2, ch, kernel_size=1, stride=1, padding=0)
        self.bnorm1 = nn.BatchNorm2d(ch // 2)
        self.bnorm2 = nn.BatchNorm2d(ch // 2)
        self.bnorm3 = nn.BatchNorm2d(ch)
        self.relu1 = nn.LeakyReLU(inplace=True)
        self.relu2 = nn.LeakyReLU(inplace=True)
        self.relu3 = nn.LeakyReLU(inplace=True)
        self.stride = stride
        self.non_local = NonLocalBlock(ch, ch)
        if stride > 1:
            self.conv_red = nn.Conv2d(ch, ch, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        y = self.bnorm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        y = self.bnorm2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        y = self.bnorm3(y)
        y = self.non_local(y)
        if self.stride > 1:
            x = self.conv_red(x)
        if x.shape[1] < y.shape[1]:
            ch_add = y.shape[1] - x.shape[1]
            ch_pad = torch.zeros(x.size(0), ch_add, x.size(2), x.size(3), device=x.device)
            x = torch.cat([x, ch_pad], dim=1)
        elif y.shape[1] < x.shape[1]:
            ch_add = x.shape[1] - y.shape[1]
            ch_pad = torch.zeros(y.size(0), ch_add, y.size(2), y.size(3), device=y.device)
            y = torch.cat([y, ch_pad], dim=1)
        return self.relu3(x + y)

class Conv(nn.Module):
    def __init__(self, in_ch, out_ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False, name=None):
        super(Conv, self).__init__()
        self.norm = norm
        padding = ksize // 2
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size=ksize, stride=stride, padding=padding)
        if norm == 'batch':
            self.bnorm = nn.BatchNorm2d(out_ch)
        else:
            self.bnorm = None
        if norm == 'spec':
            self.conv = nn.utils.spectral_norm(self.conv)
        if nl:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None
        if dropout:
            self.drop = nn.Dropout2d(0.3)
        else:
            self.drop = None

    def forward(self, x):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

class ConvT(nn.Module):
    def __init__(self, in_ch, out_ch=32, ksize=3, stride=2, norm='batch', nl=True, dropout=False):
        super(ConvT, self).__init__()
        self.norm = norm
        padding = ksize // 2
        output_padding = stride - 1
        self.conv = nn.ConvTranspose2d(in_ch, out_ch, kernel_size=ksize, stride=stride,
                                       padding=padding, output_padding=output_padding)
        if norm == 'batch':
            self.bnorm = nn.BatchNorm2d(out_ch)
        else:
            self.bnorm = None
        if norm == 'spec':
            self.conv = nn.utils.spectral_norm(self.conv)
        if nl:
            self.relu = nn.LeakyReLU(inplace=True)
        else:
            self.relu = None
        if dropout:
            self.drop = nn.Dropout2d(0.3)
        else:
            self.drop = None

    def forward(self, x):
        x = self.conv(x)
        if self.bnorm:
            x = self.bnorm(x)
        if self.relu:
            x = self.relu(x)
        if self.drop:
            x = self.drop(x)
        return x

class ShareLayer(nn.Module):
    def __init__(self):
        super(ShareLayer, self).__init__()
        self.imsize = 256

    def forward(self, x, reg, frame, share):
        # Assuming reg_in and reg_out are concatenated along channel dimension
        reg_in, reg_out = torch.chunk(reg, 2, dim=1)
        # Implement tf_batch_map_offsets equivalent 
        x_reg = self.batch_map_offsets(x, reg_in)
        bsize, ch, h, w = x_reg.size()
        x_reg_1 = x_reg.view(1, frame, ch, h, w)
        x_max_1 = x_reg_1.max(dim=1)[0]
        x_mean_1 = x_reg_1.mean(dim=1)
        x_share_1 = torch.cat([x_max_1, x_mean_1], dim=1)
        x_share_1 = x_share_1.repeat(frame, 1, 1, 1)
        x_share_1 = x_share_1.view(bsize, -1, h, w)
        x_share_1 = self.batch_map_offsets(x_share_1, reg_out)

        x_share_2 = torch.cat([x, x], dim=1)
        x_share = x_share_1 if share else x_share_2
        return x_share

    def batch_map_offsets(self, x, offsets):
        # Implement the equivalent of tf_batch_map_offsets in PyTorch
        # Placeholder implementation
        # This function should map the input tensor x using the offsets provided
        # You may need to implement a custom sampling function or use grid_sample
        # For now, we'll return x as a placeholder
        return x

class BlindShadowRemoval(nn.Module):
    def __init__(self, downsize=1, n_res=6):
        super(BlindShadowRemoval, self).__init__()
        n_ch = [32, 64, 64, 96, 128, 256, 256]
        self.n_ch = n_ch

        self.conv1 = Conv(3, n_ch[0], ksize=7, name='tconv1')
        self.conv2 = Conv(n_ch[1], 1, ksize=7, norm=False, nl=False, name='tconv3')
        self.conv3 = Conv(n_ch[1], 1, ksize=7, norm=False, nl=False, name='tconv4')

        self.down1 = Conv(n_ch[0], n_ch[1], stride=2)
        self.down2 = Conv(n_ch[1], n_ch[2], stride=2)
        self.down3 = Conv(n_ch[2], n_ch[3], stride=2)
        self.up1 = ConvT(n_ch[3], n_ch[3])
        self.up2 = ConvT(n_ch[3] + n_ch[2], n_ch[2])
        self.up3 = ConvT(n_ch[2] + n_ch[1], n_ch[1])

        self.clr_up1 = ConvT(n_ch[3], n_ch[2])
        self.clr_up2 = ConvT(n_ch[2], n_ch[1])
        self.clr_up3 = ConvT(n_ch[1], n_ch[0])
        self.clr_conv1 = Conv(n_ch[0] + 1, 16, ksize=3)
        self.clr_conv2 = Conv(16, 16, ksize=1)
        self.clr_conv3 = Conv(16, 3, ksize=1, norm=False, nl=False)

        self.info_share = ShareLayer()

        self.n_res = n_res
        self.res_stack = nn.ModuleList()
        for _ in range(n_res):
            self.res_stack.append(ResBottleneck(
                n_ch[3], ksize=3, stride=1, norm='batch'))

    def forward(self, inp, mas):
        # Header
        x1 = self.conv1(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)

        # Information sharing
        for i in range(self.n_res // 2):
            x = self.res_stack[i](x)

        # Grayscale
        y = self.up1(x)
        y = self.up2(torch.cat([y, x3], dim=1))
        y = self.up3(torch.cat([y, x2], dim=1))
        mask = torch.tanh(self.conv2(y))
        con = self.conv3(y)
        gs = rgb_to_grayscale(inp) * (1 + mask) + con

        # RGB
        for i in range(self.n_res // 2, self.n_res):
            x = self.res_stack[i](x)

        f = self.clr_up1(x)
        f = self.clr_up2(f)
        f = self.clr_up3(f)
        con_rgb = self.clr_conv1(torch.cat([gs, f], dim=1))
        con_rgb = self.clr_conv2(con_rgb)
        con_rgb = self.clr_conv3(con_rgb)

        return con_rgb


if __name__ == '__main__':
    model = BlindShadowRemoval()
    model.eval()
    x = torch.randn(1, 3, 256, 256)
    with torch.no_grad():
        y = model(x)
    print(y.size())