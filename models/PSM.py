import torch
import torch.nn as nn
import torch.nn.functional as F


def batch_map_offsets(input, offsets):
    """
    Maps input with offsets. This function uses grid_sample to apply offsets to the input tensor.
    :param input: Input tensor of shape (batch_size, channels, height, width)
    :param offsets: Offset tensor of shape (batch_size, 2, height, width)
    :return: Transformed tensor
    """
    n, c, h, w = input.size()
    # Create base grid
    grid_y, grid_x = torch.meshgrid(torch.arange(0, h), torch.arange(0, w))
    grid = torch.stack((grid_x, grid_y), 2).float().to(
        input.device)  # (h, w, 2)
    grid = grid.unsqueeze(0).repeat(n, 1, 1, 1)  # (n, h, w, 2)
    grid = grid + offsets.permute(0, 2, 3, 1)  # Apply offsets, (n, h, w, 2)
    # Normalize grid to [-1, 1]
    grid[:, :, :, 0] = 2.0 * grid[:, :, :, 0] / (w - 1) - 1.0
    grid[:, :, :, 1] = 2.0 * grid[:, :, :, 1] / (h - 1) - 1.0
    grid = grid.clamp(-2, 2)
    output = F.grid_sample(input, grid, mode='bilinear',
                           padding_mode='border', align_corners=True)
    return output


class NonLocalBlock(nn.Module):
    def __init__(self, ch=32, out_ch=None, pool=False, norm='batch'):
        super(NonLocalBlock, self).__init__()
        self.out_ch = ch if out_ch is None else out_ch
        self.g = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.phi = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.theta = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.w = nn.Conv2d(ch // 2, self.out_ch,
                           kernel_size=1, stride=1, padding=0)

        self.norm = norm
        self.bnorm = nn.BatchNorm2d(self.out_ch) if norm == 'batch' else None

        self.pool = pool
        if self.pool:
            self.pool_layer = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)

    def forward(self, x):
        bsize, in_ch, h, w = x.size()
        if self.pool:
            x_pooled = self.pool_layer(x)
            h0, w0 = x_pooled.size(2), x_pooled.size(3)
        else:
            x_pooled = x
            h0, w0 = h, w

        # g
        g_x = self.g(x)
        if self.pool:
            g_x = self.pool_layer(g_x)
        g_x = g_x.view(bsize, -1, h0 * w0).permute(0,
                                                   2, 1)  # (bsize, h0*w0, ch//2)

        # phi
        phi_x = self.phi(x)
        if self.pool:
            phi_x = self.pool_layer(phi_x)
        phi_x = phi_x.view(bsize, -1, h0 * w0)

        # theta
        theta_x = self.theta(x)
        if self.pool:
            theta_x = self.pool_layer(theta_x)
        theta_x = theta_x.view(bsize, -1, h0 * w0).permute(0, 2, 1)

        f = torch.bmm(theta_x, phi_x)  # (bsize, h0*w0, h0*w0)
        f_softmax = F.softmax(f, dim=-1)
        y = torch.bmm(f_softmax, g_x)  # (bsize, h0*w0, ch//2)
        y = y.permute(0, 2, 1).contiguous().view(
            bsize, -1, h0, w0)  # (bsize, ch//2, h0, w0)

        w_y = self.w(y)
        if self.pool:
            w_y = F.interpolate(w_y, size=(
                h, w), mode='bilinear', align_corners=False)
        if self.bnorm:
            w_y = self.bnorm(w_y)
        z = x + w_y

        return z


class Res(nn.Module):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(Res, self).__init__()
        padding = ksize // 2  # To keep 'same' padding
        self.bnorm1 = nn.BatchNorm2d(ch) if norm == 'batch' else None
        self.conv1 = nn.Conv2d(ch, ch, kernel_size=ksize,
                               stride=stride, padding=padding)
        self.relu1 = nn.LeakyReLU()
        self.bnorm2 = nn.BatchNorm2d(ch) if norm == 'batch' else None
        self.conv2 = nn.Conv2d(ch, ch, kernel_size=ksize,
                               stride=stride, padding=padding)
        self.relu2 = nn.LeakyReLU()
        self.non_local = NonLocalBlock(ch, ch)

    def forward(self, x):
        if self.bnorm1:
            y = self.bnorm1(x)
        else:
            y = x
        y = self.conv1(y)
        y = self.relu1(y)
        if self.bnorm2:
            y = self.bnorm2(y)
        y = self.conv2(y)
        y = self.relu2(x + y)
        y = self.non_local(y)
        return y


class ResBottleneck(nn.Module):
    def __init__(self, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(ResBottleneck, self).__init__()
        padding = ksize // 2  # to have 'same' padding
        self.conv1 = nn.Conv2d(ch, ch // 2, kernel_size=1, stride=1, padding=0)
        self.bnorm1 = nn.BatchNorm2d(ch // 2) if norm == 'batch' else None
        self.relu1 = nn.LeakyReLU()
        self.conv2 = nn.Conv2d(
            ch // 2, ch // 2, kernel_size=ksize, stride=stride, padding=padding)
        self.bnorm2 = nn.BatchNorm2d(ch // 2) if norm == 'batch' else None
        self.relu2 = nn.LeakyReLU()
        self.conv3 = nn.Conv2d(ch // 2, ch, kernel_size=1, stride=1, padding=0)
        self.bnorm3 = nn.BatchNorm2d(ch) if norm == 'batch' else None
        self.relu3 = nn.LeakyReLU()
        self.stride = stride
        self.non_local = NonLocalBlock(ch, ch)
        if stride > 1:
            self.conv_red = nn.Conv2d(
                ch, ch, kernel_size=1, stride=stride, padding=0)

    def forward(self, x):
        y = self.conv1(x)
        if self.bnorm1:
            y = self.bnorm1(y)
        y = self.relu1(y)
        y = self.conv2(y)
        if self.bnorm2:
            y = self.bnorm2(y)
        y = self.relu2(y)
        y = self.conv3(y)
        if self.bnorm3:
            y = self.bnorm3(y)
        y = self.non_local(y)
        if self.stride > 1:
            x = self.conv_red(x)
        x_channels = x.size(1)
        y_channels = y.size(1)
        if x_channels < y_channels:
            ch_add = y_channels - x_channels
            ch_pad = torch.zeros(x.size(0), ch_add, x.size(
                2), x.size(3), device=x.device)
            x = torch.cat([x, ch_pad], dim=1)
        elif y_channels < x_channels:
            ch_add = x_channels - y_channels
            ch_pad = torch.zeros(y.size(0), ch_add, y.size(
                2), y.size(3), device=y.device)
            y = torch.cat([y, ch_pad], dim=1)
        return self.relu3(x + y)


class Conv(nn.Module):
    def __init__(self, in_ch, ch=32, ksize=3, stride=1, norm='batch', nl=True, dropout=False):
        super(Conv, self).__init__()
        padding = ksize // 2  # To achieve 'same' padding
        self.norm = norm
        self.dropout = dropout
        if norm == 'spec':
            self.conv = nn.utils.spectral_norm(
                nn.Conv2d(in_channels=in_ch, out_channels=ch,
                          kernel_size=ksize, stride=stride, padding=padding)
            )
        else:
            self.conv = nn.Conv2d(in_channels=in_ch, out_channels=ch,
                                  kernel_size=ksize, stride=stride, padding=padding)
        self.bnorm = nn.BatchNorm2d(ch) if norm == 'batch' else None
        self.relu = nn.LeakyReLU() if nl else None
        self.drop = nn.Dropout2d(0.3) if dropout else None

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
    def __init__(self, in_ch, ch=32, ksize=3, stride=2, norm='batch', nl=True, dropout=False):
        super(ConvT, self).__init__()
        padding = ksize // 2  # To achieve 'same' padding
        output_padding = stride - 1 if stride > 1 else 0

        self.norm = norm
        if norm == 'spec':
            self.conv = nn.utils.spectral_norm(
                nn.ConvTranspose2d(in_channels=in_ch, out_channels=ch, kernel_size=ksize,
                                   stride=stride, padding=padding, output_padding=output_padding)
            )
        else:
            self.conv = nn.ConvTranspose2d(in_channels=in_ch, out_channels=ch, kernel_size=ksize,
                                           stride=stride, padding=padding, output_padding=output_padding)
        self.bnorm = nn.BatchNorm2d(ch) if norm == 'batch' else None
        self.relu = nn.LeakyReLU() if nl else None
        self.drop = nn.Dropout2d(0.3) if dropout else None

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

    def forward(self, x, reg, chuck):
        # Split along channels dimension
        reg_in, reg_out = torch.chunk(reg, 2, dim=1)
        x_reg = batch_map_offsets(x, reg_in)
        # Reshape
        chuck_bsize, ch, h, w = x_reg.size()
        x_reg = x_reg.view(chuck_bsize // chuck, chuck, ch, h, w)
        x_max = torch.max(x_reg, dim=1)[0]
        x_mean = torch.mean(x_reg, dim=1)
        x_share = torch.cat([x_max, x_mean], dim=1)
        x_share = x_share.unsqueeze(1).expand(-1, chuck, -1, -1, -1)
        x_share = x_share.contiguous().view(chuck_bsize, -1, h, w)
        x_share_dereg = batch_map_offsets(x_share, reg_out)
        return x_share_dereg


class PSM(nn.Module):
    def __init__(self, n_res=6):
        super(PSM, self).__init__()
        n_ch = [32, 64, 64, 96, 128, 256, 256]
        self.n_ch = n_ch
        self.conv1 = Conv(in_ch=3, ch=n_ch[0], ksize=7)
        self.conv2 = Conv(in_ch=n_ch[1]*2, ch=3, ksize=7, norm=False, nl=False)
        self.conv3 = Conv(in_ch=3, ch=3, ksize=7, norm=False, nl=False)

        self.down1 = Conv(in_ch=n_ch[0], ch=n_ch[1], stride=2)
        self.down2 = Conv(in_ch=n_ch[1], ch=n_ch[2], stride=2)
        self.down3 = Conv(in_ch=n_ch[2], ch=n_ch[3], stride=2)

        self.up1 = ConvT(in_ch=n_ch[3], ch=n_ch[3]*2)
        self.up2 = ConvT(in_ch=n_ch[3]*2 + n_ch[2], ch=n_ch[2]*2)
        self.up3 = ConvT(in_ch=n_ch[2]*2 + n_ch[1], ch=n_ch[1]*2)

        self.info_share = ShareLayer()

        self.n_res = n_res
        self.res_stack = nn.ModuleList()
        for i in range(n_res):
            self.res_stack.append(ResBottleneck(
                ch=n_ch[3], ksize=3, stride=1, norm='batch'))

    def forward(self, inp, mas):
        # Header
        x1 = self.conv1(inp)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x = self.down3(x3)
        b, c, h, w = x.shape

        # Information sharing
        
        for i in range(self.n_res):
            x = self.res_stack[i](x)

        # Greyscale
        y = self.up1(x)
        y = self.up2(torch.cat([y, x3], dim=1))
        y = self.up3(torch.cat([y, x2], dim=1))
        y = self.conv2(y)
        con = self.conv3(y)

        return con


if __name__ == '__main__':
    gen = PSM()
    x = torch.randn(1, 3, 256, 256)
    mask = torch.randn(1, 1, 256, 256)
    out = gen(x, mask)
    print(out.size())