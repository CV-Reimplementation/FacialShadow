import torch
import torch.nn as nn


class S3R_Net(nn.Module):
    def __init__(self, input_nc=3, output_nc=3, ngf=64, n_downsampling=3, n_blocks=4):
        super(S3R_Net, self).__init__()
        padding_type = 'reflect'
        norm_layer = nn.InstanceNorm2d
        self.n_blocks = n_blocks
        activation = nn.ReLU(True)
        self.input_nc = input_nc

        model = [nn.ReflectionPad2d(3), nn.Conv2d(
            input_nc, ngf, kernel_size=7, padding=0), norm_layer(ngf), activation]

        # downsample
        for i in range(n_downsampling):
            mult = 2 ** i
            model += [nn.Conv2d(ngf * mult, ngf * mult * 2, kernel_size=3,
                                stride=2, padding=1), norm_layer(ngf * mult * 2), activation]

        # resnet blocks
        mult = 2 ** n_downsampling
        for i in range(self.n_blocks):
            model += [ResnetBlock(ngf * mult, padding_type=padding_type,
                                  activation=activation, norm_layer=norm_layer)]

        # upsample
        for i in range(n_downsampling):
            mult = 2 ** (n_downsampling - i)
            model += [nn.ConvTranspose2d(ngf * mult, int(ngf * mult / 2), kernel_size=3, stride=2, padding=1,
                                         output_padding=1), norm_layer(int(ngf * mult / 2)), activation]

        # final layer
        # , nn.Tanh()]
        model += [nn.ReflectionPad2d(3), nn.Conv2d(ngf,
                                                   output_nc, kernel_size=7, padding=0)]
        self.model = nn.Sequential(*model)
        self.tanh = nn.Tanh()

    def forward(self, x, mas):
        res = self.model(x)
        sf_full = res + x if self.input_nc == 3 else res + x[:, :3, ...]
        sf_full = self.tanh(sf_full)
        return sf_full


# Define a resnet block
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.InstanceNorm2d, activation=nn.ReLU(True), use_dropout=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(
            dim, padding_type, norm_layer, activation, use_dropout)

    def build_conv_block(self, dim, padding_type, norm_layer, activation, use_dropout):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p),
                       norm_layer(dim), activation]  # , groups=dim
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError(
                'padding [%s] is not implemented' % padding_type)
        # , groups=dim
        conv_block += [nn.Conv2d(dim, dim, kernel_size=3,
                                 padding=p), norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256).cuda()
    model = S3R_Net().cuda()
    model.eval()
    with torch.no_grad():
        res = model(x)
        print(res.shape)