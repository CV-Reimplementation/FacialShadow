import torch
from torch import nn
from transformers import MobileNetV2Config, MobileNetV2Model


class DecoderBlock(nn.Module):
    def __init__(self, in_channels, out_channels, expansion=2, do_up_sampling=True):
        # TODO: adding skip connections
        """
        Decoder block module.

        Args:
            in_channels (int): Number of input channels.
            out_channels (int): Number of output channels.
            expansion (int, optional): Expansion factor. Default is 3.
        """
        super(DecoderBlock, self).__init__()
        self.relu = nn.ReLU(inplace=True)

        self.cnn1 = nn.Conv2d(in_channels, in_channels *
                              expansion, kernel_size=1, stride=1)
        self.bnn1 = nn.BatchNorm2d(in_channels*expansion)

        # nearest neighbor x2
        self.do_up_sampling = do_up_sampling
        self.upsample = nn.Upsample(scale_factor=2, mode='nearest')

        # DW conv/ c_in*exp x 5 x 5 x c_in*exp
        self.cnn2 = nn.Conv2d(in_channels*expansion, in_channels *
                              expansion, kernel_size=5, padding=2, stride=1)
        self.bnn2 = nn.BatchNorm2d(in_channels*expansion)

        self.cnn3 = nn.Conv2d(in_channels*expansion,
                              out_channels, kernel_size=1, stride=1)
        self.bnn3 = nn.BatchNorm2d(out_channels)

        self.identity = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        """
        Forward pass through the decoder block.

        Args:
            x (torch.Tensor): Input tensor.

        Returns:
            torch.Tensor: Output tensor.
        """
        temp_x = self.identity(x)
        x = self.cnn1(x)
        x = self.bnn1(x)
        x = self.relu(x)

        if self.do_up_sampling:
            x = self.upsample(x)
            temp_x = self.upsample(temp_x)

        x = self.cnn2(x)
        x = self.bnn2(x)
        x = self.relu(x)

        x = self.cnn3(x)
        x = self.bnn3(x)

        return x + temp_x


def get_encoderv2_layers():
    """
    Returns the encoder layers of the MobileNetV2 model.
    """
    configuration = MobileNetV2Config()
    model = MobileNetV2Model(configuration)

    list_en = nn.ModuleList()
    encoder_layers_idx = [
        [0],
        [1, 2],
        [3, 4, 5],
        [6, 7, 8, 9],
        [10, 11, 12],
        [13, 14, 15],
    ]
    for idx, layer in enumerate(encoder_layers_idx):
        list_en.append(nn.Sequential(*[model.layer[i] for i in layer]))
    list_en.append(model.conv_1x1)
    return list_en, model.conv_stem


def get_decoderv2_layers():
    """
    Returns the decoder layers of the MobileNetV2 model based on the encoder layers.
    """
    decoder_layers = nn.ModuleList()
    decoder_layers.append(DecoderBlock(1280, 320, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(320+320, 160, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(160+160, 96, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(96+96, 64, do_up_sampling=False))
    decoder_layers.append(DecoderBlock(64+64, 32, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(32+32, 24, do_up_sampling=True))
    decoder_layers.append(DecoderBlock(24+24, 16, do_up_sampling=True))

    # a conv layer to get the final output
    out_stem = nn.Sequential(
        nn.Sequential(
            nn.ConvTranspose2d(16, 32, kernel_size=(1, 1),
                               stride=(1, 1), bias=False),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
            nn.ReLU6(),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 32, kernel_size=(3, 3), stride=(
                1, 1), groups=32, bias=False, padding=1),
            nn.BatchNorm2d(32, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
            nn.ReLU6(),
        ),
        nn.Sequential(
            nn.ConvTranspose2d(32, 3, kernel_size=(3, 3), stride=(
                2, 2), bias=False, padding=1, output_padding=1),
            nn.BatchNorm2d(3, eps=0.001, momentum=0.997,
                           affine=True, track_running_stats=True),
        )
    )

    return decoder_layers, out_stem


class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6


class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)


class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        mip = max(8, inp // reduction)

        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()

        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        identity = x

        n, c, h, w = x.size()
        x_h = self.pool_h(x)
        x_w = self.pool_w(x).permute(0, 1, 3, 2)

        y = torch.cat([x_h, x_w], dim=2)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)

        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()

        out = identity * a_w * a_h

        return out


class UnetV2WithAT(nn.Module):
    def __init__(self, lr=0.5):
        super(UnetV2WithAT, self).__init__()
        self.encoder_blocks, self.first_layer = get_encoderv2_layers()
        self.decoder_blocks, self.last_layer = get_decoderv2_layers()

        self.lra = CoordAtt(3, 3)
        self.ldra = CoordAtt(3, 3)

    def forward(self, x):
        """
        Performs forward pass through the U-Net model.

        Args:
                x (torch.Tensor): Input image tensor.
                process_image (bool): Whether to preprocess input image.

        Returns:
                torch.Tensor: Output image tensor.
        """
        assert x.shape[1] == 3, "input image should have 3 channels(nx3x224x224)"

        temp_in_x = x

        x = self.first_layer(x)

        enc_outputs = []
        for indx, enc_block in enumerate(self.encoder_blocks):
            x = enc_block(x)
            enc_outputs.append(x)

        for indx, dec_block in enumerate(self.decoder_blocks):
            if indx == 0:
                x = dec_block(x)

            else:
                x = dec_block(
                    torch.cat([x, enc_outputs[len(self.decoder_blocks) - indx - 1]], dim=1))

        x = self.last_layer(x)

        # lra attention on skip connection
        temp_in_x = self.lra(temp_in_x)
        # ldra attention on output
        x = self.ldra(x)

        return x + temp_in_x


class LPIONet(nn.Module):
    """
    The LP-IONET model class.
    """

    def __init__(self, i2it_model_path, L=2, interp_mode='nearest-exact', Unet_class=UnetV2WithAT):
        """
        Initializes the LPNet model.

        Args:
            i2it_model_path (str): The path to the pre-trained I2IT model.
            L (int, optional): The number of downsampling levels. Defaults to 2.
            interp_mode (str, optional): The interpolation mode used for upsampling. Defaults to 'nearest-exact'.
            lr (float, optional): The learning rate for optimization. Defaults to 0.1.
            Unet_class (nn.Module, optional): The class of the U-Net model used for I2IT. Defaults to UnetV2WithAT.
            DEVICE (str, optional): The device to be used for computation. Defaults to 'cuda'.

        Returns:
            None
        """
        super(LPIONet, self).__init__()
        self.L = L
        self.interp_mode = interp_mode
        self.final_input_dim = None

        self.i2it_model = Unet_class()

        self.resid_refinement_net = nn.Sequential(
            # Depth-sequential network
            nn.Conv2d(in_channels=9, out_channels=32, kernel_size=1),
            nn.LeakyReLU(),

            nn.Conv2d(in_channels=32, out_channels=32,
                      kernel_size=3, padding=1, groups=32),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=1),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=64, out_channels=1, kernel_size=1),
        )

        self.next_res_net = nn.ModuleList()
        for l in range(L-1):
            layers = nn.Sequential(
                nn.Conv2d(in_channels=1, out_channels=16, kernel_size=1),
                nn.LeakyReLU(),

                nn.Conv2d(in_channels=16, out_channels=16, kernel_size=1),
                nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1),
            )

            self.next_res_net.append(layers)

        self.loss_layer = nn.L1Loss()

    def forward(self, I0):
        H = []
        shapes = []
        IL = I0
        for l in range(self.L):
            curr_dim = IL.shape
            shapes.append(curr_dim)

            # print(f'Downsampling {IL.shape}')
            IL_next = nn.functional.interpolate(
                IL, size=[curr_dim[2]//2, curr_dim[3]//2], mode=self.interp_mode)
            IL_next_up = nn.functional.interpolate(
                IL_next, size=[curr_dim[2], curr_dim[3]], mode=self.interp_mode)
            Hl = IL - IL_next_up
            H.append(Hl)

            IL = IL_next

        self.final_input_dim = IL[0].shape

        # I2IT
        IL_cap = self.i2it_model(IL)

        # Upsampling the translated image
        IL_up = nn.functional.interpolate(
            IL, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        IL_cap_up = nn.functional.interpolate(
            IL_cap, size=[shapes[self.L-1][2], shapes[self.L-1][3]], mode=self.interp_mode)
        ResNet_inp = torch.cat([IL_up, IL_cap_up, H[self.L-1]], dim=1)

        # Ml_next = torch.mean(torch.stack([IL_up, IL_cap_up, H[self.L-1]]),dim=0,keepdim=True)[0]
        Ml_next = self.resid_refinement_net(ResNet_inp)

        H_ref = H[self.L-1]*Ml_next
        Il_cap = H_ref + IL_cap_up
        # print(H_ref.shape,H[self.L-1].shape,Ml_next.shape)

        for l in range(self.L-2, -1, -1):
            Ml_next_up = nn.functional.interpolate(
                Ml_next, size=[shapes[l][2], shapes[l][3]], mode=self.interp_mode)
            Ml_next = self.next_res_net[l](Ml_next_up)

            Il_cap_up = nn.functional.interpolate(
                Il_cap, size=[shapes[l][2], shapes[l][3]], mode=self.interp_mode)

            H_ref = H[l]*Ml_next
            Il_cap = H_ref + Il_cap_up

        return Il_cap



if __name__ == '__main__':
    model = LPIONet(None)
    print(model)
    x = torch.randn(1, 3, 256, 256)
    y = model(x)
    print(y.shape)
