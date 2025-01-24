import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.layers import trunc_normal_, DropPath


class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x
        

class ConvNeXt(nn.Module):
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024,2048], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)


    def forward(self, x):
        x_layer1 = self.downsample_layers[0](x)
        x_layer1 = self.stages[0](x_layer1)

        

        x_layer2 = self.downsample_layers[1](x_layer1)
        x_layer2 = self.stages[1](x_layer2)
        

        x_layer3 = self.downsample_layers[2](x_layer2)
        out = self.stages[2](x_layer3)
          

        return x_layer1, x_layer2, out
    

class ConvNeXt0(nn.Module):
    r""" ConvNeXt
        A PyTorch impl of : `A ConvNet for the 2020s`  -
          https://arxiv.org/pdf/2201.03545.pdf
    Args:
        in_chans (int): Number of input image channels. Default: 3
        num_classes (int): Number of classes for classification head. Default: 1000
        depths (tuple(int)): Number of blocks at each stage. Default: [3, 3, 9, 3]
        dims (int): Feature dimension at each stage. Default: [96, 192, 384, 768]
        drop_path_rate (float): Stochastic depth rate. Default: 0.
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
        head_init_scale (float): Init scaling value for classifier weights and biases. Default: 1.
    """
    def __init__(self, block, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0., 
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 ):
        super().__init__()

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            trunc_normal_(m.weight, std=.02)
            nn.init.constant_(m.bias, 0)

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            x = self.stages[i](x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x
    

class PALayer(nn.Module):
    def __init__(self, channel):
        super(PALayer, self).__init__()
        self.pa = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, 1, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.pa(x)
        return x * y

class CALayer(nn.Module):
    def __init__(self, channel):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.ca = nn.Sequential(
            nn.Conv2d(channel, channel // 8, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // 8, channel, 1, padding=0, bias=True),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.avg_pool(x)
        y = self.ca(y)
        return x * y
    

class CP_Attention_block(nn.Module):
    def __init__(self, conv, dim, kernel_size):
        super(CP_Attention_block, self).__init__()
        self.conv1 = conv(dim, dim, kernel_size, bias=True)
        self.act1 = nn.ReLU(inplace=True)
        self.conv2 = conv(dim, dim, kernel_size, bias=True)
        self.calayer = CALayer(dim)
        self.palayer = PALayer(dim)
    def forward(self, x):
        res = self.act1(self.conv1(x))
        res = res + x
        res = self.conv2(res)
        res = self.calayer(res)
        res = self.palayer(res)
        res += x
        return res
    

class Block(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x
    

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)


class knowledge_adaptation_convnext(nn.Module):
    def __init__(self):
        super(knowledge_adaptation_convnext, self).__init__()
        self.encoder = ConvNeXt(Block, in_chans=3,num_classes=1000, depths=[3, 3, 27, 3], dims=[256, 512, 1024,2048], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.)
        pretrained_model = ConvNeXt0(Block, in_chans=3,num_classes=1000, depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048], drop_path_rate=0., layer_scale_init_value=1e-6, head_init_scale=1.)
        #pretrained_model=nn.DataParallel(pretrained_model)
        # checkpoint=torch.load('./weights/convnext_xlarge_22k_1k_384_ema.pth')
        #for k,v in checkpoint["model"].items():
            #print(k)
        url="https://dl.fbaipublicfiles.com/convnext/convnext_large_1k_384.pth"
        
        checkpoint = torch.hub.load_state_dict_from_url(url=url, map_location="cuda")
        pretrained_model.load_state_dict(checkpoint["model"])
        
        pretrained_dict = pretrained_model.state_dict()
        model_dict = self.encoder.state_dict()
        key_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
        model_dict.update(key_dict)
        self.encoder.load_state_dict(model_dict)


        self.up_block= nn.PixelShuffle(2)
        self.attention0 = CP_Attention_block(default_conv, 1024, 3)
        self.attention1 = CP_Attention_block(default_conv, 256, 3)
        self.attention2 = CP_Attention_block(default_conv, 192, 3)
        self.attention3 = CP_Attention_block(default_conv, 112, 3)
        self.attention4 = CP_Attention_block(default_conv, 28, 3)
        self.conv_process_1 = nn.Conv2d(28, 28, kernel_size=3,padding=1)
        self.conv_process_2 = nn.Conv2d(28, 28, kernel_size=3,padding=1)
        self.tail = nn.Sequential(nn.ReflectionPad2d(3), nn.Conv2d(28, 3, kernel_size=7, padding=0), nn.Tanh())

    def forward(self, input):
        x_layer1, x_layer2, x_output = self.encoder(input)

        x_mid = self.attention0(x_output)  #[1024,24,24]

        x = self.up_block(x_mid)      #[256,48,48]
        x = self.attention1(x)

        x = torch.cat((x, x_layer2), 1)  #[768,48,48]

        x = self.up_block(x)            #[192,96,96]
        x = self.attention2(x)
        x = torch.cat((x, x_layer1), 1)   #[448,96,96]
        x = self.up_block(x)            #[112,192,192]
        x = self.attention3(x)              
        
        x = self.up_block(x)        #[28,384,384]
        x = self.attention4(x)

        x=self.conv_process_1(x)
        out=self.conv_process_2(x)
        return out
    

class UNetCompress(nn.Module):
    def __init__(self, in_size, out_size, normalize=True,  kernel_size=4, dropout=0.33):
        super(UNetCompress, self).__init__()
        layers = [nn.Conv2d(in_size, out_size, kernel_size, 2, 1, bias=False)]
        if normalize:
            layers.append(nn.BatchNorm2d(out_size))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        if dropout > 0:
            layers.append(nn.Dropout(dropout))
        self.model = nn.Sequential(*layers)

    def forward(self, x):
        return self.model(x)


class UNetDecompress(nn.Module):
    def __init__(self, in_size, out_size, dropout=0.13):
        super(UNetDecompress, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_size, out_size, 4, 2, 1, bias=False),
            nn.BatchNorm2d(out_size),
            nn.ReLU(inplace=True),
        ]
        if dropout > 0:
            layers.append(nn.Dropout(dropout))

        self.model = nn.Sequential(*layers)

    def forward(self, x):
        f = self.model(x)
        return f
    

class LayerNormFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, weight, bias, eps):
        ctx.eps = eps
        N, C, H, W = x.size()
        mu = x.mean(1, keepdim=True)
        var = (x - mu).pow(2).mean(1, keepdim=True)
        y = (x - mu) / (var + eps).sqrt()
        ctx.save_for_backward(y, var, weight)
        y = weight.view(1, C, 1, 1) * y + bias.view(1, C, 1, 1)
        return y

    @staticmethod
    def backward(ctx, grad_output):
        eps = ctx.eps

        N, C, H, W = grad_output.size()
        y, var, weight = ctx.saved_variables
        g = grad_output * weight.view(1, C, 1, 1)
        mean_g = g.mean(dim=1, keepdim=True)

        mean_gy = (g * y).mean(dim=1, keepdim=True)
        gx = 1. / torch.sqrt(var + eps) * (g - y * mean_gy - mean_g)
        return gx, (grad_output * y).sum(dim=3).sum(dim=2).sum(dim=0), grad_output.sum(dim=3).sum(dim=2).sum(
            dim=0), None


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super(LayerNorm2d, self).__init__()
        self.register_parameter('weight', nn.Parameter(torch.ones(channels)))
        self.register_parameter('bias', nn.Parameter(torch.zeros(channels)))
        self.eps = eps

    def forward(self, x):
        return LayerNormFunction.apply(x, self.weight, self.bias, self.eps)
    

class FusedPooling(nn.Module):
    def __init__(self, nc, blend=True):
        super(FusedPooling, self).__init__()

        self.blend = blend
        self.avg_pool = nn.AvgPool2d(kernel_size=2, stride=2)
        self.max_pool = nn.MaxPool2d(kernel_size=2, stride=2)

        if self.blend:
            #self.conv = nn.Conv2d(in_channels=2 * nc, out_channels=2 * nc, kernel_size=1, padding=0, bias=False)
            self.conv = nn.Conv2d(in_channels=nc, out_channels=2*nc, kernel_size=2, stride=2) 
        else:
            self.conv = None

    def forward(self, x):
        x_ap = self.avg_pool(x)
        x_mp = self.max_pool(x)
        if self.blend:
            # return self.conv(torch.cat((x_ap, x_mp), dim=1))
            return self.conv(x)
        else:
            return torch.cat((x_ap, x_mp), dim=1)


class DWT(nn.Module):
    def __init__(self):
        super(DWT, self).__init__()
        self.requires_grad = False

    def dwt_haar(self, x):
        x01 = x[:, :, 0::2, :] / 2
        x02 = x[:, :, 1::2, :] / 2
        x1 = x01[:, :, :, 0::2]
        x2 = x02[:, :, :, 0::2]
        x3 = x01[:, :, :, 1::2]
        x4 = x02[:, :, :, 1::2]
        x_LL = x1 + x2 + x3 + x4
        x_HL = -x1 - x2 + x3 + x4
        x_LH = -x1 + x2 - x3 + x4
        x_HH = x1 - x2 - x3 + x4
        return x_LL, torch.cat((x_HL, x_LH, x_HH), 1)

    def forward(self, x):
        return self.dwt_haar(x)


class DWT_block(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.dwt = DWT()

        self.conv1x1_low = nn.Conv2d(
            in_channels, out_channels, kernel_size=1, padding=0)
        self.conv1x1_high = nn.Conv2d(
            in_channels * 3, out_channels, kernel_size=1, padding=0)

    def forward(self, x):
        dwt_low_frequency, dwt_high_frequency = self.dwt(x)
        dwt_low_frequency = self.conv1x1_low(dwt_low_frequency)
        dwt_high_frequency = self.conv1x1_high(dwt_high_frequency)
        return dwt_low_frequency, dwt_high_frequency


class SimpleGate(nn.Module):
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class CAM(nn.Module):
    def __init__(self, num_channels, compress_factor=8):
        super(CAM, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.model = nn.Sequential(
            nn.Conv2d(num_channels, num_channels //
                      compress_factor, 1, padding=0, bias=True),
            nn.PReLU(),
            nn.Conv2d(num_channels // compress_factor,
                      num_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.model(y)
        return x * y


class DynamicDepthwiseConvolution(nn.Module):
    def __init__(self, in_ch, out_ch, k=3, stride=1, num_cnvs=4, K=2):
        super(DynamicDepthwiseConvolution, self).__init__()

        self.convs = nn.ModuleList([
            nn.Conv2d(in_ch, in_ch * K, k, stride, groups=in_ch,
                      padding='same', padding_mode='reflect')
            for _ in range(num_cnvs)])

        self.weights = nn.Parameter(1 / num_cnvs * torch.ones((num_cnvs, 1), dtype=torch.float),
                                    requires_grad=True)
        self.final_conv = nn.Conv2d(
            in_channels=in_ch * K, out_channels=out_ch, kernel_size=1, padding=0, bias=True)

    def forward(self, x):
        feats = 0
        for i, conv in enumerate(self.convs):
            feats += self.weights[i] * conv(x)

        return self.final_conv(feats)


class SimplifiedDepthwiseCA(nn.Module):
    def __init__(self, num_channels, k, K):
        super().__init__()
        self.attention = CAM(num_channels)
        self.dc = DynamicDepthwiseConvolution(
            in_ch=num_channels, out_ch=num_channels, K=K, k=k)

    def forward(self, x):
        q = self.dc(x)
        w = self.attention(q)
        return torch.sigmoid(w * q)


class BlockRGB(nn.Module):
    def __init__(self, in_ch, out_ch, k_sz=3, dropout_prob=0.5):
        super(BlockRGB, self).__init__()
        self.ln = LayerNorm2d(in_ch)
        self.conv1 = nn.Conv2d(
            in_ch, out_ch // 2, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op1 = nn.LeakyReLU(0.2)
        self.dyndc = SimplifiedDepthwiseCA(
            num_channels=out_ch // 2, k=13, K=4)
        self.conv2 = nn.Conv2d(
            out_ch // 2, out_ch, k_sz, padding=k_sz // 2, padding_mode="reflect", bias=True)
        self.op2 = nn.LeakyReLU(0.2)

        self.rconv1 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch // 2, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)
        self.rconv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1, padding=0, stride=1,
                                groups=1, bias=True)

        self.a1 = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)
        self.a2 = nn.Parameter(torch.tensor(
            1.0, dtype=torch.float32), requires_grad=True)
        self.dropout1 = nn.Dropout(
            dropout_prob) if dropout_prob > 0. else nn.Identity()
        self.dropout2 = nn.Dropout(
            dropout_prob) if dropout_prob > 0. else nn.Identity()

    def forward(self, x):
        xf = self.ln(x)
        xf = self.op1(self.conv1(xf))
        xf = self.dropout1(xf)
        xf += self.a1 * self.rconv1(x)
        xf = self.dyndc(xf)
        xf = self.dropout2(xf)
        xf = self.op2(self.conv2(xf))
        return xf + self.a2 * self.rconv2(x)


class IFBlendDown(nn.Module):
    def __init__(self, in_size, rgb_in_size, out_size, dwt_size=1, dropout=0.0, default=False, blend=False):
        super().__init__()

        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_dize = dwt_size
        self.rgb_in_size = rgb_in_size

        if dwt_size > 0:
            self.dwt = DWT_block(in_channels=in_size, out_channels=dwt_size)

        self.b_unet = UNetCompress(in_size, out_size, dropout=dropout/2)

        if default:
            self.rgb_block = BlockRGB(
                3, out_size, dropout_prob=dropout)
        else:
            self.rgb_block = BlockRGB(
                rgb_in_size, out_size, dropout_prob=dropout)

        self.fp = FusedPooling(nc=out_size, blend=blend)

    def forward(self, x, rgb_img):
        xu = self.b_unet(x)
        b, c, h, w = xu.shape
        rgb_feats = self.fp(self.rgb_block(rgb_img))

        if self.dwt_dize > 0:
            lfw, hfw = self.dwt(x)
            return torch.cat((xu, rgb_feats[:, :c, :, :], lfw), dim=1), hfw, xu, rgb_feats[:, c:, :, :]
        else:
            return torch.cat((xu, rgb_feats[:, :c, :, :]), dim=1), None, xu, rgb_feats[:, c:, :, :]


class WASAM(nn.Module):
    '''
    Based on NAFNET Stereo Cross Attention Module (SCAM)
    '''

    def __init__(self, c_rgb, cr):
        super().__init__()
        self.scale = (0.5 * (c_rgb + cr)) ** -0.5

        self.norm_l = LayerNorm2d(c_rgb)
        self.l_proj_res = nn.Conv2d(
            c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj_res = nn.Conv2d(
            cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)

        self.l_proj1 = nn.Conv2d(
            c_rgb, c_rgb, kernel_size=1, stride=1, padding=0)
        self.r_proj1 = nn.Conv2d(cr, c_rgb, kernel_size=1, stride=1, padding=0)

        self.beta = nn.Parameter(torch.zeros(
            (1, c_rgb // 2, 1, 1)), requires_grad=True)
        self.gamma = nn.Parameter(torch.zeros(
            (1, c_rgb // 2, 1, 1)), requires_grad=True)

        self.l_proj2 = nn.Conv2d(
            c_rgb, c_rgb // 2, kernel_size=1, stride=1, padding=0)
        self.r_proj2 = nn.Conv2d(
            cr, c_rgb // 2, kernel_size=1, stride=1, padding=0)

    def forward(self, x_rgb, x_hfw):
        Q_l = self.l_proj1(self.norm_l(x_rgb)).permute(
            0, 2, 3, 1)  # B, H, W, c
        Q_r_T = self.r_proj1(x_hfw).permute(
            0, 2, 1, 3)  # B, H, c, W (transposed)

        V_l = self.l_proj2(x_rgb).permute(0, 2, 3, 1)  # B, H, W, c
        V_r = self.r_proj2(x_hfw).permute(0, 2, 3, 1)  # B, H, W, c

        # (B, H, W, c) x (B, H, c, W) -> (B, H, W, W)
        attention = torch.matmul(Q_l, Q_r_T) * self.scale

        F_r2l = torch.matmul(torch.softmax(
            attention, dim=-1), V_r)  # B, H, W, c
        F_l2r = torch.matmul(torch.softmax(
            attention.permute(0, 1, 3, 2), dim=-1), V_l)  # B, H, W, c

        # scale
        F_r2l = F_r2l.permute(0, 3, 1, 2) * self.beta
        F_l2r = F_l2r.permute(0, 3, 1, 2) * self.gamma
        return torch.cat((self.l_proj_res(x_rgb) + F_r2l, self.r_proj_res(x_hfw) + F_l2r), dim=1)


class IFBlendUp(nn.Module):
    def __init__(self, in_size, rgb_size, dwt_size,  out_size, dropout):
        super().__init__()
        self.in_ch = in_size
        self.out_ch = out_size
        self.dwt_size = dwt_size
        self.rgb_size = rgb_size

        self.b_unet = UNetDecompress(
            in_size + dwt_size, out_size, dropout=dropout)
        self.rgb_proj = nn.ConvTranspose2d(
            in_channels=rgb_size, out_channels=out_size, kernel_size=4, stride=2, padding=1)
        if dwt_size > 0:
            self.spfam = WASAM(rgb_size, dwt_size)

    def forward(self, x, hfw, rgb):
        if self.dwt_size > 0:
            rgb = self.spfam(rgb, hfw)

        state = self.b_unet(torch.cat((x, hfw), dim=1))
        state = state + F.relu(self.rgb_proj(rgb))
        return state


class IFBlend(nn.Module):
    def __init__(self, in_channels, use_gcb=False, blend=False):
        super().__init__()

        self.in_channels = in_channels
        self.use_gcb = use_gcb

        self.in_conv = nn.Conv2d(3, in_channels, kernel_size=3, padding=1)
        self.in_bn = nn.BatchNorm2d(in_channels)

        if self.use_gcb:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels + 28, 3, kernel_size=7,
                          padding=3, padding_mode="reflect")
            )
        else:
            self.out_conv = nn.Sequential(
                nn.Conv2d(in_channels, 3, kernel_size=7,
                          padding=3, padding_mode="reflect")
            )

        if self.use_gcb:
            self.gcb = knowledge_adaptation_convnext()

        self.c1 = IFBlendDown(in_size=in_channels, rgb_in_size=3,
                              out_size=32, dwt_size=1, dropout=0.15, default=True, blend=blend)
        self.c2 = IFBlendDown(in_size=65, rgb_in_size=32, out_size=64,
                              dwt_size=2, dropout=0.2, blend=blend)
        self.c3 = IFBlendDown(in_size=130, rgb_in_size=64, out_size=128,
                              dwt_size=4, dropout=0.25, blend=blend)
        self.c4 = IFBlendDown(in_size=260, rgb_in_size=128, out_size=256,
                              dwt_size=8, dropout=0.3, blend=blend)
        self.c5 = IFBlendDown(in_size=520, rgb_in_size=256, out_size=256,
                              dwt_size=16, dropout=0.0, blend=blend)

        self.d5 = IFBlendUp(in_size=528, dwt_size=16,
                            rgb_size=256, out_size=256, dropout=0.0)
        self.d4 = IFBlendUp(in_size=512, dwt_size=8,
                            rgb_size=256, out_size=128, dropout=0.3)
        self.d3 = IFBlendUp(in_size=256, dwt_size=4,
                            rgb_size=128, out_size=64, dropout=0.25)
        self.d2 = IFBlendUp(in_size=128, dwt_size=2,
                            rgb_size=64, out_size=32, dropout=0.2)
        self.d1 = IFBlendUp(in_size=64, dwt_size=1, rgb_size=32,
                            out_size=in_channels, dropout=0.1)

    def forward(self, x, mas):
        x_rgb = x
        xf = self.in_bn(self.in_conv(x))
        x1, s1, xs1, rgb1 = self.c1(xf, x_rgb)
        x2, s2, xs2,  rgb2 = self.c2(x1, rgb1)
        x3, s3, xs3, rgb3 = self.c3(x2, rgb2)
        x4, s4, xs4, rgb4 = self.c4(x3, rgb3)
        x5, s5, xs5, rgb5 = self.c5(x4, rgb4)
        y5 = self.d5(x5, s5, rgb5)
        y4 = self.d4(torch.cat((y5, xs4), dim=1), s4, rgb4)
        y3 = self.d3(torch.cat((y4, xs3), dim=1), s3, rgb3)
        y2 = self.d2(torch.cat((y3, xs2), dim=1), s2, rgb2)
        y1 = self.d1(torch.cat((y2, xs1), dim=1), s1, rgb1)

        if self.use_gcb:
            return torch.sigmoid(x + self.out_conv(torch.cat((y1, self.gcb(x_rgb)), dim=1)))
        else:
            return torch.sigmoid(x + self.out_conv(y1))


if __name__ == '__main__':
    inp = torch.rand(1, 3, 256, 256).cuda()
    model = IFBlend(16).cuda()
    model.eval()
    with torch.no_grad():
        out = model(inp)
        print(out.shape)
