import math
import torch
import torch.nn.functional as F
from torch import nn
from inspect import isfunction
from einops import rearrange
import numbers


def to_3d(x):
    return rearrange(x, 'b c h w -> b (h w) c')


def to_4d(x, h, w):
    return rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)

class BiasFree_LayerNorm(nn.Module):
    def __init__(self, normalized_shape):
        super(BiasFree_LayerNorm, self).__init__()
        if isinstance(normalized_shape, numbers.Integral):
            normalized_shape = (normalized_shape,)
        normalized_shape = torch.Size(normalized_shape)

        assert len(normalized_shape) == 1

        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.normalized_shape = normalized_shape

    def forward(self, x):
        sigma = x.var(-1, keepdim=True, unbiased=False)
        return x / torch.sqrt(sigma + 1e-5) * self.weight

class LayerNorm(nn.Module):
    def __init__(self, dim):
        super(LayerNorm, self).__init__()

        self.body = BiasFree_LayerNorm(dim)

    def forward(self, x):
        h, w = x.shape[-2:]
        return to_4d(self.body(to_3d(x)), h, w)


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        # print(dim)
        self.block = nn.Sequential(
            #分组归一化
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.Dropout，防止网络过拟合nn.Identity()
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, dim_out, 3, padding=1)
        )

    def forward(self, x):
        return self.block(x)
class EdgeBlock(nn.Module):
    def __init__(self, dim, dim_out, groups=32, dropout=0):
        super().__init__()
        # print(dim)
        self.block = nn.Sequential(
            #分组归一化
            nn.GroupNorm(groups, dim),
            Swish(),
            # nn.Dropout，防止网络过拟合nn.Identity()
            nn.Dropout(dropout) if dropout != 0 else nn.Identity(),
            nn.Conv2d(dim, 1, 3, padding=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.block(x)
class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out,  dropout=0, norm_groups=32):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=norm_groups)
        self.block2 = Block(dim_out, dim_out, groups=norm_groups, dropout=dropout)
        # nn.Identity()相当于一个恒等函数得到他之前的结果
        self.res_conv = nn.Conv2d(
            dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x):
        #对比ppdm多加了一个block1
        h = self.block1(x)
        h = self.block2(h)
        return h + self.res_conv(x)

##  Mixed-Scale Feed-forward Network (MSFN)
class FeedForward(nn.Module):
    def __init__(self, dim, ffn_expansion_factor, bias):
        super(FeedForward, self).__init__()

        hidden_features = int(dim * ffn_expansion_factor)

        self.project_in = nn.Conv2d(dim, hidden_features * 2, kernel_size=1, bias=bias)

        self.dwconv3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.dwconv5x5 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=5, stride=1, padding=2, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.dwconv3x3_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.dwconv5x5_1 = nn.Conv2d(hidden_features * 2, hidden_features, kernel_size=5, stride=1, padding=2, groups=hidden_features , bias=bias)

        self.relu3_1 = nn.ReLU()
        self.relu5_1 = nn.ReLU()

        self.project_out = nn.Conv2d(hidden_features * 2, dim, kernel_size=1, bias=bias)

    def forward(self, x):
        x = self.project_in(x)
        x1_3, x2_3 = self.relu3(self.dwconv3x3(x)).chunk(2, dim=1)
        x1_5, x2_5 = self.relu5(self.dwconv5x5(x)).chunk(2, dim=1)

        x1 = torch.cat([x1_3, x1_5], dim=1)
        x2 = torch.cat([x2_3, x2_5], dim=1)

        x1 = self.relu3_1(self.dwconv3x3_1(x1))
        x2 = self.relu5_1(self.dwconv5x5_1(x2))

        x = torch.cat([x1, x2], dim=1)

        x = self.project_out(x)

        return x


class SelfAttention(nn.Module):
    def __init__(self, in_channel, n_head=4, norm_groups=32):
        super().__init__()

        self.n_head = n_head
        self.temperature = nn.Parameter(torch.ones(n_head, 1, 1))
        self.norm = nn.GroupNorm(norm_groups, in_channel)
        self.qkv = nn.Conv2d(in_channel, in_channel * 3, 1, bias=False)
        self.qkv_dwconv = nn.Conv2d(in_channel*3, in_channel * 3, kernel_size=3,stride=1,padding=1,groups=in_channel*3, bias=False)
        self.out = nn.Conv2d(in_channel, in_channel, 1)
        self.attn_drop = nn.Dropout(0.)

        self.attn1 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn3 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)
        self.attn4 = torch.nn.Parameter(torch.tensor([0.2]), requires_grad=True)

    def forward(self, input):
        batch, channel, height, width = input.shape
        n_head = self.n_head
        # head_dim = channel // n_head
        head_dim = channel

        norm = self.norm(input)
        qkv = self.qkv_dwconv(self.qkv(norm)).view(batch,head_dim*3,height,width)
        query, key, value = qkv.chunk(3, dim=1)  # bhdyx
        q = rearrange(query, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        k = rearrange(key, 'b (head c) h w -> b head c (h w)', head=self.n_head)
        v = rearrange(value, 'b (head c) h w -> b head c (h w)', head=self.n_head)

        q = torch.nn.functional.normalize(q, dim=-1)

        k = torch.nn.functional.normalize(k, dim=-1)

        q_fft = torch.fft.rfft2(q.float())
        q_fft=torch.fft.fftshift(q_fft)
        # print(q_fft.shape)
        k_fft = torch.fft.rfft2(k.float())
        k_fft = torch.fft.fftshift(k_fft)


        _, _, C, _ = q.shape

        mask1 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask2 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask3 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        mask4 = torch.zeros(batch, self.n_head, C, C, device=input.device, requires_grad=False)
        attn = (q_fft @ k_fft.transpose(-2, -1)) * self.temperature
        # print(attn.shape)
        attn = torch.fft.ifftshift(attn)
        attn = torch.fft.irfft2(attn,s=(C,C))
        # print(attn.shape)
        #torch.topk用来求tensor中某个dim的前k大或者前k小的值以及对应的index。 k是维度
        index = torch.topk(attn, k=int(C / 2), dim=-1, largest=True)[1]
        #把1的数按照scatter的第一个参数叫维度，按行或按列把index的tensor，填入到mask中，
        #具体操作看https://blog.csdn.net/guofei_fly/article/details/104308528?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104308528-blog-108311046.235%5Ev38%5Epc_relevant_default_base&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1-104308528-blog-108311046.235%5Ev38%5Epc_relevant_default_base&utm_relevant_index=2
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 2 / 3), dim=-1, largest=True)[1]
        mask2.scatter_(-1, index, 1.)
        #torch.where 满足条件返回attn，不满足返回 torch.full_like(attn, float('-inf')表示负无穷的意思)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 3 / 4), dim=-1, largest=True)[1]
        mask3.scatter_(-1, index, 1.)
        attn3 = torch.where(mask3 > 0, attn, torch.full_like(attn, float('-inf')))

        index = torch.topk(attn, k=int(C * 4 / 5), dim=-1, largest=True)[1]
        mask4.scatter_(-1, index, 1.)
        attn4 = torch.where(mask4 > 0, attn, torch.full_like(attn, float('-inf')))

        attn1 = attn1.softmax(dim=-1)
        attn2 = attn2.softmax(dim=-1)
        attn3 = attn3.softmax(dim=-1)
        attn4 = attn4.softmax(dim=-1)

        out1 = (attn1 @ v)
        out2 = (attn2 @ v)
        out3 = (attn3 @ v)
        out4 = (attn4 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 + out3 * self.attn3 + out4 * self.attn4

        out = rearrange(out, 'b head c (h w) -> b (head c) h w', head=self.n_head, h=height, w=width)

        out = self.out(out)

        return out
    

class TransformerBlock(nn.Module):
    def __init__(self, dim, norm_groups,ffn_expansion_factor, bias):
        super(TransformerBlock, self).__init__()

        self.norm1 = LayerNorm(dim)
        self.attn = SelfAttention(dim,norm_groups=norm_groups)
        self.norm2 = LayerNorm(dim)
        self.ffn = FeedForward(dim, ffn_expansion_factor, bias)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.ffn(self.norm2(x))

        return x
    

class FSABlock(nn.Module):
    def __init__(self, dim, dim_out, *, norm_groups=32, dropout=0, with_attn=False):
        super().__init__()
        self.with_attn = with_attn
        self.res_block = ResnetBlock(
            dim, dim_out, norm_groups=norm_groups, dropout=dropout)
        if with_attn:
            self.attn = TransformerBlock(dim_out, norm_groups=norm_groups,ffn_expansion_factor=2.66,bias=False)

    def forward(self, x):
        x = self.res_block(x)
        if self.with_attn:
            x = self.attn(x)
        return x

class TFModule(nn.Module):
    def __init__(self, pre_channel, ffn_expansion_factor, bias):
        super(TFModule, self).__init__()

        self.norm = LayerNorm(pre_channel)
        self.avg_pool = torch.nn.AdaptiveAvgPool2d(1)

        self.xpatchembedding = nn.Conv2d(pre_channel , pre_channel , kernel_size=3, stride=1, padding=1,
                              groups=pre_channel , bias=bias)
        self.featurepatchembedding = nn.Conv2d(pre_channel , pre_channel , kernel_size=3, stride=1, padding=1,
                              groups=pre_channel , bias=bias)

        self.relu3 = nn.ReLU()
        self.xline1=nn.Conv2d(pre_channel,pre_channel//ffn_expansion_factor,1,padding=0,bias=False)
        self.xline2=nn.Conv2d(pre_channel//ffn_expansion_factor,pre_channel,1,padding=0,bias=False)
        self.faceline1=nn.Conv2d(pre_channel,pre_channel//ffn_expansion_factor,1,padding=0,bias=False)
        self.faceline2=nn.Conv2d(pre_channel//ffn_expansion_factor,pre_channel,1,padding=0,bias=False)
        self.x3x3_1 = nn.Conv2d(pre_channel , pre_channel, kernel_size=3, stride=1, padding=1,
                                groups=pre_channel, bias=bias)

        self.face3x3_1 = nn.Conv2d(pre_channel , pre_channel, kernel_size=3, stride=1, padding=1,
                                   groups=pre_channel, bias=bias)
        self.relux_1 = Swish()
        self.reluface_1 = Swish()

        self.project_out = nn.Conv2d(pre_channel * 2, pre_channel, kernel_size=1, bias=bias)

    def forward(self, x, feature):
        b, c, _, _ = x.shape
        x_1 = self.xpatchembedding(self.norm(x))
        feature_1 = self.featurepatchembedding(self.norm(feature))
        x_1 = self.avg_pool(x_1)

        feature_1 = self.avg_pool(feature_1)
        x_1 = self.relu3(self.xline1(x_1))
        feature_1 = self.relu3(self.faceline1(feature_1))

        x_1 = torch.sigmoid(self.xline2(x_1))
        feature_1 = torch.sigmoid(self.faceline2(feature_1))

        new_x = feature*x_1.expand_as(x)
        feature = x*feature_1.expand_as(feature)
        new_x = self.relux_1(self.x3x3_1(new_x))
        feature = self.reluface_1(self.face3x3_1(feature))

        g = torch.cat([new_x, feature], dim=1)

        g = self.project_out(g)

        return g



def calc_mean_std(features):
    """

    :param features: shape of features -> [batch_size, c, h, w]
    :return: features_mean, feature_s: shape of mean/std ->[batch_size, c, 1, 1]
    """

    batch_size, c = features.size()[:2]
    features_mean = features.reshape(batch_size, c, -1).mean(dim=2).reshape(batch_size, c, 1, 1)
    features_std = features.reshape(batch_size, c, -1).std(dim=2).reshape(batch_size, c, 1, 1) + 1e-6
    return features_mean, features_std

def adain(content_features, style_features):
    """
    Adaptive Instance Normalization

    :param content_features: shape -> [batch_size, c, h, w]
    :param style_features: shape -> [batch_size, c, h, w]
    :return: normalized_features shape -> [batch_size, c, h, w]
    """
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
    normalized_features = style_std * (content_features - content_mean) / content_std + style_mean
    return normalized_features

class CFModule(nn.Module):
    def __init__(self, pre_channel, ffn_expansion_factor, bias,alpha=1.0):
        super(CFModule, self).__init__()

        hidden_features = int(pre_channel * ffn_expansion_factor)
        self.alpha=alpha
        self.norm=LayerNorm(pre_channel)
        self.project_in = nn.Conv2d(pre_channel, hidden_features * 2, kernel_size=1, bias=bias)

        self.x3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.face3x3 = nn.Conv2d(hidden_features * 2, hidden_features * 2, kernel_size=3, stride=1, padding=1, groups=hidden_features * 2, bias=bias)
        self.relu3 = nn.ReLU()
        self.relu5 = nn.ReLU()

        self.x3x3_1 = nn.Conv2d(hidden_features , hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)
        self.face3x3_1 = nn.Conv2d(hidden_features , hidden_features, kernel_size=3, stride=1, padding=1, groups=hidden_features , bias=bias)

        self.relu3_1 = Swish()
        self.relu5_1 = Swish()

        self.project_out = nn.Conv2d(hidden_features*2 , pre_channel, kernel_size=1, bias=bias)


    def forward(self, x, feature):

        h = self.project_in(self.norm(x))
        feature = self.project_in(self.norm(feature))

        x1_3, x2_3 = self.relu3(self.x3x3(h)).chunk(2, dim=1)
        face1_3, face2_3 = self.relu5(self.face3x3(feature)).chunk(2, dim=1)
        t1 = adain(x1_3, face1_3)
        t2 = adain(x2_3, face2_3)
        t1 = self.alpha * t1 + (1 - self.alpha) * x1_3
        t2 = self.alpha * t2 + (1 - self.alpha) * x2_3
        # x1 = torch.cat([x1_3, face1_3], dim=1)
        # x2 = torch.cat([x2_3, face2_3], dim=1)

        h_feature = self.relu3_1(self.x3x3_1(t1))
        face_feature =  self.relu5_1(self.face3x3_1(t2))
        g = torch.cat([h_feature, face_feature], dim=1)
        # print(t.shape)
        g = self.project_out(g)

        return g+x


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


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


class Cvi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(Cvi, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels,
                              kernel_size, stride, padding, dilation, groups, bias)
        self.conv.apply(weights_init('gaussian'))  # 对卷积层进行函数的初始化

        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'LN':
            self.after = nn.InstanceNorm2d(out_channels)
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):

        if hasattr(self, 'before'):
            x = self.before(x)

        x = self.conv(x)

        if hasattr(self, 'after'):
            x = self.after(x)

        return x


class CvTi(nn.Module):
    def __init__(self, in_channels, out_channels, before=None, after=False, kernel_size=4, stride=2,
                 padding=1, dilation=1, groups=1, bias=False):
        super(CvTi, self).__init__()
        #nn.ConvTranspose2d进行上采样 反卷积
        self.up = nn.Upsample(scale_factor=2, mode="nearest")
        self.conv = nn.Conv2d(in_channels, out_channels, 3, padding=1)
        self.conv.apply(weights_init('gaussian'))

        if after == 'BN':
            self.after = nn.BatchNorm2d(out_channels)
        elif after == 'Tanh':
            self.after = torch.tanh
        elif after == 'sigmoid':
            self.after = torch.sigmoid

        if before == 'ReLU':
            self.before = nn.ReLU(inplace=True)
        elif before == 'LReLU':
            self.before = nn.LeakyReLU(negative_slope=0.2, inplace=True)

    def forward(self, x):
        if hasattr(self, 'before'):
            x = self.before(x)
        x = self.conv(self.up(x))
        if hasattr(self, 'after'):
            x = self.after(x)
        return x


class Upsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        # self.up = nn.PixelShuffle(2)
        self.up = nn.Upsample(scale_factor=2, mode="bilinear")
        self.conv = nn.Conv2d(dim, dim, 3, padding=1)

    def forward(self, x):
        return self.conv(self.up(x))


class Downsample(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.conv = nn.Conv2d(dim, dim, 3, 2, 1)

    def forward(self, x):
        return self.conv(x)


class GradientLossBlock(nn.Module):
    def __init__(self):
        super(GradientLossBlock, self).__init__()

    def forward(self, pred):
        _, cin, _, _ = pred.shape
        # _, cout, _, _ = target.shape
        assert cin == 3
        kx = torch.Tensor([[1, 0, -1], [2, 0, -2],
                           [1, 0, -1]]).view(1, 1, 3, 3).to(pred)
        ky = torch.Tensor([[1, 2, 1], [0, 0, 0],
                           [-1, -2, -1]]).view(1, 1, 3, 3).to(pred)
        kx = kx.repeat((cin, 1, 1, 1))
        ky = ky.repeat((cin, 1, 1, 1))

        pred_grad_x = F.conv2d(pred, kx, padding=1, groups=3)
        pred_grad_y = F.conv2d(pred, ky, padding=1, groups=3)
        # target_grad_x = F.conv2d(target, kx, padding=1, groups=3)
        # target_grad_y = F.conv2d(target, ky, padding=1, groups=3)

        # loss = (
        #     nn.L1Loss(reduction=self.reduction)
        #     (pred_grad_x, target_grad_x) +
        #     nn.L1Loss(reduction=self.reduction)
        #     (pred_grad_y, target_grad_y))
        return torch.sqrt(pred_grad_x ** 2 + pred_grad_y ** 2)
#res_block可以替换，原本2现在变1，先看看2的效果


class Generator(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            attn_res=16,
            res_blocks=1,
            dropout=0,
            image_size=128,
            grad=""
    ):
        super(Generator, self).__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.grad = grad

        downs = [nn.Conv2d(in_channel, inner_channel,
                           kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            #当now_res为16的时候，才进行注意力机制
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(FSABlock(
                    pre_channel, channel_mult, norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            FSABlock(pre_channel, pre_channel, norm_groups=norm_groups,
                     dropout=dropout, with_attn=True),
            CFModule(
                pre_channel,
                ffn_expansion_factor=2.66,
                bias=False, alpha=1.0),
            FSABlock(pre_channel, pre_channel,  norm_groups=norm_groups,
                     dropout=dropout, with_attn=False),
            CFModule(
                pre_channel,
                ffn_expansion_factor=2.66,
                bias=False, alpha=1.0)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for a in range(0, res_blocks+2):
                final = (a == res_blocks+1)
                if not final:
                    ups.append(FSABlock(
                        pre_channel + feat_channels.pop(), channel_mult,
                        norm_groups=norm_groups,
                        dropout=dropout, with_attn=use_attn))
                    pre_channel = channel_mult
                else:
                    ups.append(
                        TFModule(pre_channel, ffn_expansion_factor=16, bias=False))

            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(
            out_channel, in_channel), groups=norm_groups)

    def forward(self, x, facefeaturemaps, featuremaps):
        inp = x
        feats = []
        for layer in self.downs:
            x = layer(x)
            # print(x.shape)
            feats.append(x)

        for layer in self.mid:
            if isinstance(layer, CFModule):
                facefeaturemap = facefeaturemaps.pop()
                x = layer(x,facefeaturemap)
            else:
                x = layer(x)
        for layer in self.ups:
            if isinstance(layer, FSABlock):
                feat = feats.pop()
                x = layer(torch.cat((x, feat), dim=1))
            elif isinstance(layer, TFModule):
                featuremap = featuremaps.pop()
                x = layer(x, featuremap)
            else:
                x = layer(x)

        x = self.final_conv(x)
        return x + inp



class Face_UNet(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            res_blocks=1,
            attn_res=16,
            dropout=0,
            image_size=128,
            feature=False
    ):
        super(Face_UNet, self).__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.feature=feature

        downs = [nn.Conv2d(in_channel, inner_channel,
                       kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(FSABlock(
                pre_channel, channel_mult,  norm_groups=norm_groups,
                dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            FSABlock(pre_channel, pre_channel, norm_groups=norm_groups,
                           dropout=dropout, with_attn=True),
            FSABlock(pre_channel, pre_channel,  norm_groups=norm_groups,
                           dropout=dropout, with_attn=False)
        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(FSABlock(
                    pre_channel + feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
       
    def forward(self, x):
        # if self.grad== "grad":
        #    x=gradblock()(x)
        #encoder
        inp =x
        _,_,H,W=inp.shape
        feats = []
        edgefeats=[]
        facefeats = []
        for layer in self.downs:
            x = layer(x)
            feats.append(x)
            if isinstance(layer, FSABlock):

                edgefeats.append(x)

        for layer in self.mid:
            x = layer(x)
            facefeats.append(x)
        for layer in self.ups:
            if isinstance(layer, FSABlock):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1))
            else:
                x = layer(x)

        if self.feature:
            return self.final_conv(x), edgefeats
        else:
            return self.final_conv(x), facefeats
        

class Edge_UNet(nn.Module):
    def __init__(
            self,
            in_channel=3,
            out_channel=3,
            inner_channel=64,
            norm_groups=16,
            channel_mults=(1, 2, 4, 8),
            res_blocks=1,
            attn_res=16,
            dropout=0,
            image_size=128,
            feature=False
    ):
        super(Edge_UNet, self).__init__()
        num_mults = len(channel_mults)
        pre_channel = inner_channel
        feat_channels = [pre_channel]
        now_res = image_size
        self.feature=feature

        downs = [nn.Conv2d(in_channel, inner_channel,
                       kernel_size=3, padding=1)]
        for ind in range(num_mults):
            is_last = (ind == num_mults - 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks):
                downs.append(FSABlock(
                pre_channel, channel_mult,  norm_groups=norm_groups,
                dropout=dropout, with_attn=use_attn))
                feat_channels.append(channel_mult)
                pre_channel = channel_mult
            if not is_last:
                downs.append(Downsample(pre_channel))
                feat_channels.append(pre_channel)
                now_res = now_res // 2
        self.downs = nn.ModuleList(downs)

        self.mid = nn.ModuleList([
            FSABlock(pre_channel, pre_channel, norm_groups=norm_groups,
                           dropout=dropout, with_attn=True),

            FSABlock(pre_channel, pre_channel,  norm_groups=norm_groups,
                           dropout=dropout, with_attn=False),

        ])

        ups = []
        for ind in reversed(range(num_mults)):
            is_last = (ind < 1)
            use_attn = (str(now_res) in str(attn_res))
            channel_mult = inner_channel * channel_mults[ind]
            for _ in range(0, res_blocks + 1):
                ups.append(FSABlock(
                    pre_channel + feat_channels.pop(), channel_mult,
                    norm_groups=norm_groups,
                    dropout=dropout, with_attn=use_attn))
                pre_channel = channel_mult
            if not is_last:
                ups.append(Upsample(pre_channel))
                now_res = now_res * 2

        self.ups = nn.ModuleList(ups)

        self.final_conv = Block(pre_channel, default(out_channel, in_channel), groups=norm_groups)
       

    def forward(self, x):
        # if self.grad== "grad":
        #    x=gradblock()(x)
        #encoder
        inp =x
        _,_,H,W=inp.shape
        feats = []
        edgefeats=[]
        facefeats=[]
        for layer in self.downs:
            x = layer(x)
            feats.append(x)
            if isinstance(layer, FSABlock):
                edgefeats.append(x)

        for layer in self.mid:
                x = layer(x)

        for layer in self.ups:
            if isinstance(layer, FSABlock):
                feat = feats.pop()
                # if x.shape[2]!=feat.shape[2] or x.shape[3]!=feat.shape[3]:
                #     feat = F.interpolate(feat, x.shape[2:])
                x = layer(torch.cat((x, feat), dim=1))
            else:
                x = layer(x)

        if self.feature:
            return self.final_conv(x), edgefeats
        else:
            return self.final_conv(x), facefeats
        

class FSRNet(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.Face = Face_UNet(feature=False)
        self.Edge = Edge_UNet(feature=True)
        self.Generator = Generator()

    def forward(self, x):
        _, facefeaturemaps = self.Face(x)
        _, edgefeaturemaps = self.Edge(x)
        out = self.Generator(x,facefeaturemaps,edgefeaturemaps)
        return out


if __name__ == '__main__':
    model = FSRNet()
    x = torch.randn(1, 3, 256, 256)
    out = model(x)
    print(out.shape)
    # print(model)