import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import vgg19
from timm.models.swin_transformer import SwinTransformerBlock

# ------------------- 辅助模块 -------------------
class ResidualBlock(nn.Module):
    """残差块，用于细节增强"""
    def __init__(self, in_channels):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, in_channels, 3, padding=1),
            nn.InstanceNorm2d(in_channels)
        )
    
    def forward(self, x):
        return x + self.conv(x)

class MSGNet(nn.Module):
    """多尺度梯度网络"""
    def __init__(self):
        super().__init__()
        self.down1 = nn.Conv2d(3, 32, 3, stride=2, padding=1)  # 1/2
        self.down2 = nn.Conv2d(32, 64, 3, stride=2, padding=1)  # 1/4
        
        self.res_blocks = nn.Sequential(
            ResidualBlock(64),
            ResidualBlock(64)
        )
        
        self.up1 = nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        self.up2 = nn.ConvTranspose2d(32, 3, 3, stride=2, padding=1, output_padding=1)
        
    def forward(self, x):
        x = F.relu(self.down1(x))
        x = F.relu(self.down2(x))
        x = self.res_blocks(x)
        x = F.relu(self.up1(x))
        return torch.sigmoid(self.up2(x))

# ------------------- 生成器模块 -------------------
class CoarseGenerator(nn.Module):
    """带Swin-Transformer的U-Net生成器"""
    def __init__(self):
        super().__init__()
        # Encoder
        self.enc1 = nn.Sequential(
            nn.Conv2d(4, 64, 4, 2, 1),  # 224 -> 112
            nn.LeakyReLU(0.2)
        )
        self.enc2 = self._downblock(64, 128)  # 112 -> 56
        self.enc3 = self._downblock(128, 256)  # 56 -> 28
        
        # Swin-Transformer Bottleneck
        self.swin1 = SwinTransformerBlock(
            dim=256,
            input_resolution=(28, 28),
            num_heads=8,
            window_size=7,
            shift_size=0,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.
        )
        self.swin2 = SwinTransformerBlock(
            dim=256,
            input_resolution=(28, 28),
            num_heads=8,
            window_size=7,
            shift_size=3,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.
        )
        
        self.feat_norm = nn.LayerNorm(256)  # 只对通道维度做归一化
        
        # Decoder
        self.dec1 = self._upblock(256, 128)
        self.dec2 = self._upblock(128*2, 64)
        self.final = nn.Sequential(
            nn.Conv2d(64*2, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 3, 3, padding=1),
            nn.Tanh(),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        )
    
    def _downblock(self, in_c, out_c):
        return nn.Sequential(
            nn.Conv2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU(0.2)
        )
    
    def _upblock(self, in_c, out_c):
        return nn.Sequential(
            nn.ConvTranspose2d(in_c, out_c, 4, 2, 1),
            nn.BatchNorm2d(out_c),
            nn.ReLU()
        )
    
    def forward(self, x, mask):
        # 输入拼接阴影掩膜
        x = torch.cat([x, mask], dim=1)
        
        # Encoder
        e1 = self.enc1(x)    # [B, 64, 112, 112]
        e2 = self.enc2(e1)   # [B, 128, 56, 56]
        e3 = self.enc3(e2)   # [B, 256, 28, 28]
        
        # 特征形状调整
        b = e3.permute(0, 2, 3, 1)  # [B, 28, 28, 256]
        b = self.feat_norm(b)
        
        # Swin-Transformer
        b = self.swin1(b)
        b = self.swin2(b)
        
        # 转换回[B, C, H, W]格式
        b = b.permute(0, 3, 1, 2)  # [B, 256, 28, 28]
        
        # Decoder
        d1 = self.dec1(b)    # [B, 128, 56, 56]
        d1 = torch.cat([d1, e2], dim=1)
        d2 = self.dec2(d1)   # [B, 64, 112, 112]
        d2 = torch.cat([d2, e1], dim=1)
        return self.final(d2)

# ------------------- 光照校正模块 -------------------
class IlluminationCorrector(nn.Module):
    """可学习非线性光照校正"""
    def __init__(self):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(3, 128),
            nn.ReLU(),
            nn.Linear(128, 6)  # 输出3通道的scale和bias
        )
    
    def forward(self, x, params):
        # params: [B, 3]
        affines = self.mlp(params)  # [B, 6]
        scale = affines[:, :3].view(-1, 3, 1, 1)
        bias = affines[:, 3:].view(-1, 3, 1, 1)
        return x * (1 + scale) + bias  # 保持原始光照的残差形式

# ------------------- 精修模块 -------------------
class RefinementModule(nn.Module):
    def __init__(self):
        super().__init__()
        # 初始特征提取
        self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
        
        # 修改下采样层，因为输入是224x224
        self.downsample = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 224 -> 112
            nn.ReLU(),
            nn.Conv2d(64, 64, kernel_size=3, stride=2, padding=1),  # 112 -> 56
            nn.ReLU(),
            nn.BatchNorm2d(64)
        )
        
        # 第一个Swin block (56x56)
        self.norm1 = nn.LayerNorm([56, 56, 64])  # 匹配 [B, 56, 56, 64] 的输入
        self.swin1 = SwinTransformerBlock(
            dim=64,
            input_resolution=(56, 56),
            num_heads=4,
            window_size=7,
            shift_size=0,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.
        )
        
        # 下采样到28x28
        self.downsample2 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),  # 56 -> 28
            nn.ReLU(),
            nn.BatchNorm2d(128)
        )
        
        # 第二个Swin block (28x28)
        self.norm2 = nn.LayerNorm([28, 28, 128])  # 匹配 [B, 28, 28, 128] 的输入
        self.swin2 = SwinTransformerBlock(
            dim=128,
            input_resolution=(28, 28),
            num_heads=4,
            window_size=7,
            shift_size=3,
            mlp_ratio=4,
            qkv_bias=True,
            drop_path=0.
        )
        
        # 上采样回原始分辨率
        self.upsample = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),  # 28 -> 56
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 64, kernel_size=4, stride=2, padding=1),  # 56 -> 112
            nn.ReLU(),
            nn.BatchNorm2d(64),
            nn.ConvTranspose2d(64, 32, kernel_size=4, stride=2, padding=1),  # 112 -> 224
            nn.ReLU(),
            nn.BatchNorm2d(32)
        )
        
        self.final_conv = nn.Conv2d(32, 3, kernel_size=3, padding=1)
        self.illum_corrector = IlluminationCorrector()
        self.msg_net = MSGNet()
    
    def forward(self, coarse_img):
        # 打印每一步的尺寸
        print(f"\nRefinement input shape: {coarse_img.shape}")
        
        # 初始特征提取
        x = self.conv1(coarse_img)
        print(f"After conv1 shape: {x.shape}")
        
        # 下采样到56x56
        x = self.downsample(x)
        print(f"After downsample shape: {x.shape}")
        
        # 第一个Swin block
        b = x.permute(0, 2, 3, 1).contiguous()
        print(f"Before swin1 shape: {b.shape}")
        b = self.norm1(b)
        b = self.swin1(b)
        b = b.permute(0, 3, 1, 2).contiguous()
        print(f"After swin1 shape: {b.shape}")
        
        # 下采样到28x28
        x = self.downsample2(b)  # [B, 128, 28, 28]
        
        # 第二个Swin block
        b = x.permute(0, 2, 3, 1).contiguous()  # [B, 28, 28, 128]
        b = self.norm2(b)
        b = self.swin2(b)
        b = b.permute(0, 3, 1, 2).contiguous()  # [B, 128, 28, 28]
        
        # 特征聚合
        global_feat = F.adaptive_avg_pool2d(b, (1, 1)).squeeze(-1).squeeze(-1)
        
        # 上采样和最终处理
        x = self.upsample(b)  # [B, 32, 224, 224]
        x = self.final_conv(x)  # [B, 3, 224, 224]
        
        # 光照校正和细节增强
        corrected = self.illum_corrector(coarse_img, global_feat[:, :3])
        detail = self.msg_net(coarse_img)
        
        return corrected + detail + x

# ------------------- 判别器模块 -------------------
class MultiScaleDiscriminator(nn.Module):
    """带谱归一化的多尺度判别器"""
    def __init__(self):
        super().__init__()
        # 全局判别器 (输入1/2分辨率)
        self.global_disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0)
        )
        
        # 局部判别器 (原始分辨率)
        self.local_disc = nn.Sequential(
            nn.utils.spectral_norm(nn.Conv2d(3, 64, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(64, 128, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.utils.spectral_norm(nn.Conv2d(128, 256, 4, 2, 1)),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 1, 4, 1, 0)
        )
        
    def forward(self, x):
        global_input = F.interpolate(x, scale_factor=0.5)
        return [self.global_disc(global_input), self.local_disc(x)]

# ------------------- 损失函数 -------------------
class IlluminationAwareLoss(nn.Module):
    def __init__(self):
        super().__init__()
        # VGG特征提取
        self.vgg = vgg19(pretrained=True).features[:16].eval()
        # 将VGG移动到与输入相同的设备上
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.vgg = self.vgg.to(self.device)
        for param in self.vgg.parameters():
            param.requires_grad = False
            
        self.l1 = nn.L1Loss()
        self.bce = nn.BCEWithLogitsLoss()
        
    def perceptual_loss(self, pred, target):
        # 确保VGG和输入在同一设备上
        self.vgg = self.vgg.to(pred.device)
        return self.l1(self.vgg(pred), self.vgg(target))
    
    def illumination_smoothness(self, pred):
        # 光照平滑约束
        grad_x = torch.mean(torch.abs(pred[..., :-1] - pred[..., 1:]), dim=1)
        grad_y = torch.mean(torch.abs(pred[..., :-1, :] - pred[..., 1:, :]), dim=1)
        return grad_x.mean() + grad_y.mean()
    
    def forward(self, pred, target, mask_pred, mask_gt):
        # 基础损失
        l1_loss = self.l1(pred, target)
        percep_loss = self.perceptual_loss(pred, target)
        smooth_loss = self.illumination_smoothness(pred)
        mask_loss = self.bce(mask_pred, mask_gt)
        
        return (
            1.0 * l1_loss + 
            0.5 * percep_loss + 
            0.1 * smooth_loss + 
            0.2 * mask_loss
        )

# ------------------- 完整模型 -------------------
class SwinIllumGAN(nn.Module):
    """完整的阴影去除网络"""
    def __init__(self):
        super().__init__()
        # 阴影掩码预测
        self.mask_predictor = nn.Sequential(
            nn.Conv2d(3, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 1, 3, padding=1),
            nn.Sigmoid()
        )
        
        # 粗糙生成器
        self.coarse = CoarseGenerator()
        # 精修模块
        self.refinement = RefinementModule()
    
    def forward(self, x):
        # 打印输入尺寸
        print(f"Input shape: {x.shape}")
        
        # 预测阴影掩码
        shadow_mask = self.mask_predictor(x)
        print(f"Shadow mask shape: {shadow_mask.shape}")
        
        # 粗糙去除
        coarse = self.coarse(x, shadow_mask)
        print(f"Coarse output shape: {coarse.shape}")
        
        # 精修
        refined = self.refinement(coarse)
        return refined, shadow_mask

# ------------------- 训练辅助函数 -------------------
def compute_gradient_penalty(D, real_samples, fake_samples):
    """WGAN-GP梯度惩罚"""
    alpha = torch.rand(real_samples.size(0), 1, 1, 1, device=real_samples.device)
    interpolates = (alpha * real_samples + (1 - alpha) * fake_samples).requires_grad_(True)
    d_interpolates = D(interpolates)
    
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=torch.ones_like(d_interpolates),
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    
    gradients = gradients.view(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()
    return gradient_penalty

import torch


def test_model():
    # 配置测试参数
    batch_size = 2
    img_size = 224
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 初始化模型
    model = SwinIllumGAN().to(device)
    discriminator = MultiScaleDiscriminator().to(device)
    criterion = IlluminationAwareLoss()
    
    # 打印参数量
    total_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {total_params/1e6:.2f}M")
    
    # 生成测试数据
    dummy_input = torch.randn(batch_size, 3, img_size, img_size).to(device)
    
    # 前向传播测试
    try:
        with torch.no_grad():
            refined_img, shadow_mask = model(dummy_input)
            print(f"Input shape: {dummy_input.shape}")
            print(f"Refined output shape: {refined_img.shape}")
            print(f"Shadow mask shape: {shadow_mask.shape}")
            
            # 判别器测试
            fake_logits = discriminator(refined_img)
            print(f"Discriminator outputs: [Global: {fake_logits[0].shape}, Local: {fake_logits[1].shape}]")
            
            # 损失计算测试
            target = torch.rand_like(refined_img)
            mask_gt = torch.rand_like(shadow_mask)
            loss = criterion(refined_img, target, shadow_mask, mask_gt)
            print(f"Total loss: {loss.item():.4f}")
            
        print("\nAll tests passed successfully!")
        
    except Exception as e:
        print(f"\nTest failed with error: {str(e)}")
        print("\nStack trace:")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    test_model()