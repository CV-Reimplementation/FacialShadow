import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import numpy as np
from timm.models.swin_transformer import swin_tiny_patch4_window7_224

# Define Transformer Branch 
class TransformerBranch(nn.Module):
    def __init__(self):
        super(TransformerBranch, self).__init__()
        self.encoder = swin_tiny_patch4_window7_224(pretrained=True)
        self.encoder.head = nn.Identity()
        # 添加一个平均池化层来处理空间维度
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.proj = nn.Linear(768, 768)
    
    def forward(self, x):
        features = self.encoder.forward_features(x)  # [B, 7, 7, 768]
        # 调整维度顺序，将通道维度移到第二维
        features = features.permute(0, 3, 1, 2)     # [B, 768, 7, 7]
        # 使用平均池化减少空间维度
        features = self.avg_pool(features)          # [B, 768, 1, 1]
        features = features.flatten(1)              # [B, 768]
        x = self.proj(features)
        return x

# Define U-Net Generator for GAN Branch
class UNetGenerator(nn.Module):
    def __init__(self, in_channels=3, out_channels=3):
        super(UNetGenerator, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.ConvTranspose2d(64, out_channels, kernel_size=4, stride=2, padding=1),
            nn.Tanh()
        )
    
    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

# PatchGAN Discriminator
class PatchGAN(nn.Module):
    def __init__(self, in_channels=3):
        super(PatchGAN, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 1, kernel_size=4, stride=1, padding=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.model(x)

# Combined Model
class FaceShadowRemoval(nn.Module):
    def __init__(self):
        super(FaceShadowRemoval, self).__init__()
        self.transformer_branch = TransformerBranch()
        self.gan_branch = UNetGenerator()
        self.fusion = nn.Linear(768, 256)  # Feature fusion
        self.channel_adjust = nn.Conv2d(256, 3, kernel_size=1)
    
    def forward(self, x):
        global_features = self.transformer_branch(x)
        if global_features.shape[1] != 768:
            raise ValueError(f"Expected global_features shape [B, 768], but got {global_features.shape}")
        fusion_features = self.fusion(global_features).unsqueeze(-1).unsqueeze(-1)
        fusion_features = self.channel_adjust(fusion_features)
        fusion_features = fusion_features.expand(-1, -1, x.shape[2], x.shape[3]) 

        enhanced_image = self.gan_branch(x + fusion_features)
        return enhanced_image

# Define Loss Functions
# l1_loss = nn.L1Loss()
# gan_loss = nn.BCELoss()



# # 设置设备
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# # 载入模型
# model = FaceShadowRemoval().to(device)
# model.eval()  # 设置为测试模式

# # 创建随机测试输入（模拟一张3通道RGB的脸部图像）
# test_image = torch.rand(2, 3, 224, 224).to(device)  # batch_size=1, 3通道, 224x224

# # 运行模型
# with torch.no_grad():
#     output_image = model(test_image)

# # 确保输出尺寸正确
# assert output_image.shape == test_image.shape, f"Output shape {output_image.shape} does not match input {test_image.shape}"

# # 计算 L1 loss 进行简单检查
# l1_loss = torch.nn.L1Loss()
# loss = l1_loss(output_image, test_image)
# print(f"Test Passed! L1 Loss: {loss.item():.4f}")