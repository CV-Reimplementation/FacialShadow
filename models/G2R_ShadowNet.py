import torch
import torch.nn as nn


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class ResidualBlock(nn.Module):
    def __init__(self, in_features):
        super(ResidualBlock, self).__init__()

        conv_block = [  nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(in_features, in_features, 3),
                        nn.InstanceNorm2d(in_features)  ]

        self.conv_block = nn.Sequential(*conv_block)

    def forward(self, x):
        return x + self.conv_block(x)
    

class G2R_ShadowNet(nn.Module):
    def __init__(self,init_weights=False):
        super(G2R_ShadowNet, self).__init__()

        # Initial convolution block
        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(4, 64, 7),
                    nn.InstanceNorm2d(64),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(128, 256, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(256),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(ResidualBlock(256))
        self.conv5_b=nn.Sequential(ResidualBlock(256))
        self.conv6_b=nn.Sequential(ResidualBlock(256))
        self.conv7_b=nn.Sequential(ResidualBlock(256))
        self.conv8_b=nn.Sequential(ResidualBlock(256))
        self.conv9_b=nn.Sequential(ResidualBlock(256))
        self.conv10_b=nn.Sequential(ResidualBlock(256))
        self.conv11_b=nn.Sequential(ResidualBlock(256))
        self.conv12_b=nn.Sequential(ResidualBlock(256))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(256,128,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(64, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    
    @staticmethod
    def from_file(file_path: str) -> nn.Module:
        model = G2R_ShadowNet(init_weights=True)
        return model

    def forward(self, xin, mask):
        x=torch.cat((xin,mask),1)
        x=self.conv1_b(x)
        x=self.downconv2_b(x)
        x=self.downconv3_b(x)
        x=self.conv4_b(x)
        x=self.conv5_b(x)
        x=self.conv6_b(x)
        x=self.conv7_b(x)
        x=self.conv8_b(x)
        x=self.conv9_b(x)
        x=self.conv10_b(x)
        x=self.conv11_b(x)
        x=self.conv12_b(x)
        x=self.upconv13_b(x)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout=x+xin
        return xout.tanh()
    

if __name__ == '__main__':
    x = torch.rand(1, 3, 256, 256)
    m = torch.rand(1, 1, 256, 256)
    model = G2R_ShadowNet()
    model.eval()
    with torch.no_grad():
        res = model(x, m)
        print(res.shape)