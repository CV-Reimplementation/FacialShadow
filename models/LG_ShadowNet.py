import torch
import torch.nn as nn

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


def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm2d') != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class LG_ShadowNet(nn.Module):
    def __init__(self,init_weights=False):
        super(LG_ShadowNet, self).__init__()

        # Initial convolution block
        self.conv1_L=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(1, 32, 7),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True))
        self.downconv2_L=nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.downconv3_L=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.conv4_L=nn.Sequential(ResidualBlock(128))
        self.conv5_L=nn.Sequential(ResidualBlock(128))

        self.conv1_b=nn.Sequential(nn.ReflectionPad2d(3),
                    nn.Conv2d(3, 32, 7),
                    nn.InstanceNorm2d(32),
                    nn.ReLU(inplace=True))
        self.downconv2_b=nn.Sequential(nn.Conv2d(32, 64, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(64),
                                    nn.ReLU(inplace=True))
        self.downconv3_b=nn.Sequential(nn.Conv2d(64, 128, 3, stride=2, padding=1),
                                    nn.InstanceNorm2d(128),
                                    nn.ReLU(inplace=True))
        self.conv4_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv5_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv6_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv7_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv8_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv9_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv10_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv11_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.conv12_b=nn.Sequential(nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128),
                        nn.ReLU(inplace=True),
                        nn.ReflectionPad2d(1),
                        nn.Conv2d(128, 128, 3),
                        nn.InstanceNorm2d(128))
        self.upconv13_b=nn.Sequential(nn.ConvTranspose2d(128,64,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(64),
                        nn.ReLU(inplace=True))
        self.upconv14_b=nn.Sequential(nn.ConvTranspose2d(64,32,3,stride=2,padding=1,output_padding=1),
                        nn.InstanceNorm2d(32),
                        nn.ReLU(inplace=True))
        self.conv15_b=nn.Sequential(nn.ReflectionPad2d(3),nn.Conv2d(32, 3, 7))
        
        if init_weights:
            self.apply(weights_init_normal)
    

    def forward(self,xin, xinl):
        
        x1_L=self.conv1_L(xinl)
        x2_L=self.downconv2_L(x1_L)
        x3_L=self.downconv3_L(x2_L)
        x4_L=self.conv4_L(x3_L)
        x5_L=self.conv5_L(x4_L)
        
        x1=self.conv1_b(xin)
        x2=self.downconv2_b(x1)
        x3=self.downconv3_b(x2)
        
        x4=self.conv4_b(torch.mul(x3,x3_L))+x3
        x5=self.conv5_b(torch.mul(x4,x4_L))+x4
        x6=self.conv6_b(torch.mul(x5,x5_L))+x5
        x7=self.conv7_b(x6)+x6
        x8=self.conv8_b(x7)+x7
        x9=self.conv9_b(x8)+x8
        x10=self.conv10_b(x9)+x9
        x11=self.conv11_b(x10)+x10
        x12=self.conv12_b(x11)+x11

        x=self.upconv13_b(x12)
        x=self.upconv14_b(x)
        x=self.conv15_b(x)
        xout = x + xin
        return xout.tanh()
    

if __name__ == '__main__':
    model = LG_ShadowNet()
    xin = torch.randn(1,3,256,256)
    xinl = torch.randn(1,1,256,256)
    xout=model(xin,xinl)
    print(xout.shape)