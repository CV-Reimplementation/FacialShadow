import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class ResNeXt101(nn.Module):
    def __init__(self):
        super(ResNeXt101, self).__init__()
        net = models.resnext101_32x8d() # weights='DEFAULT'

        net = list(net.children())
        self.layer0 = nn.Sequential(*net[:3])
        self.layer1 = nn.Sequential(*net[3: 5])
        self.layer2 = net[5]
        self.layer3 = net[6]
        self.layer4 = net[7]

    def forward(self, x):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)
        return layer4
    

def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0, bias=False)


def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)


class Spacial_IRNN(nn.Module):
    def __init__(self, in_channels, alpha=1.0):
        super(Spacial_IRNN, self).__init__()
        self.left_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.right_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.up_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.down_weight = nn.Conv2d(in_channels, in_channels, kernel_size=1, stride=1, groups=in_channels, padding=0)
        self.left_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.right_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.up_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))
        self.down_weight.weight = nn.Parameter(torch.tensor([[[[alpha]]]]*in_channels))

    def forward(self, input):
        return irnn.apply(input, self.up_weight.weight, self.right_weight.weight, self.down_weight.weight, self.left_weight.weight, self.up_weight.bias, self.right_weight.bias, self.down_weight.bias, self.left_weight.bias)

class Attention(nn.Module):
    def __init__(self, in_channels):
        super(Attention, self).__init__()
        self.out_channels = int(in_channels / 2)
        self.conv1 = nn.Conv2d(in_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(self.out_channels, self.out_channels, kernel_size=3, padding=1, stride=1)
        self.relu2 = nn.ReLU()
        self.conv3 = nn.Conv2d(self.out_channels, 4, kernel_size=1, padding=0, stride=1)
        self.sigmod = nn.Sigmoid()
    
    def forward(self, x):
        out = self.conv1(x)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.relu2(out)
        out = self.conv3(out)
        out = self.sigmod(out)
        return out


class DSC_Module(nn.Module):
    def __init__(self, in_channels, out_channels, attention=1, alpha=1.0):
        super(DSC_Module, self).__init__()
        self.out_channels = out_channels
        self.irnn1 = Spacial_IRNN(self.out_channels, alpha)
        self.irnn2 = Spacial_IRNN(self.out_channels, alpha)
        self.conv_in = conv1x1(in_channels, in_channels)
        self.conv2 = conv1x1(in_channels * 4, in_channels)
        self.conv3 = conv1x1(in_channels * 4, in_channels)
        self.relu2 = nn.ReLU(True)
        self.attention = attention
        if self.attention:
            self.attention_layer = Attention(in_channels)

    def forward(self, x):
        if self.attention:
            weight = self.attention_layer(x)
        out = self.conv_in(x)
        top_up, top_right, top_down, top_left = self.irnn1(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv2(out)
        top_up, top_right, top_down, top_left = self.irnn2(out)
        
        # direction attention
        if self.attention:
            top_up.mul(weight[:, 0:1, :, :])
            top_right.mul(weight[:, 1:2, :, :])
            top_down.mul(weight[:, 2:3, :, :])
            top_left.mul(weight[:, 3:4, :, :])
        
        out = torch.cat([top_up, top_right, top_down, top_left], dim=1)
        out = self.conv3(out)
        out = self.relu2(out)
        
        return out

class LayerConv(nn.Module):
    def __init__(self, in_planes, out_planes, kernel_size, stride, padding, relu):
        super(LayerConv, self).__init__()
        self.conv = nn.Conv2d(in_channels=in_planes, out_channels=out_planes, kernel_size=kernel_size,
                              stride=stride, padding=padding)
        self.relu = nn.ReLU() if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.relu is not None:
            x = self.relu(x)

        return x

class Predict(nn.Module):
    def __init__(self, in_planes=32, out_planes=1, kernel_size=1):
        super(Predict, self).__init__()
        self.conv = nn.Conv2d(in_planes, out_planes, kernel_size)

    def forward(self, x):
        y = self.conv(x)

        return y

class DSC(nn.Module):
    def __init__(self):
        super(DSC, self).__init__()

        resnext = ResNeXt101()
        self.layer0 = resnext.layer0
        self.layer1 = resnext.layer1
        self.layer2 = resnext.layer2
        self.layer3 = resnext.layer3
        self.layer4 = resnext.layer4

        self.layer4_conv1 = LayerConv(2048, 512, 7, 1, 3, True)
        self.layer4_conv2 = LayerConv(512, 512, 7, 1, 3, True)
        self.layer4_dsc = DSC_Module(512, 512)
        self.layer4_conv3 = LayerConv(1024, 32, 1, 1, 0, False)

        self.layer3_conv1 = LayerConv(1024, 256, 5, 1, 2, True)
        self.layer3_conv2 = LayerConv(256, 256, 5, 1, 2, True)
        self.layer3_dsc = DSC_Module(256, 256)
        self.layer3_conv3 = LayerConv(512, 32, 1, 1, 0, False)

        self.layer2_conv1 = LayerConv(512, 128, 5, 1, 2, True)
        self.layer2_conv2 = LayerConv(128, 128, 5, 1, 2, True)
        self.layer2_dsc = DSC_Module(128, 128)
        self.layer2_conv3 = LayerConv(256, 32, 1, 1, 0, False)

        self.layer1_conv1 = LayerConv(256, 64, 3, 1, 1, True)
        self.layer1_conv2 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer1_dsc = DSC_Module(64, 64, alpha=0.8)
        self.layer1_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.layer0_conv1 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer0_conv2 = LayerConv(64, 64, 3, 1, 1, True)
        self.layer0_dsc = DSC_Module(64, 64, alpha=0.8)
        self.layer0_conv3 = LayerConv(128, 32, 1, 1, 0, False)

        self.relu = nn.ReLU()

        self.global_conv = LayerConv(160, 32, 1, 1, 0, True)

        # output channel to 3
        self.layer4_predict = Predict(32, 3, 1)
        self.layer3_predict_ori = Predict(32, 3, 1)
        self.layer3_predict = Predict(6, 3, 1)
        self.layer2_predict_ori = Predict(32, 3, 1)
        self.layer2_predict = Predict(9, 3, 1)
        self.layer1_predict_ori = Predict(32, 3, 1)
        self.layer1_predict = Predict(12, 3, 1)
        self.layer0_predict_ori = Predict(32, 3, 1)
        self.layer0_predict = Predict(15, 3, 1)
        self.global_predict = Predict(32, 3, 1)
        self.fusion_predict = Predict(18, 3, 1)

    def forward(self, x, x_non_norm):
        layer0 = self.layer0(x)
        layer1 = self.layer1(layer0)
        layer2 = self.layer2(layer1)
        layer3 = self.layer3(layer2)
        layer4 = self.layer4(layer3)

        layer4_conv1 = self.layer4_conv1(layer4)
        layer4_conv2 = self.layer4_conv2(layer4_conv1)
        layer4_dsc = self.layer4_dsc(layer4_conv2)
        layer4_context = torch.cat((layer4_conv2, layer4_dsc), 1)
        layer4_conv3 = self.layer4_conv3(layer4_context)
        layer4_up = F.interpolate(layer4_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer4_up = self.relu(layer4_up)

        layer3_conv1 = self.layer3_conv1(layer3)
        layer3_conv2 = self.layer3_conv2(layer3_conv1)
        layer3_dsc = self.layer3_dsc(layer3_conv2)
        layer3_context = torch.cat((layer3_conv2, layer3_dsc), 1)
        layer3_conv3 = self.layer3_conv3(layer3_context)
        layer3_up = F.interpolate(layer3_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer3_up = self.relu(layer3_up)

        layer2_conv1 = self.layer2_conv1(layer2)
        layer2_conv2 = self.layer2_conv2(layer2_conv1)
        layer2_dsc = self.layer2_dsc(layer2_conv2)
        layer2_context = torch.cat((layer2_conv2, layer2_dsc), 1)
        layer2_conv3 = self.layer2_conv3(layer2_context)
        layer2_up = F.interpolate(layer2_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer2_up = self.relu(layer2_up)

        layer1_conv1 = self.layer1_conv1(layer1)
        layer1_conv2 = self.layer1_conv2(layer1_conv1)
        layer1_dsc = self.layer1_dsc(layer1_conv2)
        layer1_context = torch.cat((layer1_conv2, layer1_dsc), 1)
        layer1_conv3 = self.layer1_conv3(layer1_context)
        layer1_up = F.interpolate(layer1_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer1_up = self.relu(layer1_up)

        layer0_conv1 = self.layer0_conv1(layer0)
        layer0_conv2 = self.layer0_conv2(layer0_conv1)
        layer0_dsc = self.layer0_dsc(layer0_conv2)
        layer0_context = torch.cat((layer0_conv2, layer0_dsc), 1)
        layer0_conv3 = self.layer0_conv3(layer0_context)
        layer0_up = F.interpolate(layer0_conv3, size=x.size()[2:], mode='bilinear', align_corners=True)
        layer0_up = self.relu(layer0_up)

        global_concat = torch.cat((layer0_up, layer1_up, layer2_up, layer3_up, layer4_up), 1)
        global_conv = self.global_conv(global_concat)

        layer4_predict = self.layer4_predict(layer4_up)

        layer3_predict_ori = self.layer3_predict_ori(layer3_up)
        layer3_concat = torch.cat((layer3_predict_ori, layer4_predict), 1)
        layer3_predict = self.layer3_predict(layer3_concat)

        layer2_predict_ori = self.layer2_predict_ori(layer2_up)
        layer2_concat = torch.cat((layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer2_predict = self.layer2_predict(layer2_concat)

        layer1_predict_ori = self.layer1_predict_ori(layer1_up)
        layer1_concat = torch.cat((layer1_predict_ori, layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer1_predict = self.layer1_predict(layer1_concat)

        layer0_predict_ori = self.layer0_predict_ori(layer0_up)
        layer0_concat = torch.cat((layer0_predict_ori, layer1_predict_ori, layer2_predict_ori, layer3_predict_ori, layer4_predict), 1)
        layer0_predict = self.layer0_predict(layer0_concat)

        global_predict = self.global_predict(global_conv)

        # fusion
        fusion_concat = torch.cat((layer0_predict, layer1_predict, layer2_predict, layer3_predict, layer4_predict, global_predict), 1)
        fusion_predict = self.fusion_predict(fusion_concat)


        # send x_non_norm to device
        x_non_norm = x_non_norm.to(x.device)
        layer4_predict = layer4_predict + x_non_norm
        layer3_predict = layer3_predict + x_non_norm
        layer2_predict = layer2_predict + x_non_norm
        layer1_predict = layer1_predict + x_non_norm
        layer0_predict = layer0_predict + x_non_norm
        global_predict = global_predict + x_non_norm
        fusion_predict = fusion_predict + x_non_norm
        return fusion_predict
    

class irnn(torch.autograd.Function):
    @staticmethod
    def forward(ctx, input_feature, weight_up, weight_right, weight_down, weight_left,
               bias_up, bias_right, bias_down, bias_left):
        
        batch, channels, height, width = input_feature.shape
        device = input_feature.device
        
        # Initialize outputs with same shape as input
        output_up = torch.zeros_like(input_feature)
        output_right = torch.zeros_like(input_feature)
        output_down = torch.zeros_like(input_feature)
        output_left = torch.zeros_like(input_feature)
        
        # Reshape weights and biases for broadcasting
        weight_up = weight_up.view(1, -1, 1, 1)
        weight_right = weight_right.view(1, -1, 1, 1)
        weight_down = weight_down.view(1, -1, 1, 1)
        weight_left = weight_left.view(1, -1, 1, 1)
        
        bias_up = bias_up.view(1, -1, 1, 1)
        bias_right = bias_right.view(1, -1, 1, 1)
        bias_down = bias_down.view(1, -1, 1, 1)
        bias_left = bias_left.view(1, -1, 1, 1)
        
        # Left direction - process each row in parallel
        prev = F.relu(input_feature[:, :, :, -1]).unsqueeze(-1)  # Keep last dimension
        for w in range(width-2, -1, -1):
            curr = prev * weight_left + bias_left + input_feature[:, :, :, w:w+1]
            prev = F.relu(curr)
            output_left[:, :, :, w] = prev.squeeze(-1)
        
        # Right direction - process each row in parallel
        prev = F.relu(input_feature[:, :, :, 0]).unsqueeze(-1)
        for w in range(1, width):
            curr = prev * weight_right + bias_right + input_feature[:, :, :, w:w+1]
            prev = F.relu(curr)
            output_right[:, :, :, w] = prev.squeeze(-1)
        
        # Up direction - process each column in parallel
        prev = F.relu(input_feature[:, :, -1, :]).unsqueeze(-2)  # Keep second-to-last dimension
        for h in range(height-2, -1, -1):
            curr = prev * weight_up + bias_up + input_feature[:, :, h:h+1, :]
            prev = F.relu(curr)
            output_up[:, :, h, :] = prev.squeeze(-2)
        
        # Down direction - process each column in parallel
        prev = F.relu(input_feature[:, :, 0, :]).unsqueeze(-2)
        for h in range(1, height):
            curr = prev * weight_down + bias_down + input_feature[:, :, h:h+1, :]
            prev = F.relu(curr)
            output_down[:, :, h, :] = prev.squeeze(-2)
        
        ctx.save_for_backward(input_feature, weight_up, weight_right, weight_down, weight_left,
                            output_up, output_right, output_down, output_left)
        
        return output_up, output_right, output_down, output_left

    @staticmethod
    def backward(ctx, grad_output_up, grad_output_right, grad_output_down, grad_output_left):
        input_feature, weight_up, weight_right, weight_down, weight_left, \
        output_up, output_right, output_down, output_left = ctx.saved_tensors
        
        batch, channels, height, width = input_feature.shape
        
        # Initialize gradients
        grad_input = torch.zeros_like(input_feature)
        
        # Create masks for positive outputs
        mask_up = (output_up > 0).float()
        mask_right = (output_right > 0).float()
        mask_down = (output_down > 0).float()
        mask_left = (output_left > 0).float()
        
        # Apply masks to gradients
        grad_output_up = grad_output_up * mask_up
        grad_output_right = grad_output_right * mask_right
        grad_output_down = grad_output_down * mask_down
        grad_output_left = grad_output_left * mask_left
        
        # Compute weight gradients using batch matrix multiplication
        # Reshape tensors for batch matmul
        output_reshaped = output_up.permute(0, 2, 3, 1).reshape(-1, channels)
        grad_reshaped = grad_output_up.permute(0, 2, 3, 1).reshape(-1, channels)
        
        # Compute gradients for weights and biases
        grad_weight_up = torch.sum(output_reshaped.t() @ grad_reshaped, dim=1)
        grad_weight_right = torch.sum(output_right.reshape(-1, channels).t() @ grad_output_right.reshape(-1, channels), dim=1)
        grad_weight_down = torch.sum(output_down.reshape(-1, channels).t() @ grad_output_down.reshape(-1, channels), dim=1)
        grad_weight_left = torch.sum(output_left.reshape(-1, channels).t() @ grad_output_left.reshape(-1, channels), dim=1)
        
        # Compute bias gradients
        grad_bias_up = grad_output_up.sum(dim=(0, 2, 3))
        grad_bias_right = grad_output_right.sum(dim=(0, 2, 3))
        grad_bias_down = grad_output_down.sum(dim=(0, 2, 3))
        grad_bias_left = grad_output_left.sum(dim=(0, 2, 3))
        
        # Accumulate input gradients
        grad_input = (grad_output_up + grad_output_right + grad_output_down + grad_output_left)
        
        return (grad_input, grad_weight_up, grad_weight_right, grad_weight_down, grad_weight_left,
                grad_bias_up, grad_bias_right, grad_bias_down, grad_bias_left)

# class irnn(torch.autograd.Function):
#     @staticmethod
#     def forward(ctx, input_feature, weight_up, weight_right, weight_down, weight_left,
#                bias_up, bias_right, bias_down, bias_left):
        
#         batch, channels, height, width = input_feature.shape
#         device = input_feature.device
        
#         # Initialize outputs
#         output_up = torch.zeros_like(input_feature)
#         output_right = torch.zeros_like(input_feature)
#         output_down = torch.zeros_like(input_feature)
#         output_left = torch.zeros_like(input_feature)
        
#         # Process each batch and channel
#         for b in range(batch):
#             for c in range(channels):
#                 # Left direction
#                 for h in range(height):
#                     prev = F.relu(input_feature[b, c, h, -1])
#                     for w in range(width-2, -1, -1):
#                         curr = prev * weight_left[c] + bias_left[c] + input_feature[b, c, h, w]
#                         prev = F.relu(curr)
#                         output_left[b, c, h, w] = prev
                
#                 # Right direction
#                 for h in range(height):
#                     prev = F.relu(input_feature[b, c, h, 0])
#                     for w in range(1, width):
#                         curr = prev * weight_right[c] + bias_right[c] + input_feature[b, c, h, w]
#                         prev = F.relu(curr)
#                         output_right[b, c, h, w] = prev
                
#                 # Up direction
#                 for w in range(width):
#                     prev = F.relu(input_feature[b, c, -1, w])
#                     for h in range(height-2, -1, -1):
#                         curr = prev * weight_up[c] + bias_up[c] + input_feature[b, c, h, w]
#                         prev = F.relu(curr)
#                         output_up[b, c, h, w] = prev
                
#                 # Down direction
#                 for w in range(width):
#                     prev = F.relu(input_feature[b, c, 0, w])
#                     for h in range(1, height):
#                         curr = prev * weight_down[c] + bias_down[c] + input_feature[b, c, h, w]
#                         prev = F.relu(curr)
#                         output_down[b, c, h, w] = prev
        
#         # Save tensors needed for backward
#         ctx.save_for_backward(input_feature, weight_up, weight_right, weight_down, weight_left,
#                             output_up, output_right, output_down, output_left)
        
#         return output_up, output_right, output_down, output_left

#     @staticmethod
#     def backward(ctx, grad_output_up, grad_output_right, grad_output_down, grad_output_left):
#         input_feature, weight_up, weight_right, weight_down, weight_left, \
#         output_up, output_right, output_down, output_left = ctx.saved_tensors
        
#         batch, channels, height, width = input_feature.shape
        
#         # Initialize gradients
#         grad_input = torch.zeros_like(input_feature)
#         grad_weight_up = torch.zeros_like(weight_up)
#         grad_weight_right = torch.zeros_like(weight_right)
#         grad_weight_down = torch.zeros_like(weight_down)
#         grad_weight_left = torch.zeros_like(weight_left)
#         grad_bias_up = torch.zeros_like(weight_up).reshape(weight_up.size(0))
#         grad_bias_right = torch.zeros_like(weight_right).reshape(weight_right.size(0))
#         grad_bias_down = torch.zeros_like(weight_down).reshape(weight_down.size(0))
#         grad_bias_left = torch.zeros_like(weight_left).reshape(weight_left.size(0))
        
#         # Process gradients for each batch and channel
#         for b in range(batch):
#             for c in range(channels):
#                 # Left direction
#                 for h in range(height):
#                     for w in range(width-1, -1, -1):
#                         if output_left[b, c, h, w] > 0:
#                             grad = grad_output_left[b, c, h, w]
#                             if w < width-1:
#                                 grad_weight_left[c] += output_left[b, c, h, w+1] * grad
#                                 grad_bias_left[c] += grad
#                             grad_input[b, c, h, w] += grad
                
#                 # Right direction
#                 for h in range(height):
#                     for w in range(width):
#                         if output_right[b, c, h, w] > 0:
#                             grad = grad_output_right[b, c, h, w]
#                             if w > 0:
#                                 grad_weight_right[c] += output_right[b, c, h, w-1] * grad
#                                 grad_bias_right[c] += grad
#                             grad_input[b, c, h, w] += grad
                
#                 # Up direction
#                 for w in range(width):
#                     for h in range(height-1, -1, -1):
#                         if output_up[b, c, h, w] > 0:
#                             grad = grad_output_up[b, c, h, w]
#                             if h < height-1:
#                                 grad_weight_up[c] += output_up[b, c, h+1, w] * grad
#                                 grad_bias_up[c] += grad
#                             grad_input[b, c, h, w] += grad
                
#                 # Down direction
#                 for w in range(width):
#                     for h in range(height):
#                         if output_down[b, c, h, w] > 0:
#                             grad = grad_output_down[b, c, h, w]
#                             if h > 0:
#                                 grad_weight_down[c] += output_down[b, c, h-1, w] * grad
#                                 grad_bias_down[c] += grad
#                             grad_input[b, c, h, w] += grad
        
#         return (grad_input, grad_weight_up, grad_weight_right, grad_weight_down, grad_weight_left,
#                 grad_bias_up, grad_bias_right, grad_bias_down, grad_bias_left)

if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    m = torch.rand(1, 1, 512, 512)
    model = DSC()
    model.eval()
    with torch.no_grad():
        res = model(x, m)
        print(res.shape)