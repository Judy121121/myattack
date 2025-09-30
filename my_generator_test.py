import torch
import torch.nn as nn

ngf = 64

class MyGeneratorResnet(nn.Module):
    def __init__(self, inception=False, eps=1.0, evaluate=True):
        '''
        :param inception: if True crop layer will be added to go from 3x300x300 t0 3x299x299.
        '''
        super(MyGeneratorResnet, self).__init__()
        self.inception = inception
        # 关键点处理模块 - 时空特征提取
        self.keypoint_encoder = nn.Sequential(
            nn.Linear(133 * 3, 256),
            nn.ReLU(True),
            nn.Linear(256, 256),
            nn.ReLU(True),
            nn.Linear(256, 56 * 56)  # 输出与特征图空间尺寸匹配
        )

        # 关键点注意力生成模块
        self.keypoint_attention = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(True),
            nn.Conv2d(32, 1, kernel_size=3, padding=1, bias=False),
            nn.Sigmoid()  # 生成0-1的注意力图
        )

        # 时空特征融合模块
        self.spatiotemporal_fusion = nn.Sequential(
            nn.Conv2d(ngf * 4 * 2, ngf * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            nn.Conv2d(ngf * 4, ngf * 4, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )
        # Input_size = 3, n, n
        self.block1 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(3, ngf, kernel_size=7, padding=0, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.block2 = nn.Sequential(
            nn.Conv2d(ngf, ngf * 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.block3 = nn.Sequential(
            nn.Conv2d(ngf * 2, ngf * 4, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True)
        )

        # Input size = 3, n/4, n/4
        # Residual Blocks: 6
        self.resblock1 = ResidualBlock(ngf * 4)
        self.resblock2 = ResidualBlock(ngf * 4)
        self.resblock3 = ResidualBlock(ngf * 4)
        self.resblock4 = ResidualBlock(ngf * 4)
        self.resblock5 = ResidualBlock(ngf * 4)
        self.resblock6 = ResidualBlock(ngf * 4)

        # Input size = 3, n/4, n/4
        self.upsampl_inf1 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl_inf2 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf_inf = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 3, kernel_size=7, padding=0)
        )

        # Input size = 3, n/4, n/4
        self.upsampl_01 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 4, ngf * 2, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True)
        )

        # Input size = 3, n/2, n/2
        self.upsampl_02 = nn.Sequential(
            nn.ConvTranspose2d(ngf * 2, ngf, kernel_size=3, stride=2, padding=1, output_padding=1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True)
        )

        # Input size = 3, n, n
        self.blockf_0 = nn.Sequential(
            nn.ReflectionPad2d(3),
            nn.Conv2d(ngf, 1, kernel_size=7, padding=0)
        )

        self.crop = nn.ConstantPad2d((0, -1, -1, 0), 0)

        self.eps = eps
        self.evaluate = evaluate


    def forward(self, input, grad,keypoints):
        n, _, h, w = input.shape
        # 1. 处理关键点信息 - 提取时空特征
        keypoint_features = self.keypoint_encoder(keypoints.reshape(n, -1))  # (n, 56*56)
        keypoint_features = keypoint_features.view(n, 1, 56, 56)  # (n, 1, 56, 56)

        # 2. 生成关键点注意力图
        attention_map = self.keypoint_attention(keypoint_features)  # (n, 1, 56, 56)
        x = self.block1(input)
        x = self.block2(x)
        x = self.block3(x)
        # 4. 时空特征融合
        # 复制关键点特征以匹配通道数
        keypoint_features_expanded = keypoint_features.repeat(1, ngf * 4, 1, 1)  # (n, ngf*4, 56, 56)

        # 注意力加权融合
        spatiotemporal_features = attention_map * keypoint_features_expanded + (1 - attention_map) * x

        # 拼接原始特征与时空特征
        fused_features = torch.cat([x, spatiotemporal_features], dim=1)  # (n, ngf*4*2, 56, 56)

        # 融合特征
        x = self.spatiotemporal_fusion(fused_features)  # (n, ngf*4, 56, 56)

        x = self.resblock1(x)
        x = self.resblock2(x)
        x = self.resblock3(x)
        x = self.resblock4(x)
        x = self.resblock5(x)
        x = self.resblock6(x)
        code = x
        x = self.upsampl_inf1(code)
        x = self.upsampl_inf2(x)
        x = self.blockf_inf(x)
        if self.inception:
            x = self.crop(x)
        x_inf = self.eps * torch.tanh(x)  # [-eps, eps]
        x = self.upsampl_01(code)
        x = self.upsampl_02(x)
        x = self.blockf_0(x)
        if self.inception:
            x = self.crop(x)
        x = (torch.tanh(x) + 1) / 2

        if self.evaluate:
            x_0 = torch.where(x < 0.5, torch.zeros_like(x).detach(), torch.ones_like(x).detach())
        else:
            x_0 = torch.where(torch.rand(x.shape).cuda()< 0.5, x,
                              torch.where(x < 0.5, torch.zeros_like(x), torch.ones_like(x)*grad).detach())
        x_0 = x_0 * grad

        x_out = torch.clamp((x_inf * x_0) + input, min=0, max=1)
        grad_img = torch.clamp(grad * input, min=0, max=1)
        # return x_out, x_inf, x_0, x, grad_img
        return x_out, x_0, x


class ResidualBlock(nn.Module):
    def __init__(self, num_filters):
        super(ResidualBlock, self).__init__()
        self.block = nn.Sequential(
            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters),
            nn.ReLU(True),

            nn.Dropout(0.5),

            nn.ReflectionPad2d(1),
            nn.Conv2d(in_channels=num_filters, out_channels=num_filters, kernel_size=3, stride=1, padding=0,
                      bias=False),
            nn.BatchNorm2d(num_filters)
        )

    def forward(self, x):
        residual = self.block(x)
        return x + residual