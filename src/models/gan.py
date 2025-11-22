"""
基础GAN模型实现
支持Generator和Discriminator
"""
import torch
import torch.nn as nn


class Generator(nn.Module):
    """GAN生成器 - DCGAN架构示例"""
    
    def __init__(self, nz=100, ngf=64, nc=3, img_size=32):
        """
        Args:
            nz: 输入噪声维度
            ngf: 生成器特征图数量
            nc: 输出通道数（RGB=3）
            img_size: 输出图像尺寸
        """
        super(Generator, self).__init__()
        self.nz = nz
        self.img_size = img_size
        
        # 计算初始特征图尺寸
        # 对于32x32图像，需要5层上采样（2^5 = 32）
        self.main = nn.Sequential(
            # 输入是Z，进入全连接层
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # 状态尺寸: (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # 状态尺寸: (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # 状态尺寸: (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 状态尺寸: (nc) x 32 x 32
        )
    
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    """GAN判别器 - DCGAN架构示例"""
    
    def __init__(self, nc=3, ndf=64, img_size=32):
        """
        Args:
            nc: 输入通道数（RGB=3）
            ndf: 判别器特征图数量
            img_size: 输入图像尺寸
        """
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            # 输入尺寸: (nc) x 32 x 32
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf) x 16 x 16
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf*2) x 8 x 8
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf*4) x 4 x 4
            nn.Conv2d(ndf * 4, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)


class GAN:
    """GAN包装类，包含生成器和判别器"""
    
    def __init__(self, nz=100, ngf=64, ndf=64, nc=3, img_size=32, device='cuda'):
        self.device = device
        self.nz = nz
        
        self.generator = Generator(nz, ngf, nc, img_size).to(device)
        self.discriminator = Discriminator(nc, ndf, img_size).to(device)
        
        # 初始化权重
        self.generator.apply(self._weights_init)
        self.discriminator.apply(self._weights_init)
    
    @staticmethod
    def _weights_init(m):
        """权重初始化"""
        classname = m.__class__.__name__
        if classname.find('Conv') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
        elif classname.find('BatchNorm') != -1:
            nn.init.normal_(m.weight.data, 1.0, 0.02)
            nn.init.constant_(m.bias.data, 0)
    
    def sample_noise(self, batch_size):
        """采样噪声"""
        return torch.randn(batch_size, self.nz, 1, 1, device=self.device)
    
    def generate(self, batch_size):
        """生成样本"""
        noise = self.sample_noise(batch_size)
        with torch.no_grad():
            fake_images = self.generator(noise)
        return fake_images

