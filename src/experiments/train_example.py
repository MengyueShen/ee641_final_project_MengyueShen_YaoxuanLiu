"""
示例训练脚本
展示如何使用退火化和可学习调度器训练GAN
"""
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import os
import sys

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.models.gan import GAN
from src.schedulers.annealed import AnnealedScheduler
from src.schedulers.learnable import LearnableScheduler


def train_gan_with_scheduler(gan, dataloader, scheduler, num_epochs, 
                             scheduler_type='annealed', device='cuda'):
    """
    使用调度器训练GAN
    
    Args:
        gan: GAN模型
        dataloader: 数据加载器
        scheduler: 参数调度器
        num_epochs: 训练epoch数
        scheduler_type: 'annealed' 或 'learnable'
        device: 设备
    """
    # 优化器
    optimizer_g = optim.Adam(gan.generator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    optimizer_d = optim.Adam(gan.discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
    
    # 损失函数
    criterion = nn.BCELoss()
    real_label = 1.0
    fake_label = 0.0
    
    # 训练循环
    for epoch in range(num_epochs):
        for i, (real_images, _) in enumerate(dataloader):
            batch_size = real_images.size(0)
            real_images = real_images.to(device)
            
            # ========== 更新调度器参数 ==========
            if scheduler_type == 'annealed':
                scheduler.update(epoch, num_epochs)
            elif scheduler_type == 'learnable':
                # 需要先计算一些状态信息
                # 这里简化处理，实际应该从训练过程中获取
                scheduler.update(epoch, num_epochs,
                                generator=gan.generator,
                                discriminator=gan.discriminator,
                                losses={'g': 0.0, 'd': 0.0})  # 占位符
            
            params = scheduler.get_parameters()
            noise_var = params.get('noise_var', 1.0)
            reg_weight = params.get('regularization_weight', 1.0)
            
            # ========== 训练判别器 ==========
            gan.discriminator.zero_grad()
            
            # 真实数据
            label = torch.full((batch_size,), real_label, dtype=torch.float, device=device)
            output = gan.discriminator(real_images)
            loss_d_real = criterion(output, label)
            loss_d_real.backward()
            
            # 生成数据
            noise = torch.randn(batch_size, gan.nz, 1, 1, device=device) * noise_var
            fake_images = gan.generator(noise)
            label.fill_(fake_label)
            output = gan.discriminator(fake_images.detach())
            loss_d_fake = criterion(output, label)
            loss_d_fake.backward()
            
            loss_d = loss_d_real + loss_d_fake
            optimizer_d.step()
            
            # ========== 训练生成器 ==========
            gan.generator.zero_grad()
            label.fill_(real_label)
            output = gan.discriminator(fake_images)
            loss_g = criterion(output, label)
            loss_g.backward()
            optimizer_g.step()
            
            # 打印进度
            if i % 100 == 0:
                print(f'Epoch [{epoch}/{num_epochs}] Batch [{i}/{len(dataloader)}] '
                      f'Loss_D: {loss_d.item():.4f} Loss_G: {loss_g.item():.4f} '
                      f'Noise_Var: {noise_var:.4f} Reg_Weight: {reg_weight:.4f}')
        
        # 每个epoch保存参数历史
        print(f'Epoch {epoch} completed. Parameters: {params}')


def main():
    """主函数"""
    # 配置
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    batch_size = 64
    num_epochs = 50
    dataset_name = 'CIFAR10'  # 或 'MNIST', 'CelebA' 等
    
    # 数据加载
    if dataset_name == 'CIFAR10':
        transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])
        dataset = datasets.CIFAR10(root='./data', train=True, download=True, transform=transform)
        nc = 3
        img_size = 32
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")
    
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    # 创建GAN模型
    gan = GAN(nz=100, ngf=64, ndf=64, nc=nc, img_size=img_size, device=device)
    
    # 选择调度器类型
    scheduler_type = 'annealed'  # 或 'learnable'
    
    if scheduler_type == 'annealed':
        # 退火化调度器
        config = AnnealedScheduler.create_default_config()
        scheduler = AnnealedScheduler(config)
        print("Using Annealed Scheduler")
    else:
        # 可学习调度器
        config = {
            'noise_var': {'initial': 1.0, 'bounds': (0.0, 2.0)},
            'augmentation_strength': {'initial': 0.5, 'bounds': (0.0, 1.0)},
            'regularization_weight': {'initial': 5.0, 'bounds': (0.0, 20.0)}
        }
        state_features = ['loss_g', 'loss_d', 'grad_norm_g', 'grad_norm_d', 'epoch_progress']
        scheduler = LearnableScheduler(config, state_features, device=device)
        print("Using Learnable Scheduler")
    
    # 训练
    print(f"Starting training on {device}...")
    train_gan_with_scheduler(gan, dataloader, scheduler, num_epochs, 
                            scheduler_type=scheduler_type, device=device)
    
    print("Training completed!")


if __name__ == '__main__':
    main()

