# GAN可学习训练调度机制 vs 退火化调度机制

## 项目简介

本项目研究**可学习（Learnable）训练调度机制**与**退火化（Annealed）训练调度机制**在GAN训练中的对比效果。核心思想是让GAN自动学习调节训练过程中的关键动态参数（noise、augmentation、regularization），而非依赖人工预设的退火策略。

## 项目结构

```
641_final_project/
├── src/
│   ├── models/          # GAN模型实现
│   │   └── gan.py      # Generator和Discriminator
│   ├── schedulers/      # 参数调度器
│   │   ├── base.py     # 基类
│   │   ├── annealed.py # 退火化调度器
│   │   └── learnable.py # 可学习调度器
│   ├── utils/          # 工具函数
│   └── experiments/    # 实验脚本
├── configs/            # 配置文件
├── data/              # 数据目录
├── results/           # 结果目录
│   ├── logs/         # 训练日志
│   ├── checkpoints/  # 模型检查点
│   └── figures/      # 图表
├── implementation_plan.md  # 详细实施计划
└── README.md
```

## 快速开始

### 1. 环境配置

```bash
# 创建虚拟环境
conda create -n gan_scheduler python=3.8
conda activate gan_scheduler

# 安装依赖
pip install torch torchvision
pip install numpy matplotlib
pip install pytorch-fid  # FID评估
pip install lpips        # LPIPS评估
pip install wandb       # 实验追踪（可选）
```

### 2. 基础使用

```python
from src.models.gan import GAN
from src.schedulers.annealed import AnnealedScheduler
from src.schedulers.learnable import LearnableScheduler

# 创建GAN模型
gan = GAN(nz=100, ngf=64, ndf=64, nc=3, img_size=32, device='cuda')

# 使用退火化调度器
annealed_config = AnnealedScheduler.create_default_config()
annealed_scheduler = AnnealedScheduler(annealed_config)

# 使用可学习调度器
learnable_config = {
    'noise_var': {'initial': 1.0, 'bounds': (0.0, 2.0)},
    'augmentation_strength': {'initial': 0.5, 'bounds': (0.0, 1.0)},
    'regularization_weight': {'initial': 5.0, 'bounds': (0.0, 20.0)}
}
state_features = ['loss_g', 'loss_d', 'grad_norm_g', 'grad_norm_d', 'epoch_progress']
learnable_scheduler = LearnableScheduler(learnable_config, state_features, device='cuda')

# 在训练循环中使用
for epoch in range(total_epochs):
    # 更新调度器参数
    annealed_scheduler.update(epoch, total_epochs)
    # 或
    learnable_scheduler.update(epoch, total_epochs, 
                               generator=gan.generator,
                               discriminator=gan.discriminator,
                               losses={'g': loss_g, 'd': loss_d})
    
    # 获取当前参数值
    params = scheduler.get_parameters()
    noise_var = params['noise_var']
    # ... 使用参数进行训练
```

## 实施计划

详细的实施计划请参考 [implementation_plan.md](implementation_plan.md)

## 主要特性

1. **退火化调度**：支持线性、指数、余弦等多种退火策略
2. **可学习调度**：使用元网络自动学习最优参数调度
3. **灵活配置**：易于配置和扩展
4. **完整评估**：包含FID、IS、LPIPS等评估指标

## 实验对比

项目将对比以下方面：
- **性能**：生成质量（FID、IS）
- **效率**：训练时间、收敛速度
- **稳定性**：训练稳定性、模式崩塌风险

## 贡献

欢迎提出问题和建议！

## 许可证

MIT License

