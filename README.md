# GAN可学习训练调度机制 vs 退火化调度机制

## 项目简介

本项目研究**可学习（Learnable）训练调度机制**与**退火化（Annealed）训练调度机制**在**文本到图像GAN（Text-to-Image GAN）**训练中的对比效果。

**基于论文**: "Learning Schedules for Text-to-Image GANs: A Controlled Study of Learnable and Annealed Training Dynamics"

核心思想是让GAN自动学习调节训练过程中的关键动态参数（noise、augmentation、regularization），而非依赖人工预设的退火策略。

## 项目结构

```
641_final_project/
├── src/
│   ├── models/                    # GAN模型实现
│   │   ├── gan.py                # 基础GAN（DCGAN）
│   │   └── text_to_image_gan.py  # 文本到图像GAN（主要）
│   ├── schedulers/                # 参数调度器
│   │   ├── base.py               # 基类
│   │   ├── annealed.py           # 退火化调度器
│   │   └── learnable.py          # 可学习调度器
│   ├── utils/                    # 工具函数
│   │   └── datasets.py          # 文本-图像数据集加载
│   └── experiments/              # 实验脚本
├── configs/                      # 配置文件
├── data/                        # 数据目录
│   ├── CUB_200_2011/           # CUB-200数据集（推荐）
│   └── COCO/                    # COCO数据集（可选）
├── results/                     # 结果目录
│   ├── logs/                   # 训练日志
│   ├── checkpoints/            # 模型检查点
│   └── figures/                # 图表
├── implementation_plan.md      # 详细实施计划
├── DATASET_GUIDE.md           # 数据集准备指南
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

### 2. 数据集准备

**重要**: 本项目需要文本-图像配对数据集！

推荐使用**CUB-200-2011**数据集进行实验。详细的数据集准备指南请参考 [DATASET_GUIDE.md](DATASET_GUIDE.md)

```bash
# 下载CUB-200数据集
cd data/
# 下载并解压CUB_200_2011.tgz
```

### 3. 基础使用

**文本到图像GAN**（主要使用）:
```python
from src.models.text_to_image_gan import TextToImageGAN
from src.schedulers.annealed import AnnealedScheduler
from src.schedulers.learnable import LearnableScheduler
from src.utils.datasets import CUB200Dataset, collate_fn
from torch.utils.data import DataLoader

# 创建文本到图像GAN模型
vocab_size = 2000  # 根据数据集词汇表大小设置
gan = TextToImageGAN(
    vocab_size=vocab_size,
    nz=100, ngf=64, ndf=64, nc=3,
    img_size=64, text_dim=256,
    device='cuda'
)

# 加载数据集
dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# 使用调度器训练
# ... (见训练示例)
```

**基础GAN**（用于对比）:
```python
from src.models.gan import GAN

# 创建基础GAN模型（无文本条件）
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

## 快速开始

**最快开始**：查看 [QUICK_START.md](QUICK_START.md) - 3步运行Fixed Annealing实验

**详细文档**：
- **如何运行代码？** 查看 [HOW_TO_RUN.md](HOW_TO_RUN.md) - 操作指南
- **理解项目功能？** 查看 [CODE_ROADMAP.md](CODE_ROADMAP.md) - 功能理解指南
- **数据集准备？** 查看 [DATASET_GUIDE.md](DATASET_GUIDE.md) - 数据集下载、划分、引用
- **中期报告？** 查看 [MIDTERM_REPORT_TEMPLATE.md](MIDTERM_REPORT_TEMPLATE.md) - 报告模板

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

