# GAN可学习训练调度机制 vs 退火化调度机制

## 项目简介

本项目研究**可学习（Learnable）训练调度机制**与**退火化（Annealed）训练调度机制**在**文本到图像GAN（Text-to-Image GAN）**训练中的对比效果。

**基于论文**: "Learning Schedules for Text-to-Image GANs: A Controlled Study of Learnable and Annealed Training Dynamics"

核心思想是让GAN自动学习调节训练过程中的关键动态参数（noise、augmentation、regularization），而非依赖人工预设的退火策略。

## 当前进度

✅ **已完成**：
- 三种核心调度机制实现（Fixed Annealing、Learnable Monotone、Adaptive Annealing）
- 文本到图像GAN模型（TextEncoder、ConditionalGenerator、ConditionalDiscriminator）
- CUB-200数据集加载和自动划分（train/val/test）
- 完整的可视化系统（10类可视化指标）
- Fixed Annealing完整训练流程和结果可视化

🚧 **进行中**：
- LearnableMonotone和AdaptiveAnnealed的训练逻辑完善
- 真实评估指标实现（FID、IS、CLIP Score）
- 数据集迁移到COCO（CUB-200缺乏完整文本描述）

## 项目结构

```
641_final_project/
├── src/
│   ├── models/                    # GAN模型实现
│   │   ├── gan.py                # 基础GAN（DCGAN）
│   │   └── text_to_image_gan.py  # 文本到图像GAN（主要）
│   ├── schedulers/                # 参数调度器
│   │   ├── base.py               # 抽象基类
│   │   ├── annealed.py           # Fixed Annealing调度器 ✅
│   │   ├── learnable_monotone.py # Learnable Monotone调度器 ✅
│   │   ├── adaptive_annealed.py  # Adaptive Annealing调度器 ✅
│   │   └── learnable.py          # 对比用调度器
│   ├── utils/                    # 工具函数
│   │   ├── datasets.py           # 文本-图像数据集加载（CUB-200、COCO）
│   │   └── visualization.py      # 可视化工具（10类指标）
│   └── experiments/              # 实验脚本
│       └── train_full_pipeline.py # 完整训练流程（Fixed Annealing）✅
├── configs/                      # 配置文件
├── data/                        # 数据目录
│   ├── CUB_200_2011/            # CUB-200数据集（当前使用）
│   └── processed/               # 预处理后的数据
├── results/                     # 结果目录
│   ├── logs/                    # 训练日志
│   ├── checkpoints/             # 模型检查点
│   └── figures/                 # 可视化图表
├── CODE_ROADMAP.md              # 项目功能理解指南（通俗版）
├── DATASET_GUIDE.md             # 数据集准备指南
├── RUN_FIXED_ANNEALING.md       # Fixed Annealing运行指南
├── requirements.txt             # Python依赖
└── README.md
```

## 快速开始

### 1. 环境配置

**推荐使用Python 3.8+**

```bash
# 创建虚拟环境（使用conda）
conda create -n gan_scheduler python=3.8
conda activate gan_scheduler

# 或使用venv
python -m venv venv
source venv/bin/activate  # Linux/Mac
# venv\Scripts\activate  # Windows

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据集准备

**重要**: 本项目需要文本-图像配对数据集！

**当前使用**: CUB-200-2011数据集（约1.1GB）

详细的数据集准备指南请参考 [DATASET_GUIDE.md](DATASET_GUIDE.md)

**快速步骤**：
1. 访问 https://www.vision.caltech.edu/datasets/cub_200_2011/
2. 使用浏览器下载 `CUB_200_2011.tgz`（约1.1GB）
3. 将文件移动到 `data/` 目录
4. 解压：`tar -xzf CUB_200_2011.tgz`
5. 数据集会自动划分：训练集（约4,795张）、验证集（约1,199张）、测试集（约5,794张）

**注意**：CUB-200数据集缺乏完整的文本描述，当前使用文件名作为文本输入。计划迁移到COCO数据集以获得更丰富的文本描述。

### 3. 运行实验

#### 运行Fixed Annealing实验（推荐开始）

这是当前最完整的实验流程，包含完整的训练和可视化：

```bash
# 确保在项目根目录
cd "/Users/smy/Documents/learning/studying/learning/641deeplearning system/641_final_project"

# 运行完整训练流程
python src/experiments/train_full_pipeline.py
```

**输出**：
- 训练日志保存在 `results/logs/`
- 模型检查点保存在 `results/checkpoints/`
- 可视化图表保存在 `results/figures/`：
  - `schedule_params_fixed_annealing.png` - 调度参数曲线（核心可视化）
  - `loss_curves_fixed_annealing.png` - 损失曲线
  - `grad_norms_fixed_annealing.png` - 梯度范数
  - `metrics_vs_steps_fixed_annealing.png` - 性能指标
  - `samples_grid_fixed_annealing.png` - 生成样本对比

详细运行指南请参考 [RUN_FIXED_ANNEALING.md](RUN_FIXED_ANNEALING.md)

#### 使用代码API

**文本到图像GAN + Fixed Annealing**:
```python
from src.models.text_to_image_gan import TextToImageGAN
from src.schedulers.annealed import AnnealedScheduler
from src.utils.datasets import CUB200Dataset, collate_fn
from torch.utils.data import DataLoader

# 创建模型
vocab_size = 2000  # 根据数据集词汇表大小设置
gan = TextToImageGAN(
    vocab_size=vocab_size,
    nz=100, ngf=64, ndf=64, nc=3,
    img_size=64, text_dim=256,
    device='cuda' if torch.cuda.is_available() else 'cpu'
)

# 加载数据集
dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
dataloader = DataLoader(dataset, batch_size=16, collate_fn=collate_fn)

# 创建调度器
scheduler = AnnealedScheduler({
    'noise_var': {'initial': 1.0, 'final': 0.01, 'schedule': 'exponential'},
    'augmentation_strength': {'initial': 0.8, 'final': 0.1, 'schedule': 'linear'},
    'regularization_weight': {'initial': 10.0, 'final': 1.0, 'schedule': 'cosine'}
})

# 训练循环中使用
for epoch in range(total_epochs):
    scheduler.update(epoch, total_epochs)
    params = scheduler.get_parameters()
    noise_var = params['noise_var']
    # ... 使用参数进行训练
```

## 三种核心调度机制

1. **Fixed Annealing（固定退火）** ✅ 已完成
   - 支持线性、指数、余弦、三角波四种退火策略
   - 预定义数学函数，不学习
   - 作为基准线（baseline）

2. **Learnable Monotone（可学习单调）** ✅ 已实现，训练逻辑待完善
   - 使用K-bin softmax参数化单调调度
   - 通过梯度学习最优调度曲线
   - 数学上保证单调性

3. **Adaptive Annealing（自适应退火）** ✅ 已实现，训练逻辑待完善
   - 在基础调度上叠加轻量级控制器
   - 根据训练信号（loss、梯度）动态调整
   - 最高级的调度机制

详细说明请参考 [CODE_ROADMAP.md](CODE_ROADMAP.md)

## 可视化指标

项目包含10类可视化指标：
1. 训练性能曲线（FID/IS/CLIP vs Steps）
2. Loss曲线（Generator/Discriminator/Regularization）
3. **Schedule参数曲线（σ(u), p_aug(u), λ_reg(u)）** ⭐ 核心可视化
4. 梯度范数曲线
5. 生成样本对比网格
6. 跨seed方差分析
7. CLIP-Score分布
8. Controller输出（Adaptive Annealing）
9. 多轴分析
10. Pareto front（质量 vs 算力）

## 实验对比

项目将对比以下方面：
- **性能**：生成质量（FID、IS、CLIP Score）
- **效率**：训练时间、收敛速度
- **稳定性**：训练稳定性、模式崩塌风险、跨seed方差

## 文档说明

- **理解项目功能？** 查看 [CODE_ROADMAP.md](CODE_ROADMAP.md) - 通俗易懂的功能理解指南
- **数据集准备？** 查看 [DATASET_GUIDE.md](DATASET_GUIDE.md) - 数据集下载、划分、引用说明
- **运行Fixed Annealing？** 查看 [RUN_FIXED_ANNEALING.md](RUN_FIXED_ANNEALING.md) - 详细运行指南

## 贡献

欢迎提出问题和建议！

## 许可证

MIT License

