# 如何运行代码 - 快速操作指南

## 📋 项目是什么？

研究**三种Schedule机制**在文本到图像GAN训练中的效果：
1. **Fixed Annealing**（固定退火）- 使用数学公式
2. **Learnable Monotone**（可学习单调）- K-bin softmax学习
3. **Adaptive Annealing**（自适应退火）- 固定退火+小控制器

---

## 🚀 快速开始（3步）

### 步骤1：安装依赖

```bash
pip install -r requirements.txt
```

### 步骤2：准备数据集

**选项A：使用CUB-200（推荐，快速验证）**
```bash
cd data/
# 下载地址: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
wget http://www.vision.caltech.edu/visipedia-data/CUB-200-2011/CUB_200_2011.tgz
tar -xzf CUB_200_2011.tgz
```

**目录结构**:
```
CUB_200_2011/
├── images.txt              # 图像列表
├── train_test_split.txt    # 训练/测试划分
├── images/                 # 图像文件夹
└── text/                   # 文本描述文件夹（可选）
```

**选项B：使用MS-COCO（proposal指定，较大）**
```bash
cd data/
# 下载地址: https://cocodataset.org/
# 需要下载: train2017.zip, val2017.zip, annotations_trainval2017.zip
# 解压后结构:
# COCO/
#   ├── train2017/          # 训练图像
#   ├── val2017/            # 验证图像
#   └── annotations/        # 标注文件
```

**验证数据集**:
```python
from src.utils.datasets import CUB200Dataset
dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
print(f"数据集大小: {len(dataset)}")
print(f"词汇表大小: {dataset.vocab_size}")
```

### 步骤3：运行训练

```bash
# 使用Fixed Annealing
python src/experiments/train_text2image.py

# 或修改代码中的scheduler_type来选择不同的调度器
```

---

## 📁 代码结构（简单理解）

```
src/
├── models/
│   ├── text_to_image_gan.py  # 文本到图像GAN（主要用这个）
│   └── gan.py                 # 基础GAN（对比用）
│
├── schedulers/                # Schedule机制（核心）
│   ├── annealed.py           # Fixed Annealing ✅
│   ├── learnable_monotone.py # Learnable Monotone ✅
│   ├── adaptive_annealed.py  # Adaptive Annealing ✅
│   └── learnable.py          # MLP学习（对比用）
│
├── utils/
│   └── datasets.py           # 数据集加载
│
└── experiments/
    └── train_text2image.py   # 训练脚本
```

---

## 💻 代码使用示例

### 示例1：使用Fixed Annealing

```python
from src.models.text_to_image_gan import TextToImageGAN
from src.schedulers.annealed import AnnealedScheduler
from src.utils.datasets import CUB200Dataset, collate_fn
from torch.utils.data import DataLoader

# 1. 加载数据
dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collate_fn)

# 2. 创建模型
gan = TextToImageGAN(vocab_size=dataset.vocab_size, device='cuda')

# 3. 创建调度器
config = AnnealedScheduler.create_default_config()
scheduler = AnnealedScheduler(config)

# 4. 训练循环
for epoch in range(100):
    scheduler.update(epoch, 100)  # 更新参数
    params = scheduler.get_parameters()
    
    for batch in dataloader:
        # 使用params['noise_var']等参数训练GAN
        # ... 训练代码
```

### 示例2：使用Learnable Monotone

```python
from src.schedulers.learnable_monotone import LearnableMonotoneScheduler

# 创建可学习单调调度器
config = LearnableMonotoneScheduler.create_default_config()
scheduler = LearnableMonotoneScheduler(config, k_bins=10, device='cuda')

# 训练时，schedule参数会通过GAN损失自动学习
# 需要将schedule参数加入优化器（见下方"训练逻辑"）
```

### 示例3：使用Adaptive Annealing

```python
from src.schedulers.adaptive_annealed import AdaptiveAnnealedScheduler

config = AdaptiveAnnealedScheduler.create_default_config()
state_features = ['loss_g', 'loss_d', 'grad_norm_g', 'grad_norm_d', 'epoch_progress']
scheduler = AdaptiveAnnealedScheduler(config, state_features, device='cuda')

# 更新时需要提供训练状态
scheduler.update(epoch, 100,
                generator=gan.generator,
                discriminator=gan.discriminator,
                losses={'g': loss_g, 'd': loss_d})
```

---

## ⚠️ 当前代码状态

### ✅ 可以运行的

- ✅ Fixed Annealing：完全可用
- ✅ 文本到图像GAN模型：可以训练
- ✅ 数据集加载：CUB-200和COCO都支持

### ⚠️ 需要完善的

- ⚠️ LearnableMonotone：代码完成，但**训练逻辑未实现**
  - 需要将schedule参数加入优化器
  - 通过GAN损失反向传播更新bin权重

- ⚠️ AdaptiveAnnealed：代码完成，但**训练逻辑未实现**
  - 需要实现双层优化
  - 训练小控制器

- ⚠️ 训练脚本：有基础版本，但**不完整**
  - 缺少模型保存
  - 缺少评估指标
  - 缺少可视化

---

## 🔧 如何让代码跑通（最小版本）

### 方案1：只用Fixed Annealing（最简单）

```python
# 修改 train_text2image.py
scheduler_type = 'annealed'  # 只用固定退火

# 运行
python src/experiments/train_text2image.py
```

**需要做的**：
1. 下载数据集
2. 确保代码没有语法错误
3. 运行即可

### 方案2：测试LearnableMonotone（需要完善训练逻辑）

**需要添加的代码**：
```python
# 在train_text2image.py中，将schedule参数加入优化器
optimizer_g = optim.Adam(
    list(gan.generator.parameters()) + 
    list(gan.text_encoder.parameters()) +
    list(scheduler.schedules.parameters()),  # 加入schedule参数
    lr=0.0002
)

# 训练时，schedule参数会通过loss_g.backward()自动更新
```

### 方案3：测试AdaptiveAnnealed（需要双层优化）

**需要添加的代码**：
```python
# 创建控制器优化器
optimizer_controller = optim.Adam(
    scheduler.controllers.parameters(),
    lr=0.001  # 较小的学习率
)

# 每K个epoch更新控制器
if epoch % 5 == 0:
    # 在验证集上评估
    fid_score = evaluate_fid(...)
    
    # 更新控制器
    optimizer_controller.zero_grad()
    fid_score.backward()  # 需要设计如何反向传播
    optimizer_controller.step()
```

---

## 📝 最小可运行版本（Fixed Annealing）

创建一个最简单的可运行脚本：

```python
# simple_train.py
import torch
from src.models.text_to_image_gan import TextToImageGAN
from src.schedulers.annealed import AnnealedScheduler
from src.utils.datasets import CUB200Dataset, collate_fn
from torch.utils.data import DataLoader

# 1. 数据
dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
dataloader = DataLoader(dataset, batch_size=4, collate_fn=collate_fn)

# 2. 模型
gan = TextToImageGAN(vocab_size=dataset.vocab_size, device='cuda')

# 3. 调度器
scheduler = AnnealedScheduler(AnnealedScheduler.create_default_config())

# 4. 训练（简化版）
for epoch in range(10):
    scheduler.update(epoch, 10)
    params = scheduler.get_parameters()
    print(f"Epoch {epoch}: noise_var={params['noise_var']:.4f}")
    
    # 这里可以添加实际的训练代码
```

---

## 🐛 常见问题

### Q1: 数据集找不到？

**A**: 
- 检查路径：`./data/CUB_200_2011/` 是否存在
- 检查文件：确保有`images.txt`和`train_test_split.txt`

### Q2: 词汇表错误？

**A**: 
- 确保数据集已正确加载
- 检查`dataset.vocab_size`是否正确

### Q3: CUDA out of memory？

**A**: 
- 减小batch_size（改为4或8）
- 使用CPU：`device='cpu'`

### Q4: 如何切换不同的调度器？

**A**: 修改`train_text2image.py`中的`scheduler_type`：
```python
scheduler_type = 'annealed'  # 或 'learnable_monotone' 或 'adaptive'
```

---

## 📊 三种Schedule机制对比

| 机制 | 文件 | 是否需要训练 | 难度 |
|------|------|------------|------|
| Fixed Annealing | `annealed.py` | ❌ 不需要 | ⭐ 简单 |
| Learnable Monotone | `learnable_monotone.py` | ✅ 需要 | ⭐⭐⭐ 中等 |
| Adaptive Annealing | `adaptive_annealed.py` | ✅ 需要 | ⭐⭐⭐⭐ 较难 |

**建议**：先从Fixed Annealing开始，确保代码能跑通，再逐步添加其他机制。

---

## 🎯 下一步操作（按顺序）

1. **下载数据集**（必须）
2. **测试Fixed Annealing**（确保基础功能正常）
3. **完善训练脚本**（添加保存、日志等）
4. **实现评估指标**（FID等）
5. **实现LearnableMonotone训练逻辑**
6. **实现AdaptiveAnnealed训练逻辑**

---

---

## 🚀 运行Fixed Annealing完整实验

### 步骤1：下载数据集

见 [DATASET_GUIDE.md](DATASET_GUIDE.md) - 数据集准备指南

### 步骤2：运行完整训练

```bash
python src/experiments/train_full_pipeline.py
```

**这个脚本会**：
- ✅ 自动加载数据集并划分（训练集80%，验证集20%，测试集使用官方划分）
- ✅ 训练50个epoch（使用Fixed Annealing）
- ✅ 每5个epoch在验证集上评估
- ✅ 记录所有损失、梯度、schedule参数
- ✅ 生成所有可视化图像
- ✅ 保存模型检查点

### 步骤3：查看结果

**生成的文件**：
- `results/figures/metrics_vs_steps_fixed_annealing.png` - FID/IS/CLIP vs Steps ⭐
- `results/figures/loss_curves_fixed_annealing.png` - Loss曲线 ⭐
- `results/figures/schedule_params_fixed_annealing.png` - Schedule参数曲线 ⭐⭐⭐核心
- `results/figures/grad_norms_fixed_annealing.png` - 梯度范数
- `results/checkpoints/best_model.pth` - 最佳模型
- `results/training_data.json` - 训练数据
- `results/final_results.json` - 最终结果

**这些图像可直接用于中期报告的Results to Date部分**

---

**最后更新**: 2024年12月  
**核心文件**: `src/experiments/train_full_pipeline.py` - 运行这个即可！

