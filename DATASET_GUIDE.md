# 数据集准备指南

## 📚 数据集引用

如果使用CUB-200-2011数据集，请在论文/报告中引用：

```bibtex
@techreport{WahCUB_200_2011,
    Title = {The Caltech-UCSD Birds-200-2011 Dataset},
    Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
    Year = {2011},
    Institution = {California Institute of Technology},
    Number = {CNS-TR-2011-001}
}
```

**重要提示**：数据集仅用于**非商业研究和教育目的**

---

## 📥 数据集下载

### ⚠️ 重要：必须使用浏览器下载

CUB-200数据集**较大（约1.1GB）**，直接下载链接可能失效，**必须使用浏览器下载**。

### 下载步骤

**步骤1：访问下载页面**

打开浏览器，访问：
**http://www.vision.caltech.edu/visipedia/CUB-200-2011.html**

**步骤2：下载数据集**

1. 在页面上找到 **"Download"** 部分
2. 点击下载 **"Images and annotations (1.1 GB)"**
3. 文件名为 `CUB_200_2011.tgz`
4. **文件大小应该约1.1GB**（如果只有几KB，说明下载失败）

**步骤3：移动到项目目录并解压**

```bash
# 移动到data目录（根据你的实际下载位置调整）
mv ~/Downloads/CUB_200_2011.tgz "/Users/smy/Documents/learning/studying/learning/641deeplearning system/641_final_project/data/"

# 进入data目录
cd "/Users/smy/Documents/learning/studying/learning/641deeplearning system/641_final_project/data"

# 验证文件大小（应该约1.1GB）
ls -lh CUB_200_2011.tgz

# 解压
tar -xzf CUB_200_2011.tgz
```

**步骤4：验证解压结果**

```bash
ls CUB_200_2011/
# 应该看到:
# - images.txt
# - train_test_split.txt
# - images/ (文件夹，包含很多子文件夹)
# - text/ (可选，文本描述文件夹)
```

### ⚠️ 常见问题

**Q: 下载的文件只有几KB？**

**A**: 下载失败，可能是：
- 直接访问下载链接返回了404页面
- 需要使用浏览器从网站页面下载

**解决**：必须使用浏览器访问网站页面，然后点击下载链接

**Q: tar解压失败？**

**A**: 检查：
- 文件是否完整下载（应该约1.1GB，不是几KB）
- 如果文件很小，说明下载失败，需要重新下载

---

## 📊 数据集划分说明（重要！）

### 为什么需要划分数据集？

在机器学习项目中，数据集通常需要划分为三个部分：

1. **训练集（Training Set）**：用于训练模型
2. **验证集（Validation Set）**：用于调整超参数、选择模型、早停等
3. **测试集（Test Set）**：用于最终评估模型性能（**只用一次**）

### 划分比例建议

对于GAN训练，常见的划分比例：

- **训练集**：70-80%（用于训练模型）
- **验证集**：10-15%（用于调整超参数、评估训练进度）
- **测试集**：10-15%（用于最终评估，**训练过程中不使用**）

**示例**（CUB-200数据集，共11,788张图像）：
- 训练集：~8,000张（约68%）
- 验证集：~1,800张（约15%）
- 测试集：~1,988张（约17%）

---

## 📁 CUB-200数据集划分

### 数据集自带划分

CUB-200数据集**已经提供了划分**：

- `train_test_split.txt`：包含每张图像的划分信息
  - `1` = 训练集
  - `0` = 测试集

### 如何使用

```python
from src.utils.datasets import CUB200Dataset

# 训练集
train_dataset = CUB200Dataset(
    root_dir='./data/CUB_200_2011',
    split='train',  # 使用train_test_split.txt中的训练集
    max_text_length=18
)

# 测试集
test_dataset = CUB200Dataset(
    root_dir='./data/CUB_200_2011',
    split='test',  # 使用train_test_split.txt中的测试集
    max_text_length=18
)
```

### 如何创建验证集？

**方法1：从训练集中再划分（推荐）**

```python
from torch.utils.data import random_split

# 先加载训练集
full_train_dataset = CUB200Dataset('./data/CUB_200_2011', split='train')

# 从训练集中划分出验证集（例如20%）
train_size = int(0.8 * len(full_train_dataset))
val_size = len(full_train_dataset) - train_size
train_dataset, val_dataset = random_split(full_train_dataset, [train_size, val_size])
```

**方法2：手动修改train_test_split.txt**

不推荐，因为会改变原始数据集划分。

---

## 📁 COCO数据集划分

### COCO数据集自带划分

COCO数据集已经提供了划分：

- `train2017/`：训练集图像
- `val2017/`：验证集图像（**注意：COCO的val实际上是验证集**）
- `annotations/captions_train2017.json`：训练集标注
- `annotations/captions_val2017.json`：验证集标注

### 如何使用

```python
from src.utils.datasets import COCODataset

# 训练集
train_dataset = COCODataset(
    root_dir='./data/COCO',
    split='train',
    max_text_length=18
)

# 验证集（COCO的val就是验证集）
val_dataset = COCODataset(
    root_dir='./data/COCO',
    split='val',
    max_text_length=18
)
```

### 如何创建测试集？

**方法：从验证集中划分**

```python
from torch.utils.data import random_split

# 先加载验证集
full_val_dataset = COCODataset('./data/COCO', split='val')

# 从验证集中划分出测试集（例如50%）
val_size = int(0.5 * len(full_val_dataset))
test_size = len(full_val_dataset) - val_size
val_dataset, test_dataset = random_split(full_val_dataset, [val_size, test_size])
```

---

## 🎯 在训练中的使用

### 训练集（Training Set）

**用途**：
- 训练GAN模型（Generator和Discriminator）
- 更新模型参数

**使用方式**：
```python
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

for batch in train_loader:
    # 训练模型
    train_step(batch)
```

---

### 验证集（Validation Set）

**用途**：
- **评估训练进度**：每N个epoch在验证集上评估
- **调整超参数**：选择最佳学习率、batch size等
- **早停（Early Stopping）**：如果验证集性能不再提升，停止训练
- **选择最佳模型**：保存验证集上性能最好的模型

**使用方式**：
```python
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)

# 每5个epoch评估一次
if epoch % 5 == 0:
    val_metrics = evaluate(val_loader)
    print(f"Validation FID: {val_metrics['fid']}")
    
    # 保存最佳模型
    if val_metrics['fid'] < best_fid:
        save_checkpoint(model, 'best_model.pth')
```

---

### 测试集（Test Set）

**用途**：
- **最终评估**：训练完成后，在测试集上评估模型
- **报告结果**：论文/报告中报告的性能指标
- **只使用一次**：训练过程中**不应该**使用测试集

**使用方式**：
```python
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# 训练完成后，最终评估
final_metrics = evaluate(test_loader)
print(f"Final Test FID: {final_metrics['fid']}")
print(f"Final Test IS: {final_metrics['is']}")
print(f"Final Test CLIP: {final_metrics['clip_score']}")
```

---

## ⚠️ 重要注意事项

### 1. 不要用测试集训练！

- ❌ **错误**：在训练过程中使用测试集调整模型
- ✅ **正确**：测试集只在最后评估时使用一次

### 2. 保持划分一致性

- 使用固定的随机种子（seed）确保划分可复现
- 不要在训练过程中改变划分

### 3. 验证集的作用

- 验证集用于"模拟"测试集
- 在验证集上表现好，通常测试集上也会表现好
- 但如果验证集和测试集分布不同，可能不成立

---

## 📝 数据集划分示例代码

### 完整示例（CUB-200）

```python
import torch
from torch.utils.data import DataLoader, random_split
from src.utils.datasets import CUB200Dataset, collate_fn

# 1. 加载训练集和测试集（数据集自带划分）
train_dataset = CUB200Dataset('./data/CUB_200_2011', split='train')
test_dataset = CUB200Dataset('./data/CUB_200_2011', split='test')

# 2. 从训练集中划分出验证集
train_size = int(0.8 * len(train_dataset))
val_size = len(train_dataset) - train_size
train_dataset, val_dataset = random_split(
    train_dataset, 
    [train_size, val_size],
    generator=torch.Generator().manual_seed(42)  # 固定随机种子
)

# 3. 创建DataLoader
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, collate_fn=collate_fn)

print(f"训练集: {len(train_dataset)} 张")
print(f"验证集: {len(val_dataset)} 张")
print(f"测试集: {len(test_dataset)} 张")
```

---

## 🎯 中期报告中的数据集说明

在中期报告中，你应该说明：

1. **使用的数据集**：CUB-200或COCO
2. **数据集大小**：训练集、验证集、测试集的数量
3. **划分方式**：如何划分的（数据集自带划分 or 手动划分）
4. **划分比例**：训练/验证/测试的比例

**示例文字**：

"我们使用CUB-200-2011数据集进行实验，该数据集包含11,788张鸟类图像，每张图像有10个文本描述。数据集提供了训练集和测试集的官方划分（train_test_split.txt）。我们从训练集中进一步划分出20%作为验证集，最终得到：
- 训练集：8,000张图像（用于模型训练）
- 验证集：1,800张图像（用于超参数调整和训练进度评估）
- 测试集：1,988张图像（用于最终性能评估，训练过程中不使用）"

---

**最后更新**: 2024年12月  
**关键点**: 训练集用于训练，验证集用于调整和评估，测试集只在最后用一次！

