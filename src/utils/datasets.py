"""
文本-图像数据集加载工具
支持COCO、CUB-200等常见文本-图像数据集
"""
import torch
from torch.utils.data import Dataset
from PIL import Image
import json
import os
from torchvision import transforms


class TextImageDataset(Dataset):
    """文本-图像数据集基类"""
    
    def __init__(self, root_dir, split='train', transform=None, 
                 max_text_length=18, vocab=None):
        """
        Args:
            root_dir: 数据集根目录
            split: 'train', 'test', 或 'val'（'val'需要从train中再划分）
            transform: 图像变换
            max_text_length: 最大文本长度
            vocab: 词汇表（如果为None，则从数据构建）
        """
        self.root_dir = root_dir
        self.split = split
        self.transform = transform
        self.max_text_length = max_text_length
        
        # 默认图像变换
        if self.transform is None:
            self.transform = transforms.Compose([
                transforms.Resize((64, 64)),
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
            ])
        
        # 加载数据
        self.data = self._load_data()
        
        # 构建词汇表
        if vocab is None:
            self.vocab = self._build_vocab()
        else:
            self.vocab = vocab
        
        self.vocab_size = len(self.vocab)
        self.word_to_idx = {word: idx for idx, word in enumerate(self.vocab)}
        self.idx_to_word = {idx: word for idx, word in enumerate(self.vocab)}
    
    def _load_data(self):
        """加载数据（子类需要实现）"""
        raise NotImplementedError
    
    def _build_vocab(self):
        """构建词汇表（子类需要实现）"""
        raise NotImplementedError
    
    def _text_to_ids(self, text):
        """将文本转换为token IDs"""
        # 简单的分词（实际应该使用更复杂的分词器）
        words = text.lower().split()
        ids = [self.word_to_idx.get(word, self.word_to_idx['<unk>']) 
               for word in words]
        
        # 截断或填充到固定长度
        if len(ids) > self.max_text_length:
            ids = ids[:self.max_text_length]
        else:
            ids = ids + [self.word_to_idx['<pad>']] * (self.max_text_length - len(ids))
        
        return torch.tensor(ids, dtype=torch.long)
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        """
        Returns:
            image: PIL Image或Tensor
            text_ids: [max_text_length] token IDs
            text_length: 实际文本长度
            text: 原始文本字符串
        """
        item = self.data[idx]
        
        # 加载图像
        image_path = item['image_path']
        image = Image.open(image_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        
        # 处理文本
        text = item['text']
        text_ids = self._text_to_ids(text)
        # 确保text_length至少为1（即使文本为空，也有padding token）
        text_length = max(1, min(len(text.split()), self.max_text_length))
        
        return {
            'image': image,
            'text_ids': text_ids,
            'text_length': text_length,
            'text': text
        }


class CUB200Dataset(TextImageDataset):
    """CUB-200-2011数据集
    
    鸟类图像和文本描述数据集
    
    下载地址: http://www.vision.caltech.edu/visipedia/CUB-200-2011.html
    
    引用:
        @techreport{WahCUB_200_2011,
            Title = {The Caltech-UCSD Birds-200-2011 Dataset},
            Author = {Wah, C. and Branson, S. and Welinder, P. and Perona, P. and Belongie, S.},
            Year = {2011},
            Institution = {California Institute of Technology},
            Number = {CNS-TR-2011-001}
        }
    
    注意: 数据集仅用于非商业研究和教育目的
    """
    
    def __init__(self, root_dir, split='train', transform=None, max_text_length=18, vocab=None):
        self.root_dir = root_dir
        self.images_dir = os.path.join(root_dir, 'images')
        self.text_dir = os.path.join(root_dir, 'text')
        
        super().__init__(root_dir, split, transform, max_text_length, vocab=vocab)
    
    def _load_data(self):
        """加载CUB-200数据"""
        data = []
        
        # 加载图像列表
        images_file = os.path.join(self.root_dir, 'images.txt')
        train_test_split_file = os.path.join(self.root_dir, 'train_test_split.txt')
        
        if not os.path.exists(images_file):
            raise FileNotFoundError(
                f"CUB-200数据集文件未找到: {images_file}\n"
                "请下载CUB-200-2011数据集并解压到指定目录"
            )
        
        # 读取图像列表和训练/测试划分
        image_ids = {}
        with open(images_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                image_ids[int(parts[0])] = parts[1]
        
        split_info = {}
        with open(train_test_split_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                split_info[int(parts[0])] = int(parts[1])
        
        # 加载文本描述
        for img_id, img_name in image_ids.items():
            # 检查是否属于当前split
            is_train = split_info[img_id] == 1
            
            # split可以是'train', 'test', 'val'
            # 'train'和'val'都使用is_train==1的数据（后续会再划分）
            # 'test'使用is_train==0的数据
            if self.split == 'test':
                if is_train == 1:  # 测试集只用is_train==0的数据
                    continue
            elif self.split in ['train', 'val']:
                if is_train == 0:  # 训练集和验证集只用is_train==1的数据
                    continue
            else:
                continue  # 未知的split
            
            image_path = os.path.join(self.images_dir, img_name)
            
            # 加载文本描述（CUB-200有10个描述）
            text_path = os.path.join(self.text_dir, img_name.replace('.jpg', '.txt'))
            if os.path.exists(text_path):
                with open(text_path, 'r') as f:
                    texts = [line.strip() for line in f.readlines()]
                    # 使用第一个描述（或可以随机选择）
                    text = texts[0] if texts else ""
            else:
                # 如果没有文本描述文件，使用图像类别名称作为文本
                # 图像路径格式: images/003.Sooty_Albatross/Sooty_Albatross_0002_796395.jpg
                # 提取类别名称（去掉编号前缀，如 "003.Sooty_Albatross" -> "Sooty_Albatross"）
                parts = img_name.split('/')
                if len(parts) > 1:
                    class_name = parts[0]  # 例如 "003.Sooty_Albatross"
                    # 去掉编号前缀（如 "003."）
                    if '.' in class_name:
                        class_name = class_name.split('.', 1)[1]  # "Sooty_Albatross"
                    # 将下划线替换为空格，使其更像自然语言
                    text = class_name.replace('_', ' ')  # "Sooty Albatross"
                else:
                    text = ""  # 如果无法提取，使用空字符串
            
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'text': text,
                    'image_id': img_id
                })
        
        return data
    
    def _build_vocab(self):
        """构建词汇表"""
        vocab = ['<pad>', '<unk>', '<start>', '<end>']
        word_set = set()
        
        for item in self.data:
            words = item['text'].lower().split()
            word_set.update(words)
        
        vocab.extend(sorted(word_set))
        return vocab


class COCODataset(TextImageDataset):
    """COCO数据集（简化版）
    
    需要COCO API: pip install pycocotools
    """
    
    def __init__(self, root_dir, split='train', transform=None, max_text_length=18, vocab=None):
        self.root_dir = root_dir
        self.split = split
        
        # 这里把 vocab 传给基类：如果为 None，则基类会自己根据数据构建；如果传入，就复用同一个词表
        super().__init__(root_dir, split, transform, max_text_length, vocab=vocab)
    
    def _load_data(self):
        """加载COCO数据"""
        try:
            from pycocotools.coco import COCO
        except ImportError:
            raise ImportError(
                "需要安装pycocotools: pip install pycocotools\n"
                "或者使用CUB-200数据集"
            )
        
        data = []
        
        # COCO数据路径（使用 COCO 2017）
        ann_file = os.path.join(
            self.root_dir, 
            f'annotations/captions_{self.split}2017.json'
        )
        images_dir = os.path.join(self.root_dir, f'{self.split}2017')
        
        if not os.path.exists(ann_file):
            raise FileNotFoundError(
                f"COCO数据集文件未找到: {ann_file}\n"
                "请下载COCO数据集: https://cocodataset.org/"
            )
        
        coco = COCO(ann_file)
        img_ids = coco.getImgIds()
        
        for img_id in img_ids:
            img_info = coco.loadImgs(img_id)[0]
            image_path = os.path.join(images_dir, img_info['file_name'])
            
            # 获取该图像的caption（使用第一个）
            ann_ids = coco.getAnnIds(imgIds=img_id)
            anns = coco.loadAnns(ann_ids)
            
            if anns:
                text = anns[0]['caption']
            else:
                text = ""
            
            if os.path.exists(image_path):
                data.append({
                    'image_path': image_path,
                    'text': text,
                    'image_id': img_id
                })
        
        return data
    
    def _build_vocab(self):
        """构建词汇表"""
        vocab = ['<pad>', '<unk>', '<start>', '<end>']
        word_set = set()
        
        for item in self.data:
            words = item['text'].lower().split()
            word_set.update(words)
        
        vocab.extend(sorted(word_set))
        return vocab


def collate_fn(batch):
    """自定义collate函数，处理变长文本"""
    images = torch.stack([item['image'] for item in batch])
    text_ids = torch.stack([item['text_ids'] for item in batch])
    text_lengths = torch.tensor([item['text_length'] for item in batch])
    texts = [item['text'] for item in batch]
    
    return {
        'images': images,
        'text_ids': text_ids,
        'text_lengths': text_lengths,
        'texts': texts
    }

