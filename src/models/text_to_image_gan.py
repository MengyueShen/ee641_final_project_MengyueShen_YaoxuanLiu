"""
文本到图像GAN模型实现
基于"Learning Schedules for Text-to-Image GANs"论文
支持条件生成和判别
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextEncoder(nn.Module):
    """文本编码器 - 将文本描述编码为特征向量
    
    使用RNN（LSTM/GRU）或Transformer编码文本
    """
    
    def __init__(self, vocab_size, embed_dim=256, hidden_dim=256, 
                 num_layers=1, rnn_type='LSTM'):
        """
        Args:
            vocab_size: 词汇表大小
            embed_dim: 词嵌入维度
            hidden_dim: RNN隐藏层维度
            num_layers: RNN层数
            rnn_type: 'LSTM' 或 'GRU'
        """
        super(TextEncoder, self).__init__()
        self.embed_dim = embed_dim
        self.hidden_dim = hidden_dim
        
        # 词嵌入层
        self.embedding = nn.Embedding(vocab_size, embed_dim)
        
        # RNN层
        if rnn_type == 'LSTM':
            self.rnn = nn.LSTM(embed_dim, hidden_dim, num_layers, 
                              batch_first=True, bidirectional=True)
            self.hidden_dim = hidden_dim * 2  # 双向LSTM
        else:
            self.rnn = nn.GRU(embed_dim, hidden_dim, num_layers,
                             batch_first=True, bidirectional=True)
            self.hidden_dim = hidden_dim * 2
        
        # 输出投影层（输出维度应该等于双向RNN后的hidden_dim）
        self.fc = nn.Linear(self.hidden_dim, self.hidden_dim)
    
    def forward(self, text_ids, text_lengths=None):
        """
        Args:
            text_ids: [batch_size, seq_len] 文本token IDs
            text_lengths: [batch_size] 每个文本的实际长度（用于pack_padded_sequence）
        
        Returns:
            text_features: [batch_size, hidden_dim] 文本特征向量
        """
        # 词嵌入
        embedded = self.embedding(text_ids)  # [batch_size, seq_len, embed_dim]
        
        # RNN编码
        if text_lengths is not None:
            # 确保所有长度都大于0（pack_padded_sequence的要求）
            # 如果文本为空，至少有一个padding token
            text_lengths = torch.clamp(text_lengths, min=1)
            # 使用pack_padded_sequence处理变长序列
            embedded = nn.utils.rnn.pack_padded_sequence(
                embedded, text_lengths, batch_first=True, enforce_sorted=False
            )
        
        output, (hidden, cell) = self.rnn(embedded)
        
        if text_lengths is not None:
            output, _ = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)
        
        # 使用最后一个时间步的隐藏状态
        if isinstance(self.rnn, nn.LSTM):
            # LSTM: 取前向和后向的最后一个隐藏状态并拼接
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)  # [batch_size, hidden_dim*2]
        else:
            # GRU: 类似处理
            hidden = torch.cat([hidden[-2], hidden[-1]], dim=1)
        
        # 投影到固定维度
        text_features = self.fc(hidden)  # [batch_size, self.hidden_dim] (双向RNN后的维度)
        
        return text_features


class ConditionalGenerator(nn.Module):
    """条件生成器 - 基于文本特征生成图像
    
    参考StackGAN或AttnGAN架构
    """
    
    def __init__(self, nz=100, ngf=64, nc=3, img_size=64, text_dim=256):
        """
        Args:
            nz: 噪声维度
            ngf: 生成器特征图数量
            nc: 输出通道数（RGB=3）
            img_size: 输出图像尺寸
            text_dim: 文本特征维度
        """
        super(ConditionalGenerator, self).__init__()
        self.nz = nz
        self.text_dim = text_dim
        self.img_size = img_size
        
        # 将文本特征投影到与噪声相同的空间
        self.text_projection = nn.Linear(text_dim, nz)
        
        # 条件生成网络（DCGAN风格）
        # 输入: 噪声 + 文本特征（拼接后）
        self.main = nn.Sequential(
            # 输入: (nz*2) x 1 x 1 (噪声和文本特征拼接)
            nn.ConvTranspose2d(nz * 2, ngf * 8, 4, 1, 0, bias=False),
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
            
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # 状态尺寸: (ngf) x 32 x 32
            
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # 状态尺寸: (nc) x 64 x 64
        )
    
    def forward(self, noise, text_features):
        """
        Args:
            noise: [batch_size, nz, 1, 1] 噪声向量
            text_features: [batch_size, text_dim] 文本特征
        
        Returns:
            fake_images: [batch_size, nc, img_size, img_size] 生成的图像
        """
        # 将文本特征投影到噪声空间
        text_proj = self.text_projection(text_features)  # [batch_size, nz]
        text_proj = text_proj.unsqueeze(-1).unsqueeze(-1)  # [batch_size, nz, 1, 1]
        
        # 拼接噪声和文本特征
        combined = torch.cat([noise, text_proj], dim=1)  # [batch_size, nz*2, 1, 1]
        
        # 生成图像
        fake_images = self.main(combined)
        
        return fake_images


class ConditionalDiscriminator(nn.Module):
    """条件判别器 - 判断图像和文本是否匹配
    
    使用条件GAN的架构，将文本特征融合到判别过程中
    """
    
    def __init__(self, nc=3, ndf=64, img_size=64, text_dim=256):
        """
        Args:
            nc: 输入通道数（RGB=3）
            ndf: 判别器特征图数量
            img_size: 输入图像尺寸
            text_dim: 文本特征维度
        """
        super(ConditionalDiscriminator, self).__init__()
        self.text_dim = text_dim
        
        # 图像特征提取网络
        self.img_net = nn.Sequential(
            # 输入: (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf) x 32 x 32
            
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf*2) x 16 x 16
            
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf*4) x 8 x 8
            
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # 状态尺寸: (ndf*8) x 4 x 4
        )
        
        # 文本特征投影到图像特征空间
        self.text_projection = nn.Linear(text_dim, ndf * 8 * 4 * 4)
        
        # 融合图像和文本特征
        self.fusion = nn.Sequential(
            nn.Conv2d(ndf * 8 + ndf * 8, ndf * 8, 1, 1, 0, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
        )
        
        # 最终判别
        self.classifier = nn.Sequential(
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    
    def forward(self, images, text_features):
        """
        Args:
            images: [batch_size, nc, img_size, img_size] 图像
            text_features: [batch_size, text_dim] 文本特征
        
        Returns:
            output: [batch_size] 判别结果（0-1之间的概率）
        """
        # 提取图像特征
        img_features = self.img_net(images)  # [batch_size, ndf*8, 4, 4]
        
        # 投影文本特征到图像特征空间
        text_proj = self.text_projection(text_features)  # [batch_size, ndf*8*4*4]
        text_proj = text_proj.view(-1, img_features.size(1), 
                                   img_features.size(2), img_features.size(3))
        # [batch_size, ndf*8, 4, 4]
        
        # 融合图像和文本特征
        combined = torch.cat([img_features, text_proj], dim=1)  # [batch_size, ndf*16, 4, 4]
        fused = self.fusion(combined)  # [batch_size, ndf*8, 4, 4]
        
        # 最终判别
        output = self.classifier(fused)  # [batch_size, 1, 1, 1]
        output = output.view(-1, 1).squeeze(1)  # [batch_size]
        
        return output


class TextToImageGAN:
    """文本到图像GAN包装类
    
    包含文本编码器、条件生成器和条件判别器
    """
    
    def __init__(self, vocab_size, nz=100, ngf=64, ndf=64, nc=3, 
                 img_size=64, text_dim=256, device='cuda'):
        """
        Args:
            vocab_size: 词汇表大小
            nz: 噪声维度
            ngf: 生成器特征图数量
            ndf: 判别器特征图数量
            nc: 图像通道数
            img_size: 图像尺寸
            text_dim: 文本特征维度
            device: 设备
        """
        self.device = device
        self.nz = nz
        self.text_dim = text_dim
        
        # 创建模型组件
        self.text_encoder = TextEncoder(
            vocab_size=vocab_size,
            embed_dim=256,
            hidden_dim=text_dim // 2,  # 双向RNN，所以除以2
            num_layers=1,
            rnn_type='LSTM'
        ).to(device)
        
        self.generator = ConditionalGenerator(
            nz=nz, ngf=ngf, nc=nc, img_size=img_size, text_dim=text_dim
        ).to(device)
        
        self.discriminator = ConditionalDiscriminator(
            nc=nc, ndf=ndf, img_size=img_size, text_dim=text_dim
        ).to(device)
        
        # 初始化权重
        self.text_encoder.apply(self._weights_init)
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
        elif classname.find('Linear') != -1:
            nn.init.normal_(m.weight.data, 0.0, 0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias.data, 0)
    
    def encode_text(self, text_ids, text_lengths=None):
        """编码文本"""
        return self.text_encoder(text_ids, text_lengths)
    
    def sample_noise(self, batch_size):
        """采样噪声"""
        return torch.randn(batch_size, self.nz, 1, 1, device=self.device)
    
    def generate(self, text_ids, text_lengths=None, noise=None):
        """
        根据文本生成图像
        
        Args:
            text_ids: [batch_size, seq_len] 文本token IDs
            text_lengths: [batch_size] 文本长度
            noise: [batch_size, nz, 1, 1] 噪声（可选，如果不提供则随机采样）
        
        Returns:
            fake_images: [batch_size, nc, img_size, img_size] 生成的图像
        """
        # 编码文本
        text_features = self.encode_text(text_ids, text_lengths)
        
        # 采样噪声
        if noise is None:
            noise = self.sample_noise(text_ids.size(0))
        
        # 生成图像
        with torch.no_grad():
            fake_images = self.generator(noise, text_features)
        
        return fake_images
    
    def train(self):
        """设置为训练模式"""
        self.text_encoder.train()
        self.generator.train()
        self.discriminator.train()
    
    def eval(self):
        """设置为评估模式"""
        self.text_encoder.eval()
        self.generator.eval()
        self.discriminator.eval()

