"""
可学习单调调度器（Learnable Monotone Schedule）
使用K-bin softmax实现可微分的单调调度函数
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from .base import ParameterScheduler, DynamicParameter


class MonotoneSchedule(nn.Module):
    """单调调度函数
    
    使用K-bin softmax实现可微分的单调函数
    保证参数值单调递减（或递增）
    """
    
    def __init__(self, k_bins=10, monotone_type='decreasing'):
        """
        Args:
            k_bins: bin的数量，控制调度的精细程度
            monotone_type: 'decreasing' 或 'increasing'
        """
        super(MonotoneSchedule, self).__init__()
        self.k_bins = k_bins
        self.monotone_type = monotone_type
        
        # 可学习的bin权重（通过softmax保证单调性）
        # 对于递减：前面的bin权重应该大，后面的小
        # 对于递增：前面的bin权重应该小，后面的大
        self.bin_weights = nn.Parameter(torch.ones(k_bins))
        
        # 初始化：递减时，前面的权重大；递增时，后面的权重大
        if monotone_type == 'decreasing':
            # 初始化为递减：权重从大到小
            with torch.no_grad():
                self.bin_weights.data = torch.linspace(k_bins, 1, k_bins)
        else:
            # 初始化为递增：权重从小到大
            with torch.no_grad():
                self.bin_weights.data = torch.linspace(1, k_bins, k_bins)
    
    def forward(self, progress):
        """
        根据训练进度计算参数值
        
        Args:
            progress: [batch_size] 或 scalar，训练进度 [0, 1]
        
        Returns:
            value: [batch_size] 或 scalar，参数值
        """
        if isinstance(progress, (int, float)):
            progress = torch.tensor(progress, device=self.bin_weights.device)
        
        # 将progress映射到bin索引
        # progress [0, 1] -> bin_idx [0, k_bins-1]
        bin_idx = (progress * (self.k_bins - 1)).long()
        bin_idx = torch.clamp(bin_idx, 0, self.k_bins - 1)
        
        # 计算每个bin的softmax权重（保证单调性）
        # 使用cumsum和softmax确保单调性
        if self.monotone_type == 'decreasing':
            # 递减：使用cumulative softmax
            # 前面的bin权重累积，后面的权重递减
            weights = F.softmax(self.bin_weights, dim=0)
            # 反转权重顺序，使得前面的权重大
            weights = weights.flip(0)
            # 累积权重
            cum_weights = torch.cumsum(weights, dim=0)
        else:
            # 递增：直接使用cumulative softmax
            weights = F.softmax(self.bin_weights, dim=0)
            cum_weights = torch.cumsum(weights, dim=0)
        
        # 根据progress所在的bin，插值计算值
        # 简化版本：直接使用bin的累积权重
        if progress.dim() == 0:
            value = cum_weights[bin_idx].item()
        else:
            value = cum_weights[bin_idx]
        
        return value


class LearnableMonotoneScheduler(ParameterScheduler):
    """可学习单调调度器
    
    为每个参数使用独立的MonotoneSchedule模块
    通过反向传播学习最优的单调调度策略
    """
    
    def __init__(self, parameter_configs, k_bins=10, device='cuda'):
        """
        Args:
            parameter_configs: Dict[str, dict] 参数配置
                例如: {
                    'noise_var': {
                        'initial': 1.0,
                        'final': 0.01,
                        'bounds': (0.0, 2.0),
                        'monotone_type': 'decreasing'  # 或 'increasing'
                    }
                }
            k_bins: 每个参数的bin数量
            device: 设备
        """
        parameters = {}
        schedules = {}
        
        for name, config in parameter_configs.items():
            initial = config.get('initial', 1.0)
            bounds = config.get('bounds', (0.0, 1.0))
            monotone_type = config.get('monotone_type', 'decreasing')
            
            parameters[name] = DynamicParameter(name, initial, bounds)
            
            # 为每个参数创建独立的单调调度模块
            schedules[name] = MonotoneSchedule(
                k_bins=k_bins,
                monotone_type=monotone_type
            ).to(device)
        
        super().__init__(parameters)
        self.configs = parameter_configs
        self.schedules = nn.ModuleDict(schedules)
        self.device = device
        self.k_bins = k_bins
    
    def update(self, epoch, total_epochs, **kwargs):
        """
        更新所有参数值（使用可学习的单调调度）
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            **kwargs: 其他信息（训练模式下可能需要）
        """
        progress = epoch / total_epochs  # [0, 1]
        progress_tensor = torch.tensor(progress, device=self.device)
        
        for name, param in self.parameters.items():
            config = self.configs[name]
            bounds = config.get('bounds', (0.0, 1.0))
            initial = config.get('initial', bounds[0])
            final = config.get('final', bounds[1])
            
            # 使用单调调度函数计算归一化值 [0, 1]
            normalized_value = self.schedules[name](progress_tensor)
            
            if isinstance(normalized_value, torch.Tensor):
                normalized_value = normalized_value.item()
            
            # 映射到实际参数空间
            actual_value = initial + normalized_value * (final - initial)
            
            # 确保在bounds内
            actual_value = max(bounds[0], min(bounds[1], actual_value))
            
            param.update(actual_value)
    
    def get_schedules(self):
        """获取所有调度模块（用于训练）"""
        return self.schedules
    
    def forward(self, progress):
        """
        前向传播（用于训练时的梯度计算）
        
        Args:
            progress: [batch_size] 训练进度
        
        Returns:
            params_dict: Dict[str, torch.Tensor] 参数值
        """
        params_dict = {}
        for name, schedule in self.schedules.items():
            config = self.configs[name]
            bounds = config.get('bounds', (0.0, 1.0))
            initial = config.get('initial', bounds[0])
            final = config.get('final', bounds[1])
            
            normalized_value = schedule(progress)
            actual_value = initial + normalized_value * (final - initial)
            
            params_dict[name] = actual_value
        
        return params_dict
    
    @staticmethod
    def create_default_config():
        """创建默认配置"""
        return {
            'noise_var': {
                'initial': 1.0,
                'final': 0.01,
                'bounds': (0.0, 2.0),
                'monotone_type': 'decreasing'  # 噪声应该递减
            },
            'augmentation_strength': {
                'initial': 0.8,
                'final': 0.1,
                'bounds': (0.0, 1.0),
                'monotone_type': 'decreasing'  # 增强强度应该递减
            },
            'regularization_weight': {
                'initial': 10.0,
                'final': 1.0,
                'bounds': (0.0, 20.0),
                'monotone_type': 'decreasing'  # 正则化应该递减
            }
        }

