"""
可学习调度器实现
使用元网络自动学习参数调度策略
"""
import torch
import torch.nn as nn
import numpy as np
from .base import ParameterScheduler, DynamicParameter


class MetaController(nn.Module):
    """元控制器网络
    
    输入：训练状态特征（loss、梯度、epoch进度等）
    输出：归一化的参数值（需要映射到实际参数空间）
    """
    
    def __init__(self, state_dim, num_params, hidden_dims=[128, 64]):
        """
        Args:
            state_dim: 状态特征维度
            num_params: 输出参数数量
            hidden_dims: 隐藏层维度列表
        """
        super(MetaController, self).__init__()
        
        layers = []
        input_dim = state_dim
        
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(input_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(0.1)
            ])
            input_dim = hidden_dim
        
        layers.append(nn.Linear(input_dim, num_params))
        layers.append(nn.Sigmoid())  # 输出归一化到[0,1]
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
        Returns:
            normalized_params: [batch_size, num_params] 或 [num_params]
        """
        return self.network(state)


class LearnableScheduler(ParameterScheduler):
    """可学习参数调度器"""
    
    def __init__(self, parameter_configs, state_features, device='cuda'):
        """
        Args:
            parameter_configs: Dict[str, dict] 参数配置
                例如: {
                    'noise_var': {
                        'initial': 1.0,
                        'bounds': (0.0, 2.0)
                    }
                }
            state_features: List[str] 状态特征名称列表
                例如: ['loss_g', 'loss_d', 'grad_norm_g', 'grad_norm_d', 'epoch_progress']
            device: 设备
        """
        parameters = {}
        for name, config in parameter_configs.items():
            initial = config.get('initial', 0.5)
            bounds = config.get('bounds', (0.0, 1.0))
            parameters[name] = DynamicParameter(name, initial, bounds)
        
        super().__init__(parameters)
        self.configs = parameter_configs
        self.state_features = state_features
        self.state_dim = len(state_features)
        self.device = device
        
        # 创建元控制器
        self.meta_controller = MetaController(
            state_dim=self.state_dim,
            num_params=len(parameters)
        ).to(device)
        
        # 参数名称列表（保持顺序）
        self.param_names = list(parameters.keys())
    
    def extract_state(self, generator, discriminator, losses, epoch_progress, **kwargs):
        """
        提取训练状态特征
        
        Args:
            generator: Generator模型
            discriminator: Discriminator模型
            losses: Dict包含'g'和'd'键
            epoch_progress: float [0,1] epoch进度
            **kwargs: 其他状态信息（如梯度范数等）
        
        Returns:
            state: torch.Tensor [state_dim]
        """
        state_list = []
        
        for feature_name in self.state_features:
            if feature_name == 'loss_g':
                state_list.append(losses.get('g', 0.0))
            elif feature_name == 'loss_d':
                state_list.append(losses.get('d', 0.0))
            elif feature_name == 'grad_norm_g':
                grad_norm = self._compute_grad_norm(generator)
                state_list.append(grad_norm)
            elif feature_name == 'grad_norm_d':
                grad_norm = self._compute_grad_norm(discriminator)
                state_list.append(grad_norm)
            elif feature_name == 'epoch_progress':
                state_list.append(epoch_progress)
            elif feature_name in kwargs:
                state_list.append(kwargs[feature_name])
            else:
                # 默认值
                state_list.append(0.0)
        
        # 归一化状态（可选，根据实际情况调整）
        state = torch.tensor(state_list, dtype=torch.float32, device=self.device)
        # 简单的归一化：除以一个合理的尺度
        state = state / (torch.abs(state).max() + 1e-8)
        
        return state
    
    @staticmethod
    def _compute_grad_norm(model):
        """计算模型梯度范数"""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        return total_norm ** (1. / 2)
    
    def update(self, epoch, total_epochs, generator=None, discriminator=None, 
               losses=None, **kwargs):
        """
        使用元控制器更新参数
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            generator: Generator模型（可选）
            discriminator: Discriminator模型（可选）
            losses: Dict包含损失值（可选）
            **kwargs: 其他状态信息
        """
        if generator is None or discriminator is None or losses is None:
            # 如果没有提供状态信息，使用简单的epoch进度
            epoch_progress = epoch / total_epochs
            state = torch.tensor([epoch_progress] * self.state_dim, 
                               dtype=torch.float32, device=self.device)
        else:
            epoch_progress = epoch / total_epochs
            state = self.extract_state(generator, discriminator, losses, 
                                     epoch_progress, **kwargs)
        
        # 元控制器预测归一化参数
        self.meta_controller.eval()
        with torch.no_grad():
            normalized_params = self.meta_controller(state.unsqueeze(0)).squeeze(0)
        
        # 映射到实际参数空间
        for i, name in enumerate(self.param_names):
            config = self.configs[name]
            bounds = config.get('bounds', (0.0, 1.0))
            
            # 从[0,1]映射到[bounds[0], bounds[1]]
            normalized_value = normalized_params[i].item()
            actual_value = bounds[0] + normalized_value * (bounds[1] - bounds[0])
            
            self.parameters[name].update(actual_value)
    
    def get_meta_controller(self):
        """获取元控制器（用于训练）"""
        return self.meta_controller
    
    def train_meta_controller(self, validation_metrics):
        """
        训练元控制器（基于验证集性能）
        
        Args:
            validation_metrics: Dict包含评估指标（如FID、IS等）
        
        Returns:
            meta_loss: 元损失值
        """
        # 这里需要根据实际需求设计元损失
        # 例如：负的FID改进（FID越小越好）
        # 或者：训练稳定性的奖励
        
        # 示例：简单的负FID作为奖励
        fid = validation_metrics.get('fid', 100.0)
        meta_loss = fid  # 我们希望最小化FID
        
        return meta_loss

