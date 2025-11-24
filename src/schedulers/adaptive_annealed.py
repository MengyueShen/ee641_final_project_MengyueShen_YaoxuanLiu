"""
自适应退火调度器（Adaptive Annealing Schedule）
在固定退火策略基础上，添加小控制器进行自适应调整
基于论文: "Learning Schedules for Text-to-Image GANs"
"""
import torch
import torch.nn as nn
import numpy as np
from .base import ParameterScheduler, DynamicParameter
from .annealed import AnnealedScheduler


class AdaptiveController(nn.Module):
    """自适应控制器（小网络）
    
    输入：训练状态
    输出：对退火曲线的调整因子
    """
    
    def __init__(self, state_dim, hidden_dim=32):
        """
        Args:
            state_dim: 状态特征维度
            hidden_dim: 隐藏层维度（小网络）
        """
        super(AdaptiveController, self).__init__()
        
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1),
            nn.Tanh()  # 输出[-1, 1]，作为调整因子
        )
    
    def forward(self, state):
        """
        Args:
            state: [batch_size, state_dim] 或 [state_dim]
        Returns:
            adjustment: [batch_size] 或 scalar，调整因子 [-1, 1]
        """
        return self.network(state).squeeze(-1)


class AdaptiveAnnealedScheduler(ParameterScheduler):
    """自适应退火调度器
    
    在固定退火策略基础上，使用小控制器自适应调整
    结合了固定退火的稳定性和自适应的灵活性
    """
    
    def __init__(self, parameter_configs, state_features, 
                 adaptation_strength=0.1, device='cuda'):
        """
        Args:
            parameter_configs: Dict[str, dict] 参数配置（与AnnealedScheduler相同）
            state_features: List[str] 状态特征名称列表
            adaptation_strength: float 自适应调整的强度 [0, 1]
            device: 设备
        """
        # 先创建基础退火调度器
        self.base_scheduler = AnnealedScheduler(parameter_configs)
        
        # 继承参数
        super().__init__(self.base_scheduler.parameters)
        self.configs = parameter_configs
        self.state_features = state_features
        self.state_dim = len(state_features)
        self.adaptation_strength = adaptation_strength
        self.device = device
        
        # 为每个参数创建自适应控制器
        controllers = {}
        for name in self.parameters.keys():
            controllers[name] = AdaptiveController(
                state_dim=self.state_dim,
                hidden_dim=32  # 小网络
            ).to(device)
        
        self.controllers = nn.ModuleDict(controllers)
    
    def extract_state(self, generator, discriminator, losses, epoch_progress, **kwargs):
        """提取训练状态特征（与LearnableScheduler相同）"""
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
                state_list.append(0.0)
        
        state = torch.tensor(state_list, dtype=torch.float32, device=self.device)
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
        更新所有参数值（固定退火 + 自适应调整）
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            generator: Generator模型（可选）
            discriminator: Discriminator模型（可选）
            losses: Dict包含损失值（可选）
            **kwargs: 其他状态信息
        """
        # 1. 先计算基础退火值
        self.base_scheduler.update(epoch, total_epochs)
        base_params = self.base_scheduler.get_parameters()
        
        # 2. 如果有状态信息，计算自适应调整
        if generator is not None and discriminator is not None and losses is not None:
            epoch_progress = epoch / total_epochs
            state = self.extract_state(generator, discriminator, losses,
                                     epoch_progress, **kwargs)
            
            # 3. 使用控制器计算调整因子
            for name, param in self.parameters.items():
                controller = self.controllers[name]
                controller.eval()
                with torch.no_grad():
                    adjustment = controller(state.unsqueeze(0)).squeeze(0).item()
                    # adjustment ∈ [-1, 1]
                
                # 4. 应用调整：base_value + adjustment * strength * range
                base_value = base_params[name]
                config = self.configs[name]
                bounds = config.get('bounds', None)
                
                if bounds:
                    value_range = bounds[1] - bounds[0]
                    adjustment_amount = adjustment * self.adaptation_strength * value_range
                    adjusted_value = base_value + adjustment_amount
                    
                    # 确保在bounds内
                    adjusted_value = max(bounds[0], min(bounds[1], adjusted_value))
                else:
                    # 如果没有bounds，使用相对调整
                    adjustment_amount = adjustment * self.adaptation_strength * base_value
                    adjusted_value = base_value + adjustment_amount
                
                param.update(adjusted_value)
        else:
            # 如果没有状态信息，直接使用基础退火值
            for name, param in self.parameters.items():
                param.update(base_params[name])
    
    def get_controllers(self):
        """获取所有控制器（用于训练）"""
        return self.controllers
    
    @staticmethod
    def create_default_config():
        """创建默认配置（与AnnealedScheduler相同）"""
        return AnnealedScheduler.create_default_config()

