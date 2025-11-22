"""
退火化调度器实现
支持线性、指数、余弦等退火策略
"""
import numpy as np
from .base import ParameterScheduler, DynamicParameter


class AnnealedScheduler(ParameterScheduler):
    """退火化参数调度器"""
    
    def __init__(self, parameter_configs):
        """
        Args:
            parameter_configs: Dict[str, dict] 参数配置
                例如: {
                    'noise_var': {
                        'initial': 1.0,
                        'final': 0.01,
                        'schedule': 'exponential',  # 'linear', 'exponential', 'cosine'
                        'decay_rate': 2.0  # 仅用于exponential
                    }
                }
        """
        parameters = {}
        for name, config in parameter_configs.items():
            initial = config['initial']
            bounds = config.get('bounds', None)
            parameters[name] = DynamicParameter(name, initial, bounds)
        
        super().__init__(parameters)
        self.configs = parameter_configs
    
    def update(self, epoch, total_epochs, **kwargs):
        """更新所有参数值"""
        progress = epoch / total_epochs  # 归一化到[0, 1]
        
        for name, param in self.parameters.items():
            config = self.configs[name]
            initial = config['initial']
            final = config['final']
            schedule_type = config.get('schedule', 'linear')
            
            if schedule_type == 'linear':
                value = initial + (final - initial) * progress
            elif schedule_type == 'exponential':
                decay_rate = config.get('decay_rate', 2.0)
                value = initial * ((final / initial) ** (progress ** decay_rate))
            elif schedule_type == 'cosine':
                value = final + (initial - final) * 0.5 * (1 + np.cos(np.pi * progress))
            elif schedule_type == 'step':
                # 阶梯式退火
                step_size = config.get('step_size', 0.1)
                num_steps = int(progress / step_size)
                value = initial - (initial - final) * (num_steps * step_size)
            else:
                raise ValueError(f"Unknown schedule type: {schedule_type}")
            
            param.update(value)
    
    @staticmethod
    def create_default_config():
        """创建默认配置"""
        return {
            'noise_var': {
                'initial': 1.0,
                'final': 0.01,
                'schedule': 'exponential',
                'decay_rate': 2.0,
                'bounds': (0.0, 2.0)
            },
            'augmentation_strength': {
                'initial': 0.8,
                'final': 0.1,
                'schedule': 'linear',
                'bounds': (0.0, 1.0)
            },
            'regularization_weight': {
                'initial': 10.0,
                'final': 1.0,
                'schedule': 'cosine',
                'bounds': (0.0, 20.0)
            }
        }

