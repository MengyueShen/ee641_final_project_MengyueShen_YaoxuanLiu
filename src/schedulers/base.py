"""
动态参数调度器基类
"""
from abc import ABC, abstractmethod
import torch


class DynamicParameter:
    """动态参数类"""
    
    def __init__(self, name, initial_value, bounds=None):
        """
        Args:
            name: 参数名称
            initial_value: 初始值
            bounds: (min, max) 参数范围
        """
        self.name = name
        self.value = initial_value
        self.bounds = bounds
        self.history = [initial_value]
    
    def update(self, new_value):
        """更新参数值"""
        if self.bounds:
            new_value = max(self.bounds[0], min(self.bounds[1], new_value))
        self.value = new_value
        self.history.append(new_value)
    
    def get_value(self):
        return self.value


class ParameterScheduler(ABC):
    """参数调度器基类"""
    
    def __init__(self, parameters):
        """
        Args:
            parameters: Dict[str, DynamicParameter] 参数字典
        """
        self.parameters = parameters
    
    @abstractmethod
    def update(self, epoch, total_epochs, **kwargs):
        """
        更新所有参数
        
        Args:
            epoch: 当前epoch
            total_epochs: 总epoch数
            **kwargs: 其他信息（如loss值等）
        """
        pass
    
    def get_parameters(self):
        """获取当前所有参数值"""
        return {name: param.get_value() for name, param in self.parameters.items()}
    
    def get_parameter(self, name):
        """获取特定参数值"""
        return self.parameters[name].get_value()

