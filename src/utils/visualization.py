"""
可视化工具 - 实现所有必要的可视化指标
基于proposal要求的可视化需求
"""
import matplotlib.pyplot as plt
import numpy as np
import torch
from pathlib import Path
import json
from typing import Dict, List, Optional
import seaborn as sns

# 设置中文字体（如果需要）
plt.rcParams['font.sans-serif'] = ['Arial Unicode MS', 'SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


class TrainingVisualizer:
    """训练可视化工具类"""
    
    def __init__(self, save_dir='results/figures'):
        """
        Args:
            save_dir: 图片保存目录
        """
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # 存储数据
        self.metrics_history = {}  # {schedule_name: {metric: [values]}}
        self.loss_history = {}     # {schedule_name: {'g': [...], 'd': [...], 'reg': [...]}}
        self.schedule_history = {} # {schedule_name: {param_name: [values]}}
        self.grad_norm_history = {} # {schedule_name: {'g': [...], 'd': [...]}}
    
    def log_metrics(self, schedule_name: str, metrics: Dict[str, float], step: int):
        """
        记录评估指标
        
        Args:
            schedule_name: 调度器名称（如'fixed_annealing', 'learnable_monotone'）
            metrics: {'fid': 50.2, 'is': 8.5, 'clip_score': 0.75}
            step: 当前训练步数
        """
        if schedule_name not in self.metrics_history:
            self.metrics_history[schedule_name] = {
                'steps': [],
                'fid': [],
                'is': [],
                'clip_score': []
            }
        
        self.metrics_history[schedule_name]['steps'].append(step)
        for metric_name in ['fid', 'is', 'clip_score']:
            if metric_name in metrics:
                self.metrics_history[schedule_name][metric_name].append(metrics[metric_name])
    
    def log_losses(self, schedule_name: str, losses: Dict[str, float], step: int):
        """
        记录损失值
        
        Args:
            schedule_name: 调度器名称
            losses: {'g': 2.5, 'd': 1.2, 'reg': 0.3}
            step: 当前训练步数
        """
        if schedule_name not in self.loss_history:
            self.loss_history[schedule_name] = {
                'steps': [],
                'g': [],
                'd': [],
                'reg': []
            }
        
        self.loss_history[schedule_name]['steps'].append(step)
        for loss_name in ['g', 'd', 'reg']:
            if loss_name in losses:
                self.loss_history[schedule_name][loss_name].append(losses[loss_name])
    
    def log_schedule_params(self, schedule_name: str, params: Dict[str, float], step: int):
        """
        记录调度器参数值
        
        Args:
            schedule_name: 调度器名称
            params: {'noise_var': 0.5, 'augmentation_strength': 0.3, 'regularization_weight': 5.0}
            step: 当前训练步数
        """
        if schedule_name not in self.schedule_history:
            self.schedule_history[schedule_name] = {
                'steps': [],
                'noise_var': [],
                'augmentation_strength': [],
                'regularization_weight': []
            }
        
        self.schedule_history[schedule_name]['steps'].append(step)
        for param_name in ['noise_var', 'augmentation_strength', 'regularization_weight']:
            if param_name in params:
                self.schedule_history[schedule_name][param_name].append(params[param_name])
    
    def log_grad_norms(self, schedule_name: str, grad_norms: Dict[str, float], step: int):
        """
        记录梯度范数
        
        Args:
            schedule_name: 调度器名称
            grad_norms: {'g': 0.5, 'd': 0.3}
            step: 当前训练步数
        """
        if schedule_name not in self.grad_norm_history:
            self.grad_norm_history[schedule_name] = {
                'steps': [],
                'g': [],
                'd': []
            }
        
        self.grad_norm_history[schedule_name]['steps'].append(step)
        for model_name in ['g', 'd']:
            if model_name in grad_norms:
                self.grad_norm_history[schedule_name][model_name].append(grad_norms[model_name])
    
    # ========== 核心可视化函数 ==========
    
    def plot_metrics_vs_steps(self, save_name='metrics_vs_steps.png'):
        """
        图1：FID / CLIP-Score / IS vs training steps
        
        必须有的核心可视化
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        metrics_to_plot = [
            ('fid', 'FID (越低越好)', axes[0]),
            ('clip_score', 'CLIP Score (越高越好)', axes[1]),
            ('is', 'Inception Score (越高越好)', axes[2])
        ]
        
        for metric_name, ylabel, ax in metrics_to_plot:
            for schedule_name, data in self.metrics_history.items():
                if metric_name in data and len(data[metric_name]) > 0:
                    steps = data['steps']
                    values = data[metric_name]
                    
                    # 计算移动平均（更平滑）
                    if len(values) > 10:
                        window = min(10, len(values) // 5)
                        values_smooth = np.convolve(values, np.ones(window)/window, mode='valid')
                        steps_smooth = steps[:len(values_smooth)]
                        ax.plot(steps_smooth, values_smooth, label=schedule_name, alpha=0.7, linewidth=2)
                    else:
                        ax.plot(steps, values, label=schedule_name, alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Training Steps / Epochs')
            ax.set_ylabel(ylabel)
            ax.set_title(f'{ylabel.split("(")[0].strip()} vs Steps')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_loss_curves(self, save_name='loss_curves.png'):
        """
        图2：Generator / Discriminator loss vs steps
        
        必须有的核心可视化
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        loss_types = [
            ('g', 'Generator Loss', axes[0]),
            ('d', 'Discriminator Loss', axes[1]),
            ('reg', 'Regularization Loss', axes[2])
        ]
        
        for loss_name, title, ax in loss_types:
            for schedule_name, data in self.loss_history.items():
                if loss_name in data and len(data[loss_name]) > 0:
                    steps = data['steps']
                    values = data[loss_name]
                    
                    # 移动平均
                    if len(values) > 10:
                        window = min(10, len(values) // 5)
                        values_smooth = np.convolve(values, np.ones(window)/window, mode='valid')
                        steps_smooth = steps[:len(values_smooth)]
                        ax.plot(steps_smooth, values_smooth, label=schedule_name, alpha=0.7, linewidth=2)
                    else:
                        ax.plot(steps, values, label=schedule_name, alpha=0.7, linewidth=2)
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel('Loss')
            ax.set_title(title)
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_schedule_params(self, save_name='schedule_params.png'):
        """
        图3：σ(u), p_aug(u), λ_reg(u) 曲线
        
        必须有的核心可视化 - 展示学到的schedule本身
        """
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        
        param_configs = [
            ('noise_var', 'σ(u): Generator Noise Magnitude', axes[0]),
            ('augmentation_strength', 'p_aug(u): Augmentation Probability', axes[1]),
            ('regularization_weight', 'λ_reg(u): Regularization Strength', axes[2])
        ]
        
        for param_name, ylabel, ax in param_configs:
            for schedule_name, data in self.schedule_history.items():
                if param_name in data and len(data[param_name]) > 0:
                    steps = data['steps']
                    values = data[param_name]
                    
                    # 归一化steps到[0,1]（训练进度u）
                    if len(steps) > 0:
                        steps_normalized = np.array(steps) / max(steps) if max(steps) > 0 else np.array(steps)
                        ax.plot(steps_normalized, values, label=schedule_name, linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Training Progress u ∈ [0,1]')
            ax.set_ylabel(ylabel)
            ax.set_title(ylabel.split(':')[0])
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_generated_samples_grid(self, samples_dict: Dict[str, torch.Tensor], 
                                   prompts: List[str], save_name='samples_grid.png'):
        """
        图4：不同schedule下的生成图像网格
        
        Args:
            samples_dict: {schedule_name: [batch_size, 3, H, W] tensor}
            prompts: 对应的文本提示列表
            save_name: 保存文件名
        """
        num_schedules = len(samples_dict)
        num_prompts = len(prompts)
        
        fig, axes = plt.subplots(num_schedules, num_prompts, 
                                figsize=(4*num_prompts, 4*num_schedules))
        
        if num_schedules == 1:
            axes = axes.reshape(1, -1)
        if num_prompts == 1:
            axes = axes.reshape(-1, 1)
        
        schedule_names = list(samples_dict.keys())
        
        for i, schedule_name in enumerate(schedule_names):
            samples = samples_dict[schedule_name]
            
            for j in range(num_prompts):
                ax = axes[i, j] if num_schedules > 1 else axes[j]
                
                # 转换为numpy并显示
                if isinstance(samples, torch.Tensor):
                    img = samples[j].cpu().numpy()
                    # 从[-1,1]转换到[0,1]
                    img = (img + 1) / 2
                    img = np.clip(img, 0, 1)
                    # 转换维度顺序: [C, H, W] -> [H, W, C]
                    img = np.transpose(img, (1, 2, 0))
                else:
                    img = samples[j]
                
                ax.imshow(img)
                ax.axis('off')
                
                # 第一行显示prompt
                if i == 0:
                    ax.set_title(f'Prompt: {prompts[j][:30]}...', fontsize=8)
                
                # 第一列显示schedule名称
                if j == 0:
                    ax.text(-0.1, 0.5, schedule_name, transform=ax.transAxes,
                           rotation=90, va='center', ha='right', fontsize=10, fontweight='bold')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_grad_norms(self, save_name='grad_norms.png'):
        """
        图5：||grad|| vs steps（G / D）
        
        体现稳定性的可视化
        """
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        for model_name, ax in [('g', axes[0]), ('d', axes[1])]:
            for schedule_name, data in self.grad_norm_history.items():
                if model_name in data and len(data[model_name]) > 0:
                    steps = data['steps']
                    values = data[model_name]
                    
                    # 移动平均
                    if len(values) > 10:
                        window = min(10, len(values) // 5)
                        values_smooth = np.convolve(values, np.ones(window)/window, mode='valid')
                        steps_smooth = steps[:len(values_smooth)]
                        ax.plot(steps_smooth, values_smooth, label=schedule_name, linewidth=2, alpha=0.7)
                    else:
                        ax.plot(steps, values, label=schedule_name, linewidth=2, alpha=0.7)
            
            ax.set_xlabel('Training Steps')
            ax.set_ylabel(f'Gradient Norm ||grad||')
            ax.set_title(f'{model_name.upper()} Gradient Norm')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_cross_seed_variance(self, results_dict: Dict[str, List[float]], 
                                 metric_name='fid', save_name='cross_seed_variance.png'):
        """
        图6：不同schedule的FID均值±标准差（箱线图）
        
        Args:
            results_dict: {schedule_name: [fid1, fid2, fid3, ...]} (不同seed的结果)
            metric_name: 指标名称
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        # 准备数据
        data_to_plot = []
        labels = []
        
        for schedule_name, values in results_dict.items():
            data_to_plot.append(values)
            labels.append(schedule_name)
        
        # 箱线图
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # 美化
        colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel(f'{metric_name.upper()} Score')
        ax.set_title(f'{metric_name.upper()} Distribution Across Random Seeds')
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_clip_score_distribution(self, clip_scores_dict: Dict[str, List[float]],
                                     save_name='clip_score_distribution.png'):
        """
        图7：各schedule的CLIP-Score分布（直方图/violin）
        
        Args:
            clip_scores_dict: {schedule_name: [score1, score2, ...]}
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # 直方图
        ax1 = axes[0]
        for schedule_name, scores in clip_scores_dict.items():
            ax1.hist(scores, alpha=0.6, label=schedule_name, bins=20)
        ax1.set_xlabel('CLIP Score')
        ax1.set_ylabel('Frequency')
        ax1.set_title('CLIP Score Distribution (Histogram)')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Violin图
        ax2 = axes[1]
        data_to_plot = []
        labels = []
        for schedule_name, scores in clip_scores_dict.items():
            data_to_plot.append(scores)
            labels.append(schedule_name)
        
        parts = ax2.violinplot(data_to_plot, positions=range(len(labels)), showmeans=True)
        ax2.set_xticks(range(len(labels)))
        ax2.set_xticklabels(labels, rotation=45, ha='right')
        ax2.set_ylabel('CLIP Score')
        ax2.set_title('CLIP Score Distribution (Violin Plot)')
        ax2.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_controller_output(self, controller_output_dict: Dict[str, Dict[str, List[float]]],
                              save_name='controller_output.png'):
        """
        图8：Controller输出随时间变化（用于Adaptive Annealing）
        
        Args:
            controller_output_dict: {schedule_name: {'steps': [...], 'tau': [...]}}
        """
        fig, ax = plt.subplots(1, 1, figsize=(10, 6))
        
        for schedule_name, data in controller_output_dict.items():
            if 'steps' in data and 'tau' in data:
                steps = data['steps']
                tau_values = data['tau']
                
                # 归一化steps
                if len(steps) > 0:
                    steps_normalized = np.array(steps) / max(steps) if max(steps) > 0 else np.array(steps)
                    ax.plot(steps_normalized, tau_values, label=schedule_name, linewidth=2, alpha=0.8)
        
        ax.set_xlabel('Training Progress u ∈ [0,1]')
        ax.set_ylabel('Controller Output τ(u)')
        ax.set_title('Adaptive Controller Output Over Training')
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_multi_axis_analysis(self, schedule_name: str, 
                                 loss_ema: List[float], grad_norm: List[float],
                                 tau_values: List[float], steps: List[int],
                                 save_name='multi_axis_analysis.png'):
        """
        图9：多轴曲线 - loss EMA / grad norm / τ(u)
        
        展示输入信号vs controller决策
        """
        fig, ax1 = plt.subplots(figsize=(12, 6))
        
        # 归一化steps
        steps_normalized = np.array(steps) / max(steps) if max(steps) > 0 else np.array(steps)
        
        # 第一个y轴：loss和grad norm
        color1 = 'tab:blue'
        color2 = 'tab:orange'
        ax1.set_xlabel('Training Progress u ∈ [0,1]')
        ax1.set_ylabel('Loss EMA / Gradient Norm', color='black')
        
        line1 = ax1.plot(steps_normalized, loss_ema, color=color1, label='Loss EMA', linewidth=2)
        line2 = ax1.plot(steps_normalized, grad_norm, color=color2, label='Gradient Norm', linewidth=2)
        ax1.tick_params(axis='y', labelcolor='black')
        ax1.legend(loc='upper left')
        ax1.grid(True, alpha=0.3)
        
        # 第二个y轴：tau
        ax2 = ax1.twinx()
        color3 = 'tab:red'
        ax2.set_ylabel('Controller Output τ(u)', color=color3)
        line3 = ax2.plot(steps_normalized, tau_values, color=color3, label='τ(u)', linewidth=2, linestyle='--')
        ax2.tick_params(axis='y', labelcolor=color3)
        ax2.legend(loc='upper right')
        
        plt.title(f'Multi-Axis Analysis: {schedule_name}')
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def plot_pareto_front(self, results_dict: Dict[str, Dict[str, float]],
                         save_name='pareto_front.png'):
        """
        图10：Pareto front - 质量 vs 计算成本
        
        Args:
            results_dict: {
                'schedule_name': {
                    'fid': 45.2,
                    'compute_time': 12.5,  # GPU hours
                    'total_steps': 50000
                }
            }
        """
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # 图1: FID vs Compute Time
        ax1 = axes[0]
        for schedule_name, metrics in results_dict.items():
            if 'fid' in metrics and 'compute_time' in metrics:
                ax1.scatter(metrics['compute_time'], metrics['fid'], 
                           label=schedule_name, s=100, alpha=0.7)
                ax1.annotate(schedule_name, 
                           (metrics['compute_time'], metrics['fid']),
                           fontsize=8)
        
        ax1.set_xlabel('Compute Time (GPU Hours)')
        ax1.set_ylabel('FID (越低越好)')
        ax1.set_title('Quality vs Compute Cost')
        ax1.invert_yaxis()  # FID越低越好，所以反转y轴
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 图2: FID vs Total Steps
        ax2 = axes[1]
        for schedule_name, metrics in results_dict.items():
            if 'fid' in metrics and 'total_steps' in metrics:
                ax2.scatter(metrics['total_steps'], metrics['fid'],
                           label=schedule_name, s=100, alpha=0.7)
                ax2.annotate(schedule_name,
                           (metrics['total_steps'], metrics['fid']),
                           fontsize=8)
        
        ax2.set_xlabel('Total Training Steps')
        ax2.set_ylabel('FID (越低越好)')
        ax2.set_title('Quality vs Training Steps')
        ax2.invert_yaxis()
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(self.save_dir / save_name, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"✅ 已保存: {save_name}")
    
    def generate_all_plots(self):
        """生成所有可视化图表"""
        print("开始生成可视化图表...")
        
        # 核心可视化（必须）
        if self.metrics_history:
            self.plot_metrics_vs_steps()
        
        if self.loss_history:
            self.plot_loss_curves()
        
        if self.schedule_history:
            self.plot_schedule_params()
        
        if self.grad_norm_history:
            self.plot_grad_norms()
        
        print(f"\n✅ 所有图表已保存到: {self.save_dir}")
    
    def save_data(self, save_path='results/visualization_data.json'):
        """保存所有数据（用于后续分析）"""
        data = {
            'metrics_history': self.metrics_history,
            'loss_history': self.loss_history,
            'schedule_history': self.schedule_history,
            'grad_norm_history': self.grad_norm_history
        }
        
        # 转换numpy数组为列表（JSON可序列化）
        def convert_to_list(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, (list, tuple)):
                return [convert_to_list(item) for item in obj]
            elif isinstance(obj, dict):
                return {k: convert_to_list(v) for k, v in obj.items()}
            else:
                return obj
        
        data_serializable = convert_to_list(data)
        
        with open(save_path, 'w') as f:
            json.dump(data_serializable, f, indent=2)
        
        print(f"✅ 数据已保存到: {save_path}")


# 便捷函数
def create_visualizer(save_dir='results/figures'):
    """创建可视化器实例"""
    return TrainingVisualizer(save_dir)

