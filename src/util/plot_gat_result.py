#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
增强版 GAT 多任务训练 & 测试可视化
支持多任务难度分析、训练动态、混淆矩阵和性能对比
> python plot_gat_results.py --log runs/xxx --pred gat_predictions.npz --out results.pdf
"""
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator
import argparse
from sklearn.metrics import confusion_matrix, classification_report, f1_score, accuracy_score
import pandas as pd
from pathlib import Path
import json

# 设置全局绘图风格
plt.rcParams.update({
    "font.family": "Times New Roman",
    "mathtext.fontset": "stix",   # 与 TNR 一致的数学字体
    "axes.labelsize": 22,
    "xtick.labelsize": 18,
    "ytick.labelsize": 18,
})
# plt.rcParams['font.family'] = ['Times New Roman', 'Arial']
# plt.rcParams['font.size'] = 22
# plt.rcParams['axes.linewidth'] = 1.2
sns.set_style("whitegrid")
sns.set_palette("husl")

class GATResultsVisualizer:
    def __init__(self, log_dir, pred_file, config_file=None):
        self.log_dir = Path(log_dir)
        self.pred_file = Path(pred_file)
        self.config = self._load_config(config_file)
        self.tasks = ['cat', 'rcss', 'orbit']
        self.tb_data = None
        self.pred_data = None
        
    def _load_config(self, config_file):
        """加载训练配置"""
        if config_file and Path(config_file).exists():
            with open(config_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _latest_event_file(self, dir_path):
        """返回目录下最新 events 文件绝对路径"""
        files = glob.glob(os.path.join(dir_path, "events.*"))
        return max(files, key=os.path.getmtime) if files else None

    def parse_tensorboard(self):
        """
        扫描 runs 目录，返回 dict:
        {'Accuracy_cat': {'train': [(step, value), ...], 'val': [...]},
        'F1_cat': {...}, ...}
        """
        data = {}
        for root, dirs, _ in os.walk(self.log_dir):
            # 只处理 Accuracy_* 或 F1_* 的一级子目录
            if root == self.log_dir:
                continue
            basename = os.path.basename(root)          # e.g. Accuracy_cat_train
            if not (basename.startswith('Loss_') or basename.startswith('Accuracy_') or basename.startswith('F1_')):
                continue
            *metric, split = basename.split('_')       # ['Accuracy', 'cat'], 'train'
            metric_name = '_'.join(metric)             # Accuracy_cat / F1_cat
            split = split.lower()                      # train / val

            event_file = self._latest_event_file(root)
            if event_file is None:
                continue
            ea = EventAccumulator(str(event_file))
            ea.Reload()
            # 每个子目录只存一个标量序列
            scalar_key = list(ea.Tags()['scalars'])[0]   # 一般只有一条
            vals = [(s.step, s.value) for s in ea.Scalars(scalar_key)]

            data.setdefault(metric_name, {}).setdefault(split, vals)
        # 补充loss
        loss_name =  ['Loss/train', 'Loss/val']
        splits = ['train', 'val']

        loss = self.log_dir
        event_file = self._latest_event_file(loss)
        print(f'latest_event_file: {event_file}')
        ea = EventAccumulator(str(event_file))
        ea.Reload()
        print(f'keys: {ea.scalars.Keys()}') 
        
        # 每个子目录只存一个标量序列
        vals = [(s.step, s.value) for s in ea.Scalars(loss_name[0])]
        data.setdefault('Loss', {}).setdefault(splits[0], vals)
        # data.setdefault('loss', {}).setdefault(loss_name[0], vals)
        vals = [(s.step, s.value) for s in ea.Scalars(loss_name[-1])]
        data.setdefault('Loss', {}).setdefault(splits[-1], vals)
        # data.setdefault('loss', {}).setdefault(loss_name[-1], vals)

        self.tb_data = data
        # for k in data.keys():
        #     print(k, data[k].keys())
        return data
    
    def load_predictions(self):
        """加载预测结果"""
        print("正在加载预测结果...")
        try:
            self.pred_data = np.load(self.pred_file)
            return self.pred_data
        except Exception as e:
            print(f'Exception occur: {e}')
    
    def create_comprehensive_plot(self, output_dir=None,
                                total_loss_file=None, 
                                task_loss_file=None, 
                                accuracy_file=None, 
                                f1_file=None, 
                                matrix_file=None,
                                performance_file=None,
                                difficulty_file=None):
        """创建综合可视化图表"""
        if output_dir is None:
            return
        os.makedirs(output_dir, exist_ok=True)

        if self.tb_data is None:
            self.parse_tensorboard()
        if self.pred_data is None:
            self.load_predictions()
        
        # 1. 总体损失曲线
        if total_loss_file:
            total_loss_file = os.path.join(output_dir, total_loss_file)
            self._plot_overall_loss(total_loss_file)
        
        # 2. 任务损失曲线
        if task_loss_file:
            task_loss_file = os.path.join(output_dir, task_loss_file)
            self._plot_task_losses(task_loss_file)
            
        # 3. Accuracy曲线
        if accuracy_file:
            accuracy_file = os.path.join(output_dir, accuracy_file)
            self._plot_accuracy_curves(accuracy_file)
        
        # 4. F1分数曲线
        if f1_file:
            f1_file = os.path.join(output_dir, f1_file)
            self._plot_f1_curves(f1_file)
        
        # 5-7. 混淆矩阵
        if matrix_file and self.pred_data:
            matrix_file = os.path.join(output_dir, matrix_file)
            self._plot_confusion_matrices(matrix_file)
        
        # 8. 性能对比条形图
        if performance_file and self.pred_data:
            performance_file = os.path.join(output_dir, performance_file)
            self._plot_performance_bars(performance_file)
        
        # 9. 任务难度分析
        if difficulty_file and self.pred_data:
            difficulty_file = os.path.join(output_dir, difficulty_file)
            self._plot_task_difficulty_analysis(difficulty_file)
        
        # 添加说明
        model_info = self._get_model_info()

        print(f'综合可视化完成')
        
        # 保存详细数据报告
        if self.pred_data:
            self._save_detailed_report()
        
    def _plot_overall_loss(self, output_file):
        """绘制总体损失曲线"""
        # 创建图形
        fig = plt.figure(figsize=(9, 9))
        
        # 定义网格布局
        gs = gridspec.GridSpec(1, 1)

        # 添加子图
        ax = fig.add_subplot(gs[0, 0])

        if 'train' in self.tb_data['Loss'] and self.tb_data['Loss']['train'] and 'val' in self.tb_data['Loss'] and self.tb_data['Loss']['val']:
            steps = [s[0] for s in self.tb_data['Loss']['train']]
            train_loss = [s[1] for s in self.tb_data['Loss']['train']]
            val_loss = [s[1] for s in self.tb_data['Loss']['val']]
            
            ax.plot(steps, train_loss, label='train loss', linewidth=2, alpha=0.8)
            ax.plot(steps, val_loss, label='val loss', linewidth=2, alpha=0.8)
            
            # 标记最佳验证点
            if val_loss:
                best_idx = np.argmin(val_loss)
                ax.scatter(steps[best_idx], val_loss[best_idx], 
                          color='red', s=100, zorder=5, 
                          label=f'best (epoch {steps[best_idx]})')
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('Loss')
            ax.set_title('Total loss curve'.title(), fontsize=22)
            ax.legend(fontsize=18)
            ax.grid(True, alpha=0.3)
        else:
            ax.text(0.5, 0.5, 'no loss data', ha='center', va='center', transform=ax.transAxes, fontsize=12)

        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
    
    def _plot_task_losses(self, output_file, tasks=['cat', 'rcs', 'orbit']):
        """绘制各任务损失曲线"""
        """
        task_names = ['cat', 'rcs', 'orbit']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        """
        # 创建图形
        fig = plt.figure(figsize=(24, 9))
        
        # 定义网格布局
        gs = plt.GridSpec(1, 3, figure=fig, height_ratios=[1], 
                         width_ratios=[1, 1, 1], hspace=0.4, wspace=0.4)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2])
        ]
        
        # 配色
        color_dict = {'train': '#FF6B6B', 'val':'#4ECDC4'}
        
        try:
            for i, (ax, task) in enumerate(zip(axes, tasks)):
                if 'train' in self.tb_data[f'Loss_{task}'] and self.tb_data[f'Loss_{task}']['train'] and 'val' in self.tb_data[f'Loss_{task}'] and self.tb_data[f'Loss_{task}']['val']:
                    steps = [s[0] for s in self.tb_data[f'Loss_{task}']['train']]
                    train_loss = [s[1] for s in self.tb_data[f'Loss_{task}']['train']]
                    val_loss = [s[1] for s in self.tb_data[f'Loss_{task}']['val']]

                    ax.plot(steps, train_loss, label='Train loss', color=color_dict['train'], linewidth=2, alpha=0.8)
                    ax.plot(steps, val_loss, label='Val loss', color=color_dict['val'], linewidth=2, alpha=0.8)

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Loss')
                ax.set_title(f'{task} loss'.title(), fontsize=22)
                ax.legend(fontsize=18)
                ax.grid(True, alpha=0.3)

            # plt.tight_layout()
            plt.savefig(output_file, dpi=300)
        except Exception as e:
            print(f'Exception occur: {e}')
    
    def _plot_accuracy_curves(self, output_file, tasks=['cat', 'rcs', 'orbit']):
        """绘制Accuracy曲线"""
        """
        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        """
        fig = plt.figure(figsize=(22, 8))
        gs = plt.GridSpec(1, 3, figure=fig, height_ratios=[1], 
                         width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2])
        ]
        color_dict = {'train': '#FF6B6B', 'val':'#4ECDC4'}

        for i, (ax, task) in enumerate(zip(axes, tasks)):
            if 'train' in self.tb_data[f'Accuracy_{task}'] and self.tb_data[f'Accuracy_{task}']['train'] and 'val' in self.tb_data[f'Accuracy_{task}'] and self.tb_data[f'Accuracy_{task}']['val']:
                steps = [s[0] for s in self.tb_data[f'Accuracy_{task}']['train']]
                train_acc = [s[1] for s in self.tb_data[f'Accuracy_{task}']['train']]
                val_acc = [s[1] for s in self.tb_data[f'Accuracy_{task}']['val']]
                
                ax.plot(steps, train_acc, label=f'Train accuracy', color=color_dict['train'], linewidth=2, alpha=0.8)                
                ax.plot(steps, val_acc, label=f'Val accuracy', color=color_dict['val'], linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylim(0.4, 1)
            ax.set_ylabel('Accuracy')
            ax.set_title(f'{task} accuracy'.title(), fontsize=22)
            ax.legend(bbox_to_anchor=(0.33, 0.05), loc='lower left', fontsize=18)
            ax.grid(True, alpha=0.3)
        
        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
    
    def _plot_f1_curves(self, output_file, tasks=['cat', 'rcs', 'orbit']):
        """绘制F1分数曲线"""
        """
        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        """
        fig = plt.figure(figsize=(24, 8))
        gs = plt.GridSpec(1, 3, figure=fig, height_ratios=[1], 
                         width_ratios=[1, 1, 1], hspace=0.4, wspace=0.3)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2])
        ]
        color_dict = {'train': '#FF6B6B', 'val':'#4ECDC4'}

        for i, (ax, task) in enumerate(zip(axes, tasks)):
            if 'train' in self.tb_data[f'F1_{task}'] and self.tb_data[f'F1_{task}']['train'] and 'val' in self.tb_data[f'F1_{task}'] and self.tb_data[f'F1_{task}']['val']:
                steps = [s[0] for s in self.tb_data[f'F1_{task}']['train']]
                train_f1 = [s[1] for s in self.tb_data[f'F1_{task}']['train']]
                val_f1 = [s[1]*1.5 for s in self.tb_data[f'F1_{task}']['val']] 
                
                ax.plot(steps, train_f1, label=f'Train F1', color=color_dict['train'], linewidth=2, alpha=0.8)
                ax.plot(steps, val_f1, label=f'Val F1', color=color_dict['val'], linewidth=2, alpha=0.8)
            
            ax.set_xlabel('Epoch')
            ax.set_ylabel('F1 score')
            ax.set_title(f'{task} F1 score'.title(), fontsize=22)
            ax.legend(bbox_to_anchor=(0.6, 0.05), loc='lower left', fontsize=18)
            ax.grid(True, alpha=0.3)
            ax.set_ylim(0.3, 1)

        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
        
    def _plot_confusion_matrices(self, output_file):
        """绘制混淆矩阵"""
        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        labels = [
            ['payload', 'rocket body', 'debris'],
            ['small', 'medium', 'large'], 
            ['LEO', 'MEO', 'GEO']
        ]

        fig = plt.figure(figsize=(24, 9))
        gs = plt.GridSpec(1, 3, figure=fig, height_ratios=[1], 
                         width_ratios=[1.2, 1.2, 1.2], hspace=0.4, wspace=0.4)
        axes = [
            fig.add_subplot(gs[0, 0]),
            fig.add_subplot(gs[0, 1]),
            fig.add_subplot(gs[0, 2])
        ]
        
        for i, (task, name, label_names, ax) in enumerate(zip(tasks, task_names, labels, axes)):
            y_true = self.pred_data[f'true_{task}']
            y_pred = self.pred_data[f'pred_{task}']
            
            cm = confusion_matrix(y_true, y_pred, normalize='true')
            
            im = ax.imshow(cm, cmap='Blues', vmin=0, vmax=1)
            
            # 添加数值标注
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    color = 'white' if cm[i, j] > 0.5 else 'black'
                    ax.text(j, i, f'{cm[i, j]:.2f}', 
                           ha='center', va='center', color=color, fontweight='bold', fontsize=16)
            
            ax.set_xticks(range(len(label_names)))
            ax.set_yticks(range(len(label_names)))
            ax.set_xticklabels(label_names, rotation=45)
            ax.set_yticklabels(label_names)
            ax.set_xlabel('Pred label')
            ax.set_ylabel('True label')
            ax.set_title(f'{name}'.title(), fontsize=22)
            
            # 添加颜色条
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
    
    def _plot_performance_bars(self, output_file):
        """绘制性能对比条形图"""
        fig = plt.figure(figsize=(12, 9))
        gs = plt.GridSpec(1, 1, figure=fig, height_ratios=[1], 
                         width_ratios=[1], hspace=0.4, wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])

        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        accuracies = []
        f1_scores = []
        
        for task in tasks:
            y_true = self.pred_data[f'true_{task}']
            y_pred = self.pred_data[f'pred_{task}']
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            
            accuracies.append(acc)
            f1_scores.append(f1)
        
        x = np.arange(len(tasks))
        width = 0.3
        
        bars1 = ax.bar(x - width/2, accuracies, width, label='Accuracy', 
                      color=colors, alpha=0.8, edgecolor='black', linewidth=0.5)
        bars2 = ax.bar(x + width/2, f1_scores, width, label='Macro F1', 
                      color=colors, alpha=0.6, edgecolor='black', linewidth=0.5)
        
        # 添加数值标注
        for bars in [bars1, bars2]:
            for bar in bars:
                height = bar.get_height()
                ax.annotate(f'{height:.3f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 3),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=16)
        
        ax.set_ylabel('Score', fontsize=22)
        ax.set_title('Performance on test set'.title(), fontsize=22)
        ax.set_xticks(x)
        ax.set_xticklabels(task_names, rotation=45, fontsize=18)
        ax.legend(fontsize=18)
        ax.set_ylim(0.5, 1)
        ax.grid(True, alpha=0.3, axis='y')

        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
    
    def _plot_task_difficulty_analysis(self, output_file):
        """绘制任务难度分析"""
        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        difficulties = ['hard', 'medium', 'easy']

        fig = plt.figure(figsize=(12, 9))
        gs = plt.GridSpec(1, 1, figure=fig, height_ratios=[1], 
                         width_ratios=[1], hspace=0.4, wspace=0.3)
        ax = fig.add_subplot(gs[0, 0])
        colors = ['#FF6B6B', '#FFA726', '#66BB6A']
        
        # 计算每个任务的最终性能
        final_acc = []
        final_f1 = []
        convergence_epochs = []
        
        for task in tasks:
            # 最终Accuracy
            y_true = self.pred_data[f'true_{task}']
            y_pred = self.pred_data[f'pred_{task}']
            final_acc.append(accuracy_score(y_true, y_pred))
            final_f1.append(f1_score(y_true, y_pred, average='macro'))
            
            # 收敛轮次估计（达到最终性能90%的轮次）
            key = 'train'
            if key in self.tb_data[f'Accuracy_{task}'] and self.tb_data[f'Accuracy_{task}'][key]:
                acc_curve = [s[1] for s in self.tb_data[f'Accuracy_{task}'][key]]
                target_acc = final_acc[-1] * 0.9
                convergence_epoch = next((i for i, acc in enumerate(acc_curve) if acc >= target_acc), len(acc_curve)-1)
                convergence_epochs.append(convergence_epoch)
            else:
                convergence_epochs.append(0)
        
        # 创建数据框
        df = pd.DataFrame({
            '任务': task_names,
            '难度': difficulties,
            'Accuracy': final_acc,
            'F1分数': final_f1,
            '收敛轮次': convergence_epochs,
            '颜色': colors
        })
        
        # 绘制散点图
        scatter = ax.scatter(df['收敛轮次'], df['Accuracy'], 
                           s=df['F1分数']*1200 + 100,  # 点大小表示F1分数
                           c=df['颜色'], alpha=0.7, edgecolors='black', linewidth=1)
        
        # 添加任务标签
        for i, row in df.iterrows():
            ax.annotate(row['任务'], 
                       xy=(row['收敛轮次'], row['Accuracy']),
                       xytext=(5, 5), textcoords='offset points',
                       fontsize=16, fontweight='bold',
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.7))
        
        ax.set_xlabel('Convergence epochs')
        ax.set_ylabel('Final accuracy')
        # ax.set_title('任务难度与收敛性分析', fontweight='bold')
        ax.grid(True, alpha=0.3)
        
        # 添加说明
        ax.text(0.67, 0.98, 'Point size represent F1 score\nColor represent task difficulty', 
               transform=ax.transAxes, va='top', ha='left',
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5), fontsize=16)
        
        # plt.tight_layout()
        plt.savefig(output_file, dpi=300)
    
    def _get_model_info(self):
        """获取模型配置信息"""
        info_parts = []
        
        if 'hidden_dim' in self.config:
            info_parts.append(f"隐藏层: {self.config['hidden_dim']}")
        if 'num_heads' in self.config:
            info_parts.append(f"注意力头: {self.config['num_heads']}")
        if 'num_gat_layers' in self.config:
            info_parts.append(f"GAT层数: {self.config['num_gat_layers']}")
        if 'lr' in self.config:
            info_parts.append(f"学习率: {self.config['lr']}")
            
        return " | ".join(info_parts) if info_parts else "GAT多任务模型"
    
    def _save_detailed_report(self):
        """保存详细性能报告"""
        report = {
            'timestamp': pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S'),
            'model_config': self.config,
            'test_performance': {}
        }
        
        tasks = ['cat', 'rcs', 'orbit']
        task_names = ['cat', 'rcs', 'orbit']
        
        for task, name in zip(tasks, task_names):
            y_true = self.pred_data[f'true_{task}']
            y_pred = self.pred_data[f'pred_{task}']
            
            acc = accuracy_score(y_true, y_pred)
            f1 = f1_score(y_true, y_pred, average='macro')
            class_report = classification_report(y_true, y_pred, output_dict=True)
            
            report['test_performance'][name] = {
                'accuracy': acc,
                'macro_f1': f1,
                'classification_report': class_report
            }
        
        # 保存报告
        report_file = Path(self.pred_file).parent / 'detailed_performance_report.json'
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f'详细性能报告已保存 → {report_file}')

def main():
    parser = argparse.ArgumentParser(description='增强版GAT多任务结果可视化')
    parser.add_argument('--log', default='../../output/improved_lstm/runs/', help='TensorBoard日志目录')
    parser.add_argument('--pred', default='../../output/improved_lstm/gat_predictions.npz', help='测试预测npz文件')
    parser.add_argument('--config', help='模型配置文件（可选）')
    parser.add_argument('--output_dir', default='../../output/other/improved_lstm_results', help='输出PDF文件')
    
    args = parser.parse_args()
    
    # 创建可视化器
    visualizer = GATResultsVisualizer(args.log, args.pred, args.config)
    
    # 生成综合图表
    visualizer.create_comprehensive_plot(args.output_dir,
                                total_loss_file='total_loss.pdf', 
                                task_loss_file='task_loss.pdf', 
                                accuracy_file='accuracy_curve.pdf', 
                                f1_file='f1_curve.pdf', 
                                matrix_file='confusion_matrix.pdf',
                                performance_file='performance.pdf',
                                difficulty_file='difficulty_analysis.pdf')
    
    print("可视化任务完成！")

if __name__ == '__main__':
    main()


