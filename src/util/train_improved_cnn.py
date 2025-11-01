#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
改进的训练脚本
"""
import os
import json
import torch
import argparse
from datetime import datetime
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from space_dataset import SpaceDataset, collate_fn
from improved_cnn import MultiTaskCNNClassifier, improved_collate_fn
from collections import Counter
from sklearn.metrics import classification_report, confusion_matrix

from loss import FocalLoss, MultiTaskUncertaintyLoss, calculate_class_weights
import warnings
warnings.filterwarnings('ignore')

# ---------- 训练工具函数 ----------
class EarlyStopping:
    """早停机制"""
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.val_loss_min = np.Inf
        self.delta = delta

    def __call__(self, val_loss, model, path):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(val_loss, model, path)
            self.counter = 0

    def save_checkpoint(self, val_loss, model, path):
        if self.verbose:
            print(f'Validation loss decreased ({self.val_loss_min:.6f} --> {val_loss:.6f}). Saving model...')
        torch.save(model.state_dict(), path)
        self.val_loss_min = val_loss


def run_epoch(loader, model, criterions, optimizer=None, device='cuda', task_weights=None):
    """运行一个epoch"""
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    task_correct = {task: 0 for task in ['cat', 'rcs', 'orbit']}
    task_total = {task: 0 for task in ['cat', 'rcs', 'orbit']}
    
    if task_weights is None:
        task_weights = {'cat': 0.6, 'rcs': 0.3, 'orbit': 0.1}
    
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            sequences = batch['sequences'].to(device)
            cat_labels = batch['cat_labels'].to(device)
            rcs_labels = batch['rcs_labels'].to(device)
            orbit_labels = batch['orbit_labels'].to(device)
            
            # 前向传播
            outputs = model(sequences)
            
            # 计算多任务损失
            loss_cat = criterions['cat'](outputs['cat'], cat_labels)
            loss_rcs = criterions['rcs'](outputs['rcs'], rcs_labels)
            loss_orbit = criterions['orbit'](outputs['orbit'], orbit_labels)
            
            # total_loss_batch = (task_weights['cat'] * loss_cat + 
            #                   task_weights['rcs'] * loss_rcs + 
            #                   task_weights['orbit'] * loss_orbit)
            uncertainty_w = MultiTaskUncertaintyLoss().to(device)
            losses = {'cat': loss_cat, 'rcs': loss_rcs, 'orbit': loss_orbit}
            total_loss_batch = uncertainty_w(losses)   # 或手工加权版本
            
            if is_train:
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 统计
            batch_size = len(cat_labels)
            total_loss += total_loss_batch.item() * batch_size
            
            # 计算各任务准确率
            pred_cat = outputs['cat'].argmax(1)
            pred_rcs = outputs['rcs'].argmax(1)
            pred_orbit = outputs['orbit'].argmax(1)
            
            task_correct['cat'] += (pred_cat == cat_labels).sum().item()
            task_correct['rcs'] += (pred_rcs == rcs_labels).sum().item()
            task_correct['orbit'] += (pred_orbit == orbit_labels).sum().item()
            
            task_total['cat'] += batch_size
            task_total['rcs'] += batch_size
            task_total['orbit'] += batch_size
    
    avg_loss = total_loss / len(loader.dataset)
    task_accuracies = {
        task: task_correct[task] / task_total[task] for task in task_correct
    }
    
    return avg_loss, task_accuracies

# ---------- 主流程 ----------
def main():
    parser = argparse.ArgumentParser(description='训练多任务CNN空间目标分类模型')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--base_ch', type=int, default=64)
    parser.add_argument('--num_blocks', type=int, default=4)
    parser.add_argument('--dropout', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=1440)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--task_weights', type=str, default="0.6,0.3,0.1", 
                       help="任务权重: cat,rcs,orbit")
    parser.add_argument("--train_hdf5", type=str, default='../../../../db/output/train_little_monthly.h5')
    parser.add_argument("--val_hdf5", type=str, default='../../../../db/output/val_little_monthly.h5')
    parser.add_argument("--test_hdf5", type=str, default='../../../../db/output/test_little_monthly.h5')
    parser.add_argument("--save_dir", type=str, default="../../output/improved_cnn-monthly_split-little")
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 解析任务权重
    task_weights = {
        'cat': float(args.task_weights.split(',')[0]),
        'rcs': float(args.task_weights.split(',')[1]),
        'orbit': float(args.task_weights.split(',')[2])
    }
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_h5 = args.train_hdf5
    val_h5 = args.val_hdf5
    test_h5 = args.test_hdf5
    exp_dir = args.save_dir
    os.makedirs(exp_dir, exist_ok=True)
    os.makedirs(f"{exp_dir}/runs", exist_ok=True)
    os.makedirs(f"{exp_dir}/checkpoints", exist_ok=True)
    
    # 数据加载
    print("加载数据集...")
    train_ds = SpaceDataset(train_h5, window_size=args.window_size)
    val_ds = SpaceDataset(val_h5, window_size=args.window_size)
    test_ds = SpaceDataset(test_h5, window_size=args.window_size)
    
    # 计算类别权重
    weights_cat = calculate_class_weights(train_ds, 'cat').to(device)
    weights_rcs = calculate_class_weights(train_ds, 'rcs').to(device)
    weights_orbit = calculate_class_weights(train_ds, 'orbit').to(device)
    
    print(f"类别权重 - 目标类型: {weights_cat}")
    print(f"类别权重 - RCS大小: {weights_rcs}")
    print(f"类别权重 - 轨道类型: {weights_orbit}")
    
    # 数据加载器
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=improved_collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=improved_collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=improved_collate_fn, num_workers=4, pin_memory=True
    )
    
    print(f"训练集: {len(train_ds)} 样本")
    print(f"验证集: {len(val_ds)} 样本")
    print(f"测试集: {len(test_ds)} 样本")
    
    # 模型
    model = MultiTaskCNNClassifier(
        base_ch=args.base_ch,
        num_blocks=args.num_blocks,
        dropout=args.dropout
    ).to(device)
    
    print(f"模型参数数量: {sum(p.numel() for p in model.parameters()):,}")
    
    # 多任务损失函数
    # criterions = {
    #     'cat': nn.CrossEntropyLoss(weight=weights_cat),
    #     'rcs': nn.CrossEntropyLoss(weight=weights_rcs),
    #     'orbit': nn.CrossEntropyLoss(weight=weights_orbit)
    # }
    criterions = {
        'cat': FocalLoss(alpha=weights_cat, gamma=2.0).to(device),
        'rcs': FocalLoss(alpha=weights_rcs, gamma=2.0).to(device),
        'orbit': FocalLoss(alpha=weights_orbit, gamma=2.0).to(device)
    }
    
    # 优化器
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=1e-4
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )
    
    # 早停
    early_stopping = EarlyStopping(patience=args.patience, verbose=True)
    
    # TensorBoard
    writer = SummaryWriter(f"{exp_dir}/runs")
    
    # 训练循环
    best_val_acc = 0.0
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_accs = run_epoch(
            train_loader, model, criterions, optimizer, device, task_weights
        )
        
        # 验证
        val_loss, val_accs = run_epoch(
            val_loader, model, criterions, device=device, task_weights=task_weights
        )
        
        # 学习率调度
        scheduler.step(val_loss)
        
        # 早停检查
        early_stopping(val_loss, model, f'{exp_dir}/checkpoints/best_cnn_model.pt')
        
        # 记录到TensorBoard
        writer.add_scalar('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        
        for task in ['cat', 'rcs', 'orbit']:
            writer.add_scalars(f'Accuracy/{task}', {
                'train': train_accs[task],
                'val': val_accs[task]
            }, epoch)
        
        writer.add_scalar('LR', optimizer.param_groups[0]['lr'], epoch)
        
        # 打印进度
        if epoch % 5 == 0:
            print(f'Epoch {epoch:03d}/{args.epochs}:')
            print(f'  Train Loss: {train_loss:.4f}')
            print(f'  Val Loss: {val_loss:.4f}')
            for task in ['cat', 'rcs', 'orbit']:
                print(f'  {task.capitalize()} - Train: {train_accs[task]:.4f}, Val: {val_accs[task]:.4f}')
        
        # 保存最佳模型
        current_val_acc = val_accs['cat']  # 以主要任务为准
        if current_val_acc > best_val_acc:
            best_val_acc = current_val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_accs,
                'val_loss': val_loss,
            }, f'{exp_dir}/checkpoints/best_cnn_model_full.pt')
        
        if early_stopping.early_stop:
            print("早停触发!")
            break
    
    # 加载最佳模型进行测试
    print("在测试集上评估最佳模型...")
    model.load_state_dict(torch.load(f'{exp_dir}/checkpoints/best_cnn_model.pt'))
    test_loss, test_accs = run_epoch(
        test_loader, model, criterions, device=device, task_weights=task_weights
    )
    
    # 输出测试结果
    print(f"\n测试结果:")
    print(f"测试损失: {test_loss:.4f}")
    for task in ['cat', 'rcs', 'orbit']:
        print(f"{task.capitalize()} 准确率: {test_accs[task]:.4f}")
    
    # 保存预测结果
    model.eval()
    all_predictions = {'cat': [], 'rcs': [], 'orbit': []}
    all_labels = {'cat': [], 'rcs': [], 'orbit': []}
    
    with torch.no_grad():
        for batch in test_loader:
            sequences = batch['sequences'].to(device)
            outputs = model(sequences)
            
            for task in ['cat', 'rcs', 'orbit']:
                preds = outputs[task].argmax(1).cpu().numpy()
                labels = batch[f'{task}_labels'].numpy()
                all_predictions[task].extend(preds)
                all_labels[task].extend(labels)
    
    results = {
        'test_loss': test_loss,
        'test_accuracies': test_accs,
        'predictions': all_predictions,
        'labels': all_labels
    }
    np.save(f'{exp_dir}/cnn_test_results.npy', results)
    
    print("预测结果已保存为 cnn_test_results.npy")
    
    # 关闭数据集
    train_ds.close()
    val_ds.close()
    test_ds.close()
    writer.close()

if __name__ == '__main__':
    main()