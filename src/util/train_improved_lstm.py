#!/usr/bin/env python3
import os
import json
import torch
import argparse
import numpy as np
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

from space_dataset import SpaceDataset, RandomNoise, collate_fn
from improved_lstm import MultiTaskLSTMClassifier


class EarlyStopping:
    def __init__(self, patience=10, delta=0, path='../../output/checkpoint.pt'):
        self.patience = patience
        self.delta = delta
        self.path = path
        self.counter = 0
        self.best_score = None
        self.early_stop = False

    def __call__(self, val_loss, model):
        score = -val_loss
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model)
            self.counter = 0

    def save_checkpoint(self, model):
        torch.save(model.state_dict(), self.path)

def compute_metrics(preds, targets, num_classes):
    """计算准确率、精确率、召回率、F1"""
    acc = (preds == targets).float().mean()
    
    # 对于多分类，计算宏平均F1
    f1_scores = []
    for cls in range(num_classes):
        tp = ((preds == cls) & (targets == cls)).sum().float()
        fp = ((preds == cls) & (targets != cls)).sum().float()
        fn = ((preds != cls) & (targets == cls)).sum().float()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        f1_scores.append(f1)
    
    macro_f1 = torch.stack(f1_scores).mean()
    return acc, macro_f1

def run_epoch(loader, model, criterion, optimizer=None, device='cpu', task_weights=None):
    is_train = optimizer is not None
    model.train() if is_train else model.eval()
    
    total_loss = 0.0
    task_losses = {'orbit': 0.0, 'cat': 0.0, 'rcs': 0.0}
    task_accs = {'orbit': 0.0, 'cat': 0.0, 'rcs': 0.0}
    task_f1s = {'orbit': 0.0, 'cat': 0.0, 'rcs': 0.0}
    total_samples = 0
    
    with torch.set_grad_enabled(is_train):
        for batch in loader:
            # 移动到设备
            sequences = batch['sequences'].to(device)
            seq_lens = batch['lengths'].to(device)
            orbit_labels = batch['orbit_labels'].to(device)
            rcs_labels = batch['rcs_labels'].to(device)
            cat_labels = batch['cat_labels'].to(device)
            
            batch_size = sequences.size(0)
            total_samples += batch_size
            
            # 前向传播
            outputs = model(sequences, seq_lens)
            
            # 计算损失
            orbit_loss = criterion(outputs['orbit'], orbit_labels)
            cat_loss = criterion(outputs['cat'], cat_labels)
            rcs_loss = criterion(outputs['rcs'], rcs_labels)
            
            # 加权多任务损失
            if task_weights:
                total_loss_batch = (task_weights['orbit'] * orbit_loss + 
                                    task_weights['cat'] * cat_loss + 
                                    task_weights['rcs'] * rcs_loss)
            else:
                total_loss_batch = orbit_loss + cat_loss + rcs_loss
            
            # 反向传播
            if is_train:
                optimizer.zero_grad()
                total_loss_batch.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 统计信息
            total_loss += total_loss_batch.item() * batch_size
            task_losses['orbit'] += orbit_loss.item() * batch_size
            task_losses['cat'] += cat_loss.item() * batch_size
            task_losses['rcs'] += rcs_loss.item() * batch_size
            
            # 计算指标
            for task_name in ['orbit', 'cat', 'rcs']:
                preds = outputs[task_name].argmax(dim=1)
                targets = batch[f'{task_name}_labels'].to(device)
                acc, f1 = compute_metrics(preds, targets, outputs[task_name].size(1))
                task_accs[task_name] += acc.item() * batch_size
                task_f1s[task_name] += f1.item() * batch_size
    
    # 平均指标
    avg_loss = total_loss / total_samples
    for task in task_losses:
        task_losses[task] /= total_samples
        task_accs[task] /= total_samples
        task_f1s[task] /= total_samples
    
    return avg_loss, task_losses, task_accs, task_f1s

def main():
    parser = argparse.ArgumentParser(description='训练多任务LSTM分类器')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden_dim', type=int, default=128)
    parser.add_argument('--num_layers', type=int, default=2)
    parser.add_argument('--window_size', type=int, default=1440)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_weights', type=str, default="0.6,0.3,0.1", 
                       help="任务权重: cat,rcs,orbit")
    parser.add_argument("--train_hdf5", type=str, default='../../../../db/output/train.h5')
    parser.add_argument("--val_hdf5", type=str, default='../../../../db/output/val.h5')
    parser.add_argument("--test_hdf5", type=str, default='../../../../db/output/test.h5')
    parser.add_argument("--save_dir", type=str, default="../../output/improved_lstm")
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')
    
    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    train_h5 = args.train_hdf5
    val_h5 = args.val_hdf5
    test_h5 = args.test_hdf5
    exp_dir = args.save_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(f'{exp_dir}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    os.makedirs(f"{exp_dir}/runs/", exist_ok=True)
    writer = SummaryWriter(f"{exp_dir}/runs/")
    
    # 数据加载
    print("加载数据集...")
    train_ds = SpaceDataset(train_h5, window_size=args.window_size)
    val_ds = SpaceDataset(val_h5, window_size=args.window_size)
    test_ds = SpaceDataset(test_h5, window_size=args.window_size)
    
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    test_loader = DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=collate_fn, num_workers=4, pin_memory=True
    )
    
    # 模型
    model = MultiTaskLSTMClassifier(
        input_dim=12,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        dropout=0.2
    ).to(device)
    
    # 损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', patience=5, factor=0.5
    )

    os.makedirs(f"{exp_dir}/checkpoints/", exist_ok=True)
    early_stopping = EarlyStopping(patience=args.patience, path=f'{exp_dir}/checkpoints/best_lstm_model.pt')
    
    # 解析任务权重
    task_weights = {
        'cat': float(args.task_weights.split(',')[0]),
        'rcs': float(args.task_weights.split(',')[1]),
        'orbit': float(args.task_weights.split(',')[2])
    }
    
    
    print("开始训练...")
    best_val_acc = 0.0
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_task_losses, train_task_accs, train_task_f1s = run_epoch(
            train_loader, model, criterion, optimizer, device, task_weights
        )
        
        # 验证
        val_loss, val_task_losses, val_task_accs, val_task_f1s = run_epoch(
            val_loader, model, criterion, device=device, task_weights=task_weights
        )
        
        scheduler.step(val_loss)
        early_stopping(val_loss, model)
        
        # 记录到tensorboard
        writer.add_scalar('Loss/train', train_loss, epoch)
        writer.add_scalar('Loss/val', val_loss, epoch)
        
        for task in ['orbit', 'cat', 'rcs']:
            writer.add_scalars(f'Acc/{task}', {
                'train': train_task_accs[task],
                'val': val_task_accs[task]
            }, epoch)
            writer.add_scalars(f'F1/{task}', {
                'train': train_task_f1s[task],
                'val': val_task_f1s[task]
            }, epoch)
        
        # 打印进度
        if epoch % 5 == 0:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
            for task in ['orbit', 'cat', 'rcs']:
                print(f'  {task:8} - Train Acc: {train_task_accs[task]:.3f}, '
                      f'Val Acc: {val_task_accs[task]:.3f}')
        
        # 保存最佳模型
        if val_task_accs['cat'] > best_val_acc:
            best_val_acc = val_task_accs['cat']
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_task_accs['cat'],
                'val_loss': val_loss,
            }, f"{exp_dir}/checkpoints/best_lstm_model_full.pt")
        
        if early_stopping.early_stop:
            print("早停触发!")
            break
    
    # 加载最佳模型进行测试
    model.load_state_dict(torch.load(f'{exp_dir}/checkpoints/best_lstm_model.pt'))
    test_loss, test_task_losses, test_task_accs, test_task_f1s = run_epoch(
        test_loader, model, criterion, device=device, task_weights=task_weights
    )
    
    print("\n测试结果:")
    print(f'测试损失: {test_loss:.4f}')
    for task in ['orbit', 'cat', 'rcs']:
        print(f'{task:8} - 准确率: {test_task_accs[task]:.3f}, F1: {test_task_f1s[task]:.3f}')
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_accs': test_task_accs,
        'test_f1s': test_task_f1s
    }
    with open(f'{exp_dir}/lstm_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    writer.close()
    train_ds.close()
    val_ds.close()
    test_ds.close()

if __name__ == '__main__':
    main()