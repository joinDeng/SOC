#!/usr/bin/env python3
import os
import json
import torch
import argparse
import numpy as np
from torch import nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch_geometric.loader import DataLoader as PyG_DataLoader
from datetime import datetime
from collections import Counter

from loss import FocalLoss, MultiTaskUncertaintyLoss
from space_dataset_graph import SpaceGraphDataset, GraphNoiseTransform, EdgeDropTransform
from improved_gat import ImprovedGATMultiTaskSinCosPE, ImprovedGATMultiTaskRoPE, register_nan_hook
from trans_spec_gat import SpectrumEnhancedGAT

class MultiTaskLoss(nn.Module):
    """多任务损失函数"""
    def __init__(self, task_weights=None, label_smoothing=0.1):
        super().__init__()
        self.task_weights = task_weights or {'cat': 1.0, 'rcs': 0.8, 'orbit': 0.6}
        self.label_smoothing = label_smoothing
        
    def forward(self, outputs, targets):
        total_loss = 0.0
        task_losses = {}
        
        for task_name in outputs.keys():
            if task_name in targets:
                logits = outputs[task_name]
                labels = targets[task_name]
                
                if self.label_smoothing > 0:
                    loss = F.cross_entropy(
                        logits, labels, 
                        label_smoothing=self.label_smoothing
                    )
                else:
                    loss = F.cross_entropy(logits, labels)
                
                weight = self.task_weights.get(task_name, 1.0)
                total_loss += loss * weight
                task_losses[task_name] = loss.item()
        
        return total_loss, task_losses

def calculate_class_weights(dataset, target_type='cat'):
    """计算类别权重"""
    class_counts = Counter()
    cat_label = {0: 'payload', 1: 'rocket body', 2: 'debris'}
    rcs_label = {0: 'small', 1: 'medium', 2: 'large'}
    orbit_label = {0: 'LEO', 1: 'MEO', 2: 'HEO'}
    for i in range(len(dataset)):
        sample = dataset[i]
        if target_type == 'cat':
            label = cat_label[int(sample.y_cat)]
        elif target_type == 'rcs':
            label = cat_label[int(sample.y_rcs)]
        else:  # orbit
            label = cat_label[int(sample.y_orbit)]
        class_counts[label] += 1
    print(f"class_counts: {class_counts}")
    
    total_samples = sum(class_counts.values())
    if target_type == 'cat':
        classes = ['payload', 'rocket body', 'debris']
    elif target_type == 'rcs':
        classes = ['small', 'medium', 'large']
    else:  # orbit
        classes = ['LEO', 'MEO', 'HEO']
    
    weights = torch.tensor([
        total_samples / (class_counts[cls] + 1e-8) for cls in classes
    ], dtype=torch.float32)
    
    return weights / weights.sum()

class EarlyStopping:
    def __init__(self, patience=10, delta=0, checkpoint_dir='checkpoints'):
        self.patience = patience
        self.delta = delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.checkpoint_dir = checkpoint_dir
        os.makedirs(checkpoint_dir, exist_ok=True)
        
    def __call__(self, val_loss, model, epoch):
        score = -val_loss
        
        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(model, epoch, val_loss)
        elif score < self.best_score + self.delta:
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(model, epoch, val_loss)
            self.counter = 0
            
    def save_checkpoint(self, model, epoch, val_loss):
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'val_loss': val_loss,
            'best_score': self.best_score
        }
        torch.save(checkpoint, os.path.join(self.checkpoint_dir, 'best_gat_model.pt'))
        print(f'Checkpoint saved at epoch {epoch} with val_loss {val_loss.item():.4f}')

def compute_metrics(outputs, targets, task_names):
    """计算多任务评估指标"""
    metrics = {}
    
    for task_name in task_names:
        if task_name in outputs and task_name in targets:
            preds = outputs[task_name].argmax(dim=1)
            labels = targets[task_name]
            
            # 准确率
            accuracy = (preds == labels).float().mean().item()
            
            # F1分数（宏平均）
            f1_scores = []
            num_classes = outputs[task_name].size(1)
            
            for cls in range(num_classes):
                tp = ((preds == cls) & (labels == cls)).sum().item()
                fp = ((preds == cls) & (labels != cls)).sum().item()
                fn = ((preds != cls) & (labels == cls)).sum().item()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                f1_scores.append(f1)
            
            macro_f1 = np.mean(f1_scores)
            
            metrics[f'{task_name}_accuracy'] = accuracy
            metrics[f'{task_name}_f1'] = macro_f1
    
    return metrics

def train_epoch(model, dataloader, criterions, optimizer, device, scheduler=None):
    """训练一个epoch"""
    model.train()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in ['cat', 'rcs', 'orbit']}
    all_metrics = {f'{task}_{metric}': 0.0 for task in ['cat', 'rcs', 'orbit'] for metric in ['accuracy', 'f1']}
    num_batches = 0
    
    for batch in dataloader:
        # 移动到设备
        batch = batch.to(device)

        # print("输入 x 范围:", batch.x.min().item(), batch.x.max().item())

        # 准备目标
        targets = {
            'cat': batch.y_cat,
            'rcs': batch.y_rcs, 
            'orbit': batch.y_orbit
        }
        
        # 前向传播
        optimizer.zero_grad()
        outputs = model(batch)
        
        # 计算损失
        # loss, batch_task_losses = criterions(outputs, targets)

        # # 反向传播
        # loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        # optimizer.step()

        # if scheduler:
        #     scheduler.step()
        
        # # 统计信息
        # total_loss += loss.item()
        # for task, task_loss in batch_task_losses.items():
        #     task_losses[task] += task_loss

        # 计算损失
        cat_loss = criterions['cat'](outputs['cat'], targets['cat'])
        rcs_loss = criterions['rcs'](outputs['rcs'], targets['rcs'])
        orbit_loss = criterions['orbit'](outputs['orbit'], targets['orbit'])
        
        uncertainty_w = MultiTaskUncertaintyLoss().to(device)
        losses = {'cat': cat_loss, 'rcs': rcs_loss, 'orbit': orbit_loss}
        total_loss_batch = uncertainty_w(losses)   # 或手工加权版本

        # 反向传播
        total_loss_batch.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        if scheduler:
            scheduler.step()

        # 统计信息
        total_loss += total_loss_batch
        task_losses['cat'] += cat_loss
        task_losses['rcs'] += rcs_loss
        task_losses['orbit'] += orbit_loss
        

        # 计算指标
        batch_metrics = compute_metrics(outputs, targets, ['cat', 'rcs', 'orbit'])
        for metric_name, metric_value in batch_metrics.items():
            all_metrics[metric_name] += metric_value
        
        num_batches += 1
    
    # 计算平均值
    avg_loss = total_loss / num_batches
    for task in task_losses:
        task_losses[task] /= num_batches
    for metric_name in all_metrics:
        all_metrics[metric_name] /= num_batches
    
    return avg_loss, task_losses, all_metrics

def validate_epoch(model, dataloader, criterions, device):
    """验证一个epoch"""
    model.eval()
    total_loss = 0.0
    task_losses = {task: 0.0 for task in ['cat', 'rcs', 'orbit']}
    all_metrics = {f'{task}_{metric}': 0.0 for task in ['cat', 'rcs', 'orbit'] for metric in ['accuracy', 'f1']}
    num_batches = 0
    
    with torch.no_grad():
        for batch in dataloader:
            batch = batch.to(device)
            
            targets = {
                'cat': batch.y_cat,
                'rcs': batch.y_rcs,
                'orbit': batch.y_orbit
            }
            
            outputs = model(batch)

            # # 交叉熵损失函数
            # loss, batch_task_losses = criterions(outputs, targets)
            # total_loss += loss.item()
            # for task, task_loss in batch_task_losses.items():
            #     task_losses[task] += task_loss

            # 焦点损失函数
            cat_loss = criterions['cat'](outputs['cat'], targets['cat'])
            rcs_loss = criterions['rcs'](outputs['rcs'], targets['rcs'])
            orbit_loss = criterions['orbit'](outputs['orbit'], targets['orbit'])
            
            uncertainty_w = MultiTaskUncertaintyLoss().to(device)
            losses = {'cat': cat_loss, 'rcs': rcs_loss, 'orbit': orbit_loss}
            total_loss_batch = uncertainty_w(losses)   # 或手工加权版本

            total_loss += total_loss_batch
            task_losses['cat'] += cat_loss
            task_losses['rcs'] += rcs_loss
            task_losses['orbit'] += orbit_loss

            
            batch_metrics = compute_metrics(outputs, targets, ['cat', 'rcs', 'orbit'])
            for metric_name, metric_value in batch_metrics.items():
                all_metrics[metric_name] += metric_value
            
            num_batches += 1
    
        avg_loss = total_loss / num_batches
        for task in task_losses:
            task_losses[task] /= num_batches
        for metric_name in all_metrics:
            all_metrics[metric_name] /= num_batches

    return avg_loss, task_losses, all_metrics

def main():
    parser = argparse.ArgumentParser(description='训练多任务GAT模型')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--hidden_dim', type=int, default=64)
    parser.add_argument('--node_dim', type=int, default=64)
    parser.add_argument('--num_gat_layers', type=int, default=1)
    parser.add_argument('--num_heads', type=int, default=4)
    parser.add_argument('--drop_out', type=float, default=0.3)
    parser.add_argument('--window_size', type=int, default=1440)
    parser.add_argument('--subwindow_size', type=int, default=256)
    parser.add_argument('--stride', type=int, default=128)
    parser.add_argument('--graph_type', type=str, default='temporal', 
                        help="可选构图类型 ['temporal', 'temporal_bidirectional', 'similarity']")
    parser.add_argument('--patience', type=int, default=15)
    parser.add_argument('--gpu', type=int, default=0)
    parser.add_argument('--task_weights', type=str, default="0.6,0.3,0.1", 
                        help="任务权重: cat,rcs,orbit")
    parser.add_argument("--ncf_data", type=str, default="../../../../db/ncf_20220101-20230101.h5")
    parser.add_argument("--train_json", type=str, default='../../../../db/intermediate/split_samples_little_monthly_train.json')
    parser.add_argument("--val_json", type=str, default='../../../../db/intermediate/split_samples_little_monthly_val.json')
    parser.add_argument("--test_json", type=str, default='../../../../db/intermediate/split_samples_little_monthly_test.json')
    parser.add_argument("--save_dir", type=str, default="../../output/improved_gat-monthly_split-trans_spec-sincos-focalloss-little")
    
    args = parser.parse_args()
    
    # 设备设置
    device = torch.device(f'cuda:{args.gpu}' if torch.cuda.is_available() else 'cpu')
    print(f'使用设备: {device}')

    # 创建实验目录
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    ncf_data = args.ncf_data
    train_json = args.train_json
    val_json = args.val_json
    test_json = args.test_json
    exp_dir = args.save_dir
    os.makedirs(exp_dir, exist_ok=True)
    
    # 保存配置
    with open(f'{exp_dir}/config.json', 'w') as f:
        json.dump(vars(args), f, indent=2)

    os.makedirs(f"{exp_dir}/runs", exist_ok=True)
    writer = SummaryWriter(f"{exp_dir}/runs/")
    
    # 数据加载
    print("加载数据集...")
    train_transforms = GraphNoiseTransform(noise_std=0.01)  # [GraphNoiseTransform(noise_std=0.01), EdgeDropTransform(drop_prob=0.1)]
    
    train_ds = SpaceGraphDataset(
        ncf_data,
        train_json,  # 需要确保这个文件存在
        window_size=args.window_size,
        subwindow_size=args.subwindow_size,
        stride=args.stride,
        graph_type=args.graph_type,
        transform=train_transforms,
        use_cache=True
    )
    
    val_ds = SpaceGraphDataset(
        ncf_data,
        val_json,
        window_size=args.window_size,
        subwindow_size=args.subwindow_size,
        stride=args.stride,
        graph_type=args.graph_type,
        use_cache=True
    )
    
    test_ds = SpaceGraphDataset(
        ncf_data,
        test_json, 
        window_size=args.window_size,
        subwindow_size=args.subwindow_size,
        stride=args.stride,
        graph_type=args.graph_type,
        use_cache=True
    )

    train_loader = PyG_DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4
    )
    val_loader = PyG_DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    test_loader = PyG_DataLoader(
        test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4
    )
    
    print(f"训练样本: {len(train_ds)}, 验证样本: {len(val_ds)}, 测试样本: {len(test_ds)}")
    
    # 模型
    # model = ImprovedGATMultiTask(
    #     input_dim=16,  # 19,  # 增加了时间特征
    #     node_dim=args.node_dim,
    #     hidden_dim=args.hidden_dim,
    #     n_cat=3,
    #     n_rcs=3,
    #     n_orbit=3,
    #     n_heads=args.num_heads,
    #     num_gat_layers=args.num_gat_layers,
    #     dropout=args.drop_out
    # ).to(device)

    model = SpectrumEnhancedGAT(
        seq_len=args.subwindow_size, 
        input_dim=16, 
        temporal_dim=args.node_dim, 
        spec_dim=8,
        hidden_dim=args.hidden_dim,
        n_cat=3, 
        n_rcs=3, 
        n_orbit=3, 
        fusion_method='concat', 
        gat_heads=args.num_heads, 
        gat_layers=args.num_gat_layers, 
        dropout=args.drop_out
    ).to(device)

    
    print(f"模型参数量: {sum(p.numel() for p in model.parameters()):,}")

    register_nan_hook(model)
    
    # 损失函数和优化器
    # 解析任务权重
    task_weights = {
        'cat': float(args.task_weights.split(',')[0]),
        'rcs': float(args.task_weights.split(',')[1]),
        'orbit': float(args.task_weights.split(',')[2])
    }

    # criterions = MultiTaskLoss(task_weights=task_weights, label_smoothing=0.1)

    # 计算类别权重
    weights_cat = calculate_class_weights(train_ds, 'cat').to(device)
    weights_rcs = calculate_class_weights(train_ds, 'rcs').to(device)
    weights_orbit = calculate_class_weights(train_ds, 'orbit').to(device)

    # 损失函数和优化器
    criterions = {
            'cat': FocalLoss(alpha=weights_cat, gamma=2.0).to(device),
            'rcs': FocalLoss(alpha=weights_rcs, gamma=2.0).to(device),
            'orbit': FocalLoss(alpha=weights_orbit, gamma=2.0).to(device)
        }

    
    
    optimizer = torch.optim.AdamW(
        model.parameters(), 
        lr=args.lr, 
        weight_decay=0.01,
        betas=(0.9, 0.999)
    )
    
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=10, T_mult=2, eta_min=1e-6
    )
    
    early_stopping = EarlyStopping(
        patience=args.patience, 
        delta=0.001,
        checkpoint_dir=os.path.join(exp_dir, 'checkpoints')
    )
    
    print("开始训练...")
    best_val_loss = float('inf')
    
    for epoch in range(1, args.epochs + 1):
        # 训练
        train_loss, train_task_losses, train_metrics = train_epoch(
            model, train_loader, criterions, optimizer, device, scheduler
        )
        
        # 验证
        val_loss, val_task_losses, val_metrics = validate_epoch(
            model, val_loader, criterions, device
        )
        
        # 学习率调度
        scheduler.step()
        
        # 早停检查
        early_stopping(val_loss, model, epoch)
        
        # 记录到tensorboard
        writer.add_scalar('Loss', {'train': train_loss, 'val': val_loss}, epoch)
        
        for task in ['cat', 'rcs', 'orbit']:
            writer.add_scalars(f'Loss/{task}', {
                'train': train_task_losses[task],
                'val': val_task_losses[task]
            }, epoch)
            
            writer.add_scalars(f'Accuracy/{task}', {
                'train': train_metrics[f'{task}_accuracy'],
                'val': val_metrics[f'{task}_accuracy']
            }, epoch)
            
            writer.add_scalars(f'F1/{task}', {
                'train': train_metrics[f'{task}_f1'],
                'val': val_metrics[f'{task}_f1']
            }, epoch)
        
        
        # 打印进度
        if epoch % 5 == 0 or epoch == 1:
            print(f'Epoch {epoch:03d}:')
            print(f'  Train Loss: {train_loss.item():.4f}, Val Loss: {val_loss.item():.4f}')
            print(f'  Train Acc - cat: {train_metrics["cat_accuracy"]:.3f}, '
                  f'RCS: {train_metrics["rcs_accuracy"]:.3f}, '
                  f'Orbit: {train_metrics["orbit_accuracy"]:.3f}')
            print(f'  Val Acc   - cat: {val_metrics["cat_accuracy"]:.3f}, '
                  f'RCS: {val_metrics["rcs_accuracy"]:.3f}, '
                  f'Orbit: {val_metrics["orbit_accuracy"]:.3f}')
            print(f'  Learning Rate: {optimizer.param_groups[0]["lr"]:.2e}')

        if epoch % 10 == 0:
            print(torch.cuda.memory_summary(device, abbreviated=True))

        if early_stopping.early_stop:
            print("早停触发!")
            break
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
    
    # 加载最佳模型进行测试
    checkpoint = torch.load(os.path.join(exp_dir, 'checkpoints', 'best_gat_model.pt'))
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"加载最佳模型 (epoch {checkpoint['epoch']}, val_loss {checkpoint['val_loss'].item():.4f})")
    
    # 测试
    test_loss, test_task_losses, test_metrics = validate_epoch(
        model, test_loader, criterions, device
    )
    
    print("\n测试结果:")
    print(f'测试损失: {test_loss:.4f}')
    for task in ['cat', 'rcs', 'orbit']:
        print(f'{task:8} - 准确率: {test_metrics[f"{task}_accuracy"]:.3f}, '
              f'F1: {test_metrics[f"{task}_f1"]:.3f}')
    
    # 保存测试结果
    test_results = {
        'test_loss': test_loss,
        'test_task_losses': test_task_losses,
        'test_metrics': test_metrics
    }
    with open(f'{exp_dir}/gat_test_results.json', 'w') as f:
        json.dump(test_results, f, indent=2)
    
    # 保存预测结果
    print("生成预测结果...")
    model.eval()
    all_predictions = {'cat': [], 'rcs': [], 'orbit': []}
    all_targets = {'cat': [], 'rcs': [], 'orbit': []}
    
    with torch.no_grad():
        for batch in test_loader:
            batch = batch.to(device)
            outputs = model(batch)
            
            for task in ['cat', 'rcs', 'orbit']:
                preds = outputs[task].argmax(dim=1).cpu().numpy()
                targets = getattr(batch, f'y_{task}').cpu().numpy()
                all_predictions[task].extend(preds)
                all_targets[task].extend(targets)
    
    # 保存预测
    predictions_file = f'{exp_dir}/gat_predictions.npz'
    np.savez(
        predictions_file,
        **{f'pred_{k}': v for k, v in all_predictions.items()},
        **{f'true_{k}': v for k, v in all_targets.items()}
    )
    print(f'预测结果已保存到: {predictions_file}')
    
    writer.close()
    train_ds.close()
    val_ds.close()
    test_ds.close()

if __name__ == '__main__':
    main()