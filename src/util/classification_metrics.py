import numpy as np

def compute_detailed_metrics(outputs: Dict[str, torch.Tensor], 
                           targets: Dict[str, torch.Tensor],
                           task_names: List[str]) -> Dict[str, Dict]:
    """计算详细的评估指标"""
    metrics = {}
    
    for task_name in task_names:
        if task_name in outputs and task_name in targets:
            preds = outputs[task_name].argmax(dim=1).cpu().numpy()
            labels = targets[task_name].cpu().numpy()
            num_classes = outputs[task_name].size(1)
            
            # 基础指标
            accuracy = (preds == labels).mean()
            
            # 每个类别的精确率、召回率、F1
            class_metrics = {}
            for cls in range(num_classes):
                tp = ((preds == cls) & (labels == cls)).sum()
                fp = ((preds == cls) & (labels != cls)).sum()
                fn = ((preds != cls) & (labels == cls)).sum()
                
                precision = tp / (tp + fp + 1e-8)
                recall = tp / (tp + fn + 1e-8)
                f1 = 2 * precision * recall / (precision + recall + 1e-8)
                
                class_metrics[cls] = {
                    'precision': precision,
                    'recall': recall, 
                    'f1': f1,
                    'support': (labels == cls).sum()
                }
            
            # 宏平均和微平均
            macro_precision = np.mean([m['precision'] for m in class_metrics.values()])
            macro_recall = np.mean([m['recall'] for m in class_metrics.values()])
            macro_f1 = np.mean([m['f1'] for m in class_metrics.values()])
            
            metrics[task_name] = {
                'accuracy': accuracy,
                'macro_precision': macro_precision,
                'macro_recall': macro_recall,
                'macro_f1': macro_f1,
                'per_class': class_metrics
            }
    
    return metrics