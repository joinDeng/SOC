import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F
from collections import Counter

class FocalLoss(torch.nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.gamma = gamma
        self.alpha = alpha          # tensor 形状 [C]
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)    # 概率
        focal = (1 - pt) ** self.gamma * ce_loss
        if self.reduction == 'mean':
            return focal.mean()
        return focal

class MultiTaskUncertaintyLoss(torch.nn.Module):
    def __init__(self, task_names=['cat', 'rcs', 'orbit']):
        super().__init__()
        self.log_vars = torch.nn.ParameterDict({
            name: torch.nn.Parameter(torch.zeros(1)) for name in task_names
        })

    def forward(self, losses):
        """
        losses: dict {'cat': tensor, 'rcs': tensor, 'orbit': tensor}
        return: 加权总损失 + 正则项
        """
        total = 0.0
        for name, loss in losses.items():
            precision = torch.exp(-self.log_vars[name])
            total += precision * loss + self.log_vars[name]
        return total

def calculate_class_weights(dataset, target_type='cat'):
    """计算类别权重"""
    class_counts = Counter()
    for i in range(len(dataset)):
        sample = dataset[i]
        if target_type == 'cat':
            label = sample['grid']['final_cat']
        elif target_type == 'rcs':
            label = sample['grid']['final_rcs']
        else:  # orbit
            label = sample['grid']['orbit_class']
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
        total_samples / class_counts[cls] for cls in classes
    ], dtype=torch.float32)
    
    return weights / weights.sum()
