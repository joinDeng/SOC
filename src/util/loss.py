import torch
from torch.nn import CrossEntropyLoss
import torch.nn.functional as F

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

        