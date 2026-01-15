import torch
import torch.nn as nn

class DiceLoss(nn.Module):
    def __init__(self, smooth=1e-5):
        super().__init__()
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum()
        union = probs.sum() + targets.sum()
        dice = (2. * intersection + self.smooth) / (union + self.smooth)
        return 1 - dice
    

class TverskyLoss(nn.Module):
    def __init__(self, alpha=0.7, beta=0.3, smooth=1e-5):
        """
        alpha: 假阴性惩罚权重（召回率）
        beta: 假阳性惩罚权重（精确率）
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.smooth = smooth
        
    def forward(self, logits, targets):
        probs = torch.sigmoid(logits)
        tp = (probs * targets).sum()
        fp = ((1 - targets) * probs).sum()
        fn = (targets * (1 - probs)).sum()
        score = (tp + self.smooth) / (tp + self.alpha * fn + self.beta * fp + self.smooth)
        return 1 - score