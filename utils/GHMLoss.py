import torch
import torch.nn as nn
import torch.nn.functional as F

class GHM_Loss(nn.Module):
    def __init__(self, bins=10, alpha=0.5, reduction='mean'):
        """
        Gradient Harmonizing Mechanism Loss (GHM-Loss)
        
        Args:
            bins (int): 分桶数量 (10-30)
            alpha (float): 平滑系数 (0.25-0.75)
            reduction (str): 损失聚合方式 ('mean', 'sum', 'none')
        """
        super().__init__()
        self.bins = bins
        self.alpha = alpha
        self.reduction = reduction
        
        # 预注册缓冲区
        self.register_buffer('edges', torch.linspace(0, 1, bins + 1))
        self.register_buffer('zero_tensor', torch.tensor(0.))
        
    def forward(self, pred, target):
        """
        Args:
            pred (torch.Tensor): 模型输出 (logits)
            target (torch.Tensor): 真实标签
                - 二分类: float tensor (0/1) 形状同 pred
                - 多分类: long tensor (类别索引) 形状 [N]
        Returns:
            torch.Tensor: GHM 损失值
        """
        # ==================== 1. 计算梯度模长 g ====================
        if target.dtype == torch.long:  # 多分类
            # 获取正确类别的预测概率
            probs = F.softmax(pred, dim=1)
            class_probs = probs[torch.arange(pred.size(0)), target]
            g = (1 - class_probs).unsqueeze(1)  # [N, 1]
        else:  # 二分类
            probs = torch.sigmoid(pred)
            g = torch.abs(probs - target)  # 形状同 pred
        
        # ==================== 2. 计算梯度密度权重 ====================
        # 展平所有样本的 g 值
        flat_g = g.flatten()
        
        # 计算每个桶的样本数量 (向量化)
        counts = torch.histc(flat_g, bins=self.bins, min=0, max=1)
        
        # 计算梯度密度权重 (避免除零)
        weights = (flat_g.numel() / (counts + 1e-6)) * self.alpha
        
        # ==================== 3. 分配样本权重 ====================
        # 计算每个 g 值所属的桶索引
        bin_idx = torch.floor(flat_g * self.bins).long().clamp(0, self.bins-1)
        
        # 获取每个样本对应的权重
        sample_weights = weights[bin_idx].view_as(g)
        
        # ==================== 4. 计算加权损失 ====================
        if target.dtype == torch.long:  # 多分类
            base_loss = F.cross_entropy(pred, target, reduction='none')
            loss = base_loss * sample_weights.squeeze(1)
        else:  # 二分类
            base_loss = F.binary_cross_entropy_with_logits(
                pred, target, reduction='none'
            )
            loss = base_loss * sample_weights
        
        # ==================== 5. 聚合损失 ====================
        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        return loss
    

if __name__ == '__main__':
    criterion_bin = GHM_Loss(bins=5, alpha=0.5)
    logits_bin = torch.randn(4, 1)
    targets_bin = torch.randint(0, 2, (4, 1)).float()
    loss_bin = criterion_bin(logits_bin, targets_bin)
    print(f"Binary GHM Loss: {loss_bin.item():.4f}")
    
    # 测试多分类
    criterion_multi = GHM_Loss(bins=5, alpha=0.5)
    logits_multi = torch.randn(4, 3)
    targets_multi = torch.tensor([0, 2, 1, 0])
    loss_multi = criterion_multi(logits_multi, targets_multi)
    print(f"Multi-class GHM Loss: {loss_multi.item():.4f}")
    
    # 测试极端情况 (所有样本相同)
    logits_identical = torch.zeros(100, 1)
    targets_identical = torch.ones(100, 1)
    loss_identical = criterion_bin(logits_identical, targets_identical)
    print(f"Identical samples loss: {loss_identical.item():.4f}")
    
    # 测试权重计算
    flat_g = torch.cat([torch.ones(90)*0.1, torch.ones(10)*0.9])
    bin_idx = torch.floor(flat_g * 5).long().clamp(0, 4)
    counts = torch.histc(flat_g, bins=5, min=0, max=1)
    weights = (flat_g.numel() / (counts + 1e-6)) * 0.5
    print(f"Weights for sparse samples: {weights[4].item():.2f} (expected >1.0)")
