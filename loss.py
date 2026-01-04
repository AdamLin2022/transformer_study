import torch
import torch.nn as nn
import torch.nn.functional as F

class LabelSmoothingLoss(nn.Module):
    """
    Label Smoothing Loss function.
    Proposed in "Attention Is All You Need" (Vaswani et al., 2017).
    """
    def __init__(self, size, padding_idx, smoothing=0.1):
        super(LabelSmoothingLoss, self).__init__()
        self.criterion = nn.KLDivLoss(reduction='sum')
        self.padding_idx = padding_idx
        self.confidence = 1.0 - smoothing
        self.smoothing = smoothing
        self.size = size
        self.true_dist = None

    def forward(self, x, target):
        """
        x: (batch_size * seq_len, vocab_size) - Logits from the model
        target: (batch_size * seq_len) - Ground truth indices
        """
        assert x.size(1) == self.size
        
        # Create mask for padding positions (don't compute loss for padding)
        mask = (target != self.padding_idx)
        n_valid = mask.sum().item()
        
        if n_valid == 0:
            # If all positions are padding, return 0 loss
            return torch.tensor(0.0, device=x.device, requires_grad=True)
        
        # x are logits, KLDivLoss requires log-probabilities
        x = F.log_softmax(x, dim=-1)
        
        # Initialize true_dist: for each position, distribute smoothing mass
        # over all tokens except padding
        true_dist = torch.zeros_like(x)
        # For non-padding positions, distribute smoothing over (size - 1) tokens
        # (excluding padding token)
        true_dist.fill_(self.smoothing / (self.size - 1))
        # Set padding token probability to 0
        true_dist[:, self.padding_idx] = 0
        
        # For each position, assign confidence to the correct target
        # target.unsqueeze(1) shape: (batch_size * seq_len, 1)
        true_dist.scatter_(1, target.unsqueeze(1), self.confidence)
        
        # Normalize: ensure each row sums to 1 (except padding rows which should be 0)
        # For padding positions, set entire row to 0
        padding_mask = (target == self.padding_idx)
        if padding_mask.any():
            true_dist[padding_mask] = 0.0
        
        # Normalize non-padding rows to sum to 1
        row_sums = true_dist.sum(dim=1, keepdim=True)
        # Avoid division by zero for padding rows
        row_sums = torch.clamp(row_sums, min=1e-8)
        true_dist = true_dist / row_sums
        # Set padding rows back to 0
        if padding_mask.any():
            true_dist[padding_mask] = 0.0
        
        self.true_dist = true_dist
        
        # Calculate KL Divergence
        # KLDivLoss with reduction='sum' will sum over all elements
        # We need to mask out padding positions
        loss = self.criterion(x, true_dist)
        
        # Normalize by number of valid (non-padding) positions
        return loss / n_valid
