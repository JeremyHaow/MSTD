import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchOperations(nn.Module):
    """
    实现图像的Patch操作
    """
    def __init__(self, patch_size=32, stride=16):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
        self.unfold = nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=stride
        )
        
        # 创建可学习的位置编码
        self.use_position_embedding = True
        if self.use_position_embedding:
            self.position_embedding = nn.Parameter(
                torch.randn(1, 100, patch_size * patch_size * 3) * 0.02  # 假设最多100个patches
            )
    
    def patchify(self, x, shuffle=False):
        """
        将图像分解成patches
        Args:
            x: [B, C, H, W] 输入图像
            shuffle: 是否打乱patches
        Returns:
            patches: [B, N, C, patch_size, patch_size] patch列表
        """
        B, C, H, W = x.shape
        
        # 计算padding，确保图像可以被完整地分解成patches
        pad_h = (self.stride - (H - self.patch_size) % self.stride) % self.stride
        pad_w = (self.stride - (W - self.patch_size) % self.stride) % self.stride
        
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, (0, pad_w, 0, pad_h))
            H, W = x.shape[2], x.shape[3]
        
        # 展开成patches
        patches = self.unfold(x)  # [B, C*patch_size*patch_size, N]
        
        # 计算patch数量
        n_h = (H - self.patch_size) // self.stride + 1
        n_w = (W - self.patch_size) // self.stride + 1
        N = n_h * n_w
        
        # 重塑为[B, N, C*patch_size*patch_size]
        patches = patches.transpose(-1, -2)  # [B, N, C*patch_size*patch_size]
        
        # 添加位置编码
        if self.use_position_embedding and N <= self.position_embedding.shape[1]:
            patches = patches + self.position_embedding[:, :N, :]
        
        # 重塑为[B, N, C, patch_size, patch_size]
        patches = patches.reshape(B, N, C, self.patch_size, self.patch_size)
        
        if shuffle:
            # 打乱patches顺序
            indices = torch.randperm(N, device=patches.device)
            patches = patches[:, indices]
            
        return patches
    
    def reconstruct(self, patches, output_size):
        """
        将patches重构回图像
        Args:
            patches: [B, N, C, patch_size, patch_size] patch列表
            output_size: (H, W) 输出图像大小
        Returns:
            reconstructed: [B, C, H, W] 重构后的图像
        """
        B, N, C, _, _ = patches.shape
        
        # 重塑为[B, N, C*patch_size*patch_size]
        patches_flat = patches.reshape(B, N, -1)
        
        # 转置为[B, C*patch_size*patch_size, N]
        patches_flat = patches_flat.transpose(-1, -2)
        
        # 使用fold操作重构
        fold = nn.Fold(
            output_size=output_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.stride
        )
        
        # 处理归一化因子，解决重叠区域
        normalizer = torch.ones_like(patches_flat)
        norm_map = fold(normalizer)
        norm_map = torch.where(norm_map == 0, torch.ones_like(norm_map), norm_map)
        
        reconstructed = fold(patches_flat) / norm_map
        
        return reconstructed
