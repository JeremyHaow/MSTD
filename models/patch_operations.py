import torch
import torch.nn as nn
import torch.nn.functional as F

class PatchOperations(nn.Module):
    """
    实现图像的Smash和Patches操作
    """
    def __init__(self, patch_size=32, stride=16):
        super().__init__()
        self.patch_size = patch_size
        self.stride = stride
        
        self.unfold = nn.Unfold(
            kernel_size=(patch_size, patch_size),
            stride=stride
        )
    
    def smash_and_patch(self, x):
        """
        将输入图像分解成patches
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            patches: [B, N, C*patch_size*patch_size] patch列表
        """
        B, C, H, W = x.shape
        
        # 展开成patches
        patches = self.unfold(x)  # [B, C*patch_size*patch_size, N]
        patches = patches.transpose(-1, -2)  # [B, N, C*patch_size*patch_size]
        
        return patches
    
    def patchify(self, x, shuffle=False):
        """
        实现Patchify操作（与框架图对应）
        Args:
            x: [B, C, H, W] 输入图像
            shuffle: 是否打乱patches
        Returns:
            patches: [B, N, C, patch_size, patch_size] patch列表
        """
        B = x.shape[0]
        patches = self.smash_and_patch(x)
        patches = patches.reshape(B, -1, 3, self.patch_size, self.patch_size)
        
        if shuffle:
            # 打乱patches顺序
            N = patches.shape[1]
            indices = torch.randperm(N)
            patches = patches[:, indices]
            
        return patches
    
    def reconstruct(self, patches, output_size):
        """
        将patches重构回图像
        Args:
            patches: [B, N, C*patch_size*patch_size] patch列表
            output_size: (H, W) 输出图像大小
        Returns:
            reconstructed: [B, C, H, W] 重构后的图像
        """
        B = patches.shape[0]
        patches = patches.transpose(-1, -2)  # [B, C*patch_size*patch_size, N]
        
        fold = nn.Fold(
            output_size=output_size,
            kernel_size=(self.patch_size, self.patch_size),
            stride=self.stride
        )
        
        reconstructed = fold(patches)
        
        return reconstructed
