import torch
import torch.nn as nn
import numpy as np

def generate_dct_matrix(size):
    """生成DCT变换矩阵"""
    m = [[(np.sqrt(1./size) if i == 0 else np.sqrt(2./size)) * 
         np.cos((j + 0.5) * np.pi * i / size) for j in range(size)] 
         for i in range(size)]
    return torch.tensor(m).float()

class DCTTransform(nn.Module):
    """DCT变换基础模块"""
    def __init__(self, patch_size=32):
        super().__init__()
        self.patch_size = patch_size
        
        # 生成DCT变换矩阵
        dct_matrix = generate_dct_matrix(patch_size)
        self.register_buffer('dct_matrix', dct_matrix)
        self.register_buffer('dct_matrix_T', dct_matrix.T)
    
    def dct2d(self, x):
        """2D DCT变换"""
        return self.dct_matrix @ x @ self.dct_matrix_T
    
    def idct2d(self, x):
        """2D IDCT逆变换"""
        return self.dct_matrix_T @ x @ self.dct_matrix
    
    def forward(self, x):
        """
        批量处理DCT变换
        Args:
            x: [..., H, W] 输入tensor
        Returns:
            dct_coeffs: [..., H, W] DCT系数
        """
        orig_shape = x.shape
        x = x.reshape(-1, self.patch_size, self.patch_size)
        dct_coeffs = self.dct2d(x)
        return dct_coeffs.reshape(orig_shape)
    
    def inverse(self, dct_coeffs):
        """
        批量处理IDCT逆变换
        Args:
            dct_coeffs: [..., H, W] DCT系数
        Returns:
            x: [..., H, W] 重构数据
        """
        orig_shape = dct_coeffs.shape
        dct_coeffs = dct_coeffs.reshape(-1, self.patch_size, self.patch_size)
        x = self.idct2d(dct_coeffs)
        return x.reshape(orig_shape)

    def get_frequency_patches(self, dct_coeffs, k_highest=9, k_lowest=9):
        """
        选择最高和最低频率的patches
        Args:
            dct_coeffs: [B, N, C, patch_size, patch_size] DCT系数
            k_highest: int, 选择的最高频率patch数量
            k_lowest: int, 选择的最低频率patch数量
        Returns:
            highest_patches: [B, k, C, patch_size, patch_size] 最高频率patches
            lowest_patches: [B, k, C, patch_size, patch_size] 最低频率patches
        """
        B, N, C, H, W = dct_coeffs.shape
        
        # 计算每个patch的频率能量
        energy = torch.sum(torch.abs(dct_coeffs), dim=[2,3,4])  # [B, N]
        
        # 选择最高/最低频率patches
        _, highest_idx = torch.topk(energy, k=k_highest, dim=1)  # [B, k]
        _, lowest_idx = torch.topk(energy, k=k_lowest, dim=1, largest=False)  # [B, k]
        
        # 收集patches
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, k_highest)  # [B, k]
        highest_patches = dct_coeffs[batch_idx, highest_idx]  # [B, k, C, H, W]
        
        batch_idx = torch.arange(B).unsqueeze(1).expand(-1, k_lowest)  # [B, k]
        lowest_patches = dct_coeffs[batch_idx, lowest_idx]  # [B, k, C, H, W]
        
        return highest_patches, lowest_patches
