import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .srm_filter_kernel import square_3x3, all_normalized_hpf_list

class FrequencyAnalysis(nn.Module):
    """频率分析模块"""
    def __init__(self):
        super().__init__()
    
    def get_frequency_patches(self, dct_coeffs, k_highest=9, k_lowest=9):
        """
        选择最高和最低频率的patches
        Args:
            dct_coeffs: [B, N, ...] DCT系数
            k_highest: int, 选择的最高频率patch数量
            k_lowest: int, 选择的最低频率patch数量
        Returns:
            highest_patches: [B, k, ...] 最高频率patches
            lowest_patches: [B, k, ...] 最低频率patches
        """
        B = dct_coeffs.shape[0]
        N = dct_coeffs.shape[1]
        
        # 计算频率能量
        energy = torch.sum(torch.abs(dct_coeffs.flatten(2)), dim=-1)  # [B, N]
        
        # 选择最高/最低频率patches
        _, highest_idx = torch.topk(energy, k=k_highest, dim=1)  # [B, k]
        _, lowest_idx = torch.topk(energy, k=k_lowest, dim=1, largest=False)  # [B, k]
        
        # 收集patches
        batch_idx = torch.arange(B, device=dct_coeffs.device).unsqueeze(1)
        highest_patches = dct_coeffs[batch_idx.expand(-1, k_highest), highest_idx]
        lowest_patches = dct_coeffs[batch_idx.expand(-1, k_lowest), lowest_idx]
        
        return highest_patches, lowest_patches

class SRMConv(nn.Module):
    """使用srm_filter_kernel.py中的滤波器实现SRM卷积层"""
    def __init__(self, in_channels=3, use_multiple_filters=False):
        super().__init__()
        
        if use_multiple_filters:
            # 使用多个滤波器
            kernels = []
            for hpf in all_normalized_hpf_list[:5]:  # 使用前5个滤波器作为示例
                if len(hpf.shape) == 2:
                    h, w = hpf.shape
                    kernel = torch.from_numpy(hpf).float().view(1, 1, h, w)
                    kernel = kernel.repeat(in_channels, 1, 1, 1)
                    kernels.append(kernel)
            
            self.register_buffer('kernel', torch.cat(kernels, dim=0))
            self.groups = 1
            self.out_channels = len(kernels) * in_channels
        else:
            # 使用单个经典滤波器（square_3x3）
            kernel = torch.from_numpy(square_3x3).float().view(1, 1, 3, 3)
            self.register_buffer('kernel', kernel.repeat(in_channels, 1, 1, 1))
            self.groups = in_channels
            self.out_channels = in_channels
    
    def forward(self, x):
        """应用SRM滤波"""
        return F.conv2d(x, self.kernel, padding=1, groups=self.groups) 