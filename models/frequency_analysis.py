import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .srm_filter_kernel import (
    square_3x3, 
    square_5x5, 
    all_normalized_hpf_list,
    normalized_hpf_3x3_list,
    normalized_hpf_5x5_list
)

class FrequencyAnalysis(nn.Module):
    """频率分析模块"""
    def __init__(self):
        super().__init__()
    
    def get_frequency_patches(self, dct_coeffs, patch_grades=None, k_highest=9, k_lowest=9):
        """
        根据DCT分数选择最高和最低频率的patches
        Args:
            dct_coeffs: [B, N, C, H, W] DCT系数
            patch_grades: [B, N] 每个patch的分数，如果为None则计算频率能量
            k_highest: int, 选择的最高频率patch数量
            k_lowest: int, 选择的最低频率patch数量
        Returns:
            highest_patches: [B, k, C, H, W] 最高频率patches
            lowest_patches: [B, k, C, H, W] 最低频率patches
        """
        B = dct_coeffs.shape[0]
        N = dct_coeffs.shape[1]
        
        # 如果没有提供patch_grades，则基于DCT系数计算频率能量
        if patch_grades is None:
            # 计算频率能量
            patch_grades = torch.sum(torch.abs(dct_coeffs), dim=[2, 3, 4])  # [B, N]
        
        # 选择分数最高和最低的patches
        _, highest_idx = torch.topk(patch_grades, k=min(k_highest, N), dim=1)  # [B, k]
        _, lowest_idx = torch.topk(patch_grades, k=min(k_lowest, N), dim=1, largest=False)  # [B, k]
        
        # 收集patches
        batch_idx = torch.arange(B, device=dct_coeffs.device).unsqueeze(1)
        highest_patches = dct_coeffs[batch_idx.expand(-1, k_highest), highest_idx]
        lowest_patches = dct_coeffs[batch_idx.expand(-1, k_lowest), lowest_idx]
        
        return highest_patches, lowest_patches

class SRMConv(nn.Module):
    """使用srm_filter_kernel.py中的噪声残差模型进行图像预处理"""
    def __init__(self, in_channels=3, use_multiple_filters=True):
        super().__init__()
        
        if use_multiple_filters:
            # 分别处理3x3和5x5滤波器
            kernels_3x3 = []
            kernels_5x5 = []
            
            # 处理3x3滤波器
            for hpf in normalized_hpf_3x3_list:
                kernel = torch.from_numpy(hpf).float().view(1, 1, 3, 3)
                kernel = kernel.repeat(1, in_channels, 1, 1)
                kernels_3x3.append(kernel)
            
            # 处理5x5滤波器
            for hpf in normalized_hpf_5x5_list:
                kernel = torch.from_numpy(hpf).float().view(1, 1, 5, 5)
                kernel = kernel.repeat(1, in_channels, 1, 1)
                kernels_5x5.append(kernel)
            
            # 创建两个卷积层，一个用于3x3滤波器，一个用于5x5滤波器
            if kernels_3x3:
                self.register_buffer('kernel_3x3', torch.cat(kernels_3x3, dim=0))
                self.has_3x3 = True
            else:
                self.has_3x3 = False
                
            if kernels_5x5:
                self.register_buffer('kernel_5x5', torch.cat(kernels_5x5, dim=0))
                self.has_5x5 = True
            else:
                self.has_5x5 = False
            
            # 计算输出通道数 - 每个滤波器产生1个输出通道
            self.out_channels = 0
            if self.has_3x3:
                self.out_channels += len(kernels_3x3)  # 每个滤波器产生1个输出通道
            if self.has_5x5:
                self.out_channels += len(kernels_5x5)  # 每个滤波器产生1个输出通道
                
            if self.out_channels == 0:
                # 如果没有滤波器，使用默认square_3x3
                kernel = torch.from_numpy(square_3x3).float().view(1, 1, 3, 3)
                self.register_buffer('kernel', kernel.repeat(in_channels, 1, 1, 1))
                self.groups = in_channels
                self.out_channels = in_channels
                self.use_single = True
            else:
                self.groups = 1  # 使用标准卷积
                self.use_single = False
        else:
            # 使用单个经典滤波器（square_3x3）
            kernel = torch.from_numpy(square_3x3).float().view(1, 1, 3, 3)
            self.register_buffer('kernel', kernel.repeat(in_channels, 1, 1, 1))
            self.groups = in_channels
            self.out_channels = in_channels
            self.use_single = True
        
        # 添加批归一化层
        self.bn = nn.BatchNorm2d(self.out_channels)
        
        # 添加可学习的缩放因子
        self.scale = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        """应用SRM滤波"""
        if hasattr(self, 'use_single') and self.use_single:
            # 使用单个滤波器
            srm_features = F.conv2d(x, self.kernel, padding=1, groups=self.groups)
        else:
            # 使用多个滤波器
            features = []
            
            if hasattr(self, 'has_3x3') and self.has_3x3:
                # 应用3x3滤波器 - 每个滤波器应用于所有输入通道
                features_3x3 = F.conv2d(x, self.kernel_3x3, padding=1, groups=1)
                features.append(features_3x3)
            
            if hasattr(self, 'has_5x5') and self.has_5x5:
                # 应用5x5滤波器 - 每个滤波器应用于所有输入通道
                features_5x5 = F.conv2d(x, self.kernel_5x5, padding=2, groups=1)
                features.append(features_5x5)
            
            # 拼接所有特征
            srm_features = torch.cat(features, dim=1)
        
        # 应用批归一化和激活函数
        srm_features = self.bn(srm_features)
        srm_features = F.relu(srm_features)
        
        # 应用缩放因子
        srm_features = srm_features * self.scale
        
        return srm_features 