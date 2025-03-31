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
    def __init__(self, patch_size=32, num_filters=4):
        super().__init__()
        self.patch_size = patch_size
        self.num_filters = num_filters  # 带通滤波器数量K
        
        # 生成DCT变换矩阵
        dct_matrix = generate_dct_matrix(patch_size)
        self.register_buffer('dct_matrix', dct_matrix)
        self.register_buffer('dct_matrix_T', dct_matrix.T)
        
        # 生成K个带通滤波器
        self.create_bandpass_filters()
    
    def create_bandpass_filters(self):
        """创建K个带通滤波器"""
        filters = []
        N = self.patch_size
        K = self.num_filters
        
        for k in range(K):
            # 创建第k个带通滤波器
            filter_k = torch.zeros((N, N), dtype=torch.float32)
            
            for i in range(N):
                for j in range(N):
                    # 根据公式(3)定义带通滤波器
                    if (2*N/K)*k <= i+j < (2*N/K)*(k+1):
                        filter_k[i, j] = 1.0
            
            filters.append(filter_k)
        
        # 注册为模型缓冲区
        self.register_buffer('bandpass_filters', torch.stack(filters))  # [K, N, N]
    
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
            x: [B, N, C, H, W] 输入patches
        Returns:
            dct_coeffs: [B, N, C, H, W] DCT系数
            patch_grades: [B, N] 每个patch的分数
        """
        B, N_patches, C, H, W = x.shape
        x_reshaped = x.reshape(-1, C, H, W)  # [B*N_patches, C, H, W]
        
        # 对每个通道分别进行DCT变换
        dct_coeffs = []
        for c in range(C):
            channel_data = x_reshaped[:, c]  # [B*N_patches, H, W]
            channel_dct = torch.stack([self.dct2d(patch) for patch in channel_data])
            dct_coeffs.append(channel_dct)
        
        # 重组成原始形状
        dct_coeffs = torch.stack(dct_coeffs, dim=1)  # [B*N_patches, C, H, W]
        dct_coeffs_orig_shape = dct_coeffs.reshape(B, N_patches, C, H, W)
        
        # 计算每个patch的分数
        # 先计算log(|x_dct| + 1)
        log_abs_dct = torch.log(torch.abs(dct_coeffs) + 1)  # [B*N_patches, C, H, W]
        
        # 使用K个带通滤波器计算分数
        patch_grades = torch.zeros((B, N_patches), device=x.device)
        
        for b in range(B):
            for n in range(N_patches):
                patch_score = 0
                for k in range(self.num_filters):
                    # 根据公式(4): 2^k * 滤波结果
                    for c in range(C):
                        # 应用滤波器
                        filtered = log_abs_dct[b*N_patches + n, c] * self.bandpass_filters[k]
                        patch_score += (2**k) * filtered.sum()
                
                patch_grades[b, n] = patch_score
        
        return dct_coeffs_orig_shape, patch_grades
    
    def inverse(self, dct_coeffs):
        """
        批量处理IDCT逆变换
        Args:
            dct_coeffs: [B, N, C, H, W] DCT系数
        Returns:
            x: [B, N, C, H, W] 重构数据
        """
        B, N, C, H, W = dct_coeffs.shape
        coeffs_reshaped = dct_coeffs.reshape(-1, C, H, W)  # [B*N, C, H, W]
        
        # 对每个通道分别进行IDCT逆变换
        reconstructed = []
        for c in range(C):
            channel_coeffs = coeffs_reshaped[:, c]  # [B*N, H, W]
            channel_idct = torch.stack([self.idct2d(coeff) for coeff in channel_coeffs])
            reconstructed.append(channel_idct)
        
        # 重组成原始形状
        reconstructed = torch.stack(reconstructed, dim=1)  # [B*N, C, H, W]
        reconstructed = reconstructed.reshape(B, N, C, H, W)
        
        return reconstructed

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
