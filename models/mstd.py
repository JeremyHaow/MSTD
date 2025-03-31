# models/mstd.py
# MSTD主模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import torchvision.models as models
from transformers import CLIPVisionModel, CLIPConfig
import numpy as np

from .patch_operations import PatchOperations
from .dct_transform import DCTTransform
from .frequency_analysis import FrequencyAnalysis, SRMConv

class ResNet50(nn.Module):
    """完整的ResNet50网络"""
    def __init__(self, in_channels=3, feature_dim=256):
        super().__init__()
        # 加载预训练的ResNet50
        self.resnet = models.resnet50(pretrained=True)
        
        # 修改第一层以适应不同的输入通道数
        if in_channels != 3:
            self.resnet.conv1 = nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False)
            
        # 替换最后的全连接层
        self.resnet.fc = nn.Linear(2048, feature_dim)
    
        # 添加一个直接的fc属性作为resnet.fc的引用
        self.fc = self.resnet.fc

    def forward(self, x):
        return self.resnet(x)

class FrequencyBranch(nn.Module):
    """频率特征提取分支"""
    def __init__(self, in_channels=3, feature_dim=256, use_fire=True):
        super().__init__()
        # 使用SRM滤波器
        self.srm = SRMConv(in_channels, use_multiple_filters=True)
        
        # 根据SRM输出通道数调整输入通道数
        resnet_in_channels = self.srm.out_channels
        
        # 使用完整的ResNet50
        self.resnet = ResNet50(resnet_in_channels, feature_dim)
        self.use_fire = use_fire  # 用于区分高低频路径
    
    def forward(self, x):
        x = self.srm(x)
        features = self.resnet(x)
        return features

class CLIPVisionEncoder(nn.Module):
    """真实的CLIP ViT-L/14视觉编码器"""
    def __init__(self, output_dim=256):
        super().__init__()
        # 加载CLIP ViT-L/14模型
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14")
        
        # 添加映射层，将CLIP的特征维度映射到所需的特征维度
        self.mapper = nn.Linear(1024, output_dim)  # CLIP ViT-L/14的输出维度是1024
        
        # 冻结CLIP模型参数
        for param in self.model.parameters():
            param.requires_grad = False
    
    def forward(self, x):
        # 获取CLIP视觉特征
        with torch.no_grad():
            clip_features = self.model(x).pooler_output  # [B, 1024]
        
        # 映射到所需的特征维度
        mapped_features = self.mapper(clip_features)
        return mapped_features

class FCLayer(nn.Module):
    """全连接层用于CLIP分支"""
    def __init__(self, input_dim, output_dim=256):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(input_dim, input_dim * 2),
            nn.LayerNorm(input_dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.Dropout(0.1)
        )
    
    def forward(self, x):
        return self.fc(x)

class MultiLayerPerceptron(nn.Module):
    """输出头MLP"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.LayerNorm(hidden_dim // 2),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim // 2, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x)

class MSTD(nn.Module):
    """
    Multi-Scale Semantic-Texture Detector (MSTD)
    具体实现双分支AI图像检测网络
    """
    def __init__(self, 
                 patch_size=32,
                 stride=16,
                 patch_grid_size=3,  # 重组的图片大小 (3x3, 4x4等)
                 num_views=10,       # CLIP分支的视图数量
                 feature_dim=256):
        super().__init__()
        
        # 基础组件
        self.patch_ops = PatchOperations(patch_size, stride)
        self.dct = DCTTransform(patch_size)
        self.freq_analysis = FrequencyAnalysis()
        
        # 频率分支
        self.high_freq_branch = FrequencyBranch(3, feature_dim, use_fire=True)  # 复杂纹理
        self.low_freq_branch = FrequencyBranch(3, feature_dim, use_fire=False)  # 简单纹理
        
        # CLIP视觉编码器分支
        self.clip_encoder = CLIPVisionEncoder(output_dim=feature_dim)
        self.fc_layer = FCLayer(feature_dim, feature_dim)
        
        # 模型参数
        self.patch_size = patch_size
        self.patch_grid_size = patch_grid_size
        self.num_views = num_views
        self.feature_dim = feature_dim
        
        # MLP输出头
        self.mlp_head = MultiLayerPerceptron(feature_dim * 3, 512, 1)
    
    def extract_patches(self, x):
        """图像分块"""
        B = x.shape[0]
        patches = self.patch_ops.patchify(x)
        return patches
    
    def reconstruct_to_grid_image(self, patches):
        """将patches重组为网格图像"""
        B, num_patches, C, H, W = patches.shape
        
        # 重组为grid_size x grid_size的正方形图像
        grid_size = int(np.sqrt(num_patches))
        patches = patches[:, :grid_size*grid_size]  # 确保patches数量是完全平方数
        
        grid_patches = patches.reshape(B, grid_size, grid_size, C, H, W)
        grid_images = grid_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        grid_images = grid_images.reshape(B, C, grid_size * H, grid_size * W)
        
        return grid_images
    
    def process_frequency_branch(self, x):
        """处理频率分支"""
        # 1. 提取patches
        patches = self.extract_patches(x)
        
        # 2. DCT变换和评分
        dct_coeffs, patch_grades = self.dct(patches)
        
        # 3. 根据评分选择高频/低频patches
        high_freq_patches, low_freq_patches = self.freq_analysis.get_frequency_patches(
            dct_coeffs, 
            patch_grades,
            k_highest=self.patch_grid_size * self.patch_grid_size, 
            k_lowest=self.patch_grid_size * self.patch_grid_size
        )
        
        # 4. 逆变换还原图像内容
        high_freq_patches = self.dct.inverse(high_freq_patches)
        low_freq_patches = self.dct.inverse(low_freq_patches)
        
        # 5. 重组为正方形图像
        highest_images = self.reconstruct_to_grid_image(high_freq_patches)
        lowest_images = self.reconstruct_to_grid_image(low_freq_patches)
        
        # 6. 调整大小到256x256像素（按照论文要求）
        highest_images = F.interpolate(highest_images, size=(256, 256), mode='bilinear', align_corners=False)
        lowest_images = F.interpolate(lowest_images, size=(256, 256), mode='bilinear', align_corners=False)
        
        # 7. 通过SRM+ResNet处理
        high_freq_features = self.high_freq_branch(highest_images)
        low_freq_features = self.low_freq_branch(lowest_images)
        
        return high_freq_features, low_freq_features
    
    def create_random_views(self, patches, num_views=10):
        """创建随机重组的视图"""
        B, N, C, H, W = patches.shape
        views = []
        
        for i in range(num_views):
            # 随机打乱patches
            indices = torch.randperm(N, device=patches.device)[:self.patch_grid_size * self.patch_grid_size]
            selected_patches = patches[:, indices]
            
            # 重组为单一图像
            view = selected_patches.reshape(B, self.patch_grid_size, self.patch_grid_size, C, H, W)
            view = view.permute(0, 3, 1, 4, 2, 5).contiguous()
            view = view.reshape(B, C, self.patch_grid_size * H, self.patch_grid_size * W)
            
            # 调整大小以适应CLIP ViT-L/14的输入要求 (224x224)
            view = F.interpolate(view, size=(224, 224), mode='bilinear', align_corners=False)
            
            views.append(view)
        
        return views
    
    def process_clip_branch(self, x):
        """处理CLIP分支"""
        # 1. 提取patches
        patches = self.extract_patches(x)
        
        # 2. 创建随机视图
        views = self.create_random_views(patches, self.num_views)
        
        # 3. 通过CLIP编码器处理每个视图
        clip_features = []
        for view in views:
            features = self.clip_encoder(view)
            clip_features.append(features)
        
        # 4. 求和
        clip_sum = torch.stack(clip_features).sum(dim=0)
        
        # 5. 通过FC Layer
        clip_output = self.fc_layer(clip_sum)
        
        return clip_output
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            score: [B] 真实性得分(0-1)，0表示AI生成，1表示真实图像
        """
        # 1. 处理频率分支
        high_freq_features, low_freq_features = self.process_frequency_branch(x)
        
        # 2. 处理CLIP分支
        clip_features = self.process_clip_branch(x)
        
        # 3. 特征融合 (Channel Concatenate)
        combined_features = torch.cat([
            high_freq_features,
            low_freq_features,
            clip_features
        ], dim=1)
        
        # 4. MLP头输出分数
        score = self.mlp_head(combined_features).squeeze(-1)
        
        return score