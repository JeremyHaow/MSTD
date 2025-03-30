# models/mstd.py
# MSTD主模型定义

import torch
import torch.nn as nn
import torch.nn.functional as F

from .patch_operations import PatchOperations
from .dct_transform import DCTTransform
from .frequency_analysis import FrequencyAnalysis, SRMConv

class ResNetBlock(nn.Module):
    """ResNet基础块"""
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, 3, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, 1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
    
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = F.relu(out)
        return out

class SimpleResNet50(nn.Module):
    """简化版ResNet50"""
    def __init__(self, in_channels=3, num_classes=256):
        super().__init__()
        self.conv1 = nn.Conv2d(in_channels, 64, 7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.maxpool = nn.MaxPool2d(3, stride=2, padding=1)
        
        # 简化的层结构
        self.layer1 = self._make_layer(64, 64, 2)
        self.layer2 = self._make_layer(64, 128, 2, stride=2)
        self.layer3 = self._make_layer(128, 256, 2, stride=2)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(256, num_classes)
    
    def _make_layer(self, in_channels, out_channels, num_blocks, stride=1):
        layers = [ResNetBlock(in_channels, out_channels, stride)]
        for _ in range(1, num_blocks):
            layers.append(ResNetBlock(out_channels, out_channels))
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

class FrequencyBranch(nn.Module):
    """频率特征提取分支"""
    def __init__(self, in_channels=3, feature_dim=256, use_fire=True):
        super().__init__()
        # 使用srm_filter_kernel.py中的滤波器
        self.srm = SRMConv(in_channels, use_multiple_filters=False)
        
        # 根据SRM输出通道数调整输入通道数
        resnet_in_channels = self.srm.out_channels
        
        self.resnet = SimpleResNet50(resnet_in_channels, feature_dim)
        self.use_fire = use_fire  # 是否使用"火焰"标记（图中红色标记）
    
    def forward(self, x):
        x = self.srm(x)
        features = self.resnet(x)
        return features

class CLIPVisualEncoder(nn.Module):
    """CLIP视觉编码器"""
    def __init__(self, input_dim, hidden_dim=512, output_dim=256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, output_dim)
        )
    
    def forward(self, x):
        return self.encoder(x)

class MultiLayerPerceptron(nn.Module):
    """输出头MLP"""
    def __init__(self, input_dim, hidden_dim=128, output_dim=1):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, output_dim),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        return self.mlp(x)

class MSTD(nn.Module):
    """
    Patches is All You Need for AI-generate Images Detection (PAID)
    基于论文中展示的框架图实现
    """
    def __init__(self, 
                 patch_size=32,
                 stride=16,
                 num_high_freq_patches=9,
                 num_low_freq_patches=9,
                 feature_dim=256):
        super().__init__()
        
        # 基础组件
        self.patch_ops = PatchOperations(patch_size, stride)
        self.dct = DCTTransform(patch_size)
        self.freq_analysis = FrequencyAnalysis()
        
        # 频率分支
        self.high_freq_branch = FrequencyBranch(3, feature_dim, use_fire=True)
        self.low_freq_branch = FrequencyBranch(3, feature_dim, use_fire=False)
        
        # CLIP视觉编码器
        self.clip_encoder = CLIPVisualEncoder(patch_size * patch_size * 3, 512, feature_dim)
        
        # 模型参数
        self.patch_size = patch_size
        self.num_high_freq_patches = num_high_freq_patches
        self.num_low_freq_patches = num_low_freq_patches
        self.feature_dim = feature_dim
        
        # MLP输出头
        self.mlp_head = MultiLayerPerceptron(feature_dim * 3, 128, 1)
    
    def extract_patches(self, x):
        """图像分块"""
        B = x.shape[0]
        patches = self.patch_ops.smash_and_patch(x)
        patches = patches.reshape(B, -1, 3, self.patch_size, self.patch_size)
        return patches
    
    def process_frequency_branches(self, patches):
        """处理频率分支"""
        B = patches.shape[0]
        
        # DCT变换
        dct_coeffs = self.dct(patches)
        
        # 获取高频和低频patches
        high_freq_patches, low_freq_patches = self.freq_analysis.get_frequency_patches(
            dct_coeffs, 
            k_highest=self.num_high_freq_patches, 
            k_lowest=self.num_low_freq_patches
        )
        
        # 逆变换还原图像内容
        high_freq_patches = self.dct.inverse(high_freq_patches)
        low_freq_patches = self.dct.inverse(low_freq_patches)
        
        # 重塑为图像批次进行处理
        high_freq_patches = high_freq_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        low_freq_patches = low_freq_patches.reshape(-1, 3, self.patch_size, self.patch_size)
        
        # 通过SRM+ResNet50处理
        high_freq_features = self.high_freq_branch(high_freq_patches)
        low_freq_features = self.low_freq_branch(low_freq_patches)
        
        # 重塑回批次并平均
        high_freq_features = high_freq_features.reshape(B, -1, self.feature_dim).mean(dim=1)
        low_freq_features = low_freq_features.reshape(B, -1, self.feature_dim).mean(dim=1)
        
        return high_freq_features, low_freq_features
    
    def process_clip_branch(self, patches):
        """处理CLIP分支"""
        B = patches.shape[0]
        N = patches.shape[1]
        
        # 随机打乱patches
        indices = torch.randperm(N)
        shuffled_patches = patches[:, indices]
        
        # 展平patch并通过CLIP编码器
        flat_patches = shuffled_patches.reshape(B, N, -1)
        
        # 对每个patch使用CLIP编码器
        clip_features = []
        for i in range(min(N, 10)):  # 限制处理的patch数量
            patch_feature = self.clip_encoder(flat_patches[:, i])
            clip_features.append(patch_feature)
        
        # 融合特征
        clip_features = torch.stack(clip_features, dim=1)
        clip_features = clip_features.mean(dim=1)
        
        return clip_features
    
    def forward(self, x):
        """
        前向传播
        Args:
            x: [B, C, H, W] 输入图像
        Returns:
            score: [B] 真实性得分(0-1)，0表示AI生成，1表示真实图像
        """
        # 1. 图像分块 (Smash & Patches)
        patches = self.extract_patches(x)
        
        # 2. 处理频率分支 (高频和低频)
        high_freq_features, low_freq_features = self.process_frequency_branches(patches)
        
        # 3. 处理CLIP分支
        clip_features = self.process_clip_branch(patches)
        
        # 4. 连接特征 (Channel Concatenate)
        combined_features = torch.cat([
            high_freq_features,
            low_freq_features,
            clip_features
        ], dim=1)
        
        # 5. MLP头输出分数
        score = self.mlp_head(combined_features).squeeze(-1)
        
        return score