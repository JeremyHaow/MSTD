"""
数据集加载和预处理
"""

import os
import glob
import random
import torch
from torch.utils.data import Dataset
from PIL import Image

class ImageDataset(Dataset):
    """
    AI生成图像检测数据集
    支持从目录结构中加载数据
    
    数据集结构:
    dataset_path/
        0_real/
            real_image1.jpg
            real_image2.jpg
            ...
        1_fake/
            fake_image1.jpg
            fake_image2.jpg
            ...
    """
    def __init__(self, root, transform=None, is_train=True, balance=True):
        """
        初始化数据集
        
        Args:
            root: 数据根目录
            transform: 图像变换
            is_train: 是否为训练集
            balance: 是否平衡数据集
        """
        self.root = root
        self.transform = transform
        self.is_train = is_train
        
        # 定义真实和生成图像的路径
        real_dir = os.path.join(root, "0_real")
        fake_dir = os.path.join(root, "1_fake")
        
        if not os.path.exists(real_dir) or not os.path.exists(fake_dir):
            raise ValueError(f"目录结构不正确! 需要在{root}下有0_real和1_fake文件夹。")
        
        # 加载真实图像
        self.real_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.real_images.extend(glob.glob(os.path.join(real_dir, '**', ext), recursive=True))
            
        # 加载假图像
        self.fake_images = []
        for ext in ['*.jpg', '*.jpeg', '*.png']:
            self.fake_images.extend(glob.glob(os.path.join(fake_dir, '**', ext), recursive=True))
        
        # 确保数据集均衡
        if balance:
            min_size = min(len(self.real_images), len(self.fake_images))
            if is_train:
                # 训练集使用全部数据
                self.real_images = self.real_images[:min_size]
                self.fake_images = self.fake_images[:min_size]
            else:
                # 验证/测试集使用部分数据，如果数据较少可以全部使用
                val_size = min(min_size//5, min_size)
                self.real_images = self.real_images[:val_size]
                self.fake_images = self.fake_images[:val_size]
        
        # 创建图像路径和标签列表
        self.image_paths = self.real_images + self.fake_images
        self.labels = [0] * len(self.real_images) + [1] * len(self.fake_images)  # 0:真实, 1:AI生成
        
        # 检查数据集大小
        if len(self.image_paths) == 0:
            raise RuntimeError(f"未找到图像文件，请检查路径: {root}")
        
        phase = "训练" if is_train else "验证/测试"
        print(f"{phase}数据集加载完成，共 {len(self.real_images)} 真实图像和 {len(self.fake_images)} AI生成图像")
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        """获取单个数据样本"""
        # 加载图像
        img_path = self.image_paths[idx]
        try:
            img = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"加载图像 {img_path} 时出错: {e}")
            # 如果图像损坏，返回随机图像
            return self.__getitem__(random.randint(0, self.__len__() - 1))
        
        # 应用变换
        if self.transform:
            img = self.transform(img)
        
        # 获取标签
        label = self.labels[idx]
        
        return img, label 