## MSTD
### 模块

2025年3月28日 第一次提交

整合三篇论文的核心思想，并提出一个创新的AI生成图像检测方法。这些论文分别是：

1. SFLD (主论文)：使用PatchShuffle整合高级语义和低级纹理信息，减少内容偏见
2. Sanity Check：提出了Chameleon数据集，揭示了现有检测器在高度逼真的AI生成图像上的失败，并提出了AIDE方法
3. PatchCraft：利用丰富和贫乏纹理区域之间的纹理对比

提出一种新的AI生成图像检测方法——多尺度语义-纹理融合检测器（Multi-Scale Semantic-Texture Detector，简称MSTD）。该方法整合了三篇论文的优势，并添加了创新性元素，以提高检测器对高度逼真AI生成图像的鲁棒性和泛化能力。

MSTD主要创新点在于：

1. **自适应多尺度特征提取**：不同于SFLD固定的patch大小，MSTD使用自适应策略动态选择最优patch尺寸
2. **层次化特征融合**：从低级频率特征到高级语义特征建立层次化特征表示
3. **对抗性特征增强**：引入对抗训练模块增强模型的鲁棒性
4. **自适应注意力机制**：根据图像内容动态调整不同特征的权重

### 创新点

1. **自适应patch选择**：动态确定最优patch大小，而不是使用固定大小。
2. **层次化特征融合**：建立从低级频率特征到高级语义特征的层次化表示。
3. **对抗性特征增强**：通过对抗训练提高模型的鲁棒性。
4. **自适应注意力机制**：根据图像内容动态调整不同特征的权重。
5. **频率域一致性分析**：添加空间域和频率域之间的一致性检查。

### 项目结构

```
mstd/
├── config.py             # 配置参数
├── main.py               # 主入口文件
├── train.py              # 训练脚本
├── evaluate.py           # 评估脚本
├── models/
│   ├── __init__.py
│   ├── mstd.py           # MSTD主模型定义
│   ├── patch_module.py   # Patch操作相关模块
│   ├── texture_module.py # 纹理分析模块
│   └── fusion_module.py  # 特征融合模块
├── utils/
│   ├── __init__.py
│   ├── losses.py         # 损失函数定义
│   ├── metrics.py        # 评估指标
│   └── adversarial.py    # 对抗训练工具
└── datasets/
    ├── __init__.py
    ├── dataloader.py     # 数据加载器
    └── augmentation.py   # 数据增强方法
```

### 架构图v1.0



![alt text](data/module.png)