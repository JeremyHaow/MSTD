"""
计算模型性能指标的工具函数
"""

import numpy as np
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

def calculate_metrics(predictions, targets, threshold=0.5):
    """
    计算模型性能指标
    
    Args:
        predictions: 模型预测的概率值
        targets: 真实标签 (1=虚假图像, 0=真实图像)
        threshold: 二分类阈值
    
    Returns:
        包含各种性能指标的字典
    """
    # 应用阈值
    binary_preds = (predictions > threshold).astype(int)
    
    # 计算基本指标
    accuracy = accuracy_score(targets, binary_preds)
    
    # 处理可能的边缘情况
    try:
        auc = roc_auc_score(targets, predictions)
    except:
        auc = 0.5  # 当只有一个类时的默认AUC值
    
    try:
        precision = precision_score(targets, binary_preds)
    except:
        precision = 0.0
    
    try:
        recall = recall_score(targets, binary_preds)
    except:
        recall = 0.0
    
    try:
        f1 = f1_score(targets, binary_preds)
    except:
        f1 = 0.0
    
    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(targets, binary_preds, labels=[0, 1]).ravel()
    
    # 计算虚假图像召回率和准确率
    fake_recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0  # 虚假图像识别率
    fake_precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0  # 虚假图像预测准确率
    
    # 返回所有指标
    return {
        'accuracy': accuracy,
        'auc': auc,
        'precision': precision,
        'recall': recall,
        'f1': f1,
        'fake_recall': fake_recall,  # 虚假图像识别率
        'fake_precision': fake_precision,  # 虚假图像预测准确率
        'true_negative': tn,  # 真实图像正确分类数
        'false_positive': fp,  # 真实图像错误分类为虚假数
        'false_negative': fn,  # 虚假图像错误分类为真实数
        'true_positive': tp    # 虚假图像正确分类数
    } 