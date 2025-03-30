"""
日志记录和检查点保存工具
"""

import os
import logging
import torch

def setup_logger(log_dir, log_name='main.log'):
    """
    设置日志记录器
    
    Args:
        log_dir: 日志目录
        log_name: 日志文件名
    
    Returns:
        配置好的logger对象
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    log_file = os.path.join(log_dir, log_name)
    
    # 创建logger
    logger = logging.getLogger('PAID')
    logger.setLevel(logging.INFO)
    
    # 创建文件处理器
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    
    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    
    # 创建格式器
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)
    
    # 添加处理器
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger

def save_checkpoint(state, is_best, save_dir, filename='checkpoint.pth'):
    """
    保存训练检查点
    
    Args:
        state: 包含模型状态等信息的字典
        is_best: 是否是当前最佳模型
        save_dir: 保存目录
        filename: 保存文件名
    """
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # 只保存模型权重
    torch.save(state['model'], os.path.join(save_dir, filename))
    
    if is_best:
        # 保存最佳模型权重
        torch.save(state['model'], os.path.join(save_dir, 'best_model.pth'))
        
    # 可选：单独保存重要的训练状态到文本文件，方便查看
    with open(os.path.join(save_dir, 'training_info.txt'), 'w') as f:
        f.write(f"Epoch: {state['epoch']}\n")
        f.write(f"Best Metric: {state['best_metric']}\n")
        f.write(f"Learning Rate: {state['optimizer']['param_groups'][0]['lr']}\n")

def log_metrics(logger, phase, epoch, metrics):
    """
    记录性能指标
    
    Args:
        logger: 日志记录器
        phase: 阶段名称（Train/Validation）
        epoch: 当前epoch
        metrics: 指标字典
    """
    logger.info(
        f"{phase} Epoch {epoch} - "
        f"Accuracy: {metrics['accuracy']:.4f}, "
        f"AUC: {metrics['auc']:.4f}, "
        f"Fake-Recall: {metrics['fake_recall']:.4f}, "  # 虚假图像识别率
        f"Fake-Precision: {metrics['fake_precision']:.4f}, "  # 虚假图像预测准确率
        f"F1: {metrics['f1']:.4f}, "
        f"TP/FP/TN/FN: {metrics['true_positive']}/{metrics['false_positive']}/"
        f"{metrics['true_negative']}/{metrics['false_negative']}"
    ) 