#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
PAID (Patches is All You Need for AI-generate Images Detection) 主文件
包含命令行参数解析、训练和评估功能
"""

import os
import time
import argparse
import random
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.backends.cudnn as cudnn
from torch.utils.tensorboard import SummaryWriter

from models import MSTD
from utils.dataset import ImageDataset
from utils.metrics import calculate_metrics
from utils.logging import setup_logger, save_checkpoint, log_metrics

def parse_args():
    """命令行参数解析"""
    parser = argparse.ArgumentParser(description='PAID模型训练与评估')
    
    # 基本参数
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'eval', 'inference'],
                        help='运行模式: 训练、评估或推理')
    parser.add_argument('--data_root', type=str, default='./data',
                        help='数据集根目录')
    parser.add_argument('--save_dir', type=str, default='./output',
                        help='模型和日志保存目录')
    parser.add_argument('--resume', type=str, default=None,
                        help='恢复训练的检查点路径')
    parser.add_argument('--seed', type=int, default=42,
                        help='随机种子')
    parser.add_argument('--gpu', type=str, default='0',
                        help='GPU ID')
    
    # 数据集参数
    parser.add_argument('--batch_size', type=int, default=32,
                        help='批次大小')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='数据加载器的工作线程数')
    parser.add_argument('--image_size', type=int, default=256,
                        help='输入图像大小')
    
    # 模型参数
    parser.add_argument('--patch_size', type=int, default=32,
                        help='图像patch大小')
    parser.add_argument('--stride', type=int, default=16,
                        help='图像patch提取步长')
    parser.add_argument('--num_high_freq_patches', type=int, default=9,
                        help='高频patch数量')
    parser.add_argument('--num_low_freq_patches', type=int, default=9,
                        help='低频patch数量')
    parser.add_argument('--feature_dim', type=int, default=256,
                        help='特征维度')
    
    # 训练参数
    parser.add_argument('--epochs', type=int, default=50,
                        help='训练轮数')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='初始学习率')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='权重衰减')
    parser.add_argument('--lr_scheduler', type=str, default='cosine',
                        choices=['step', 'cosine', 'plateau'],
                        help='学习率调度器类型')
    parser.add_argument('--lr_step_size', type=int, default=10,
                        help='学习率调度器步长(仅用于step模式)')
    parser.add_argument('--lr_gamma', type=float, default=0.1,
                        help='学习率衰减系数(仅用于step和plateau模式)')
    parser.add_argument('--warmup_epochs', type=int, default=5,
                        help='学习率预热轮数')
    
    # 日志和检查点参数
    parser.add_argument('--print_freq', type=int, default=10,
                        help='打印频率(迭代次数)')
    parser.add_argument('--save_freq', type=int, default=5,
                        help='保存检查点频率(轮数)')
    parser.add_argument('--eval_freq', type=int, default=1,
                        help='评估频率(轮数)')
    
    # 推理参数
    parser.add_argument('--image', type=str, default=None,
                        help='用于推理模式的图像文件路径')
    
    return parser.parse_args()

def set_seed(seed):
    """设置随机种子以确保可重现性"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        cudnn.deterministic = True
        cudnn.benchmark = False

def get_data_loaders(args):
    """创建数据加载器"""
    # 数据增强和预处理
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = ImageDataset(
        root=os.path.join(args.data_root, 'train'),
        transform=train_transform,
        is_train=True
    )
    
    val_dataset = ImageDataset(
        root=os.path.join(args.data_root, 'val'),
        transform=val_transform,
        is_train=False
    )
    
    # 创建数据加载器
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True
    )
    
    return train_loader, val_loader

def build_model(args):
    """构建MSTD模型"""
    model = MSTD(
        patch_size=args.patch_size,
        stride=args.stride,
        num_high_freq_patches=args.num_high_freq_patches,
        num_low_freq_patches=args.num_low_freq_patches,
        feature_dim=args.feature_dim
    )
    
    return model

def get_optimizer_and_scheduler(args, model, train_loader):
    """获取优化器和学习率调度器"""
    # 优化器
    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # 学习率调度器
    if args.lr_scheduler == 'step':
        scheduler = optim.lr_scheduler.StepLR(
            optimizer,
            step_size=args.lr_step_size,
            gamma=args.lr_gamma
        )
    elif args.lr_scheduler == 'cosine':
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=args.epochs - args.warmup_epochs
        )
    elif args.lr_scheduler == 'plateau':
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='max',
            factor=args.lr_gamma,
            patience=5,
            verbose=True
        )
    
    # 学习率预热
    if args.warmup_epochs > 0:
        warmup_scheduler = optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=0.1,
            end_factor=1.0,
            total_iters=args.warmup_epochs * len(train_loader)
        )
    else:
        warmup_scheduler = None
    
    return optimizer, scheduler, warmup_scheduler

def train_one_epoch(model, train_loader, criterion, optimizer, warmup_scheduler, device, epoch, args, logger, writer):
    """训练一个epoch"""
    model.train()
    
    losses = []
    all_preds = []
    all_targets = []
    
    pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Train]')
    for i, (images, targets) in enumerate(pbar):
        # 将数据移至GPU
        images = images.to(device)
        targets = targets.float().to(device)
        
        # 前向传播
        outputs = model(images)
        loss = criterion(outputs, targets)
        
        # 反向传播和优化
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # 如果在预热阶段，更新学习率
        if warmup_scheduler is not None and epoch < args.warmup_epochs:
            warmup_scheduler.step()
        
        # 计算准确率
        batch_preds = (outputs > 0.5).float()
        batch_acc = (batch_preds == targets).float().mean().item()
        
        # 记录损失和预测
        losses.append(loss.item())
        all_preds.extend(outputs.detach().cpu().numpy())
        all_targets.extend(targets.detach().cpu().numpy())
        
        # 更新进度条，显示损失和准确率
        avg_loss = sum(losses) / len(losses)
        pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{batch_acc:.4f}'})
        
        # 定期打印日志
        if (i + 1) % args.print_freq == 0:
            logger.info(f'Epoch [{epoch+1}/{args.epochs}] Batch [{i+1}/{len(train_loader)}] Loss: {avg_loss:.4f}, Acc: {batch_acc:.4f}')
            # 记录到TensorBoard
            step = epoch * len(train_loader) + i
            writer.add_scalar('train/loss', avg_loss, step)
            writer.add_scalar('train/batch_acc', batch_acc, step)
    
    # 计算epoch级别的度量
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    
    # 记录训练度量
    log_metrics(logger, 'Train', epoch+1, metrics)
    for key, value in metrics.items():
        writer.add_scalar(f'train/{key}', value, epoch)
    
    return metrics, np.mean(losses)

def validate(model, val_loader, criterion, device, epoch, args, logger, writer):
    """在验证集上评估模型"""
    model.eval()
    
    losses = []
    all_preds = []
    all_targets = []
    
    with torch.no_grad():
        pbar = tqdm(val_loader, desc=f'Epoch {epoch+1}/{args.epochs} [Val]')
        for images, targets in pbar:
            # 将数据移至GPU
            images = images.to(device)
            targets = targets.float().to(device)
            
            # 前向传播
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # 计算准确率
            batch_preds = (outputs > 0.5).float()
            batch_acc = (batch_preds == targets).float().mean().item()
            
            # 记录损失和预测
            losses.append(loss.item())
            all_preds.extend(outputs.detach().cpu().numpy())
            all_targets.extend(targets.detach().cpu().numpy())
            
            # 更新进度条，显示损失和准确率
            avg_loss = sum(losses) / len(losses)
            pbar.set_postfix({'loss': f'{avg_loss:.4f}', 'acc': f'{batch_acc:.4f}'})
    
    # 计算epoch级别的度量
    metrics = calculate_metrics(np.array(all_preds), np.array(all_targets))
    
    # 记录验证度量
    log_metrics(logger, 'Validation', epoch+1, metrics)
    for key, value in metrics.items():
        writer.add_scalar(f'val/{key}', value, epoch)
    writer.add_scalar('val/loss', np.mean(losses), epoch)
    
    return metrics, np.mean(losses)

def main():
    """主函数：解析参数并启动训练或评估"""
    args = parse_args()
    
    # 设置随机种子
    set_seed(args.seed)
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 创建保存目录
    timestamp = time.strftime('%Y%m%d_%H%M%S', time.localtime())
    args.save_dir = os.path.join(args.save_dir, f'PAID_{timestamp}')
    os.makedirs(args.save_dir, exist_ok=True)
    
    # 设置日志记录
    logger = setup_logger(args.save_dir, 'main.log')
    logger.info(f"Arguments: {args}")
    logger.info(f"Using device: {device}")
    
    # 创建TensorBoard写入器
    writer = SummaryWriter(log_dir=os.path.join(args.save_dir, 'tensorboard'))
    
    # 获取数据加载器
    train_loader, val_loader = get_data_loaders(args)
    logger.info(f"Training dataset size: {len(train_loader.dataset)}")
    logger.info(f"Validation dataset size: {len(val_loader.dataset)}")
    
    # 构建模型
    model = build_model(args)
    model = model.to(device)
    
    # 定义损失函数
    criterion = nn.BCELoss()  # 如果模型输出已经过sigmoid
    # 或者
    # criterion = nn.BCEWithLogitsLoss()  # 如果模型输出未经过sigmoid
    
    # 获取优化器和学习率调度器
    optimizer, lr_scheduler, warmup_scheduler = get_optimizer_and_scheduler(args, model, train_loader)
    
    # 从检查点恢复（如果指定）
    start_epoch = 0
    best_metric = 0
    if args.resume:
        if os.path.isfile(args.resume):
            logger.info(f"Loading checkpoint '{args.resume}'")
            # 只加载模型权重
            model_weights = torch.load(args.resume)
            model.load_state_dict(model_weights)
            # 读取训练信息文件(如果存在)
            info_file = os.path.join(os.path.dirname(args.resume), 'training_info.txt')
            if os.path.exists(info_file):
                with open(info_file, 'r') as f:
                    lines = f.readlines()
                    for line in lines:
                        if line.startswith('Epoch:'):
                            start_epoch = int(line.split(':')[1].strip()) + 1
                        elif line.startswith('Best Metric:'):
                            best_metric = float(line.split(':')[1].strip())
            logger.info(f"Loaded model weights from '{args.resume}'")
            logger.info(f"Resuming from epoch {start_epoch}")
        else:
            logger.info(f"No checkpoint found at '{args.resume}'")
    
    # 训练模式
    if args.mode == 'train':
        logger.info("Starting training...")
        
        for epoch in range(start_epoch, args.epochs):
            # 训练一个epoch
            train_metrics, train_loss = train_one_epoch(
                model, train_loader, criterion, optimizer, 
                warmup_scheduler, device, epoch, args, logger, writer
            )
            
            # 在验证集上评估模型
            if (epoch + 1) % args.eval_freq == 0:
                val_metrics, val_loss = validate(
                    model, val_loader, criterion, 
                    device, epoch, args, logger, writer
                )
                
                # 更新学习率（如果使用ReduceLROnPlateau）
                if args.lr_scheduler == 'plateau':
                    lr_scheduler.step(val_metrics['auc'])
                else:
                    lr_scheduler.step()
                
                # 保存最佳模型
                if val_metrics['auc'] > best_metric:
                    best_metric = val_metrics['auc']
                    save_checkpoint({
                        'epoch': epoch,
                        'model': model.state_dict(),
                        'optimizer': optimizer.state_dict(),
                        'scheduler': lr_scheduler.state_dict(),
                        'best_metric': best_metric,
                    }, is_best=True, save_dir=args.save_dir)
                    logger.info(f"New best model saved with AUC: {best_metric:.4f}")
            else:
                # 如果这个epoch不评估，仍需要更新学习率（除非使用ReduceLROnPlateau）
                if args.lr_scheduler != 'plateau':
                    lr_scheduler.step()
            
            # 定期保存检查点
            if (epoch + 1) % args.save_freq == 0:
                save_checkpoint({
                    'epoch': epoch,
                    'model': model.state_dict(),
                    'optimizer': optimizer.state_dict(),
                    'scheduler': lr_scheduler.state_dict(),
                    'best_metric': best_metric,
                }, is_best=False, save_dir=args.save_dir)
                logger.info(f"Checkpoint saved at epoch {epoch+1}")
            
            # 记录当前学习率
            current_lr = optimizer.param_groups[0]['lr']
            writer.add_scalar('train/lr', current_lr, epoch)
            logger.info(f"Epoch {epoch+1} completed. Current LR: {current_lr:.6f}")
        
        logger.info("Training completed!")
        writer.close()
    
    # 评估模式
    elif args.mode == 'eval':
        logger.info("Starting evaluation...")
        val_metrics, val_loss = validate(
            model, val_loader, criterion, 
            device, 0, args, logger, writer
        )
        logger.info(f"Evaluation completed. Loss: {val_loss:.4f}")
        logger.info(f"Metrics: {val_metrics}")
        writer.close()
    
    # 推理模式
    elif args.mode == 'inference':
        logger.info("Starting inference...")
        # 加载单张图像
        if args.image:
            from PIL import Image
            
            transform = transforms.Compose([
                transforms.Resize((args.image_size, args.image_size)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            image = Image.open(args.image).convert('RGB')
            image_tensor = transform(image).unsqueeze(0).to(device)
            
            with torch.no_grad():
                output = model(image_tensor)
                prob = output.item()
                
            result = "AI生成的虚假图像" if prob > 0.5 else "真实图像"
            confidence = prob if prob > 0.5 else 1 - prob
            
            logger.info(f"图像: {args.image}")
            logger.info(f"预测结果: {result}")
            logger.info(f"置信度: {confidence:.4f}")
            logger.info(f"原始输出分数: {prob:.4f}")
        else:
            logger.info("请使用 --image 参数指定要推理的图像文件")

if __name__ == '__main__':
    main()