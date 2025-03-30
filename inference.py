"""
MSTD模型单张图像推理脚本
"""

import os
import argparse
import torch
import torch.serialization
torch.serialization.add_safe_globals(['numpy.core.multiarray.scalar'])
from PIL import Image
from torchvision import transforms
from models import MSTD

def parse_args():
    parser = argparse.ArgumentParser(description='MSTD模型推理')
    parser.add_argument('--image', type=str, required=True, help='输入图像路径')
    parser.add_argument('--model', type=str, required=True, help='模型权重路径')
    parser.add_argument('--image_size', type=int, default=256, help='图像大小')
    parser.add_argument('--patch_size', type=int, default=32, help='patch大小')
    parser.add_argument('--stride', type=int, default=16, help='patch步长')
    parser.add_argument('--gpu', type=str, default='0', help='GPU ID')
    return parser.parse_args()

def main():
    args = parse_args()
    
    # 设置设备
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 加载模型
    model = MSTD(
        patch_size=args.patch_size,
        stride=args.stride,
        num_high_freq_patches=9,
        num_low_freq_patches=9,
        feature_dim=256
    )
    
    # 加载权重 - 支持.pth和.pth.tar两种格式
    try:
        if args.model.endswith('.pth'):
            # 如果是.pth格式，直接加载模型权重
            model_weights = torch.load(args.model, map_location=device)
            model.load_state_dict(model_weights)
            print(f"成功从.pth文件加载模型: {args.model}")
        else:
            # 如果是.pth.tar格式，从字典中提取模型权重
            checkpoint = torch.load(args.model, map_location=device)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print(f"成功从检查点文件加载模型: {args.model}")
    except Exception as e:
        print(f"加载模型时出错: {e}")
        print("尝试替代加载方法...")
        try:
            checkpoint = torch.load(args.model, map_location=device, weights_only=False)
            if 'model' in checkpoint:
                model.load_state_dict(checkpoint['model'])
            else:
                model.load_state_dict(checkpoint)
            print("使用weights_only=False成功加载模型")
        except Exception as e2:
            print(f"替代加载方法也失败: {e2}")
            return
    
    model = model.to(device)
    model.eval()
    
    # 加载图像
    image = Image.open(args.image).convert('RGB')
    
    # 图像预处理
    transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    
    # 推理
    with torch.no_grad():
        output = model(image_tensor)
        
    probability = output.item()
    
    # 打印结果
    print(f"图像路径: {args.image}")
    print(f"AI生成概率: {probability:.4f}")
    print(f"预测结果: {'AI生成' if probability > 0.5 else '真实图像'}")
    
if __name__ == '__main__':
    main() 