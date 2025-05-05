import os
import yaml
from ultralytics import YOLO
import cv2
import numpy as np
from pathlib import Path

def load_config(config_path):
    """加载配置文件"""
    with open(config_path, 'r', encoding='utf-8') as f:
        config = yaml.safe_load(f)
    return config

def train(config_path):
    """训练模型"""
    # 加载配置
    config = load_config(config_path)
    
    # 初始化模型
    model = YOLO(f"yolov8{config['model']['size']}.pt")
    
    # 训练模型
    results = model.train(
        data=config_path,
        epochs=config['epochs'],
        imgsz=config['img_size'],
        batch=config['batch_size'],
        workers=config['workers'],
        device='0',  # 使用GPU，如果使用CPU则设为'cpu'
        project='runs/train',
        name='damage_detection'
    )
    
    return results

def validate(model_path, val_data_path, conf_thres=0.25):
    """验证模型"""
    model = YOLO(model_path)
    results = model.val(
        data=val_data_path,
        conf=conf_thres,
    )
    return results

if __name__ == '__main__':
    config_path = 'configs/config.yaml'
    
    # 训练模型
    print("开始训练模型...")
    results = train(config_path)
    print(f"训练完成！模型保存在: {results.save_dir}")
    
    # 验证模型
    print("\n开始验证模型...")
    val_results = validate(
        model_path=str(Path(results.save_dir) / 'weights' / 'best.pt'),
        val_data_path=config_path
    )
    print("验证完成！") 