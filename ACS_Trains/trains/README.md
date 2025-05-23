# 结构损伤检测项目

这个项目使用YOLOv8模型来检测飞机和汽车表面的结构损伤。

## 项目结构

```
trains/
├── data/
│   ├── images/          # 存放训练、验证和测试图片
│   └── labels/          # 存放标注文件
├── models/              # 存放训练好的模型
├── utils/              # 工具函数
├── configs/            # 配置文件
├── train.py           # 训练脚本
├── detect.py          # 推理脚本
└── requirements.txt   # 项目依赖
```

## 环境配置

1. 创建虚拟环境（推荐）：
```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows
```

2. 安装依赖：
```bash
pip install -r requirements.txt
```

## 数据准备

1. 将训练图片放在 `data/images/train/` 目录下
2. 将验证图片放在 `data/images/val/` 目录下
3. 将测试图片放在 `data/images/test/` 目录下
4. 对应的标注文件放在 `data/labels/` 对应子目录下

标注格式为YOLO格式：
```
<class> <x_center> <y_center> <width> <height>
```
其中：
- class: 0表示损伤，1表示正常
- 所有坐标值都是相对于图片大小的归一化值（0-1之间）

## 训练模型

```bash
python train.py
```

训练完成后，最佳模型将保存在 `runs/train/damage_detection/weights/` 目录下。

## 使用模型进行检测

```bash
python detect.py
```

检测结果将显示在屏幕上，并保存为图片或视频文件。

## 注意事项

1. 确保有足够的GPU内存进行训练
2. 可以在 `configs/config.yaml` 中调整训练参数
3. 检测时可以调整置信度阈值来平衡准确率和召回率

## 模型效果

- 支持检测飞机和汽车表面的各类损伤
- 可以实时显示损伤位置和置信度
- 支持图片、视频和实时检测 