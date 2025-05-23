# 高级识别系统（Advanced Recognition System）

本项目为一个集成了结构损伤检测与垃圾识别的综合系统，包含Web端管理与展示、模型训练与推理、数据库管理等模块。适用于如飞机、汽车表面损伤检测及多类垃圾识别等场景。

## 目录结构

```
.
├── ACS_Web/         # Web端前后端一体化服务（Flask）
├── ACS_Trains/      # 结构损伤与垃圾检测模型训练与推理
├── ACS_Test/        # 测试脚本
├── ACS_Install/     # 安装相关（预留）
├── venv/            # Python虚拟环境
├── README.md        # 项目说明文档
└── ...
```

## 主要功能

### 1. Web端（ACS_Web）
- 用户注册、登录、权限管理（支持管理员与普通用户）
- 图像、视频、摄像头实时检测
- 检测结果可视化与历史记录管理
- 检测警报、风险分级、垃圾清理状态管理
- 科普内容、问答、测试题管理
- RESTful API接口
- 支持多用户并发与SocketIO实时通信

### 2. 训练与推理（ACS_Trains/trains）
- 基于YOLOv8的结构损伤/垃圾检测模型训练与推理
- 支持自定义数据集、YOLO格式标注
- 训练参数可配置，支持多类别检测
- 推理支持图片、视频、实时流

## 安装与环境配置

### 1. 克隆项目
```bash
git clone <your-repo-url>
cd advanced-recognition-system
```

### 2. 创建虚拟环境并安装依赖

#### Web端依赖
```bash
cd ACS_Web
python -m venv venv
venv\Scripts\activate  # Windows
# source venv/bin/activate  # Linux/Mac
pip install -r requirements.txt
```

#### 训练模块依赖
```bash
cd ../ACS_Trains/trains
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

## 数据准备（训练模块）

- 训练图片放在 `data/images/train/`
- 验证图片放在 `data/images/val/`
- 测试图片放在 `data/images/test/`
- 标注文件放在 `data/labels/`，格式为YOLO格式

## 训练模型

```bash
python train.py
```
训练完成后，模型保存在 `runs/train/damage_detection/weights/`。

## 推理检测

```bash
python detect.py
```
支持图片、视频、实时流检测，结果自动保存。

## 启动Web服务

```bash
cd ACS_Web
venv\Scripts\activate
python app.py
```
浏览器访问 http://127.0.0.1:5000

## 数据库说明

- 使用SQLite，默认数据库文件为`detection_records.db`
- 主要表结构包括：用户（User）、检测记录（DetectionRecord）、警报（Alert）、科普内容（ScientificContent）、问答（ScientificFAQ）、测试题（ScientificQuestion）等
- 数据库模型定义详见`ACS_Web/models.py`

## 配置说明

- 主要配置项在`ACS_Web/config.py`，包括模型路径、类别定义、上传目录等
- 支持自定义模型与类别

## 其他说明

- 支持多用户并发、权限分级
- 检测类别可根据实际需求扩展
- 科普内容、问答、测试题可在后台管理

## 致谢

- 本项目部分功能基于[ultralytics/yolov8](https://github.com/ultralytics/ultralytics)实现
