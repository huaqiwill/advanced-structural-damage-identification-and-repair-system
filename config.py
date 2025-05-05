# coding:utf-8
import os

# 基础配置
class Config:
    # Flask配置
    SECRET_KEY = 'your-secret-key-keep-it-secret'
    SQLALCHEMY_DATABASE_URI = 'sqlite:///detection_records.db'
    SQLALCHEMY_TRACK_MODIFICATIONS = False
    
    # 文件上传配置
    UPLOAD_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'static/uploads')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 最大16MB
    ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
    ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'webm'}

    # 检测模型配置
    MODEL_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    MODEL_PATH = os.path.join(MODEL_DIR, 'best.pt')
    SAVE_PATH = os.path.join(UPLOAD_FOLDER, 'results')

    # 类别配置
    NAMES = {
        0: 'bad_fruit', 
        1: 'bottle', 
        2: 'branch', 
        3: 'can', 
        4: 'glass_bottle', 
        5: 'grass', 
        6: 'leaf', 
        7: 'milk_box', 
        8: 'plastic_bag', 
        9: 'plastic_box'
    }
    
    CH_NAMES = [
        '腐烂水果', 
        '塑料瓶', 
        '树枝', 
        '易拉罐', 
        '玻璃瓶', 
        '杂草', 
        '叶子', 
        '牛奶盒', 
        '塑料袋', 
        '塑料箱'
    ]
    
    ADMIN_KEY = "admin"

    @staticmethod
    def init_app(app):
        # 确保必要目录存在
        os.makedirs(Config.UPLOAD_FOLDER, exist_ok=True)
        os.makedirs(Config.SAVE_PATH, exist_ok=True)
        os.makedirs(Config.MODEL_DIR, exist_ok=True) 