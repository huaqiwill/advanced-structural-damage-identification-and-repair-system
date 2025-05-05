from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import json

db = SQLAlchemy()

class User(db.Model):
    """用户模型"""
    __tablename__ = 'users'
    
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password = db.Column(db.String(120), nullable=False)
    is_admin = db.Column(db.Boolean, default=False)  # 新增管理员标识字段
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # 关联检测记录
    detection_records = db.relationship('DetectionRecord', backref='user', lazy=True)
    # 关联警报记录
    alerts = db.relationship('Alert', backref='user', lazy=True)

class DetectionRecord(db.Model):
    """检测记录模型"""
    __tablename__ = 'detection_records'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)
    source_type = db.Column(db.String(20))  # 'image', 'video' or 'camera'
    source_name = db.Column(db.String(255))
    original_path = db.Column(db.String(255))  # 原始文件路径
    result_path = db.Column(db.String(255))    # 结果文件路径
    duration = db.Column(db.Float, default=0.0)  # 检测持续时间（秒）
    total_objects = db.Column(db.Integer, default=0)  # 检测到的总目标数
    detection_results = db.Column(db.Text)  # JSON格式存储检测结果
    is_cleaned = db.Column(db.Boolean, default=False)  # 垃圾是否已清理
    
    # 关联警报
    alerts = db.relationship('Alert', backref='detection_record', lazy=True)
    
    def set_detection_results(self, results_dict):
        """设置检测结果"""
        self.detection_results = json.dumps(results_dict)
    
    def get_detection_results(self):
        """获取检测结果"""
        try:
            if self.detection_results:
                return json.loads(self.detection_results)
            else:
                return {"objects": [], "avg_confidence": 0}
        except Exception as e:
            print(f"解析检测结果出错(ID:{self.id}): {str(e)}")
            return {"objects": [], "avg_confidence": 0, "error": str(e)}
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'source_type': self.source_type,
            'source_name': self.source_name,
            'original_path': self.original_path,
            'result_path': self.result_path,
            'duration': self.duration,
            'total_objects': self.total_objects,
            'detection_results': self.get_detection_results(),
            'is_cleaned': self.is_cleaned
        }

class Alert(db.Model):
    """警报模型"""
    __tablename__ = 'alerts'
    
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('users.id'), nullable=False)
    detection_record_id = db.Column(db.Integer, db.ForeignKey('detection_records.id'), nullable=False)
    alert_type = db.Column(db.String(50), nullable=False)  # 'image', 'video', 'multi_image', 'camera'
    source = db.Column(db.String(255))  # 来源文件名
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)  # 警报时间
    garbage_count = db.Column(db.Integer, default=0)  # 检测到的垃圾数量
    risk_level = db.Column(db.String(20), default='medium')  # 风险级别: 'high', 'medium', 'low'
    status = db.Column(db.String(20), default='pending')  # 处理状态: 'pending', 'processed'
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'detection_record_id': self.detection_record_id,
            'alert_type': self.alert_type,
            'source': self.source,
            'timestamp': self.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'garbage_count': self.garbage_count,
            'risk_level': self.risk_level,
            'status': self.status
        }

class ScientificContent(db.Model):
    """科普内容模型"""
    __tablename__ = 'scientific_contents'
    
    id = db.Column(db.Integer, primary_key=True)
    section_id = db.Column(db.String(100), unique=True, nullable=False)  # 用于前端识别的唯一ID
    title = db.Column(db.String(200), nullable=False)  # 内容标题
    content = db.Column(db.Text, nullable=False)  # 内容正文，可以是HTML格式
    content_type = db.Column(db.String(50), default='text')  # 'text', 'faq', 'tips', 'quiz'
    image_path = db.Column(db.String(255))  # 相关图片路径
    position = db.Column(db.Integer, default=0)  # 显示位置顺序
    status = db.Column(db.String(20), default='active')  # 'active', 'hidden'
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # 内容关联问答（如FAQs）
    faqs = db.relationship('ScientificFAQ', backref='section', lazy=True)
    # 内容关联测试题目
    questions = db.relationship('ScientificQuestion', backref='section', lazy=True)
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'section_id': self.section_id,
            'title': self.title,
            'content': self.content,
            'content_type': self.content_type,
            'image_path': self.image_path,
            'position': self.position,
            'status': self.status,
            'created_at': self.created_at.strftime('%Y-%m-%d %H:%M:%S'),
            'updated_at': self.updated_at.strftime('%Y-%m-%d %H:%M:%S')
        }

class ScientificFAQ(db.Model):
    """科普问答模型"""
    __tablename__ = 'scientific_faqs'
    
    id = db.Column(db.Integer, primary_key=True)
    section_id = db.Column(db.Integer, db.ForeignKey('scientific_contents.id'), nullable=False)
    question = db.Column(db.String(500), nullable=False)  # 问题
    answer = db.Column(db.Text, nullable=False)  # 回答
    position = db.Column(db.Integer, default=0)  # 显示位置顺序
    status = db.Column(db.String(20), default='active')  # 'active', 'hidden'
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'section_id': self.section_id,
            'question': self.question,
            'answer': self.answer,
            'position': self.position,
            'status': self.status
        }

class ScientificQuestion(db.Model):
    """科普测试题目模型"""
    __tablename__ = 'scientific_questions'
    
    id = db.Column(db.Integer, primary_key=True)
    section_id = db.Column(db.Integer, db.ForeignKey('scientific_contents.id'), nullable=True)
    question = db.Column(db.String(500), nullable=False)  # 题目
    options = db.Column(db.Text, nullable=False)  # JSON格式存储选项数组
    answer = db.Column(db.String(10), nullable=False)  # 正确答案
    explanation = db.Column(db.Text)  # 解析
    position = db.Column(db.Integer, default=0)  # 显示位置顺序
    status = db.Column(db.String(20), default='active')  # 'active', 'hidden'
    
    def set_options(self, options_list):
        """设置选项"""
        self.options = json.dumps(options_list)
    
    def get_options(self):
        """获取选项"""
        try:
            if self.options:
                return json.loads(self.options)
            else:
                return []
        except Exception as e:
            print(f"解析题目选项出错(ID:{self.id}): {str(e)}")
            return []
    
    def to_dict(self):
        """转换为字典格式"""
        return {
            'id': self.id,
            'section_id': self.section_id,
            'question': self.question,
            'options': self.get_options(),
            'answer': self.answer,
            'explanation': self.explanation,
            'position': self.position,
            'status': self.status
        } 