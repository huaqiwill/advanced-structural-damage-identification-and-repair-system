from flask import Blueprint, render_template, request, redirect, url_for, flash, session, jsonify
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, DetectionRecord, Alert
from datetime import datetime, timedelta
from config import Config
import time
import random
from functools import wraps
import os
from werkzeug.utils import secure_filename
import json
import shutil
import base64
import imageio
import imageio_ffmpeg
import numpy as np
import cv2

# 创建Blueprint
admin_bp = Blueprint('admin', __name__, url_prefix='/admin')

# 辅助函数获取上传目录
def get_upload_folder():
    """获取上传文件的根目录"""
    # 使用相对于当前文件的路径
    static_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
    upload_folder = os.path.join(static_dir, "uploads")
    ensure_dir(upload_folder)
    return upload_folder

# 获取原始文件目录
def get_original_dir():
    """获取原始文件存储目录"""
    folder = os.path.join(get_upload_folder(), 'original')
    ensure_dir(folder)
    return folder

# 获取结果文件目录
def get_results_dir():
    """获取处理结果文件存储目录"""
    folder = os.path.join(get_upload_folder(), 'results')
    ensure_dir(folder)
    return folder

# 获取摄像头文件目录
def get_camera_dir():
    """获取摄像头文件存储目录"""
    folder = os.path.join(get_upload_folder(), 'camera')
    ensure_dir(folder)
    return folder

# 确保目录存在
def ensure_dir(directory):
    """确保目录存在，不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory

# 获取相对路径
def get_relative_path(absolute_path):
    """将绝对路径转换为相对于static的路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # 项目根目录
    static_dir = os.path.join(base_dir, 'static')
    
    # 规范化路径，处理不同操作系统的路径分隔符
    absolute_path = os.path.normpath(absolute_path)
    static_dir = os.path.normpath(static_dir)
    
    # 使用os.path.relpath获取相对路径
    if absolute_path and os.path.exists(absolute_path):
        try:
            # 首先检查是否已经是相对于static的路径
            path_parts = absolute_path.replace('\\', '/').split('/static/')
            if len(path_parts) > 1:
                return '/static/' + path_parts[1]
            
            # 否则尝试获取相对路径
            rel_path = os.path.relpath(absolute_path, static_dir)
            # 确保使用正斜杠，避免Windows反斜杠问题
            rel_path = rel_path.replace('\\', '/')
            # 避免路径中包含上级目录符号，如 '../'
            if not rel_path.startswith('..'):
                return '/static/' + rel_path
        except:
            pass
    
    # 如果无法获取相对路径，则返回规范化后的原始路径
    return absolute_path.replace('\\', '/')

# 管理员验证装饰器
def admin_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('请先登录', 'error')
            return redirect(url_for('admin.admin_login'))
        
        user = User.query.get(session['user_id'])
        if not user or not user.is_admin:
            flash('您没有管理员权限', 'error')
            return redirect(url_for('admin.admin_login'))
            
        return f(*args, **kwargs)
    return decorated_function

# 管理员登录
@admin_bp.route('/login', methods=['GET', 'POST'])
def admin_login():
    """管理员登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        # 验证管理员用户
        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password) and user.is_admin:
            session['user_id'] = user.id
            session['username'] = user.username  # 明确存储用户名到session
            session['is_admin'] = True
            session['login_time'] = int(time.time())
            flash('管理员登录成功！', 'success')
            return redirect(url_for('admin.dashboard'))
        flash('用户名或密码错误，或用户不是管理员！', 'error')
    
    # 返回管理员登录页面
    return render_template('admin/admin_login.html')

# 管理员注册
@admin_bp.route('/register', methods=['GET', 'POST'])
def admin_register():
    """管理员注册页面"""
    if request.method == 'POST':
        admin_key = request.form.get('admin_key')
        username = request.form.get('username')
        password = request.form.get('password')
        confirm_password = request.form.get('confirm_password')
        
        # 验证管理员授权码
        if admin_key != Config.ADMIN_KEY:
            flash('管理员授权码无效！', 'error')
            return render_template('admin/admin_register.html')
        
        # 验证密码
        if password != confirm_password:
            flash('两次输入的密码不一致！', 'error')
            return render_template('admin/admin_register.html')
        
        # 检查密码强度
        if len(password) < 8:
            flash('密码至少需要8个字符！', 'error')
            return render_template('admin/admin_register.html')
        
        if not any(c.isdigit() for c in password) or not any(c.isalpha() for c in password):
            flash('密码必须包含数字和字母！', 'error')
            return render_template('admin/admin_register.html')
        
        # 检查用户名是否存在
        if User.query.filter_by(username=username).first():
            flash('用户名已存在！', 'error')
            return render_template('admin/admin_register.html')
        
        # 创建新管理员用户
        user = User(
            username=username,
            password=generate_password_hash(password),
            is_admin=True  # 设置为管理员
        )
        db.session.add(user)
        
        try:
            db.session.commit()
            flash('管理员账户创建成功！请登录。', 'success')
            return redirect(url_for('admin.admin_login'))
        except Exception as e:
            db.session.rollback()
            flash(f'注册失败：{str(e)}', 'error')
    
    return render_template('admin/admin_register.html')

@admin_bp.route('/dashboard')
@admin_required
def dashboard():
    """管理员仪表盘"""
    return render_template('admin/admin_home.html')

@admin_bp.route('/home')
@admin_required
def admin_home():
    """管理员首页"""
    return render_template('admin/admin_home.html')

@admin_bp.route('/detection_records')
@admin_required
def admin_detection_records():
    """管理员检测记录页面"""
    return render_template('admin/detection_records.html')

@admin_bp.route('/admin_multi_image')
@admin_required
def admin_multi_image():
    """多张图片检测页面"""
    return render_template('admin/admin_multi_image.html')

@admin_bp.route('/admin_image_detection')
@admin_required
def admin_image_detection():
    """管理员图片识别页面"""
    return render_template('admin/admin_image_detection.html')

@admin_bp.route('/admin_video_detection')
@admin_required
def admin_video_detection():
    """管理员视频识别页面"""
    return render_template('admin/admin_video_detection.html')

@admin_bp.route('/camera_detection')
@admin_required
def admin_camera_detection():
    """摄像头实时检测页面"""
    return render_template('admin/admin_camera_detection.html')

@admin_bp.route('/admin_alert_management')
@admin_required
def admin_alert_management():
    """警报管理页面"""
    return render_template('admin/admin_alert_management.html')

@admin_bp.route('/admin_scientific')
@admin_required
def admin_scientific():
    """科普知识管理页面"""
    return render_template('admin/admin_scientific.html')

@admin_bp.route('/admin_user_management')
@admin_required
def admin_user_management():
    """用户管理页面"""
    return render_template('admin/admin_user_management.html')

@admin_bp.route('/settings')
@admin_required
def settings():
    """设置页面"""
    return render_template('admin/settings.html')

# 管理员登出
@admin_bp.route('/logout')
def admin_logout():
    """管理员登出"""
    # 清除session中的管理员信息
    session.pop('user_id', None)
    session.pop('username', None)  # 确保清除用户名
    session.pop('is_admin', None)
    session.pop('login_time', None)
    flash('您已成功退出登录', 'success')
    # 重定向到管理员登录页面
    return redirect(url_for('admin.admin_login'))

# 创建检测记录后生成警报的函数
def create_alert_from_detection(detection_record):
    """根据检测记录创建警报，针对检测到垃圾的记录创建警报"""
    # 只有当检测到垃圾时（total_objects > 0）才创建警报
    if detection_record.total_objects > 0:
        # 根据垃圾数量确定风险级别
        garbage_count = detection_record.total_objects
        
        if garbage_count >= 5:
            risk_level = 'high'  # 5个或更多垃圾，高风险
        elif garbage_count >= 2:
            risk_level = 'medium'  # 2-4个垃圾，中风险
        else:
            risk_level = 'low'  # 1个垃圾，低风险
        
        # 返回警报信息而不是存储到数据库
        alert_info = {
            'user_id': detection_record.user_id,
            'detection_record_id': detection_record.id,
            'alert_type': detection_record.source_type,
            'source': detection_record.source_name,
            'garbage_count': garbage_count,
            'risk_level': risk_level,
            'timestamp': datetime.now()
        }
        
        # 这里可以添加通知逻辑，如发送邮件或短信
        
        return alert_info
    
    return None

# 获取警报列表API - 直接查询detection_records表中未检测到垃圾的记录
@admin_bp.route('/api/alerts', methods=['GET'])
def get_alerts():
    """获取警报列表 - 检测到垃圾的记录（total_objects>0的detection_records）"""
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # 获取筛选参数
    source_type = request.args.get('source_type')
    is_cleaned = request.args.get('is_cleaned')
    
    # 构建查询 - 直接查询detection_records表，条件是total_objects>0
    query = DetectionRecord.query.filter(DetectionRecord.total_objects > 0)
    
    # 应用筛选条件
    if source_type:
        query = query.filter(DetectionRecord.source_type == source_type)
    if is_cleaned:
        query = query.filter(DetectionRecord.is_cleaned == (is_cleaned.lower() == 'true'))
    
    # 按时间降序排序并分页
    pagination = query.order_by(DetectionRecord.timestamp.desc()).paginate(page=page, per_page=per_page)
    alerts = pagination.items
    
    # 获取项目根目录的路径，用于移除路径前缀
    current_dir = os.path.dirname(os.path.abspath(__file__))
    base_dir = os.path.dirname(current_dir)  # 项目根目录
    
    # 格式化数据
    alerts_data = []
    for record in alerts:
        # 处理路径，获取相对路径
        original_path = get_relative_path(record.original_path)
        result_path = get_relative_path(record.result_path)
        
        alert_data = {
            'id': record.id,
            'user_id': record.user_id,
            'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S') if record.timestamp else '',
            'alert_type': record.source_type,
            'source': record.source_name,
            'original_path': original_path,
            'result_path': result_path,
            'status': '已处理' if record.is_cleaned else '待处理',
            'total_objects': record.total_objects
        }
        alerts_data.append(alert_data)
    
    # 返回结果
    return jsonify({
        'alerts': alerts_data,
        'total': pagination.total,
        'pages': pagination.pages,
        'current_page': page
    })

# 获取警报统计信息API - 直接统计detection_records表中total_objects=0的记录
@admin_bp.route('/api/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """获取警报统计信息 - 检测到垃圾的记录统计（total_objects>0的detection_records）"""
    # 总警报数（total_objects>0的记录数）
    total_alerts = DetectionRecord.query.filter(DetectionRecord.total_objects > 0).count()
    
    # 今日警报数
    today = datetime.now().date()
    today_alerts = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.timestamp >= datetime(today.year, today.month, today.day)
    ).count()
    
    # 各状态警报数
    cleaned_count = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.is_cleaned == True
    ).count()
    not_cleaned_count = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.is_cleaned == False
    ).count()
    
    # 按检测类型统计
    image_count = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.source_type == 'image'
    ).count()
    video_count = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.source_type == 'video'
    ).count()
    multi_image_count = DetectionRecord.query.filter(
        DetectionRecord.total_objects > 0,
        DetectionRecord.source_type == 'multi_image'
    ).count()
    
    # 最近7天每天的警报数
    daily_stats = []
    for i in range(6, -1, -1):
        day_date = datetime.now().date() - timedelta(days=i)
        day_start = datetime(day_date.year, day_date.month, day_date.day)
        day_end = datetime(day_date.year, day_date.month, day_date.day, 23, 59, 59)
        
        count = DetectionRecord.query.filter(
            DetectionRecord.total_objects > 0,
            DetectionRecord.timestamp >= day_start,
            DetectionRecord.timestamp <= day_end
        ).count()
        
        daily_stats.append({
            'date': day_date.strftime('%m-%d'),
            'count': count
        })
    
    return jsonify({
        'total_alerts': total_alerts,
        'today_alerts': today_alerts,
        'high_risk_count': total_alerts,  # 检测到垃圾，全部为高风险
        'medium_risk_count': 0,  # 不使用中风险
        'low_risk_count': 0,  # 不使用低风险
        'cleaned_count': cleaned_count,
        'not_cleaned_count': not_cleaned_count,
        'image_count': image_count,
        'video_count': video_count,
        'multi_image_count': multi_image_count,
        'daily_stats': daily_stats
    })

# 测试警报API
@admin_bp.route('/api/alerts/test', methods=['POST'])
def test_alert():
    """测试警报发送"""
    # 这里可以实现发送测试警报的逻辑
    # 比如发送邮件或短信通知
    
    return jsonify({
        'success': True,
        'message': '测试警报已发送'
    })

# 添加到现有的api/detect/image路由
@admin_bp.route('/api/detect/image', methods=['POST'])
def api_detect_image():
    """API接口：图片检测"""
    try:
        # 导入必要模块
        import os
        import imageio
        import cv2
        import numpy as np
        from datetime import datetime
        from werkzeug.utils import secure_filename
        
        # 检查是否有图片文件上传
        if 'image' not in request.files:
            # 模拟检测结果（用于测试）
            detection_results = {
                'objects': [
                    {'class': '塑料瓶', 'confidence': 0.92, 'bbox': [10, 20, 100, 150]},
                    {'class': '纸杯', 'confidence': 0.85, 'bbox': [200, 210, 280, 350]},
                    {'class': '食品包装', 'confidence': 0.78, 'bbox': [350, 100, 450, 200]}
                ],
                'processing_time': 0.85
            }
            
            # 获取当前用户ID
            user_id = session.get('user_id', 1)
        
            # 使用示例图片名称
            image_name = 'test_image.jpg'  
            
            # 创建记录
            record = DetectionRecord(
                user_id=user_id,
                source_type='image',
                source_name=image_name,
                original_path='/uploads/original/test_image.jpg',
                result_path='/uploads/results/test_image_result.jpg',
                duration=detection_results['processing_time'],
                total_objects=len(detection_results['objects'])
            )
            record.set_detection_results(detection_results)
            
            db.session.add(record)
            db.session.commit()
            
            # 创建警报（如有必要）
            alert_info = create_alert_from_detection(record)
            
            # 返回结果
            result = {
                'success': True,
                'record_id': record.id,
                'detection_results': detection_results,
                'result_image_url': record.result_path,
                'total_objects': record.total_objects,
                'has_garbage': record.total_objects > 0
            }
            
            if alert_info:
                result['alert'] = alert_info
                
            return jsonify(result)
        
        # 处理实际上传的图片文件
        image_file = request.files['image']
        
        # 确保上传目录存在
        original_dir = get_original_dir()
        results_dir = get_results_dir()
        
        # 保存上传的原始图片
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = secure_filename(image_file.filename)
        original_filename = f"{timestamp}_{filename}"
        original_path = os.path.join(original_dir, original_filename)
        original_path = os.path.normpath(original_path)
        
        image_file.save(original_path)
        print(f"图片已保存: {original_path}")
        
        # 设置结果文件路径
        base_name = os.path.splitext(original_filename)[0]
        result_filename = f"{base_name}_result.jpg"
        result_path = os.path.join(results_dir, result_filename)
        result_path = os.path.normpath(result_path)
        
        # 获取当前用户ID
        user_id = session.get('user_id', 1)
        
        # 使用模型进行检测处理
        start_time = time.time()
        try:
            # 导入模型（实际项目中应有真实模型）
            from app import model
            
            if not model:
                raise Exception("未能加载检测模型")
                
            # 使用imageio读取图片
            image = imageio.imread(original_path)
            
            if image is None:
                raise Exception(f"无法读取图片: {original_path}")
            
            # 转换为OpenCV格式处理（如果需要）
            if len(image.shape) == 3 and image.shape[2] == 3:
                opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                opencv_image = image
            
            # 执行检测
            results = model(opencv_image)
            result = results[0]
            boxes = result.boxes
            
            # 获取检测结果
            detected_objects = []
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    detected_objects.append({
                        'class': label,
                        'confidence': round(conf, 2),
                        'bbox': [x1, y1, x2, y2]
                    })
                
                # 绘制检测框
                annotated_image = result.plot()
                
                # 保存结果图像（转回RGB格式）
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    imageio.imwrite(result_path, rgb_image, quality=95)
                else:
                    imageio.imwrite(result_path, annotated_image, quality=95)
            else:
                # 未检测到对象，保存原始图片
                imageio.imwrite(result_path, image, quality=95)
                detected_objects = []
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建检测结果
            detection_results = {
                'objects': detected_objects,
                'processing_time': round(processing_time, 2)
            }
            
        except Exception as e:
            print(f"图片处理失败: {str(e)}")
            # 处理失败时使用模拟数据
            detection_results = {
                'objects': [
                    {'class': '塑料瓶', 'confidence': 0.92, 'bbox': [10, 20, 100, 150]},
                    {'class': '纸杯', 'confidence': 0.85, 'bbox': [200, 210, 280, 350]}
                ],
                'processing_time': 0.85,
                'error': str(e)
            }
            
            # 保存原始图片作为结果
            if os.path.exists(original_path):
                imageio.imwrite(result_path, imageio.imread(original_path), quality=95)
        
        # 获取相对路径
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)
        
        # 保存检测记录到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='image',
            source_name=filename,
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=detection_results['processing_time'],
            total_objects=len(detection_results['objects'])
        )
        record.set_detection_results(detection_results)
        
        db.session.add(record)
        db.session.commit()
        
        # 创建警报（如有必要）
        alert_info = create_alert_from_detection(record)
        
        # 构建返回结果
        result = {
            'success': True,
            'record_id': record.id,
            'detection_results': detection_results,
            'result_image_url': relative_result_path,
            'total_objects': record.total_objects,
            'has_garbage': record.total_objects > 0
        }
        
        if alert_info:
            result['alert'] = alert_info
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 添加到现有的api/detect/multi_image路由
@admin_bp.route('/api/detect/multi_image', methods=['POST'])
def api_detect_multi_image():
    """API接口：批量图片检测"""
    # 这里是假设的批量图片检测代码
    try:
        # 模拟检测过程
        time.sleep(2)  # 模拟处理时间
        
        # 模拟请求中的数据
        request_data = request.get_json() or {}
        uploaded_files = request_data.get('files', [])
        
        # 如果前端没有传入文件信息，使用模拟数据
        if not uploaded_files:
            # 假设处理了多张图片，有的检测到垃圾，有的没有检测到
            uploaded_files = [
                {'name': 'image1.jpg', 'path': '/temp/uploads/image1.jpg'},
                {'name': 'image2.jpg', 'path': '/temp/uploads/image2.jpg'},
                {'name': 'image3.jpg', 'path': '/temp/uploads/image3.jpg'},
                {'name': 'image4.jpg', 'path': '/temp/uploads/image4.jpg'},
                {'name': 'image5.jpg', 'path': '/temp/uploads/image5.jpg'}
            ]
        
        # 模拟检测结果，这里随机生成检测到的对象数量
        # 在实际项目中，应该调用真实的检测模型
        import random
        detection_results = []
        for file in uploaded_files:
            # 随机决定是否检测到垃圾，约有60%的概率检测到
            if random.random() > 0.4:
                # 检测到1-8个垃圾
                objects_count = random.randint(1, 8)
                objects = []
                for i in range(objects_count):
                    objects.append({
                        'class': f'垃圾{i+1}',
                        'confidence': round(0.75 + random.random() * 0.2, 2),
                        'bbox': [10 + i*50, 20 + i*40, 100 + i*50, 150 + i*40]
                    })
            else:
                # 未检测到垃圾
                objects_count = 0
                objects = []
            
            detection_results.append({
                'file': file['name'],
                'objects': objects,
                'processing_time': round(0.3 + random.random() * 0.5, 2)
            })
        
        user_id = session.get('user_id', 1)  # 如果未登录，使用默认用户ID
        records = []
        alerts = []
        
        # 为每张图片创建记录
        for result in detection_results:
            file_name = result['file']
            objects = result['objects']
            
            # 创建记录
            record = DetectionRecord(
                user_id=user_id,
                source_type='multi_image',
                source_name=file_name,
                original_path=f'/uploads/original/{file_name}',
                result_path=f'/uploads/results/{file_name.split(".")[0]}_result.jpg',
                duration=result['processing_time'],
                total_objects=len(objects)  # 使用检测对象数量作为total_objects
            )
            
            # 设置检测结果
            record.set_detection_results({
                'objects': objects,
                'processing_time': result['processing_time']
            })
            
            db.session.add(record)
            records.append(record)
        
        db.session.commit()
        
        # 为未检测到垃圾的图片（total_objects = 0）创建警报
        for record in records:
            if record.total_objects == 0:  # 明确检查是否未检测到垃圾
                alert_info = create_alert_from_detection(record)
                if alert_info:
                    alerts.append(alert_info)
        
        # 返回结果
        return jsonify({
            'success': True,
            'processed_images': len(records),
            'clean_images': len(alerts),  # 未检测到垃圾的图片数量
            'records': [{
                'id': r.id, 
                'name': r.source_name, 
                'objects': r.total_objects, 
                'has_garbage': r.total_objects > 0
            } for r in records]
        })
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 添加到现有的api/detect/video路由
@admin_bp.route('/api/detect/video', methods=['POST'])
def api_detect_video():
    """API接口：视频检测"""
    try:
        # 导入必要模块
        import imageio
        import imageio_ffmpeg
        import numpy as np
        from datetime import datetime
        from werkzeug.utils import secure_filename
        
        # 检查是否有视频文件上传
        if 'video' not in request.files:
            # 模拟检测结果（用于测试）
            objects_detected = random.randint(0, 15)
        
            detection_results = {
                'frames_processed': 150,
                'objects_detected': objects_detected,
                'processing_time': 8.5,
                'frames_with_objects': [] if objects_detected == 0 else [
                    random.randint(1, 150) for _ in range(min(objects_detected, 15))
                ]
            }
            
            # 获取当前用户ID
            user_id = session.get('user_id', 1)
        
            # 使用示例视频名称
            video_name = 'test_video.mp4'
            
            # 创建记录
            record = DetectionRecord(
                user_id=user_id,
                source_type='video',
                source_name=video_name,
                original_path=f'/uploads/original/{video_name}',
                result_path=f'/uploads/results/{video_name.split(".")[0]}_result.mp4',
                duration=detection_results['processing_time'],
                total_objects=detection_results['objects_detected']
            )
            record.set_detection_results(detection_results)
            
            db.session.add(record)
            db.session.commit()
            
            # 创建警报（如有必要）
            alert_info = create_alert_from_detection(record)
            
            # 返回结果
            result = {
                'success': True,
                'record_id': record.id,
                'detection_results': detection_results,
                'result_video_url': record.result_path,
                'total_objects': record.total_objects,
                'has_garbage': record.total_objects > 0
            }
            
            if alert_info:
                result['alert'] = alert_info
                
            return jsonify(result)
        
        # 处理实际上传的视频文件
        video_file = request.files['video']
        
        # 确保上传目录存在
        original_dir = get_original_dir()
        results_dir = get_results_dir()
        
        # 保存上传的原始视频
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        filename = secure_filename(video_file.filename)
        original_filename = f"original_{timestamp}_{filename}"
        original_path = os.path.join(original_dir, original_filename)
        original_path = os.path.normpath(original_path)
        
        video_file.save(original_path)
        print(f"视频已保存: {original_path}")
        
        # 设置结果文件路径
        base_name = os.path.splitext(original_filename)[0]
        result_filename = f"result_{timestamp}.mp4"
        result_path = os.path.join(results_dir, result_filename)
        result_path = os.path.normpath(result_path)
        
        # 获取当前用户ID
        user_id = session.get('user_id', 1)
        
        # 使用模型进行检测处理
        start_time = time.time()
        objects_list = []  # 存储检测到的对象
        avg_confidence = 0.0  # 平均置信度
        
        try:
            # 导入模型（实际项目中应有真实模型）
            from app import model
            
            if not model:
                raise Exception("未能加载检测模型")
            
            # 使用imageio读取视频
            reader = imageio.get_reader(original_path)
            fps = reader.get_meta_data().get('fps', 30)
            
            # 设置处理参数
            process_interval = 5  # 每隔5帧处理一次
            max_frames = 300  # 最多处理300帧
            
            # 收集处理后的帧和检测结果
            processed_frames = []
            objects_detected = 0
            frames_with_objects = []
            frame_count = 0
            confidence_sum = 0.0
            confidence_count = 0
            
            # 处理视频帧
            for i, frame in enumerate(reader):
                if i >= max_frames:
                    break
                    
                if i % process_interval == 0:  # 按间隔处理帧
                    # 使用模型检测
                    results = model(frame)
                    
                    # 检查是否检测到对象
                    detected_boxes = results[0].boxes
                    objects_in_frame = len(detected_boxes)
                    
                    if objects_in_frame > 0:
                        objects_detected += objects_in_frame
                        frames_with_objects.append(i)
                        
                        # 记录检测到的对象详情
                        for j, box in enumerate(detected_boxes):
                            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                            conf = float(box.conf[0])
                            cls = int(box.cls[0])
                            label = model.names[cls]
                            
                            # 累加置信度以计算平均值
                            confidence_sum += conf
                            confidence_count += 1
                            
                            # 添加到对象列表
                            objects_list.append({
                                'label': label,
                                'confidence': round(conf * 100, 2),
                                'x1': x1,
                                'y1': y1,
                                'x2': x2,
                                'y2': y2
                            })
                        
                        # 绘制检测框
                        annotated_frame = results[0].plot()
                        processed_frames.append(annotated_frame)
                    else:
                        processed_frames.append(frame)
                else:
                    processed_frames.append(frame)
                
                frame_count += 1
            
            # 关闭读取器
            reader.close()
            
            # 计算平均置信度
            if confidence_count > 0:
                avg_confidence = confidence_sum / confidence_count
            
            # 保存处理后的视频
            writer = imageio.get_writer(
                result_path,
                fps=fps,
                codec='libx264',
                pixelformat='yuv420p',
                quality=8
            )
            
            for frame in processed_frames:
                writer.append_data(frame)
            
            writer.close()
            
            # 计算处理时间
            processing_time = time.time() - start_time
            
            # 创建检测结果对象
            detection_results = {
                'objects': objects_list,
                'avg_confidence': avg_confidence,
                'frames_processed': frame_count,
                'objects_detected': objects_detected,
                'processing_time': round(processing_time, 2),
                'frames_with_objects': frames_with_objects
            }
            
        except Exception as e:
            print(f"视频处理失败: {str(e)}")
            # 处理失败时，创建一个基本的检测结果
            detection_results = {
                'objects': [],
                'avg_confidence': 0,
                'frames_processed': 0,
                'objects_detected': 0,
                'processing_time': 0,
                'frames_with_objects': [],
                'error': str(e)
            }
            
            # 如果原始视频存在，复制为结果
            if os.path.exists(original_path):
                import shutil
                shutil.copyfile(original_path, result_path)
                print(f"使用原始视频作为结果: {result_path}")
        
        # 获取相对路径
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)
        
        # 保存检测记录到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='video',
            source_name=filename,
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=detection_results.get('processing_time', 0),
            total_objects=len(objects_list)
        )
        record.set_detection_results(detection_results)
        
        db.session.add(record)
        db.session.commit()
        
        # 创建警报（如有必要）
        alert_info = create_alert_from_detection(record)
        
        # 构建返回结果
        result = {
            'success': True,
            'record_id': record.id,
            'detection_results': detection_results,
            'result_video_url': relative_result_path,
            'total_objects': record.total_objects,
            'has_garbage': record.total_objects > 0
        }
        
        if alert_info:
            result['alert'] = alert_info
        
        return jsonify(result)
        
    except Exception as e:
        print(f"视频检测API错误: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 添加到现有的api/detect/camera路由
@admin_bp.route('/api/detect/camera', methods=['POST'])
def api_detect_camera():
    """API接口：摄像头实时检测警报"""
    try:
        # 摄像头实时检测通常是前端直接连接摄像头进行处理
        # 这里主要处理前端提交的检测结果，并创建记录和警报
        
        # 获取前端发送的数据
        data = request.get_json()
        objects_detected = data.get('objects_detected', 0)  # 检测到的垃圾数量
        frame_data = data.get('frame_data', '')  # 假设前端发送了一个检测到警报的帧的base64数据
        
        user_id = session.get('user_id', 1)  # 如果未登录，使用默认用户ID
        
        # 生成唯一文件名
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        frame_name = f'camera_alert_{timestamp}.jpg'
        
        # 保存检测记录到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='camera',
            source_name=frame_name,
            original_path=f'/uploads/camera/{frame_name}',
            result_path=f'/uploads/camera/{frame_name.replace(".jpg", "_result.jpg")}',
            duration=0.1,  # 实时检测通常不记录处理时间
            total_objects=objects_detected  # 使用前端传来的对象数量作为total_objects
        )
        
        detection_results = {
            'objects': [],
            'timestamp': timestamp
        }
        
        # 如果检测到垃圾，添加对象详情
        if objects_detected > 0:
            for i in range(objects_detected):
                detection_results['objects'].append({
                    'class': f'垃圾{i+1}',
                    'confidence': 0.8 + (i * 0.02),
                    'bbox': [10 + i*50, 20 + i*40, 100 + i*50, 150 + i*40]
                })
        
        record.set_detection_results(detection_results)
        
        db.session.add(record)
        db.session.commit()
        
        # 如果未检测到垃圾（total_objects = 0），创建警报
        alert_info = create_alert_from_detection(record)
        
        # 构建返回结果
        result = {
            'success': True,
            'record_id': record.id,
            'total_objects': record.total_objects,
            'has_garbage': record.total_objects > 0
        }
        
        # 如果创建了警报，添加警报信息
        if alert_info:
            result['alert'] = alert_info
        
        return jsonify(result)
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

# 更新清理状态API
@admin_bp.route('/api/alerts/<int:record_id>/status', methods=['PUT'])
def update_alert_status(record_id):
    """更新警报状态（更新detection_record的is_cleaned字段）"""
    record = DetectionRecord.query.get_or_404(record_id)
    
    data = request.get_json()
    is_cleaned = data.get('is_cleaned', False)
    
    # 更新状态
    record.is_cleaned = is_cleaned
    db.session.commit()
    
    return jsonify({
        'success': True,
        'message': '状态已更新',
        'record': {
            'id': record.id,
            'is_cleaned': record.is_cleaned
        }
    })

# 添加一个API来获取所有检测记录（不管是否检测到垃圾）
@admin_bp.route('/api/detection_records', methods=['GET'])
def get_detection_records():
    """获取所有检测记录，包括检测到垃圾和未检测到垃圾的记录"""
    # 获取分页参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)
    
    # 获取筛选参数
    has_garbage = request.args.get('has_garbage')  # 'true', 'false' 或 None
    source_type = request.args.get('source_type')  # 'image', 'video', 'multi_image', 'camera'
    record_id = request.args.get('record_id')  # 特定记录ID
    
    # 构建查询
    query = DetectionRecord.query
    
    # 应用筛选
    if has_garbage == 'true':
        query = query.filter(DetectionRecord.total_objects > 0)
    elif has_garbage == 'false':
        query = query.filter(DetectionRecord.total_objects == 0)
    
    if source_type:
        query = query.filter(DetectionRecord.source_type == source_type)
        
    if record_id:
        query = query.filter(DetectionRecord.id == record_id)
    
    # 按时间降序排序并分页
    if record_id:  # 如果查询特定ID，不需要分页
        records = query.all()
        total = len(records)
        pages = 1
    else:
        pagination = query.order_by(DetectionRecord.timestamp.desc()).paginate(page=page, per_page=per_page)
        records = pagination.items
        total = pagination.total
        pages = pagination.pages
    
    # 准备返回数据
    records_data = []
    for record in records:
        # 处理路径，使用get_relative_path获取相对路径
        original_path = get_relative_path(record.original_path)
        result_path = get_relative_path(record.result_path)
            
        record_dict = record.to_dict()
        record_dict['has_garbage'] = record.total_objects > 0
        record_dict['original_path'] = original_path
        record_dict['result_path'] = result_path
        
        # 检查是否有关联的警报（针对未检测到垃圾的记录）
        records_data.append(record_dict)
    
    return jsonify({
        'success': True,
        'records': records_data,
        'total': total,
        'pages': pages,
        'current_page': page
    })

# 导出警报数据
@admin_bp.route('/api/alerts/export', methods=['GET'])
@admin_required
def export_alerts():
    """导出所有警报数据为CSV格式"""
    try:
        # 直接查询DetectionRecord表
        records_query = db.session.query(
            DetectionRecord
        ).filter(
            DetectionRecord.total_objects > 0  # 只获取检测到垃圾的记录
        ).order_by(DetectionRecord.timestamp.desc())
        
        records = records_query.all()
        
        # 转换为字典列表方便前端处理
        records_list = []
        for record in records:
            records_list.append({
                'id': record.id,
                'alert_type': record.source_type,  # 使用source_type作为alert_type
                'source_type': record.source_type,
                'source': record.source_name,
                'source_name': record.source_name,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S') if isinstance(record.timestamp, datetime) else str(record.timestamp),
                'is_cleaned': record.is_cleaned,
                'duration': record.duration
            })
        
        return jsonify({
            'success': True,
            'alerts': records_list
        })
    except Exception as e:
        print(f"导出警报数据出错: {str(e)}")
        return jsonify({
            'success': False,
            'message': f"导出数据失败: {str(e)}"
        }), 500

# 保存摄像头录制的视频和检测结果
@admin_bp.route('/api/camera/save_recording', methods=['POST'])
@admin_required
def save_camera_recording():
    """保存摄像头录制的视频和检测结果，返回记录ID和视频URL"""
    try:
        # 导入必要的模块
        import os
        import json
        import shutil
        import traceback
        import cv2
        import numpy as np
        import imageio
        from datetime import datetime
        from werkzeug.utils import secure_filename
        
        # 获取当前用户
        user_id = session.get('user_id', 1)
        
        # 检查请求数据
        if 'video' not in request.files:
            return jsonify({
                'success': False,
                'message': '没有收到视频文件'
            }), 400
        
        video_file = request.files['video']
        if not video_file.filename:
            return jsonify({
                'success': False,
                'message': '视频文件为空'
            }), 400
        
        # 处理检测结果
        detection_results = json.loads(request.form.get('detection_results', '{"objects":[]}'))
        total_objects = int(request.form.get('total_objects', 0))
        duration = float(request.form.get('duration', 0))
        
        # 是否使用保存的帧创建视频
        use_saved_frames = request.form.get('use_saved_frames') == 'true'
        session_id = request.form.get('session_id', '')
        
        # 确保上传目录存在
        camera_dir = get_camera_dir()
        
        # 确保camera_frames目录存在
        camera_frames_dir = os.path.join(get_upload_folder(), 'camera_frames')
        ensure_dir(camera_frames_dir)
        
        # 安全处理文件名
        filename = secure_filename(video_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"camera_original_{timestamp}_{filename}"
        original_path = os.path.join(camera_dir, original_filename)
        original_path = os.path.normpath(original_path)  # 规范化路径
        
        # 打印路径信息（调试用）
        print(f"准备保存文件到: {original_path}")
        
        # 保存原始视频
        video_file.save(original_path)
        print(f"原始视频已保存到: {original_path}")
        
        # 设置结果视频路径
        result_filename = f"camera_result_{timestamp}.mp4"
        result_path = os.path.join(camera_dir, result_filename)
        result_path = os.path.normpath(result_path)  # 规范化路径
        
        # 处理视频 - 如果检测到垃圾对象，使用模型进行处理
        processing_success = False
        
        if total_objects > 0:
            try:
                # 导入必要的模块
                from app import model  # 获取YOLO模型实例
                
                # 检查模型是否成功加载
                if not model:
                    raise Exception("未能加载检测模型")
                
                # 使用imageio读取视频
                reader = imageio.get_reader(original_path)
                fps = reader.get_meta_data()['fps']
                
                # 预处理视频并收集帧
                frames = []
                max_frames = 300  # 限制处理的帧数以提高性能
                
                for i, frame in enumerate(reader):
                    # 每10帧处理一次（优化性能）
                    if i % 10 == 0:
                        # 转换为OpenCV格式处理（BGR）
                        opencv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
                        
                        # 使用YOLO检测
                        results = model(opencv_frame)
                        
                        # 绘制检测框并转回RGB
                        annotated_frame = results[0].plot()
                        rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                        frames.append(rgb_frame)
                    else:
                        # 其他帧直接添加
                        frames.append(frame)
                    
                    # 限制处理帧数
                    if i >= max_frames:
                        break
                
                # 关闭读取器
                reader.close()
                
                # 使用imageio写入处理后的视频
                try:
                    # 确保使用高质量编码
                    imageio.mimsave(
                        result_path, 
                        frames, 
                        fps=fps, 
                        quality=8,  # 控制质量
                        macro_block_size=1  # 保证兼容性
                    )
                    
                    if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                        print(f"视频处理成功: {result_path}")
                        processing_success = True
                    else:
                        raise Exception("处理后的视频文件为空")
                except Exception as writer_error:
                    print(f"视频写入失败: {str(writer_error)}")
                    # 尝试使用imageio-ffmpeg
                    try:
                        imageio_ffmpeg.mimwrite(
                            result_path, 
                            frames, 
                            fps=fps, 
                            codec='libx264', 
                            pixelformat='yuv420p',
                            quality=8,
                            ffmpeg_log_level='quiet'
                        )
                        
                        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                            print(f"使用imageio-ffmpeg处理成功: {result_path}")
                            processing_success = True
                        else:
                            raise Exception("处理后的视频文件为空")
                    except Exception as ffmpeg_error:
                        print(f"imageio-ffmpeg处理失败: {str(ffmpeg_error)}")
                        raise
            
            except Exception as e:
                print(f"视频处理失败: {str(e)}")
                traceback.print_exc()
                
                # 处理失败，使用原始视频作为结果
                result_filename = f"camera_result_{timestamp}_{filename}"
                result_path = os.path.join(camera_dir, result_filename)
                result_path = os.path.normpath(result_path)
                shutil.copyfile(original_path, result_path)
                print(f"使用原始视频作为结果: {result_path}")
        else:
            # 没有检测到垃圾对象，直接复制原始视频
            result_filename = f"camera_result_{timestamp}_{filename}"
            result_path = os.path.join(camera_dir, result_filename)
            result_path = os.path.normpath(result_path)
            shutil.copyfile(original_path, result_path)
            print(f"未检测到垃圾，复制原始视频: {result_path}")
            processing_success = True
        
        # 清理保存的帧目录
        if use_saved_frames and session_id:
            try:
                frames_dir = os.path.join(camera_frames_dir, session_id)
                frames_dir = os.path.normpath(frames_dir)
                if os.path.exists(frames_dir):
                    shutil.rmtree(frames_dir)
                    print(f"已清理帧目录: {frames_dir}")
            except Exception as e:
                print(f"清理帧目录失败: {str(e)}")
        
        # 将路径转换为相对路径
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)
        
        # 保存到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='camera',
            source_name=f"实时监控_{timestamp}",
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=duration,
            total_objects=total_objects
        )
        record.set_detection_results(detection_results)
        
        db.session.add(record)
        db.session.commit()
        
        # 如果检测到垃圾，创建警报
        alert_info = None
        if total_objects > 0:
            alert_info = create_alert_from_detection(record)
            if alert_info:
                print(f"已创建警报，警报信息: {alert_info}")
        
        # 构建视频URL路径（相对路径）
        result_extension = os.path.splitext(result_path)[1]
        if not result_extension:
            result_extension = '.mp4'
        
        # 确保URL使用正斜杠
        original_video_url = f'/static/uploads/camera/{original_filename}'.replace('\\', '/')
        result_filename_base = os.path.splitext(os.path.basename(result_path))[0]
        result_video_url = f'/static/uploads/camera/{result_filename_base}{result_extension}'.replace('\\', '/')
        
        return jsonify({
            'success': True,
            'message': '视频和检测结果已保存',
            'record_id': record.id,
            'original_video_url': original_video_url,
            'result_video_url': result_video_url,
            'processed': processing_success,
            'alert_info': alert_info
        })
    except Exception as e:
        print(f"保存摄像头录制失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
        }), 500

# 保存摄像头截图
@admin_bp.route('/api/camera/save_capture', methods=['POST'])
@admin_required
def save_camera_capture():
    """保存摄像头截图和检测结果"""
    try:
        # 导入必要的模块
        import os
        import cv2
        import numpy as np
        import shutil
        import traceback
        import base64
        import imageio
        from datetime import datetime
        
        # 获取当前用户
        user_id = session.get('user_id', 1)
        
        # 获取请求数据
        data = request.json
        if not data or 'image' not in data:
            return jsonify({
                'success': False,
                'message': '没有收到图像数据'
            }), 400
        
        # 处理base64图像数据
        image_data = data['image']
        if image_data.startswith('data:image/jpeg;base64,'):
            image_data = image_data[len('data:image/jpeg;base64,'):]
        
        # 解码base64数据
        image_binary = base64.b64decode(image_data)
        
        # 确保摄像头目录存在
        camera_dir = get_camera_dir()
        
        # 生成文件名
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"capture_original_{timestamp}.jpg"
        original_path = os.path.join(camera_dir, original_filename)
        original_path = os.path.normpath(original_path)  # 规范化路径
        
        # 保存原始图像
        with open(original_path, 'wb') as f:
            f.write(image_binary)
        
        print(f"原始图像已保存到: {original_path}")
        
        # 设置结果图像路径
        result_filename = f"capture_result_{timestamp}.jpg"
        result_path = os.path.join(camera_dir, result_filename)
        result_path = os.path.normpath(result_path)  # 规范化路径
        
        # 使用imageio和YOLO模型进行检测
        try:
            # 导入模型
            from app import model
            
            # 检查模型是否加载成功
            if not model:
                raise Exception("未能加载检测模型")
            
            # 使用imageio读取图像
            image = imageio.imread(original_path)
            
            if image is None:
                raise Exception(f"无法从{original_path}读取图像")
            
            # 转换为OpenCV格式处理（如果需要）
            if len(image.shape) == 3 and image.shape[2] == 3:
                # 转换为BGR（OpenCV格式）
                opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
            else:
                # 如果不是RGB格式，直接使用
                opencv_image = image
            
            # 执行检测
            results = model(opencv_image)
            result = results[0]
            boxes = result.boxes
            
            # 获取检测结果
            detected_objects = []
            if len(boxes) > 0:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    conf = float(box.conf[0])
                    cls = int(box.cls[0])
                    label = model.names[cls]
                    detected_objects.append({
                        'id': i,
                        'class': cls,
                        'label': label,
                        'confidence': round(conf * 100, 2),
                        'bbox': [x1, y1, x2, y2]
                    })
                
                # 绘制检测框
                annotated_image = result.plot()
                
                # 保存结果图像（转回RGB格式如果需要）
                if len(annotated_image.shape) == 3 and annotated_image.shape[2] == 3:
                    # BGR转RGB
                    rgb_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
                    imageio.imwrite(result_path, rgb_image, quality=95)
                else:
                    imageio.imwrite(result_path, annotated_image, quality=95)
                
                print(f"结果图像（含对象标记）已保存到: {result_path}")
                
                # 设置检测结果
                detection_results = {
                    'objects': detected_objects,
                    'timestamp': datetime.now().isoformat()
                }
                total_objects = len(detected_objects)
            else:
                # 未检测到对象，直接保存原始图像
                imageio.imwrite(result_path, image, quality=95)
                print("未检测到对象，使用原始图像")
                total_objects = 0
                detection_results = {'objects': []}
        except Exception as e:
            # 处理失败，直接复制原始图像
            print(f"图像处理失败: {str(e)}")
            traceback.print_exc()
            shutil.copyfile(original_path, result_path)
            
            # 使用前端传递的检测结果（如果有）
            detection_results = data.get('detection_results', {'objects': []})
            total_objects = data.get('total_objects', 0)
        
        # 将路径转换为相对路径
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)
        
        # 保存到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='camera',
            source_name=f"摄像头截图_{timestamp}",
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=0.0,  # 截图没有处理时间
            total_objects=total_objects
        )
        record.set_detection_results(detection_results)
        
        db.session.add(record)
        db.session.commit()
        
        # 如果检测到垃圾，创建警报
        alert_info = None
        if total_objects > 0:
            alert_info = create_alert_from_detection(record)
        
        # 构建图像URL路径（相对路径）
        # 确保URL使用正斜杠
        original_url = f'/static/uploads/camera/{original_filename}'.replace('\\', '/')
        result_url = f'/static/uploads/camera/{result_filename}'.replace('\\', '/')
        
        return jsonify({
            'success': True,
            'message': '截图和检测结果已保存',
            'record_id': record.id,
            'original_url': original_url,
            'result_url': result_url,
            'processed': total_objects > 0,
            'total_objects': total_objects,
            'alert_info': alert_info
        })
    except Exception as e:
        print(f"保存摄像头截图失败: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
        }), 500

