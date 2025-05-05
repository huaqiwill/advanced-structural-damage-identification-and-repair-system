from flask import Blueprint, request, jsonify, session
from werkzeug.security import generate_password_hash, check_password_hash
from models import db, User, DetectionRecord
from datetime import datetime, timedelta
import time
import os
import json

# 创建Blueprint
api_bp = Blueprint('api', __name__, url_prefix='/api')

# 辅助函数获取上传目录
def get_upload_folder():
    """获取上传文件的根目录"""
    # 使用绝对路径确保正确的目录位置
    current_file = os.path.abspath(__file__)  # api.py的绝对路径
    current_dir = os.path.dirname(current_file)  # api.py所在目录
    base_dir = os.path.dirname(current_dir)  # 项目根目录
    upload_folder = os.path.join(base_dir, 'static', 'uploads')
    print(f"API Upload folder path: {upload_folder}")  # 调试信息
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
    
    # 使用os.path.relpath获取相对路径
    if absolute_path and os.path.exists(absolute_path):
        try:
            # 首先检查是否已经是相对于static的路径
            if '/static/' in absolute_path or '\\static\\' in absolute_path:
                for part in absolute_path.replace('\\', '/').split('/'):
                    if part == 'static':
                        rel_path_parts = absolute_path.replace('\\', '/').split('/static/')
                        if len(rel_path_parts) > 1:
                            return '/static/' + rel_path_parts[1]
            
            # 否则尝试获取相对路径
            rel_path = os.path.relpath(absolute_path, static_dir)
            # 避免路径中包含上级目录符号，如 '../'
            if not rel_path.startswith('..'):
                return '/static/' + rel_path
        except:
            pass
    
    # 如果无法获取相对路径，则返回原始路径
    return absolute_path

# 获取所有用户
@api_bp.route('/users', methods=['GET'])
def get_users():
    try:
        users = User.query.all()
        result = []
        for user in users:
            # 查询用户最后登录时间
            last_login = "未登录"
            # 查询用户最后的检测记录
            latest_record = DetectionRecord.query.filter_by(
                user_id=user.id).order_by(DetectionRecord.timestamp.desc()).first()
            if latest_record:
                last_activity = latest_record.timestamp.strftime(
                    '%Y-%m-%d %H:%M:%S')
            else:
                last_activity = "无活动"

            # 特殊处理：admin用户当做管理员
            is_admin = (user.username == 'admin')

            result.append({
                'id': user.id,
                'username': user.username,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'last_login': last_login,
                'last_activity': last_activity,
                'is_admin': is_admin
            })
        return jsonify({'success': True, 'data': result})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 添加用户
@api_bp.route('/users', methods=['POST'])
def add_user():
    try:
        data = request.json
        username = data.get('username')
        password = data.get('password')
        
        # 验证数据
        if not username or not password:
            return jsonify({'success': False, 'message': '用户名和密码不能为空'})

        # 检查用户名是否已存在
        existing_user = User.query.filter_by(username=username).first()
        if existing_user:
            return jsonify({'success': False, 'message': '用户名已存在'})

        # 创建新用户
        new_user = User(
            username=username,
            password=generate_password_hash(password)
        )

        db.session.add(new_user)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户添加成功',
            'data': {
                'id': new_user.id,
                'username': new_user.username,
                'created_at': new_user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'is_admin': (new_user.username == 'admin')
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

# 获取单个用户
@api_bp.route('/users/<int:user_id>', methods=['GET'])
def get_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'})

        # 查询用户的检测记录数量
        record_count = DetectionRecord.query.filter_by(user_id=user.id).count()
        
        # 查询用户最后的检测记录
        latest_record = DetectionRecord.query.filter_by(
            user_id=user.id).order_by(DetectionRecord.timestamp.desc()).first()
        last_activity = latest_record.timestamp.strftime('%Y-%m-%d %H:%M:%S') if latest_record else "无活动"

        # 特殊处理：admin用户当做管理员
        is_admin = (user.username == 'admin')

        return jsonify({
            'success': True,
            'data': {
                'id': user.id,
                'username': user.username,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'record_count': record_count,
                'last_activity': last_activity,
                'is_admin': is_admin
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 更新用户信息
@api_bp.route('/users/<int:user_id>', methods=['PUT'])
def update_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'})

        data = request.json

        # 更新用户名
        if 'username' in data and data['username'] != user.username:
            # 检查是否与其他用户名冲突
            existing_user = User.query.filter_by(
                username=data['username']).first()
            if existing_user and existing_user.id != user_id:
                return jsonify({'success': False, 'message': '用户名已存在'})
            user.username = data['username']

        # 更新密码
        if 'password' in data and data['password']:
            user.password = generate_password_hash(data['password'])

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '用户信息更新成功',
            'data': {
                'id': user.id,
                'username': user.username,
                'created_at': user.created_at.strftime('%Y-%m-%d %H:%M:%S'),
                'is_admin': (user.username == 'admin')
            }
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

# 删除用户
@api_bp.route('/users/<int:user_id>', methods=['DELETE'])
def delete_user(user_id):
    try:
        user = User.query.get(user_id)
        if not user:
            return jsonify({'success': False, 'message': '用户不存在'})

        # 删除用户关联的检测记录
        DetectionRecord.query.filter_by(user_id=user.id).delete()
        db.session.delete(user)
        db.session.commit()

        return jsonify({'success': True, 'message': '用户删除成功'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'message': str(e)})

# 用户统计
@api_bp.route('/users/statistics', methods=['GET'])
def user_statistics():
    try:
        # 用户总数
        total_users = User.query.count()

        # 今日新增用户
        today = datetime.now().date()
        new_users_today = User.query.filter(
            db.func.date(User.created_at) == today).count()

        # 活跃用户数（有检测记录的用户）
        active_users = db.session.query(
            DetectionRecord.user_id).distinct().count()

        # 用户类型统计
        admin_count = User.query.filter_by(username='admin').count()

        return jsonify({
            'success': True,
            'data': {
                'total_users': total_users,
                'new_users_today': new_users_today,
                'active_users': active_users,
                'admin_count': admin_count,
                'regular_users': total_users - admin_count
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 实时摄像头帧处理API
@api_bp.route('/detect/camera_frame', methods=['POST'])
def detect_camera_frame():
    """处理摄像头单帧图像"""
    try:
        # 获取BASE64编码的图像
        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': '未接收到图像数据'})
            
        image_data = data['image']
        if not image_data.startswith('data:image'):
            return jsonify({'success': False, 'message': '图像数据格式不正确'})
            
        # 获取检测阈值
        confidence = float(data.get('confidence', 0.5))
        # 是否保存帧
        save_frames = data.get('save_frames', False)
        # 检测会话ID
        session_id = data.get('session_id', '')
            
        # 解码BASE64图像
        import base64
        import numpy as np
        import cv2
        import os
        from config import Config
        from datetime import datetime
        import time
        from ultralytics import YOLO
        
        # 获取YOLO模型实例
        from app import model
        
        # 解码图像
        image_b64 = image_data.split(',')[1]
        binary = base64.b64decode(image_b64)
        image = np.asarray(bytearray(binary), dtype="uint8")
        image = cv2.imdecode(image, cv2.IMREAD_COLOR)
        
        # 开始计时
        start_time = time.time()
        
        # 确保上传目录存在
        upload_dir = get_upload_folder()
        
        # 图像临时保存
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        temp_path = os.path.join(upload_dir, f'temp_frame_{timestamp}.jpg')
        cv2.imwrite(temp_path, image)
        
        # 执行检测（直接使用内存中的图像，不通过文件）
        results = model(image)[0]
        detection_time = time.time() - start_time
        
        # 处理检测结果
        objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            
            # 如果置信度低于阈值，跳过
            if conf < confidence:
                continue
                
            objects.append({
                'label': Config.CH_NAMES[cls],
                'confidence': round(conf * 100, 2),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })
        
        # 绘制结果
        result_img = results.plot()
        
        # 如果需要保存帧且有会话ID
        if save_frames and session_id:
            # 确保camera_frames目录存在
            camera_frames_dir = os.path.join(get_upload_folder(), 'camera_frames')
            ensure_dir(camera_frames_dir)
            
            # 创建会话专用目录
            frames_dir = os.path.join(camera_frames_dir, session_id)
            if not os.path.exists(frames_dir):
                os.makedirs(frames_dir, exist_ok=True)
                
            # 生成帧文件名（使用递增序号确保正确排序）
            frame_count_file = os.path.join(frames_dir, 'frame_count.txt')
            frame_count = 0
            
            # 读取当前帧计数
            if os.path.exists(frame_count_file):
                with open(frame_count_file, 'r') as f:
                    frame_count = int(f.read().strip() or '0')
                    
            # 递增帧计数
            frame_count += 1
            
            # 保存当前帧计数
            with open(frame_count_file, 'w') as f:
                f.write(str(frame_count))
                
            # 保存处理后的帧，使用6位数字作为文件名前缀确保正确排序
            frame_filename = f'{frame_count:06d}.jpg'
            frame_path = os.path.join(frames_dir, frame_filename)
            cv2.imwrite(frame_path, result_img)
            
            # 打印日志
            print(f"已保存处理后的帧: {frame_path}, 帧编号: {frame_count}")
        
        # 将结果图像转为BASE64
        _, buffer = cv2.imencode('.jpg', result_img)
        result_b64 = base64.b64encode(buffer).decode('utf-8')
        
        # 返回结果
        return jsonify({
            'success': True,
            'detection_time': round(detection_time, 3),
            'fps': round(1.0 / detection_time, 1),
            'objects': objects,
            'result_image': f'data:image/jpeg;base64,{result_b64}'
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

# 控制面板统计数据API
@api_bp.route('/dashboard/statistics', methods=['GET'])
def dashboard_statistics():
    """获取控制面板的统计数据"""
    try:
        from datetime import datetime, timedelta
        from sqlalchemy import func, and_, or_
        from config import Config
        
        # 获取统计时间范围（默认最近7天）
        days = int(request.args.get('days', 7))
        end_date = datetime.now()
        start_date = end_date - timedelta(days=days)
        
        # 获取总检测次数
        total_detections = DetectionRecord.query.count()
        
        # 获取各类型检测计数
        image_detections = DetectionRecord.query.filter(
            DetectionRecord.source_type.in_(['image', 'multi_image'])
        ).count()
        
        video_detections = DetectionRecord.query.filter(
            DetectionRecord.source_type == 'video'
        ).count()
        
        camera_detections = DetectionRecord.query.filter(
            DetectionRecord.source_type == 'camera'
        ).count()
        
        # 计算水面垃圾检测次数（含有检测到的物体的记录数）
        garbage_detections = DetectionRecord.query.filter(
            DetectionRecord.total_objects > 0
        ).count()
        
        # 获取每日检测统计
        daily_stats = []
        date_labels = []
        
        # 生成日期范围
        current_date = start_date.date()
        end_date_only = end_date.date()
        
        while current_date <= end_date_only:
            date_str = current_date.strftime('%m-%d')
            date_labels.append(date_str)
            
            # 当天开始和结束时间
            day_start = datetime.combine(current_date, datetime.min.time())
            day_end = datetime.combine(current_date, datetime.max.time())
            
            # 当天总检测数
            day_total = DetectionRecord.query.filter(
                DetectionRecord.timestamp.between(day_start, day_end)
            ).count()
            
            # 当天检测到物体的数量
            day_garbage = DetectionRecord.query.filter(
                and_(
                    DetectionRecord.timestamp.between(day_start, day_end),
                    DetectionRecord.total_objects > 0
                )
            ).count()
            
            daily_stats.append({
                'date': date_str,
                'total': day_total,
                'garbage': day_garbage
            })
            
            # 前进到下一天
            current_date += timedelta(days=1)
        
        # 统计文件类型分布
        file_extensions = {}
        records = DetectionRecord.query.all()
        
        for record in records:
            if record.source_name:
                ext = record.source_name.split('.')[-1].lower() if '.' in record.source_name else 'unknown'
                
                # 判断文件类型
                if ext in Config.ALLOWED_IMAGE_EXTENSIONS:
                    file_type = 'image'
                elif ext in Config.ALLOWED_VIDEO_EXTENSIONS:
                    file_type = 'video'
                else:
                    file_type = 'other'
                
                if file_type not in file_extensions:
                    file_extensions[file_type] = 0
                file_extensions[file_type] += 1
        
        # 获取最近的5条检测记录
        recent_records = []
        latest_records = DetectionRecord.query.order_by(
            DetectionRecord.timestamp.desc()
        ).limit(5).all()
        
        for record in latest_records:
            # 获取用户信息
            user = User.query.get(record.user_id)
            username = user.username if user else "未知用户"
            
            # 解析检测结果
            detection_results = {}
            if record.detection_results:
                try:
                    detection_results = json.loads(record.detection_results)
                except:
                    pass
            
            # 判断是否检测到物体
            has_objects = record.total_objects > 0
            
            recent_records.append({
                'id': record.id,
                'user': username,
                'source_type': record.source_type,
                'source_name': record.source_name,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'duration': round(record.duration, 3) if record.duration else 0,
                'total_objects': record.total_objects,
                'has_objects': has_objects
            })
        
        # 构建响应数据
        response_data = {
            'summary': {
                'total_detections': total_detections,
                'garbage_detections': garbage_detections,
                'image_detections': image_detections,
                'video_detections': video_detections,
                'camera_detections': camera_detections
            },
            'trend': {
                'labels': date_labels,
                'total': [stat['total'] for stat in daily_stats],
                'garbage': [stat['garbage'] for stat in daily_stats]
            },
            'distribution': {
                'file_types': file_extensions
            },
            'recent_records': recent_records
        }
        
        return jsonify({
            'success': True,
            'data': response_data
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

# 获取检测记录列表
@api_bp.route('/detection_records', methods=['GET'])
def get_detection_records():
    """获取检测记录列表，支持分页和筛选"""
    try:
        # 分页参数
        page = int(request.args.get('page', 1))
        per_page = int(request.args.get('per_page', 10))
        
        # 筛选参数
        source_type = request.args.get('source_type')
        has_objects = request.args.get('has_objects')
        search_term = request.args.get('search')
        
        # 构建查询
        query = DetectionRecord.query
        
        # 应用筛选条件
        if source_type and source_type != '所有类型':
            query = query.filter(DetectionRecord.source_type == source_type)
            
        if has_objects is not None:
            if has_objects == 'true':
                query = query.filter(DetectionRecord.total_objects > 0)
            elif has_objects == 'false':
                query = query.filter(DetectionRecord.total_objects == 0)
                
        if search_term:
            query = query.filter(DetectionRecord.source_name.ilike(f'%{search_term}%'))
        
        # 按时间降序排序
        query = query.order_by(DetectionRecord.timestamp.desc())
        
        # 执行分页查询
        total_records = query.count()
        records = query.paginate(page=page, per_page=per_page, error_out=False)
        
        # 处理结果
        results = []
        for record in records.items:
            user = User.query.get(record.user_id)
            username = user.username if user else "未知用户"
            
            # 获取有用的元数据
            metadata = {}
            if record.detection_results:
                try:
                    metadata = json.loads(record.detection_results)
                except:
                    pass
            
            # 确保路径是前端可访问的路径
            original_path = record.original_path
            result_path = record.result_path
            
            # 转换为相对路径
            if original_path and not original_path.startswith('/static'):
                original_path = '/static/uploads/' + os.path.basename(original_path)
            
            if result_path and not result_path.startswith('/static'):
                result_path = '/static/uploads/results/' + os.path.basename(result_path)
            
            # 构建记录数据
            result = {
                'id': record.id,
                'user_id': record.user_id,
                'username': username,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'source_type': record.source_type,
                'source_name': record.source_name,
                'original_path': original_path,
                'result_path': result_path,
                'duration': round(record.duration, 3) if record.duration else 0,
                'total_objects': record.total_objects,
                'is_cleaned': record.is_cleaned,
                'has_objects': record.total_objects > 0,
                'metadata': metadata
            }
            results.append(result)
        
        # 计算分页信息
        total_pages = (total_records + per_page - 1) // per_page
        has_prev = page > 1
        has_next = page < total_pages
        
        # 返回结果
        return jsonify({
            'success': True,
            'data': {
                'records': results,
                'pagination': {
                    'page': page,
                    'per_page': per_page,
                    'total_records': total_records,
                    'total_pages': total_pages,
                    'has_prev': has_prev,
                    'has_next': has_next
                }
            }
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

# 获取单条检测记录详情
@api_bp.route('/detection_records/<int:record_id>', methods=['GET'])
def get_detection_record(record_id):
    """获取单条检测记录详情"""
    try:
        record = DetectionRecord.query.get_or_404(record_id)
        
        # 获取用户信息
        user = User.query.get(record.user_id)
        username = user.username if user else "未知用户"
        
        # 解析检测结果
        detection_results = {}
        if record.detection_results:
            try:
                detection_results = json.loads(record.detection_results)
            except:
                pass
        
        # 判断是否为视频文件
        is_video = False
        if record.source_name:
            video_extensions = ['mp4', 'webm', 'ogg', 'mov', 'avi', 'wmv', 'flv', 'mkv']
            ext = record.source_name.split('.')[-1].lower() if '.' in record.source_name else ''
            is_video = ext in video_extensions
        
        # 构建URL
        original_url = None
        result_url = None
        
        if record.original_path:
            if record.original_path.startswith('/static'):
                original_url = record.original_path
            else:
                filename = os.path.basename(record.original_path)
                original_url = f'/static/uploads/original/{filename}'
        
        if record.result_path:
            if record.result_path.startswith('/static'):
                result_url = record.result_path
            else:
                result_filename = os.path.basename(record.result_path)
                if is_video:
                    # 视频结果可能位于不同目录
                    result_url = f'/static/uploads/results/{result_filename}'
                else:
                    result_url = f'/static/uploads/results/{result_filename}'
        
        # 构建记录详情
        result = {
            'id': record.id,
            'user_id': record.user_id,
            'username': username,
            'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'source_type': record.source_type,
            'source_name': record.source_name,
            'original_path': record.original_path,
            'result_path': record.result_path,
            'duration': round(record.duration, 3) if record.duration else 0,
            'total_objects': record.total_objects,
            'is_cleaned': record.is_cleaned,
            'detection_results': detection_results,
            'has_objects': record.total_objects > 0,
            'is_video': is_video,
            
            # 添加图像URL，前端可以直接访问
            'original_url': original_url,
            'result_url': result_url
        }
        
        return jsonify({
            'success': True,
            'data': result
        })
        
    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return jsonify({'success': False, 'message': str(e)})

# 警报统计数据
@api_bp.route('/alerts/statistics', methods=['GET'])
def get_alert_statistics():
    """获取警报统计信息"""
    try:
        # 获取所有检测到垃圾的记录（total_objects>0的记录）
        total_alerts = DetectionRecord.query.filter(DetectionRecord.total_objects > 0).count()
        
        # 今日警报数量
        today = datetime.now().date()
        today_alerts = DetectionRecord.query.filter(
            DetectionRecord.total_objects > 0
        ).filter(
            db.func.date(DetectionRecord.timestamp) == today
        ).count()
        
        # 按检测类型统计警报数量
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
        
        camera_count = DetectionRecord.query.filter(
            DetectionRecord.total_objects > 0, 
            DetectionRecord.source_type == 'camera'
        ).count()
        
        # 获取最近7天的每日警报数据
        end_date = datetime.now().date()
        start_date = end_date - timedelta(days=7)
        
        daily_stats_query = db.session.query(
            db.func.date(DetectionRecord.timestamp).label('date'), 
            db.func.count(DetectionRecord.id).label('count')
        ).filter(
            DetectionRecord.total_objects > 0,
            db.func.date(DetectionRecord.timestamp) >= start_date,
            db.func.date(DetectionRecord.timestamp) <= end_date
        ).group_by(
            db.func.date(DetectionRecord.timestamp)
        ).order_by(
            db.func.date(DetectionRecord.timestamp)
        ).all()
        
        daily_stats = [{'date': str(item.date), 'count': item.count} for item in daily_stats_query]
        
        return jsonify({
            'success': True,
            'data': {
                'total_alerts': total_alerts,
                'today_alerts': today_alerts,
                'image_count': image_count,
                'video_count': video_count,
                'multi_image_count': multi_image_count,
                'camera_count': camera_count,
                'daily_stats': daily_stats
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 获取警报列表
@api_bp.route('/alerts', methods=['GET'])
def get_alerts():
    """获取检测到垃圾的警报列表"""
    try:
        # 获取分页参数
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 10, type=int)
        
        # 查询所有total_objects>0的记录（有垃圾记录）
        query = DetectionRecord.query.filter(
            DetectionRecord.total_objects > 0
        ).order_by(
            DetectionRecord.timestamp.desc()
        )
        
        # 分页查询
        pagination = query.paginate(page=page, per_page=per_page)
        
        alerts = []
        for record in pagination.items:
            # 处理原始图像和结果图像的路径
            original_path = record.original_path
            if original_path and not original_path.startswith('/static'):
                original_path = f'/static/uploads/{os.path.basename(original_path)}'
                
            result_path = record.result_path
            if result_path and not result_path.startswith('/static'):
                result_path = f'/static/uploads/results/{os.path.basename(result_path)}'
            
            # 根据垃圾数量确定风险等级
            garbage_count = record.total_objects
            if garbage_count >= 5:
                risk_level = '高风险'
            elif garbage_count >= 2:
                risk_level = '中风险'
            else:
                risk_level = '低风险'
            
            alerts.append({
                'id': record.id,
                'alert_type': record.source_type,
                'source': record.source_name,
                'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                'status': '已处理' if record.is_cleaned else '待处理',
                'original_path': original_path,
                'result_path': result_path,
                'garbage_count': garbage_count,
                'risk_level': risk_level
            })
        
        return jsonify({
            'success': True,
            'data': {
                'alerts': alerts,
                'current_page': page,
                'pages': pagination.pages,
                'total': pagination.total
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})

# 获取指定警报记录详情
@api_bp.route('/detection_records', methods=['GET'])
def get_detection_record_detail():
    """获取检测记录详情"""
    try:
        record_id = request.args.get('record_id', type=int)
        
        if not record_id:
            return jsonify({'success': False, 'message': '记录ID不能为空'})
            
        record = DetectionRecord.query.get(record_id)
        if not record:
            return jsonify({'success': False, 'message': '记录不存在'})
            
        # 构建记录详情数据
        records = [{
            'id': record.id,
            'user_id': record.user_id,
            'source_type': record.source_type,
            'source_name': record.source_name,
            'timestamp': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'duration': record.duration,
            'total_objects': record.total_objects,
            'is_cleaned': record.is_cleaned,
            'detection_results': json.loads(record.detection_results) if record.detection_results else None
        }]
        
        return jsonify({
            'success': True,
            'data': {
                'records': records
            }
        })
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})
