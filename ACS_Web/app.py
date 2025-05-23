import traceback

from flask import Flask, render_template, request, redirect, url_for, flash, session, jsonify, Response, \
    send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import cv2
import numpy as np
from ultralytics import YOLO
import time
import json
from datetime import datetime, timedelta
from config import Config
from models import db, User, DetectionRecord, ScientificContent, ScientificFAQ, ScientificQuestion
from collections import Counter, defaultdict
import base64
import sqlite3
import csv
from io import StringIO
from api import api_bp  # 导入API Blueprint
from admin import admin_bp, create_alert_from_detection  # 导入Admin Blueprint和alert创建函数
import logging
import imageio
from io import BytesIO

# 配置日志
logging.basicConfig()
# 设置SQLAlchemy的logger为DEBUG级别，输出SQL语句
logging.getLogger('sqlalchemy.engine').setLevel(logging.INFO)

# 创建Flask应用
app = Flask(__name__)
app.config.from_object(Config)
Config.init_app(app)

# 初始化扩展
CORS(app)
db.init_app(app)
socketio = SocketIO(app, cors_allowed_origins="*")
app.register_blueprint(api_bp)  # 注册API Blueprint
app.register_blueprint(admin_bp)  # 注册Admin Blueprint


# 辅助函数获取上传目录
def get_upload_folder():
    """获取上传文件的根目录"""
    # 使用相对于当前文件的路径而非绝对路径
    current_dir = os.path.dirname(os.path.abspath(__file__))
    upload_folder = os.path.join(current_dir, 'static', 'uploads')
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


# 获取相对路径
def get_relative_path(absolute_path):
    """将绝对路径转换为相对于static的路径"""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    static_dir = os.path.join(current_dir, 'static')

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


# 加载YOLOv8模型
model = YOLO(Config.MODEL_PATH)
print("\n=== 模型加载的类别信息 ===")
print(model.names)
print("========================\n")


def allowed_file(filename, allowed_extensions):
    """检查文件扩展名是否允许"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in allowed_extensions


def ensure_dir(directory):
    """确保目录存在，如果不存在则创建"""
    if not os.path.exists(directory):
        os.makedirs(directory, exist_ok=True)
    return directory


# 确保上传和保存目录存在
ensure_dir(get_upload_folder())
ensure_dir(get_original_dir())
ensure_dir(get_results_dir())
ensure_dir(get_camera_dir())


@app.route('/')
def index():
    """主页"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('home.html')

@app.route('/app')
def app_home():
    return render_template('app/home.html')

@app.route('/app/detection')
def app_detection():
    return render_template('app/detection.html')

@app.route('/app/records')
def app_records():
    return render_template('app/records.html')

@app.route('/app/profile')
def app_profile():
    return render_template('app/profile.html')

@app.route('/login', methods=['GET', 'POST'])
def login():
    """登录页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        user = User.query.filter_by(username=username).first()
        if user and check_password_hash(user.password, password):
            session['user_id'] = user.id
            session['login_time'] = int(time.time())  # 记录登录时间
            session['is_admin'] = user.is_admin  # 记录用户类型
            session['username'] = username  # 记录用户名，用于前端显示

            flash('登录成功！', 'success')

            # 根据用户类型重定向到不同页面
            if user.is_admin:
                # 管理员重定向到管理员页面
                return redirect(url_for('admin.dashboard'))  # 假设管理员使用admin蓝图
            else:
                # 普通用户重定向到首页
                return redirect(url_for('index'))

        flash('用户名或密码错误！', 'error')

    return render_template('login.html')


@app.route('/register', methods=['GET', 'POST'])
def register():
    """注册页面"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')

        if User.query.filter_by(username=username).first():
            flash('用户名已存在！', 'error')
            return render_template('register.html')

        user = User(
            username=username,
            password=generate_password_hash(password),
            is_admin=False  # 普通注册用户默认不是管理员
        )
        db.session.add(user)

        try:
            db.session.commit()
            flash('注册成功！请登录。', 'success')
            return redirect(url_for('login'))
        except Exception as e:
            db.session.rollback()
            flash('注册失败，请稍后重试！', 'error')

    return render_template('register.html')


@app.route('/logout')
def logout():
    """登出"""
    session.pop('user_id', None)
    flash('您已成功登出！', 'success')
    return redirect(url_for('login'))


@app.route('/image_detection')
def image_detection():
    """图片检测页面"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('image_detection.html')


@app.route('/api/detect/image', methods=['POST'])
def detect_image():
    """图片检测API"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    if 'image' not in request.files:
        return jsonify({'success': False, 'message': '未上传图片'})

    file = request.files['image']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择图片'})

    if not allowed_file(file.filename, Config.ALLOWED_IMAGE_EXTENSIONS):
        return jsonify({'success': False, 'message': '不支持的图片格式'})

    try:
        # 保存原始图片
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_path = os.path.join(get_original_dir(), f'original_{timestamp}_{filename}')
        file.save(original_path)

        # 获取检测参数
        confidence = float(request.form.get('confidence', 0.5))
        show_labels = request.form.get('show_labels', 'true').lower() == 'true'
        show_confidence = request.form.get('show_confidence', 'true').lower() == 'true'

        # 执行检测
        start_time = time.time()
        results = model(original_path)[0]
        detection_time = time.time() - start_time

        # 处理检测结果
        boxes = results.boxes
        objects = []
        for box in boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            objects.append({
                'label': Config.CH_NAMES[cls],
                'confidence': round(conf * 100, 2),
                'x1': x1,
                'y1': y1,
                'x2': x2,
                'y2': y2
            })

        # 保存检测结果图片
        result_img = results.plot()
        result_path = os.path.join(get_results_dir(), f'result_{timestamp}_{filename}')
        cv2.imwrite(result_path, result_img)

        # 获取相对路径用于数据库存储
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)

        # 保存检测记录
        record = DetectionRecord(
            user_id=session['user_id'],
            source_type='image',
            source_name=filename,
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=detection_time,
            total_objects=len(objects),
            is_cleaned=False
        )
        record.set_detection_results({
            'objects': objects,
            'confidence_threshold': confidence,
            'show_labels': show_labels,
            'show_confidence': show_confidence
        })
        try:
            db.session.add(record)
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            print(f"保存检测记录时出错: {str(e)}")

        # 返回结果
        return jsonify({
            'success': True,
            'detection_time': round(detection_time, 3),
            'timestamp': datetime.now().isoformat(),
            'objects': objects,
            'result_image': f'/static/uploads/results/result_{timestamp}_{filename}'
        })

    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/video_detection')
def video_detection():
    return render_template('video_detection.html')


@app.route('/api/detect/video', methods=['POST'])
def detect_video():
    if 'video' not in request.files:
        return jsonify({'success': False, 'message': '未上传视频'})

    file = request.files['video']
    if file.filename == '':
        return jsonify({'success': False, 'message': '未选择视频'})

    if not allowed_file(file.filename, Config.ALLOWED_VIDEO_EXTENSIONS):
        return jsonify({'success': False, 'message': '不支持的视频格式'})

    try:
        # 导入必要的模块
        import os
        import imageio
        import cv2
        import numpy as np

        # 保存原始视频
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_path = os.path.join(get_original_dir(), f'original_{timestamp}_{filename}')
        original_path = os.path.normpath(original_path)  # 规范化路径
        file.save(original_path)

        # 开始处理时间
        start_time = time.time()

        # 使用imageio读取视频
        reader = imageio.get_reader(original_path)
        meta_data = reader.get_meta_data()
        fps = meta_data.get('fps', 30)

        # 收集处理后的帧和检测结果
        frames = []
        all_objects = []
        total_confidence = 0
        total_detections = 0
        frame_count = 0
        total_frames = min(300, len(reader))  # 限制最大处理帧数

        # 每5帧处理一次以提高性能
        process_interval = 5

        # 处理视频帧
        for i, frame in enumerate(reader):
            # 发送进度更新
            if i % 10 == 0:  # 减少进度更新频率
                progress = int((i / total_frames) * 100)
                socketio.emit('detection_progress', {
                    'progress': progress,
                    'current_frame': i,
                    'total_frames': total_frames
                })

            # 转换为OpenCV格式处理
            if i % process_interval == 0:
                opencv_frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)

                # 执行检测
                results = model(opencv_frame)[0]

                # 绘制检测框
                annotated_frame = results.plot()
                rgb_frame = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
                frames.append(rgb_frame)

                # 收集检测结果
                for box in results.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    cls = int(box.cls[0])
                    conf = float(box.conf[0])
                    all_objects.append({
                        'label': Config.CH_NAMES[cls],
                        'confidence': round(conf * 100, 2),
                        'x1': x1,
                        'y1': y1,
                        'x2': x2,
                        'y2': y2
                    })
                    total_confidence += conf
                    total_detections += 1
            else:
                # 未处理的帧直接添加
                frames.append(frame)

            frame_count += 1
            if frame_count >= total_frames:
                break

        # 关闭读取器
        reader.close()

        # 设置结果视频路径
        result_filename = f'result_{timestamp}.mp4'
        result_path = os.path.join(get_results_dir(), result_filename)
        result_path = os.path.normpath(result_path)

        # 使用imageio写入视频
        try:
            # 使用imageio直接保存视频
            imageio.mimsave(
                result_path,
                frames,
                fps=fps,
                quality=8,
                macro_block_size=1  # 提高兼容性
            )

            # 检查视频是否成功生成
            if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                raise Exception("视频文件生成失败")

        except Exception as writer_error:
            print(f"imageio视频写入失败: {str(writer_error)}")
            # 尝试使用imageio-ffmpeg
            try:
                import imageio_ffmpeg
                writer = imageio.get_writer(
                    result_path,
                    fps=fps,
                    codec='libx264',
                    pixelformat='yuv420p',
                    quality=8,
                    ffmpeg_log_level='quiet'
                )
                for frame in frames:
                    writer.append_data(frame)
                writer.close()

                if not os.path.exists(result_path) or os.path.getsize(result_path) == 0:
                    raise Exception("视频文件生成失败")
            except Exception as ffmpeg_error:
                print(f"imageio-ffmpeg处理失败: {str(ffmpeg_error)}")
                raise

        # 构建结果视频的URL
        result_url = f'/static/uploads/results/{result_filename}'.replace('\\', '/')

        # 计算处理时间和平均置信度
        processing_time = time.time() - start_time
        avg_confidence = (total_confidence / total_detections) if total_detections > 0 else 0

        # 获取相对路径用于数据库存储
        relative_original_path = get_relative_path(original_path)
        relative_result_path = get_relative_path(result_path)

        # 保存检测记录
        record = DetectionRecord(
            user_id=session['user_id'],
            source_type='video',
            source_name=filename,
            original_path=relative_original_path,
            result_path=relative_result_path,
            duration=processing_time,
            total_objects=len(all_objects)
        )

        # 设置检测结果
        detection_results = {
            'objects': all_objects,
            'avg_confidence': avg_confidence
        }
        record.set_detection_results(detection_results)
        db.session.add(record)
        db.session.commit()

        # 统计检测结果分布
        detection_counts = Counter(obj['label'] for obj in all_objects)

        return jsonify({
            'success': True,
            'message': '视频处理完成',
            'result_url': result_url,
            'processing_time': round(processing_time, 2),
            'total_objects': len(all_objects),
            'avg_confidence': round(avg_confidence * 100, 2),
            'object_distribution': {label: count for label, count in detection_counts.items()}
        })

    except Exception as e:
        print(f"视频处理失败: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@app.route('/scientific')
def scientific():
    """科普知识页面"""
    try:
        # 获取活跃的科普内容，按位置排序
        sections = ScientificContent.query.filter_by(status='active').order_by(ScientificContent.position).all()

        # 获取所有FAQ
        faqs = ScientificFAQ.query.filter_by(status='active').order_by(ScientificFAQ.position).all()

        # 获取所有测试题目
        questions = ScientificQuestion.query.filter_by(status='active').order_by(ScientificQuestion.position).all()

        # 如果数据库中没有内容，返回静态模板
        if not sections:
            return render_template('scientific.html')

        # 将科普内容组织成字典
        content_data = {
            'sections': [section.to_dict() for section in sections],
            'faqs': [faq.to_dict() for faq in faqs],
            'questions': [question.to_dict() for question in questions]
        }

        return render_template('scientific.html', content_data=content_data)
    except Exception as e:
        print(f"获取科普内容时出错: {str(e)}")
        # 出错时返回静态模板
        return render_template('scientific.html')


@app.route('/realtime_detection')
def realtime_detection():
    """实时检测页面"""
    if 'user_id' not in session:
        return redirect(url_for('login'))
    return render_template('realtime_detection.html')


@app.route('/detection_records')
def detection_records():
    """检测记录页面"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    # 获取当前用户ID
    user_id = session['user_id']

    # 获取页码和每页记录数参数
    page = request.args.get('page', 1, type=int)
    per_page = request.args.get('per_page', 10, type=int)

    # 限制per_page参数范围，防止过大请求
    if per_page <= 0:
        per_page = 10
    elif per_page > 1000:  # 设置一个上限以防止资源滥用
        per_page = 1000

    # 获取当前用户的所有检测记录，并按时间降序排列
    records_pagination = DetectionRecord.query.filter_by(user_id=user_id).order_by(
        DetectionRecord.timestamp.desc()).paginate(page=page, per_page=per_page)

    # 统计总数、已处理和未处理记录
    total_records = DetectionRecord.query.filter_by(user_id=user_id).count()
    cleaned_records = DetectionRecord.query.filter_by(user_id=user_id, is_cleaned=True).count()
    uncleaned_records = total_records - cleaned_records

    # 统计各类型记录
    types = ['image', 'video', 'multi_image']
    type_counts = {}
    for type_name in types:
        type_counts[type_name] = DetectionRecord.query.filter_by(user_id=user_id, source_type=type_name).count()

    # 统计每日记录数
    daily_stats_query = db.session.query(
        db.func.date(DetectionRecord.timestamp).label('date'),
        db.func.count(DetectionRecord.id).label('count')
    ).filter_by(user_id=user_id).group_by(db.func.date(DetectionRecord.timestamp)).order_by(
        db.func.date(DetectionRecord.timestamp)).all()

    daily_stats = [{'date': str(item.date), 'count': item.count} for item in daily_stats_query]

    # 统计检测到的垃圾类型分布
    garbage_types = {}
    records_with_results = DetectionRecord.query.filter_by(user_id=user_id).all()
    for record in records_with_results:
        if record.detection_results:
            results = json.loads(record.detection_results)
            if 'objects' in results:
                for obj in results['objects']:
                    label = obj['label']
                    garbage_types[label] = garbage_types.get(label, 0) + 1

    # 将字典转换为列表
    garbage_distribution = [{'name': k, 'value': v} for k, v in garbage_types.items()]

    # 按值降序排序
    garbage_distribution.sort(key=lambda x: x['value'], reverse=True)

    # 统计各检测类型的清理状态
    type_cleaned_stats = {}
    for type_name in types:
        cleaned = DetectionRecord.query.filter_by(user_id=user_id, source_type=type_name, is_cleaned=True).count()
        total = type_counts[type_name]
        uncleaned = total - cleaned
        type_cleaned_stats[type_name] = {
            'cleaned': cleaned,
            'uncleaned': uncleaned,
            'total': total
        }

    return render_template(
        'detection_records.html',
        records=records_pagination.items,
        pagination=records_pagination,
        stats={
            'total': total_records,
            'cleaned': cleaned_records,
            'uncleaned': uncleaned_records,
            'types': type_counts,
            'daily_stats': daily_stats,
            'garbage_distribution': garbage_distribution,
            'type_cleaned_stats': type_cleaned_stats
        }
    )


@app.route('/alert')
def alert():
    """水面清洁警报页面"""
    if 'user_id' not in session:
        return redirect(url_for('login'))

    return render_template('alert.html')


@app.route('/api/detection_detail/<int:id>')
def detection_detail(id):
    """获取检测记录详情"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    try:
        record = DetectionRecord.query.get(id)
        if not record:
            return jsonify({'error': '未找到记录'}), 404

        results = record.get_detection_results()

        return jsonify({
            'detection_time': record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
            'type': record.source_type,
            'file_name': record.source_name,
            'confidence': round(results.get('avg_confidence', 0) * 100, 2),
            'processing_time': round(record.duration * 1000, 2),  # 转换为毫秒
            'results': results.get('objects', [])
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@app.route('/api/delete_record/<int:id>', methods=['DELETE'])
def delete_record(id):
    """删除检测记录"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    try:
        # 获取用户
        user = User.query.get(session['user_id'])
        # 判断是否为管理员用户
        is_admin = user.is_admin

        if not is_admin:
            return jsonify({'success': False, 'message': '您没有权限执行此操作'})

        record = DetectionRecord.query.get(id)
        if record:
            db.session.delete(record)
            db.session.commit()
            return jsonify({'success': True})
        return jsonify({'success': False, 'message': '记录不存在'})
    except Exception as e:
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/export_records')
def export_records():
    """导出检测记录"""
    if 'user_id' not in session:
        flash('请先登录', 'error')
        return redirect(url_for('login'))

    try:
        # 查询当前用户的所有检测记录
        records = DetectionRecord.query.filter_by(
            user_id=session['user_id']
        ).order_by(
            DetectionRecord.timestamp.desc()
        ).all()

        if not records:
            flash('没有可导出的记录', 'warning')
            return redirect(url_for('detection_records'))

        # 创建CSV响应，添加UTF-8 BOM标记，使Excel能正确识别编码
        output = StringIO()
        output.write('\ufeff')  # UTF-8 BOM
        writer = csv.writer(output)

        # 写入CSV头部
        writer.writerow(['ID', '检测时间', '检测类型', '文件名', '检测结果', '置信度', '处理时间(ms)', '是否已清理'])

        # 写入数据行
        for record in records:
            try:
                results = record.get_detection_results()

                # 根据检测类型处理检测结果
                if record.source_type == 'video':
                    # 视频结果可能包含大量对象，统计类别计数
                    if 'objects' in results and isinstance(results['objects'], list):
                        # 计算每种垃圾的数量
                        label_counts = {}
                        for obj in results['objects']:
                            if 'label' in obj:
                                label = obj['label']
                                label_counts[label] = label_counts.get(label, 0) + 1

                        # 格式化为"类别:数量"的形式
                        detected_objects = ', '.join(f"{label}:{count}" for label, count in label_counts.items())
                    else:
                        detected_objects = '无检测结果'
                else:
                    # 图片和其他类型的检测
                    detected_objects = ', '.join(
                        obj['label'] for obj in results.get('objects', [])
                    ) if results.get('objects') else '无检测结果'

                # 计算平均置信度
                confidence = round(results.get('avg_confidence', 0) * 100, 2) if results.get('avg_confidence') else 0

                # 写入行数据
                writer.writerow([
                    record.id,
                    record.timestamp.strftime('%Y-%m-%d %H:%M:%S'),
                    record.source_type,
                    record.source_name,
                    detected_objects,
                    f"{confidence}%",
                    f"{round(record.duration * 1000, 2)}",
                    '已清理' if record.is_cleaned else '未清理'
                ])
            except Exception as row_error:
                # 如果单行处理失败，继续处理下一行
                print(f"处理记录ID {record.id} 失败: {row_error}")
                continue

        # 准备响应
        output.seek(0)

        # 记录导出成功日志
        print(f"用户 {session['user_id']} 成功导出了 {len(records)} 条检测记录")

        # 返回CSV响应
        return Response(
            output.getvalue(),
            mimetype='text/csv',
            headers={
                'Content-Disposition': f'attachment; filename="detection_records_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv"',
                'Content-Type': 'text/csv; charset=utf-8'
            }
        )
    except Exception as e:
        print(f"导出记录时发生错误: {str(e)}")
        flash(f'导出失败: {str(e)}', 'error')
        return redirect(url_for('detection_records'))


@socketio.on('detect_frame')
def handle_frame(data):
    """处理实时检测帧"""
    try:
        # 解码图像数据
        image_data = base64.b64decode(data['image'].split(',')[1])

        # 保存临时文件
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], 'temp_frame.jpg')
        with open(temp_path, 'wb') as f:
            f.write(image_data)

        # 记录开始时间
        start_time = time.time()

        # 执行检测
        results = model(temp_path)[0]
        detection_time = time.time() - start_time
        fps = 1.0 / detection_time

        # 处理检测结果
        objects = []
        boxes = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            w, h = x2 - x1, y2 - y1
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            label = Config.CH_NAMES[cls]

            objects.append({
                'label': label,
                'confidence': round(conf * 100, 2)
            })

            boxes.append({
                'x': x1,
                'y': y1,
                'width': w,
                'height': h,
                'category': label
            })

        # 发送结果
        emit('detection_result', {
            'timestamp': datetime.now().isoformat(),
            'fps': round(fps, 1),
            'objects': objects,
            'boxes': boxes
        })

    except Exception as e:
        print(f"Error in handle_frame: {str(e)}")
        emit('detection_error', {'message': str(e)})


@app.route('/api/save_capture', methods=['POST'])
def save_capture():
    """保存实时检测截图"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    try:
        # 导入必要模块
        import imageio
        import cv2
        import numpy as np
        import base64
        from io import BytesIO

        data = request.json
        if not data or 'image' not in data:
            return jsonify({'success': False, 'message': '未接收到图像数据'})

        # 解码base64图像数据
        try:
            if ',' in data['image']:
                image_data = data['image'].split(',')[1]
            else:
                image_data = data['image']
            image_bytes = base64.b64decode(image_data)
        except Exception as e:
            return jsonify({'success': False, 'message': f'图像数据解码失败: {str(e)}'})

        # 保存图片
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f'capture_{timestamp}.jpg'
        filepath = os.path.join(get_camera_dir(), filename)

        # 使用BytesIO加载图像数据，避免文件IO
        image = imageio.imread(BytesIO(image_bytes))
        imageio.imwrite(filepath, image, quality=95)

        # 转换为OpenCV格式处理
        opencv_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        # 执行检测
        results = model(opencv_image)[0]

        # 处理检测结果
        objects = []
        for box in results.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
            conf = float(box.conf[0])
            cls = int(box.cls[0])
            objects.append({
                'label': Config.CH_NAMES[cls],
                'confidence': round(conf * 100, 2),
                'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2
            })

        # 绘制检测框
        result_img = results.plot()

        # 转回RGB格式并保存
        result_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)
        result_path = os.path.join(get_results_dir(), f'result_{filename}')
        imageio.imwrite(result_path, result_rgb, quality=95)

        # 保存记录
        record = DetectionRecord(
            user_id=session['user_id'],
            source_type='camera',
            source_name=filename,
            original_path=get_relative_path(filepath),
            result_path=get_relative_path(result_path),
            duration=0.0,
            total_objects=len(results.boxes),
            is_cleaned=False
        )

        # 设置检测结果
        detection_results = {
            'objects': objects,
            'avg_confidence': sum(obj['confidence'] for obj in objects) / len(objects) if objects else 0
        }
        record.set_detection_results(detection_results)

        # 保存到数据库
        db.session.add(record)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '截图已保存并进行检测',
            'filename': filename,
            'total_objects': len(objects),
            'objects': objects
        })

    except Exception as e:
        print(f"保存截图时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)})


@app.route('/api/update_clean_status/<int:record_id>', methods=['POST'])
def update_clean_status(record_id):
    """更新垃圾清理状态"""
    if 'user_id' not in session:
        return jsonify({'success': False, 'message': '请先登录'})

    try:
        # 获取用户
        user = User.query.get(session['user_id'])
        # 判断是否为管理员用户
        is_admin = user.is_admin

        if not is_admin:
            return jsonify({'success': False, 'message': '您没有权限执行此操作'})

        record = DetectionRecord.query.get_or_404(record_id)
        record.is_cleaned = True
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '更新成功'
        })

    except Exception as e:
        db.session.rollback()
        return jsonify({
            'success': False,
            'message': str(e)
        })


@app.route('/favicon.ico')
def favicon():
    """提供网站图标"""
    return send_from_directory(
        os.path.join(app.root_path, 'static', 'img'),
        'favicon.ico', mimetype='image/vnd.microsoft.icon'
    )


def init_database():
    """初始化数据库"""
    with app.app_context():
        # 创建所有表（如果不存在）
        db.create_all()

        # 创建测试用户（如果不存在）
        if not User.query.filter_by(username='admin').first():
            admin = User(
                username='admin',
                password=generate_password_hash('admin123'),
                is_admin=True  # 设置为管理员
            )
            db.session.add(admin)
            db.session.commit()
            print("创建管理员用户成功")

        # 初始化科普内容（如果不存在）
        if not ScientificContent.query.filter_by(section_id='intro').first():
            # 创建水面漂浮垃圾的危害板块
            intro_section = ScientificContent(
                section_id='intro',
                title='水面漂浮垃圾的危害',
                content='''
                <p>水面漂浮垃圾是指漂浮在水体表面的各类人为废弃物，如塑料瓶、食品包装、泡沫塑料、废纸等。这些漂浮垃圾不仅影响水体景观，更对水生态系统和环境造成严重危害：</p>
                <ul>
                    <li><strong>生态危害</strong>：威胁水生动物生存，导致误食和缠绕伤害</li>
                    <li><strong>环境污染</strong>：分解过程释放有害物质，污染水源</li>
                    <li><strong>资源浪费</strong>：大量可回收资源未得到合理利用</li>
                    <li><strong>景观破坏</strong>：影响水域景观和旅游资源价值</li>
                </ul>
                <p>通过科学的垃圾分类和及时清理，我们可以有效减少水面漂浮垃圾，保护珍贵的水资源。</p>
                ''',
                content_type='text',
                image_path='/static/img/water_pollution.jpg',
                position=1,
                status='active'
            )
            db.session.add(intro_section)
            db.session.commit()

            # 创建垃圾分类指南板块
            classification_section = ScientificContent(
                section_id='classification',
                title='水面漂浮垃圾分类指南',
                content='''
                <p>根据我国垃圾分类标准和水面垃圾特点，水面漂浮垃圾主要可分为以下几类：</p>
                ''',
                content_type='text',
                position=2,
                status='active'
            )
            db.session.add(classification_section)
            db.session.commit()

            # 创建水环境保护小贴士板块
            protection_section = ScientificContent(
                section_id='protection',
                title='水环境保护小贴士',
                content='''
                <p>保护水环境，预防水面垃圾产生，从我们的日常行为做起：</p>
                ''',
                content_type='tips',
                position=3,
                status='active'
            )
            db.session.add(protection_section)
            db.session.commit()

            # 创建常见问题解答板块
            faq_section = ScientificContent(
                section_id='faq',
                title='常见问题解答',
                content='',
                content_type='faq',
                position=4,
                status='active'
            )
            db.session.add(faq_section)
            db.session.commit()

            # 添加FAQ问题
            faqs = [
                {
                    'question': '什么是微塑料，它对水生态有什么危害？',
                    'answer': '''
                    <p>微塑料是指直径小于5毫米的塑料颗粒，通常由大型塑料垃圾在环境中分解而来。它们对水生态的危害主要包括：</p>
                    <ul>
                        <li>被水生生物误食，导致其消化系统堵塞或造成内部伤害</li>
                        <li>微塑料表面可吸附有毒物质，通过食物链传递和富集，最终可能危害人类健康</li>
                        <li>影响水体自净能力，破坏水生态系统平衡</li>
                    </ul>
                    '''
                },
                {
                    'question': '为什么塑料垃圾在水中分解需要很长时间？',
                    'answer': '''
                    <p>塑料垃圾在水中分解缓慢主要因为：</p>
                    <ul>
                        <li>塑料由人工合成的聚合物组成，自然界中很少有微生物能够有效分解这些化学结构</li>
                        <li>水环境温度相对稳定且较低，减缓了塑料的光降解和热降解过程</li>
                        <li>大部分塑料设计初衷就是为了耐用，具有优良的稳定性</li>
                    </ul>
                    <p>不同类型的塑料在水环境中的分解时间从数十年到数百年不等。例如，塑料袋约需20年，塑料瓶约需450年，渔网约需600年。</p>
                    '''
                },
                {
                    'question': '人工智能技术如何应用于水面垃圾检测？',
                    'answer': '''
                    <p>人工智能技术在水面垃圾检测中的应用主要包括：</p>
                    <ul>
                        <li><strong>计算机视觉识别</strong>：通过深度学习算法训练模型识别不同类型的水面垃圾</li>
                        <li><strong>无人机巡查</strong>：结合无人机和AI系统实现大范围水域垃圾自动监测</li>
                        <li><strong>智能清理系统</strong>：基于AI识别结果，指导自动化清理设备进行精准清理</li>
                        <li><strong>数据分析预测</strong>：分析垃圾分布规律，预测垃圾聚集热点，提高清理效率</li>
                    </ul>
                    <p>本系统正是利用深度学习技术实现了对水面垃圾的自动检测和分类。</p>
                    '''
                }
            ]

            for i, faq_data in enumerate(faqs):
                faq = ScientificFAQ(
                    section_id=faq_section.id,
                    question=faq_data['question'],
                    answer=faq_data['answer'],
                    position=i + 1,
                    status='active'
                )
                db.session.add(faq)

            # 创建环保知识测试板块
            quiz_section = ScientificContent(
                section_id='quiz',
                title='环保知识小测试',
                content='''
                <p>测试一下您对水面垃圾分类和环保知识的掌握程度：</p>
                ''',
                content_type='quiz',
                position=5,
                status='active'
            )
            db.session.add(quiz_section)
            db.session.commit()

            # 添加测试题目
            questions = [
                {
                    'question': '下列哪种垃圾在水中分解时间最长？',
                    'options': ['A. 纸巾', 'B. 塑料袋', 'C. 玻璃瓶', 'D. 苹果核'],
                    'answer': 'C',
                    'explanation': '玻璃瓶在自然环境中分解需要约100万年，远长于塑料袋（10-20年）、纸巾（2-4周）和苹果核（1-2个月）。'
                },
                {
                    'question': '以下哪类不属于水面漂浮垃圾中的可回收物？',
                    'options': ['A. 塑料瓶', 'B. 烟头', 'C. 饮料罐', 'D. 纸板箱'],
                    'answer': 'B',
                    'explanation': '烟头含有多种有害物质，属于其他垃圾，不可回收。塑料瓶、饮料罐和纸板箱均可回收利用。'
                },
                {
                    'question': '关于微塑料的说法，错误的是：',
                    'options': ['A. 微塑料可以被鱼类摄入体内', 'B. 微塑料主要来源于大型塑料垃圾的分解',
                                'C. 微塑料在短时间内会自然分解无害', 'D. 微塑料可能通过食物链影响人类健康'],
                    'answer': 'C',
                    'explanation': '微塑料不会在短时间内自然分解无害，相反，它们会在环境中持续存在很长时间。'
                }
            ]

            for i, question_data in enumerate(questions):
                question = ScientificQuestion(
                    section_id=quiz_section.id,
                    question=question_data['question'],
                    answer=question_data['answer'],
                    explanation=question_data['explanation'],
                    position=i + 1,
                    status='active'
                )
                question.set_options(question_data['options'])
                db.session.add(question)

            db.session.commit()
            print("初始化科普内容成功")

        print("数据库初始化完成")


# 科普内容管理API路由
@app.route('/api/scientific/contents', methods=['GET'])
def get_scientific_contents():
    """获取所有科普内容"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        # 获取所有科普内容
        sections = ScientificContent.query.order_by(ScientificContent.position).all()
        return jsonify({
            'success': True,
            'data': [section.to_dict() for section in sections]
        })
    except Exception as e:
        print(f"获取科普内容列表时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/content/<string:section_id>', methods=['GET'])
def get_scientific_content(section_id):
    """获取特定科普内容"""
    try:
        # 获取特定科普内容
        section = ScientificContent.query.filter_by(section_id=section_id).first()
        if not section:
            return jsonify({'success': False, 'message': '内容不存在'}), 404

        # 获取关联的FAQs和问题
        section_data = section.to_dict()
        section_data['faqs'] = [faq.to_dict() for faq in section.faqs]
        section_data['questions'] = [question.to_dict() for question in section.questions]

        return jsonify({
            'success': True,
            'data': section_data
        })
    except Exception as e:
        print(f"获取科普内容详情时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/content', methods=['POST'])
def create_scientific_content():
    """创建科普内容"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 验证必要字段
        if not data or not data.get('section_id') or not data.get('title') or not data.get('content'):
            return jsonify({'success': False, 'message': '缺少必要字段'}), 400

        # 检查section_id是否已存在
        if ScientificContent.query.filter_by(section_id=data['section_id']).first():
            return jsonify({'success': False, 'message': 'section_id已存在'}), 400

        # 创建科普内容
        section = ScientificContent(
            section_id=data['section_id'],
            title=data['title'],
            content=data['content'],
            content_type=data.get('content_type', 'text'),
            image_path=data.get('image_path', ''),
            position=data.get('position', 0),
            status=data.get('status', 'active')
        )

        db.session.add(section)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '创建成功',
            'data': section.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"创建科普内容时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/content/<string:section_id>', methods=['PUT'])
def update_scientific_content(section_id):
    """更新科普内容"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 获取科普内容
        section = ScientificContent.query.filter_by(section_id=section_id).first()
        if not section:
            return jsonify({'success': False, 'message': '内容不存在'}), 404

        # 更新字段
        if 'title' in data:
            section.title = data['title']
        if 'content' in data:
            section.content = data['content']
        if 'content_type' in data:
            section.content_type = data['content_type']
        if 'image_path' in data:
            section.image_path = data['image_path']
        if 'position' in data:
            section.position = data['position']
        if 'status' in data:
            section.status = data['status']

        # 更新时间戳
        section.updated_at = datetime.utcnow()

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '更新成功',
            'data': section.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"更新科普内容时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/content/<string:section_id>', methods=['DELETE'])
def delete_scientific_content(section_id):
    """删除科普内容"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        # 获取科普内容
        section = ScientificContent.query.filter_by(section_id=section_id).first()
        if not section:
            return jsonify({'success': False, 'message': '内容不存在'}), 404

        # 删除关联的FAQs和问题
        ScientificFAQ.query.filter_by(section_id=section.id).delete()
        ScientificQuestion.query.filter_by(section_id=section.id).delete()

        # 删除科普内容
        db.session.delete(section)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '删除成功'
        })
    except Exception as e:
        db.session.rollback()
        print(f"删除科普内容时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/faq', methods=['POST'])
def create_scientific_faq():
    """创建科普FAQ"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 验证必要字段
        if not data or not data.get('section_id') or not data.get('question') or not data.get('answer'):
            return jsonify({'success': False, 'message': '缺少必要字段'}), 400

        # 获取关联的科普内容
        section = ScientificContent.query.filter_by(id=data['section_id']).first()
        if not section:
            return jsonify({'success': False, 'message': '关联的科普内容不存在'}), 404

        # 创建FAQ
        faq = ScientificFAQ(
            section_id=section.id,
            question=data['question'],
            answer=data['answer'],
            position=data.get('position', 0),
            status=data.get('status', 'active')
        )

        db.session.add(faq)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '创建成功',
            'data': faq.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"创建科普FAQ时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/question', methods=['POST'])
def create_scientific_question():
    """创建科普测试题目"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 验证必要字段
        if not data or not data.get('question') or not data.get('options') or not data.get('answer'):
            return jsonify({'success': False, 'message': '缺少必要字段'}), 400

        # 创建测试题目
        question = ScientificQuestion(
            section_id=data.get('section_id'),
            question=data['question'],
            answer=data['answer'],
            explanation=data.get('explanation', ''),
            position=data.get('position', 0),
            status=data.get('status', 'active')
        )

        # 设置选项
        question.set_options(data['options'])

        db.session.add(question)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '创建成功',
            'data': question.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"创建科普测试题目时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/faq/<int:faq_id>', methods=['PUT'])
def update_scientific_faq(faq_id):
    """更新科普FAQ"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 获取FAQ
        faq = ScientificFAQ.query.get(faq_id)
        if not faq:
            return jsonify({'success': False, 'message': 'FAQ不存在'}), 404

        # 更新字段
        if 'question' in data:
            faq.question = data['question']
        if 'answer' in data:
            faq.answer = data['answer']
        if 'position' in data:
            faq.position = data['position']
        if 'status' in data:
            faq.status = data['status']

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '更新成功',
            'data': faq.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"更新科普FAQ时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/faq/<int:faq_id>', methods=['DELETE'])
def delete_scientific_faq(faq_id):
    """删除科普FAQ"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        # 获取FAQ
        faq = ScientificFAQ.query.get(faq_id)
        if not faq:
            return jsonify({'success': False, 'message': 'FAQ不存在'}), 404

        # 删除FAQ
        db.session.delete(faq)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '删除成功'
        })
    except Exception as e:
        db.session.rollback()
        print(f"删除科普FAQ时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/question/<int:question_id>', methods=['PUT'])
def update_scientific_question(question_id):
    """更新科普测试题目"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        data = request.json

        # 获取测试题目
        question = ScientificQuestion.query.get(question_id)
        if not question:
            return jsonify({'success': False, 'message': '测试题目不存在'}), 404

        # 更新字段
        if 'question' in data:
            question.question = data['question']
        if 'answer' in data:
            question.answer = data['answer']
        if 'explanation' in data:
            question.explanation = data['explanation']
        if 'position' in data:
            question.position = data['position']
        if 'status' in data:
            question.status = data['status']
        if 'options' in data:
            question.set_options(data['options'])

        db.session.commit()

        return jsonify({
            'success': True,
            'message': '更新成功',
            'data': question.to_dict()
        })
    except Exception as e:
        db.session.rollback()
        print(f"更新科普测试题目时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/question/<int:question_id>', methods=['DELETE'])
def delete_scientific_question(question_id):
    """删除科普测试题目"""
    if 'user_id' not in session or not session.get('is_admin'):
        return jsonify({'success': False, 'message': '无权限访问'}), 403

    try:
        # 获取测试题目
        question = ScientificQuestion.query.get(question_id)
        if not question:
            return jsonify({'success': False, 'message': '测试题目不存在'}), 404

        # 删除测试题目
        db.session.delete(question)
        db.session.commit()

        return jsonify({
            'success': True,
            'message': '删除成功'
        })
    except Exception as e:
        db.session.rollback()
        print(f"删除科普测试题目时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/faqs/<int:section_id>', methods=['GET'])
def get_scientific_faqs(section_id):
    """获取特定板块的FAQ列表"""
    try:
        faqs = ScientificFAQ.query.filter_by(section_id=section_id).order_by(ScientificFAQ.position).all()
        return jsonify({
            'success': True,
            'data': [faq.to_dict() for faq in faqs]
        })
    except Exception as e:
        print(f"获取FAQ列表时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/questions/<int:section_id>', methods=['GET'])
def get_scientific_questions(section_id):
    """获取特定板块的测试题目列表"""
    try:
        questions = ScientificQuestion.query.filter_by(section_id=section_id).order_by(
            ScientificQuestion.position).all()
        return jsonify({
            'success': True,
            'data': [question.to_dict() for question in questions]
        })
    except Exception as e:
        print(f"获取测试题目列表时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


@app.route('/api/scientific/question/<int:question_id>', methods=['GET'])
def get_scientific_question(question_id):
    """获取单个科普测试题目"""
    try:
        question = ScientificQuestion.query.get(question_id)
        if not question:
            return jsonify({'success': False, 'message': '测试题目不存在'}), 404

        return jsonify({
            'success': True,
            'data': question.to_dict()
        })
    except Exception as e:
        print(f"获取测试题目时出错: {str(e)}")
        return jsonify({'success': False, 'message': str(e)}), 500


# 为普通用户提供的保存摄像头录制功能API
@app.route('/api/user/save_recording', methods=['POST'])
# @login_required
def user_save_camera_recording():
    """保存普通用户摄像头录制的视频和检测结果，返回记录ID和视频URL"""
    try:
        # 导入必要的模块
        import os
        import json
        import shutil
        import traceback
        import cv2
        import numpy as np
        import imageio
        import glob
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
        uploads_dir = os.path.join('static', 'uploads', 'camera')
        os.makedirs(uploads_dir, exist_ok=True)

        # 安全处理文件名
        filename = secure_filename(video_file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        original_filename = f"user_camera_original_{timestamp}_{filename}"
        original_path = os.path.join(uploads_dir, original_filename)
        original_path = os.path.normpath(original_path)  # 规范化路径

        # 打印路径信息用于调试
        print(f"准备保存文件到: {original_path}")

        # 保存原始视频
        video_file.save(original_path)
        print(f"原始视频已保存到: {original_path}")

        # 设置结果视频路径
        result_filename = f"user_camera_result_{timestamp}.mp4"
        result_path = os.path.join(uploads_dir, result_filename)
        result_path = os.path.normpath(result_path)  # 规范化路径

        # 处理状态变量
        processing_success = False

        # 使用保存的帧创建视频
        if use_saved_frames and session_id:
            # 构建帧目录路径并规范化
            frames_dir = os.path.join('static', 'uploads', 'camera_frames', session_id)
            frames_dir = os.path.normpath(frames_dir)  # 规范化路径

            if os.path.exists(frames_dir):
                try:
                    # 获取所有保存的帧（按文件名排序）
                    frame_files = sorted(glob.glob(os.path.join(frames_dir, '*.jpg')))
                    if not frame_files:
                        print(f"没有找到会话 {session_id} 的帧文件")
                        raise Exception(f"会话 {session_id} 没有保存的帧")

                    print(f"找到 {len(frame_files)} 个保存的帧，开始创建视频...")

                    # 使用imageio读取所有帧
                    frames = []
                    for i, frame_file in enumerate(frame_files):
                        # 每20帧打印一次进度
                        if i % 20 == 0:
                            print(f"正在处理帧 {i + 1}/{len(frame_files)}...")

                        # 读取帧并添加到列表
                        frame = imageio.imread(frame_file)
                        frames.append(frame)

                    # 设置帧率
                    fps = 15  # 增加帧率使视频更流畅

                    # 使用imageio直接保存视频
                    try:
                        imageio.mimsave(
                            result_path,
                            frames,
                            fps=fps,
                            quality=8,
                            macro_block_size=1  # 提高兼容性
                        )

                        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                            print(f"视频处理成功: {result_path}")
                            processing_success = True
                        else:
                            raise Exception("处理后的视频文件为空")

                    except Exception as writer_error:
                        print(f"imageio视频写入失败: {str(writer_error)}")

                        # 尝试使用imageio-ffmpeg
                        try:
                            import imageio_ffmpeg
                            writer = imageio.get_writer(
                                result_path,
                                fps=fps,
                                codec='libx264',
                                pixelformat='yuv420p',
                                quality=8,
                                ffmpeg_log_level='quiet'
                            )

                            for frame in frames:
                                writer.append_data(frame)
                            writer.close()

                            if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                                print(f"使用imageio-ffmpeg处理成功: {result_path}")
                                processing_success = True
                            else:
                                raise Exception("处理后的视频文件为空")

                        except Exception as ffmpeg_error:
                            print(f"imageio-ffmpeg处理失败: {str(ffmpeg_error)}")
                            # 如果所有尝试都失败，回退到复制原始视频
                            raise

                    # 处理完成后清理帧文件夹
                    try:
                        shutil.rmtree(frames_dir)
                        print(f"已清理帧目录: {frames_dir}")
                    except Exception as e:
                        print(f"清理帧目录失败: {str(e)}")

                except Exception as e:
                    print(f"使用保存的帧创建视频失败: {str(e)}")
                    # 如果使用保存的帧失败，尝试使用上传的视频文件
                    processing_success = False

        # 如果未使用保存的帧或处理失败，处理上传的视频文件
        if not use_saved_frames or not processing_success:
            # 如果有检测到对象，使用视频处理
            if total_objects > 0:
                try:
                    # 导入模型
                    from app import model

                    # 检查模型是否成功加载
                    if not model:
                        raise Exception("未能加载检测模型")

                    # 使用imageio读取视频
                    reader = imageio.get_reader(original_path)
                    fps = reader.get_meta_data().get('fps', 30)

                    # 预处理视频并收集帧
                    frames = []
                    max_frames = 300  # 限制处理的帧数以提高性能

                    for i, frame in enumerate(reader):
                        # 每10帧处理一次（优化性能）
                        if i % 10 == 0:
                            # 转换为OpenCV格式处理
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
                        imageio.mimsave(
                            result_path,
                            frames,
                            fps=fps,
                            quality=8,
                            macro_block_size=1  # 提高兼容性
                        )

                        if os.path.exists(result_path) and os.path.getsize(result_path) > 0:
                            print(f"视频处理成功: {result_path}")
                            processing_success = True
                        else:
                            raise Exception("处理后的视频文件为空")

                    except Exception as writer_error:
                        print(f"imageio视频写入失败: {str(writer_error)}")

                        # 尝试使用imageio-ffmpeg
                        try:
                            import imageio_ffmpeg
                            writer = imageio.get_writer(
                                result_path,
                                fps=fps,
                                codec='libx264',
                                pixelformat='yuv420p',
                                quality=8,
                                ffmpeg_log_level='quiet'
                            )

                            for frame in frames:
                                writer.append_data(frame)
                            writer.close()

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
                    # 处理失败，直接复制原始视频
                    shutil.copyfile(original_path, result_path)
                    print(f"使用原始视频作为结果: {result_path}")
                    processing_success = False
            else:
                # 如果没有检测到对象，直接复制原始视频
                shutil.copyfile(original_path, result_path)
                print(f"未检测到垃圾，复制原始视频: {result_path}")
                processing_success = True

        # 构建相对路径URL
        original_url = f'/static/uploads/camera/{original_filename}'.replace('\\', '/')
        result_url = f'/static/uploads/camera/{result_filename}'.replace('\\', '/')

        # 保存到数据库
        record = DetectionRecord(
            user_id=user_id,
            source_type='camera',
            source_name=f"实时监控_{timestamp}",
            original_path=original_url,
            result_path=result_url,
            duration=duration,
            total_objects=total_objects,
            is_cleaned=False
        )
        record.set_detection_results(detection_results)

        db.session.add(record)
        db.session.commit()

        # 如果检测到垃圾（total_objects > 0），创建警报
        alert_info = None
        if total_objects > 0:
            alert_info = create_alert_from_detection(record)

        return jsonify({
            'success': True,
            'message': '视频和检测结果已保存',
            'record_id': record.id,
            'original_video_url': original_url,
            'result_video_url': result_url,
            'processed': processing_success,
            'alert_info': alert_info
        })

    except Exception as e:
        print(f"保存摄像头录制失败: {str(e)}")
        traceback.print_exc()
        return jsonify({
            'success': False,
            'message': f'保存失败: {str(e)}'
        }), 500


if __name__ == '__main__':
    init_database()
    socketio.run(app, debug=True, host='0.0.0.0', port=8888)
