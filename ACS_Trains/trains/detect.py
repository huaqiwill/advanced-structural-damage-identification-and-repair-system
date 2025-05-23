import cv2
import numpy as np
from ultralytics import YOLO
from pathlib import Path
import time

class DamageDetector:
    def __init__(self, model_path, conf_thres=0.25, iou_thres=0.45):
        """
        初始化损伤检测器
        :param model_path: 模型路径
        :param conf_thres: 置信度阈值
        :param iou_thres: IOU阈值
        """
        self.model = YOLO(model_path)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
    def detect_image(self, image_path):
        """
        检测单张图片
        :param image_path: 图片路径
        :return: 处理后的图片和检测结果
        """
        # 读取图片
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"无法读取图片: {image_path}")
            
        # 进行检测
        results = self.model(image, conf=self.conf_thres, iou=self.iou_thres)[0]
        
        # 在图片上绘制检测结果
        annotated_image = self._draw_results(image.copy(), results)
        
        return annotated_image, results
    
    def detect_video(self, video_path, output_path=None):
        """
        检测视频
        :param video_path: 视频路径
        :param output_path: 输出视频路径
        """
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise ValueError(f"无法打开视频: {video_path}")
            
        # 获取视频信息
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        
        # 创建视频写入器
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
                
            # 进行检测
            results = self.model(frame, conf=self.conf_thres, iou=self.iou_thres)[0]
            
            # 绘制结果
            annotated_frame = self._draw_results(frame.copy(), results)
            
            if output_path:
                out.write(annotated_frame)
            
            yield annotated_frame, results
            
        cap.release()
        if output_path:
            out.release()
    
    def _draw_results(self, image, results):
        """
        在图片上绘制检测结果
        :param image: 原始图片
        :param results: 检测结果
        :return: 标注后的图片
        """
        boxes = results.boxes
        for box in boxes:
            # 获取边界框坐标
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            
            # 获取类别和置信度
            cls = int(box.cls[0])
            conf = float(box.conf[0])
            
            # 根据类别选择颜色
            color = (0, 0, 255) if cls == 0 else (0, 255, 0)  # 红色表示损伤，绿色表示正常
            
            # 绘制边界框
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # 添加标签
            label = f"{'损伤' if cls == 0 else '正常'} {conf:.2f}"
            cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return image

def main():
    # 示例用法
    model_path = "runs/train/damage_detection/weights/best.pt"
    detector = DamageDetector(model_path)
    
    # 图片检测示例
    image_path = "data/test/test_image.jpg"
    try:
        result_image, results = detector.detect_image(image_path)
        cv2.imwrite("result.jpg", result_image)
        print(f"检测完成，结果已保存至 result.jpg")
    except Exception as e:
        print(f"图片检测出错: {str(e)}")
    
    # 视频检测示例
    video_path = "data/test/test_video.mp4"
    output_path = "result.mp4"
    try:
        for frame, results in detector.detect_video(video_path, output_path):
            # 实时显示结果（可选）
            cv2.imshow("Detection", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"视频检测出错: {str(e)}")
    finally:
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main() 