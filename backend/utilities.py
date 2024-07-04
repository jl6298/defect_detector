import base64
import cv2
import time
import numpy as np
from defect_detector import YoloPredictor

def process_image(image_data, tags, cv_model):
    # 将base64图像转化为numpy数列图像
    start = time.time()
    image_bytes = base64.b64decode(image_data.split(',')[1])
    image_np = np.frombuffer(image_bytes, np.uint8)
    image_cv = cv2.imdecode(image_np, cv2.IMREAD_COLOR)
    # 执行推理
    labels_count_dict, frame = cv_model.inference(image_cv, tags)
    # 将数列图像转化回base64图像
    _, frame = cv2.imencode('.jpg', frame )
    frame = base64.b64encode(frame).decode('utf-8')
    end = time.time()
    fps = int(1 / (end - start))
    return {'message': 'detection complete', 'processed_image': frame, "detection_results":labels_count_dict, "fps": fps}