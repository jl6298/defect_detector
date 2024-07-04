from flask import Flask, request, jsonify, Response, render_template, send_file
import mysql.connector
import cv2
import requests
from ultralytics import YOLO
import io
import numpy as np

app = Flask(__name__)

# MySQL database configuration
db_config = {
    'user': 'root',
    'password': '',
    'host': 'localhost',
    'database': 'test',
    'port': 3307,# Specify the database name here
}

# Load YOLOv8 model
model = YOLO('yolov8n.pt')  # Use the appropriate YOLOv8 model

# Function to connect to the database
def get_db_connection():
    return mysql.connector.connect(**db_config)

@app.route('/')
def index():
    return render_template('index.html')



@app.route('/get_image')
def get_image():
    image_id = request.args.get('id')
    timestamp = request.args.get('timestamp')

    query = None
    params = None

    if image_id:
        query = "SELECT image_data FROM images WHERE id = %s"
        params = (image_id,)
    elif timestamp:
        query = "SELECT image_data FROM images WHERE timestamp = %s"
        params = (timestamp,)

    if query:
        connection = get_db_connection()
        cursor = connection.cursor()
        try:
            cursor.execute(query, params)
            result = cursor.fetchone()
            if result:
                image_data = result[0]
                return send_file(io.BytesIO(image_data), mimetype='image/jpeg')
        finally:
            cursor.fetchall()
            cursor.close()
            connection.close()

    return "No images found", 404

@app.route('/cameras')
def list_cameras():
    cameras = []
    index = 0
    while True:
        cap = cv2.VideoCapture(index)
        if not cap.read()[0]:
            break
        else:
            cameras.append(index)
        cap.release()
        index += 1
    return jsonify(cameras)
def draw_boxes(image, results):
    for result in results:
        boxes = result.boxes
        for box in boxes:
            x1, y1, x2, y2 = box.xyxy[0].numpy()
            label = model.names[int(box.cls[0])]
            confidence = box.conf[0].numpy()
            
            # Draw the bounding box
            cv2.rectangle(image, (int(x1), int(y1)), (int(x2), int(y2)), (0, 255, 0), 2)
            # Put the label and confidence
            cv2.putText(image, f'{label} {confidence:.2f}', (int(x1), int(y1) - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)
    return image

def capture_tags(results, frame):
    connection = get_db_connection()
    cursor = connection.cursor()
    capture_image = False
    tags_to_capture = {"person", "wheel"}
    for result in results:
        boxes = result.boxes
        for box in boxes:
            label = model.names[int(box.cls[0])]
            if label in tags_to_capture:
                capture_image = True
                break
        if capture_image:
            break
    if capture_image:
        try:
            add_image = ("INSERT INTO images (image_data) VALUES (%s)")
            cursor.execute(add_image, (frame,))
            connection.commit()
        finally:
            cursor.close()
            connection.close()
            
def generate_frames(camera_index):
    cap = cv2.VideoCapture(camera_index)
    while True:
        success, frame = cap.read()
        if not success:
            break
        else:
            # Process the frame using YOLOv8
            results = model(frame)

            # Draw the bounding boxes on the frame
            frame = draw_boxes(frame, results)

            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()
            capture_tags(results,frame)
            yield (b'--frame\r\n'
                   b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/video_feed')
def video_feed():
    camera_index = int(request.args.get('camera_index', 0))
    return Response(generate_frames(camera_index),
                    mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)
