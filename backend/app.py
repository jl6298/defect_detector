
import tensorflow as tf
import numpy as np
import os
import mysql.connector
from flask import Flask, request, jsonify, render_template
from defect_detector import YoloPredictor
from input_optimization import input_optimization
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from utilities import process_image
 
app = Flask(__name__)


# Mysql数据库配置,内容可在.env文件中更改
db_config = {
    'user': os.getenv('MYSQL_USER'),# 用户名
    'password': os.getenv('MYSQL_PASSWORD'),# 用户密码
    'host': os.getenv('MYSQL_HOST'),# host
    'database': os.getenv('MYSQL_DB'),# 数据库名
    'port': os.getenv('MYSQL_PORT'),# 端口
}

# 载入YOLOv8模型
model = YoloPredictor('yolov8n.pt')
# 载入神经网络
predicator = load_model("./models/defect_prediction.h5")
predicator.compile(optimizer=Adam(), loss='sparse_categorical_crossentropy', metrics=['accuracy'])



# 连接数据库
def get_db_connection():
    return mysql.connector.connect(**db_config)

def capture_tags_db( frame):
    try:
        connection = get_db_connection() # 连接数据库
    except:
        return "fail to connect to database"
    cursor = connection.cursor()
    # 查找特定标签，如检测结果中包含特定标签则存入数据库
    try:
        add_image = ("INSERT INTO images (image_data) VALUES (%s)")
        cursor.execute(add_image, (frame,))
        connection.commit()
    finally:
        cursor.close()
        connection.close() # 结束连接
    return "success"

@app.route('/')
def index():
    """
    生成测试前端。
       
    Returns:
        生成的前端html界面。
    """
    return render_template('test.html')

@app.route('/process_image', methods=['POST']) # 实时图像识别接口
def receive_image():
    """
    接受前端发送的图片、处理、并将处理后的图片以及检测目标和fps送至前端。
        
    Returns:
        变为jsonified后的base64图片, 识别目标，以及fps。
    """
    data = request.get_json()
    image_data = data['image_data']
    tags = data['tags']
    result = process_image(image_data, tags, model)
    print(db_config)
    if (result["detection_results"] != {}):
        if (data['check_value'] == "checkedValue"):
            try:
                capture_tags_db(result["processed_image"])
            except:
                print("fail to import image to database")

    return jsonify(result)

@app.route('/process_parameter', methods=['POST']) # 离线参数优化接口
def receive_parameters():
    data = request.get_json()
    received_list = data['list']
    target = data["target"]
    x = tf.convert_to_tensor((np.array(received_list)).reshape(1,-1))
    iop = input_optimization(predicator, x)
    output = iop.optimize(target)
    output = output.numpy().flatten().tolist()
    return jsonify({'status': 'success', 'op_input': output})


@app.route('/db_config', methods=['POST'])
def receive_config():
    data = request.json
    # 提取数据库配置信息Extract database configuration details
    try:
        db_config = {
        'host': data.get('host'),
        'port': data.get('port'),
        'user': data.get('user'),
        'password': data.get('password'),
        'database': data.get('database')
        }
    except:
        return jsonify({'status': 'fail, 数据格式不正确', 'received_config': {}})
    
    # 返回
    return jsonify({'status': 'success', 'received_config': db_config})


if __name__ == '__main__':
    app.run(debug=True)
