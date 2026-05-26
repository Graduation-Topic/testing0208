import cv2
import time
import os
from flask import Flask, render_template, Response, jsonify, send_file
from threading import Thread
from detector import AIVisionEngine

app = Flask(__name__)

# 建立 AI 視覺引擎實例
engine = AIVisionEngine(model_path='face_landmarker.task')

def generate_frames():
    """讀取鏡頭，並產生 MJPEG 串流供網頁使用"""
    cap = cv2.VideoCapture(0)
    prev_frame_time = 0

    while True:
        success, frame = cap.read()
        if not success:
            break

        # 鏡像翻轉
        frame = cv2.flip(frame, 1)
        new_frame_time = time.time()

        # 1. 交給我們的 AI 引擎處理
        processed_frame = engine.process_frame(frame)

        # 2. 計算 FPS 並存入引擎供前端讀取
        if prev_frame_time != 0:
            fps = int(1 / (new_frame_time - prev_frame_time))
            engine.current_fps = fps
        prev_frame_time = new_frame_time

        # 3. 將 OpenCV 的圖像編碼轉換為 JPEG，並利用 Generator 產出串流
        ret, buffer = cv2.imencode('.jpg', processed_frame)
        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/')
def index():
    """載入首頁範本"""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """負責吐出串流影像的 API 路徑"""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/force_recalibrate', methods=['POST'])
def force_recalibrate():
    engine.force_recalibrate()
    return jsonify({"status": "success", "message": "重新校正中，請直視前方..."})

@app.route('/toggle_landmarks', methods=['POST'])
def toggle_landmarks():
    status = engine.toggle_landmarks()
    msg = "✅ 特徵點顯示已開啟" if status else "❌ 特徵點顯示已關閉"
    return jsonify({"status": "success", "message": msg})

@app.route('/generate_report', methods=['POST'])
def generate_report():
    # 將耗時的報表生成任務放到背景執行緒，避免網頁請求被卡住
    thread = Thread(target=engine.generate_report)
    thread.start()
    return jsonify({"status": "success", "message": "✅ 指令已送出！報表將於背景生成..."})

@app.route('/report_image')
def report_image():
    """回傳生成的圖表給網頁前端顯示"""
    if os.path.exists('fatigue_report.png'):
        return send_file('fatigue_report.png', mimetype='image/png')
    return "尚未生成報告", 404

@app.route('/realtime_data')
def realtime_data():
    """回傳即時狀態數據給前端畫圖 (擷取最新 100 筆)"""
    return jsonify(engine.get_realtime_data(100))

if __name__ == '__main__':
    print("伺服器啟動中，請打開瀏覽器前往 http://localhost:5000")
    # 這裡務必設定 debug=False，以避免 MediaPipe 與 Flask 的 reload 執行緒衝突
    app.run(host='0.0.0.0', port=5000, debug=False)