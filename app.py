from flask import Flask, render_template, request, jsonify
import datetime

app = Flask(__name__)

# 暫存違規紀錄的 List (未來你可以把它改成存入 SQLite 或是寫成 CSV 檔案)
fatigue_logs = []

@app.route('/')
def index():
    """載入前端網頁 (AI 現在完全在這個網頁裡面跑)"""
    return render_template('index.html')

@app.route('/api/log_warning', methods=['POST'])
def log_warning():
    """接收前端 (JavaScript) 傳來的疲勞或違規事件"""
    data = request.json
    
    # 建立一筆新的違規紀錄
    event = {
        "time": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "warning_type": data.get("type", "Unknown"),
        "spirit_score": data.get("score", 0),
        "perclos": data.get("perclos", 0)
    }
    
    fatigue_logs.append(event)
    
    # 在後端終端機印出警告，證明伺服器有收到
    print(f"🚨 [伺服器接收] {event['time']} | 警告: {event['warning_type']} | 分數: {event['spirit_score']}")
    
    return jsonify({"status": "success", "message": "已將紀錄存入伺服器"})

@app.route('/api/get_report', methods=['GET'])
def get_report():
    """(擴充功能) 供車隊後台查看今天的違規紀錄"""
    return jsonify({
        "total_warnings": len(fatigue_logs),
        "logs": fatigue_logs
    })

if __name__ == '__main__':
    print("🚀 輕量化 API 伺服器啟動中... 請前往 http://127.0.0.1:5000")
    # 開啟 debug=True 方便開發
    app.run(host='0.0.0.0', port=5000, debug=True)