import cv2
import time
import mediapipe as mp
import numpy as np
from collections import deque

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# 全域變數
latest_result = None

# 回傳函數
def print_result(result, _output_image, _timestamp_ms):
    global latest_result
    latest_result = result

# ==========================================
# 功能函式區塊 (Functions)
# ==========================================
def apply_night_mode(frame, clahe):
    """判斷畫面亮度，若太暗則套用 CLAHE 夜間增強模式"""
    average_brightness = np.mean(frame)
    if average_brightness < 80:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        cv2.putText(frame, "CLAHE Night Mode ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def generate_fatigue_report(timestamps, alertness, perclos, nod, jaw, turn, is_calibrated, nod_thr, yawn_thr):
    """程式結束後，將收集到的數據匯出為 CSV 與折線圖"""
    print("\n正在生成分析報告...")
    try:
        import csv
        with open('fatigue_data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time(s)', 'Alertness Score', 'PERCLOS(%)', 'Nod Ratio', 'Jaw Open', 'Turn Ratio'])
            for i in range(len(timestamps)):
                writer.writerow([f"{timestamps[i]:.2f}", alertness[i], f"{perclos[i]:.2f}", f"{nod[i]:.3f}", f"{jaw[i]:.3f}", f"{turn[i]:.3f}"])
        print("✅ 已成功匯出數據報表：fatigue_data.csv")

        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 支援中文顯示 (微軟正黑體)
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.subplot(4, 1, 1)
        plt.plot(timestamps, alertness, label='精神分數 (Alertness)', color='green')
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('分數')
        plt.title('駕駛疲勞偵測分析報告')
        plt.legend(loc='upper right')

        plt.subplot(4, 1, 2)
        plt.plot(timestamps, perclos, label='閉眼率 PERCLOS (%)', color='blue')
        plt.axhline(y=15, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('百分比 %')
        plt.legend(loc='upper right')

        plt.subplot(4, 1, 3)
        plt.plot(timestamps, nod, label='低頭比例 (Nod Ratio)', color='purple')
        plt.axhline(y=nod_thr if is_calibrated else 0.3, color='red', linestyle='--', alpha=0.7, label='低頭門檻')
        plt.ylabel('比例')
        plt.legend(loc='upper right')

        plt.subplot(4, 1, 4)
        plt.plot(timestamps, jaw, label='嘴巴張開 (Jaw Open)', color='brown')
        plt.axhline(y=yawn_thr if is_calibrated else 0.8, color='red', linestyle='--', alpha=0.7, label='打哈欠門檻')
        plt.xlabel('時間 (秒)')
        plt.ylabel('分數')
        plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('fatigue_report.png', dpi=150)
        print("✅ 已成功匯出分析圖表：fatigue_report.png")
        plt.show() # 自動彈出視窗顯示圖表
    except ImportError:
        print("⚠️ 無法繪製圖表：缺少 matplotlib 模組。")
        print("提示: 請開啟終端機，輸入 'pip install matplotlib' 來安裝繪圖套件！")

class DriverFatigueDetector:
    def __init__(self):
        # PERCLOS 滑動視窗設定 (視窗大小 150 幀)
        self.perclos_window_size = 150
        self.eye_closure_history = deque(maxlen=self.perclos_window_size)
        self.open_mouth_start_time = None
        self.yawn_timestamps = deque()
        self.current_yawn_counted = False

        # 動態校正設定
        self.CALIBRATION_FRAMES = 100
        self.calibration_counter = 0
        self.jaw_open_baseline_scores = []
        self.eye_blink_baseline_scores = []
        self.nod_ratio_baseline_scores = []
        self.turn_ratio_baseline_scores = []
        self.yawn_threshold = 0.8
        self.eye_close_threshold = 0.40
        self.nod_threshold = 0.30
        self.turn_center_baseline = 0.5
        self.TURN_MARGIN = 0.25
        self.is_calibrated = False

        self.last_calibration_time = time.time()
        self.RECALIBRATION_INTERVAL = 300

        # 紀錄數據供輸出圖表使用
        self.log_timestamps = []
        self.log_alertness = []
        self.log_perclos = []
        self.log_nod_ratio = []
        self.log_jaw_open = []
        self.log_turn_ratio = []
        self.distraction_start_time = None
        self.start_record_time = time.time()

    def force_recalibrate(self):
        """手動強制重新校正"""
        self.last_calibration_time = 0

    def analyze_and_draw(self, frame, blendshapes, landmarks):
        """核心邏輯：接收特徵點，計算疲勞數值並直接繪製在畫面上"""
        current_time = time.time()
        
        eyeBlinkLeft = blendshapes[9].score
        eyeBlinkRight = blendshapes[10].score
        JawOpen = blendshapes[25].score
        
        # 抓取臉部特徵點，計算低頭比例
        nose_y = landmarks[1].y
        chin_y = landmarks[152].y
        forehead_y = landmarks[10].y
        face_height = chin_y - forehead_y
        nod_ratio = (chin_y - nose_y) / face_height if face_height > 0 else 0.5
        
        # 抓取臉部特徵點，計算左右轉頭比例
        left_x = landmarks[234].x
        right_x = landmarks[454].x
        nose_x = landmarks[1].x
        face_width = right_x - left_x
        turn_ratio = (nose_x - left_x) / face_width if face_width > 0 else 0.5

        # 動態判斷狀態
        is_nodding_off = nod_ratio < self.nod_threshold
        nod_penalty = 40 if is_nodding_off else 0

        is_eyes_closed = (eyeBlinkLeft >= self.eye_close_threshold and eyeBlinkRight >= self.eye_close_threshold)
        self.eye_closure_history.append(1 if is_eyes_closed else 0)
        current_perclos = sum(self.eye_closure_history) / len(self.eye_closure_history) if self.eye_closure_history else 0
        
        is_looking_away = abs(turn_ratio - self.turn_center_baseline) > self.TURN_MARGIN
        if is_looking_away:
            if self.distraction_start_time is None:
                self.distraction_start_time = current_time
            distraction_duration = current_time - self.distraction_start_time
        else:
            self.distraction_start_time = None
            distraction_duration = 0
            
        distraction_penalty = 40 if distraction_duration > 3.0 else 0

        # 精神分數計算與紀錄
        spirit_score = 100 - (current_perclos / 0.15) * 50 - (JawOpen * 20) - nod_penalty - distraction_penalty
        spirit_score = max(0, min(100, int(spirit_score)))
        
        self.log_timestamps.append(current_time - self.start_record_time)
        self.log_alertness.append(spirit_score)
        self.log_perclos.append(current_perclos * 100)
        self.log_nod_ratio.append(nod_ratio)
        self.log_jaw_open.append(JawOpen)
        self.log_turn_ratio.append(turn_ratio)

        # ---------------- 畫面繪製區塊 ---------------- #
        score_color = (0, 255, 0) if spirit_score > 60 else (0, 255, 255) if spirit_score > 30 else (0, 0, 255)
        cv2.putText(frame, f"Alertness: {spirit_score}/100", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        cv2.putText(frame, f"Nod Ratio: {nod_ratio:.3f} (Thr: {self.nod_threshold:.3f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
        cv2.putText(frame, f"Turn Ratio: {turn_ratio:.3f} (Center: {self.turn_center_baseline:.3f})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
        
        if is_nodding_off: cv2.putText(frame, "WARNING: HEAD DROP!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        if distraction_duration > 3.0: cv2.putText(frame, "WARNING: DISTRACTED (LOOKING AWAY)!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
        elif distraction_duration > 1.5: cv2.putText(frame, "Pay Attention to the Road...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        if len(self.eye_closure_history) == self.perclos_window_size:
            cv2.putText(frame, f"PERCLOS: {current_perclos*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if current_perclos >= 0.15:
                print(f"[PERCLOS 警告] 閉眼率達 {current_perclos*100:.1f}%, 左眼: {eyeBlinkLeft:.3f}, 右眼: {eyeBlinkRight:.3f}")
                cv2.putText(frame, "WARNING: FATIGUE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            cv2.putText(frame, f"PERCLOS: Calibrating ({len(self.eye_closure_history)}/{self.perclos_window_size})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        while self.yawn_timestamps and current_time - self.yawn_timestamps[0] > 60: self.yawn_timestamps.popleft()
        cv2.putText(frame, f"Yawns (1min): {len(self.yawn_timestamps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # ---------------- 動態校正區塊 ---------------- #
        if self.is_calibrated and (current_time - self.last_calibration_time > self.RECALIBRATION_INTERVAL):
            self.is_calibrated, self.calibration_counter = False, 0
            for lst in [self.jaw_open_baseline_scores, self.eye_blink_baseline_scores, self.nod_ratio_baseline_scores, self.turn_ratio_baseline_scores]: lst.clear()
            print(f"\n[{self.RECALIBRATION_INTERVAL}秒定時] 環境可能改變，啟動自動重新校正...")

        if not self.is_calibrated:
            cv2.putText(frame, f"Calibrating... Look Straight ({self.calibration_counter}/{self.CALIBRATION_FRAMES})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.calibration_counter < self.CALIBRATION_FRAMES:
                self.jaw_open_baseline_scores.append(JawOpen)
                self.eye_blink_baseline_scores.append(max(eyeBlinkLeft, eyeBlinkRight))
                self.nod_ratio_baseline_scores.append(nod_ratio)
                self.turn_ratio_baseline_scores.append(turn_ratio)
                self.calibration_counter += 1
            elif self.jaw_open_baseline_scores: # 校正完成
                self.yawn_threshold = np.mean(self.jaw_open_baseline_scores) + 0.4 
                self.eye_close_threshold = np.mean(self.eye_blink_baseline_scores) + 0.3
                self.nod_threshold = np.mean(self.nod_ratio_baseline_scores) - 0.06
                self.turn_center_baseline = np.mean(self.turn_ratio_baseline_scores)
                self.is_calibrated = True
                self.last_calibration_time = current_time
                print(f"校正完成！低頭門檻: {self.nod_threshold:.3f} | 直視基準: {self.turn_center_baseline:.3f}")
        elif JawOpen >= self.yawn_threshold: # 偵測打哈欠
            if self.open_mouth_start_time is None:
                self.open_mouth_start_time, self.current_yawn_counted = current_time, False
            yawn_duration = current_time - self.open_mouth_start_time
            if yawn_duration >= 1.5:
                cv2.putText(frame, "WARNING: SEVERE YAWNING!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                if not self.current_yawn_counted:
                    self.yawn_timestamps.append(current_time)
                    self.current_yawn_counted = True
                    print(f"[警告] 偵測到嚴重打哈欠！過去 1 分鐘內累計: {len(self.yawn_timestamps)} 次")
            elif yawn_duration >= 0.5: cv2.putText(frame, "WARNING: YAWNING...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            self.open_mouth_start_time, self.current_yawn_counted = None, False

# 模型路徑 (請確認 face_landmarker.task 在旁邊)
model_path = 'face_landmarker.task'

# 3. 啟動偵測器
try:
    options = FaceLandmarkerOptions(
        base_options=BaseOptions(model_asset_path=model_path),
        running_mode=VisionRunningMode.LIVE_STREAM,
        result_callback=print_result,
        num_faces=1,
        output_face_blendshapes=True,
        # [針對 NITYMED 調整] 降低臉部偵測門檻，讓模型在紅外線/昏暗影像中不容易跟丟人臉
        min_face_detection_confidence=0.4,
        min_face_presence_confidence=0.4,
        min_tracking_confidence=0.4
    )
    detector = FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"錯誤：模型載入失敗！請確認 {model_path} 檔案存在。")
    print(f"詳細錯誤: {e}")
    exit()

# 開啟攝影機
cap = cv2.VideoCapture(0)
print("系統啟動成功！(手動繪圖模式)")

# 初始化疲勞偵測物件
fatigue_detector = DriverFatigueDetector()

# 特徵點顯示開關 (預設為開啟)
show_landmarks = True

# [效能優化] 在迴圈外先建立好 CLAHE 增強器，避免每幀重複建立浪費資源
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 鏡像並轉 RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 套用夜間模式增強 (已獨立成函式)
    frame = apply_night_mode(frame, clahe)

    # 傳送給模型
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)

    # 看看有沒有收到表情成績單
    if latest_result and latest_result.face_blendshapes:
        # 拿出畫面中第一個人的所有表情分數
        blendshapes = latest_result.face_blendshapes[0]
        landmarks = latest_result.face_landmarks[0]
        
        # 將資料交給物件進行分析與畫面繪製
        fatigue_detector.analyze_and_draw(frame, blendshapes, landmarks)

    


    # 4. 手動把點畫出來 (不用官方繪圖工具，避免報錯)
    if show_landmarks and latest_result and latest_result.face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in latest_result.face_landmarks:
            
            # 遍歷 478 個特徵點
            for idx, landmark in enumerate(face_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 畫綠色小點
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # 畢業專題特效：特別標註鼻尖 (1) 與 左右臉頰邊界 (234, 454)
                if idx in [1, 234, 454]:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('Final Project Demo', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # 按 ESC 退出
        break
    elif key == ord('c'): # [新增] 按下小寫 'c' 鍵，強制手動觸發重新校正
        fatigue_detector.force_recalibrate() # 故意把最後校正時間歸零，下個迴圈就會觸發校正
        print("\n[手動觸發] 啟動重新校正...")
    elif key == ord('l'): # [新增] 按下小寫 'l' 鍵，切換特徵點顯示狀態
        show_landmarks = not show_landmarks
        print(f"\n[切換顯示] 特徵點顯示狀態: {'開啟' if show_landmarks else '關閉'}")

detector.close()
cap.release()
cv2.destroyAllWindows()

# 呼叫生成報表函式
generate_fatigue_report(fatigue_detector.log_timestamps, fatigue_detector.log_alertness, fatigue_detector.log_perclos, fatigue_detector.log_nod_ratio, fatigue_detector.log_jaw_open, fatigue_detector.log_turn_ratio, fatigue_detector.is_calibrated, fatigue_detector.nod_threshold, fatigue_detector.yawn_threshold)