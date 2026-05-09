import cv2
import time
import mediapipe as mp
import numpy as np
from collections import deque
import winsound

from mediapipe.tasks import python
from mediapipe.tasks.python import vision

BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

def apply_night_mode(frame, clahe):
    """判斷畫面亮度，若太暗則套用 CLAHE 夜間增強模式"""
    h, w = frame.shape[:2]
    center_region = frame[h//4 : 3*h//4, w//4 : 3*w//4]
    average_brightness = np.mean(center_region)
    
    cv2.putText(frame, f"Brightness: {average_brightness:.1f}", (10, h - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (200, 200, 200), 2)

    if average_brightness < 70:
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        cl = clahe.apply(l)
        merged_lab = cv2.merge((cl, a, b))
        frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        cv2.putText(frame, "CLAHE Night Mode ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    return frame

def generate_fatigue_report(timestamps, alertness, perclos, nod, jaw, turn, is_calibrated, nod_thr, yawn_thr):
    """程式結束或網頁觸發後，將收集到的數據匯出為 CSV 與折線圖"""
    print("\n正在生成分析報告...")
    try:
        import csv
        with open('fatigue_data.csv', 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(['Time(s)', 'Alertness Score', 'PERCLOS(%)', 'Nod Ratio', 'Jaw Open', 'Turn Ratio'])
            for i in range(len(timestamps)):
                writer.writerow([f"{timestamps[i]:.2f}", alertness[i], f"{perclos[i]:.2f}", f"{nod[i]:.3f}", f"{jaw[i]:.3f}", f"{turn[i]:.3f}"])
        print("✅ 已成功匯出數據報表：fatigue_data.csv")

        import matplotlib
        matplotlib.use('Agg') # 在 Flask 伺服器環境中，避免繪圖視窗卡死主執行緒
        import matplotlib.pyplot as plt
        plt.figure(figsize=(12, 10))
        plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 支援中文顯示
        plt.rcParams['axes.unicode_minus'] = False
        
        plt.subplot(4, 1, 1)
        plt.plot(timestamps, alertness, label='精神分數 (Alertness)', color='green')
        plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
        plt.axhline(y=30, color='red', linestyle='--', alpha=0.7)
        plt.ylabel('分數')
        plt.title('駕駛疲勞偵測分析報告')
        plt.legend(loc='upper right')

        for i, (data, label, color, thr) in enumerate([
            (perclos, '閉眼率 PERCLOS (%)', 'blue', 15),
            (nod, '低頭比例 (Nod Ratio)', 'purple', nod_thr if is_calibrated else 0.3),
            (jaw, '嘴巴張開 (Jaw Open)', 'brown', yawn_thr if is_calibrated else 0.8)
        ], start=2):
            plt.subplot(4, 1, i)
            plt.plot(timestamps, data, label=label, color=color)
            plt.axhline(y=thr, color='red', linestyle='--', alpha=0.7)
            plt.legend(loc='upper right')

        plt.tight_layout()
        plt.savefig('fatigue_report.png', dpi=150)
        print("✅ 已成功匯出分析圖表：fatigue_report.png (已存檔，請至資料夾查看)")
    except ImportError:
        print("⚠️ 無法繪製圖表：缺少 matplotlib 模組。請安裝 'pip install matplotlib'")

class DriverFatigueDetector:
    """疲勞數值與邏輯運算的核心物件"""
    def __init__(self):
        self.perclos_window_size = 150
        self.eye_closure_history = deque(maxlen=self.perclos_window_size)
        self.open_mouth_start_time = None
        self.yawn_timestamps = deque()
        self.current_yawn_counted = False

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

        self.log_timestamps = []
        self.log_alertness = []
        self.log_perclos = []
        self.log_nod_ratio = []
        self.log_jaw_open = []
        self.log_turn_ratio = []
        self.distraction_start_time = None
        self.last_beep_time = 0
        self.start_record_time = time.time()

    def force_recalibrate(self):
        self.last_calibration_time = 0

    def trigger_audio_alarm(self, current_time):
        if current_time - self.last_beep_time > 1.0:
            winsound.PlaySound("SystemHand", winsound.SND_ALIAS | winsound.SND_ASYNC)
            self.last_beep_time = current_time

    def analyze_and_draw(self, frame, blendshapes, landmarks):
        current_time = time.time()
        
        eyeBlinkLeft = blendshapes[9].score
        eyeBlinkRight = blendshapes[10].score
        JawOpen = blendshapes[25].score
        
        nose_y = landmarks[1].y
        chin_y = landmarks[152].y
        forehead_y = landmarks[10].y
        face_height = chin_y - forehead_y
        nod_ratio = (chin_y - nose_y) / face_height if face_height > 0 else 0.5
        
        left_x = landmarks[234].x
        right_x = landmarks[454].x
        nose_x = landmarks[1].x
        face_width = right_x - left_x
        turn_ratio = (nose_x - left_x) / face_width if face_width > 0 else 0.5

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

        spirit_score = 100 - (current_perclos / 0.15) * 50 - (JawOpen * 20) - nod_penalty - distraction_penalty
        spirit_score = max(0, min(100, int(spirit_score)))
        
        self.log_timestamps.append(current_time - self.start_record_time)
        self.log_alertness.append(spirit_score)
        self.log_perclos.append(current_perclos * 100)
        self.log_nod_ratio.append(nod_ratio)
        self.log_jaw_open.append(JawOpen)
        self.log_turn_ratio.append(turn_ratio)

        score_color = (0, 255, 0) if spirit_score > 60 else (0, 255, 255) if spirit_score > 30 else (0, 0, 255)
        cv2.putText(frame, f"Alertness: {spirit_score}/100", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        cv2.putText(frame, f"Nod Ratio: {nod_ratio:.3f} (Thr: {self.nod_threshold:.3f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
        cv2.putText(frame, f"Turn Ratio: {turn_ratio:.3f} (Center: {self.turn_center_baseline:.3f})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
        
        if is_nodding_off: 
            cv2.putText(frame, "WARNING: HEAD DROP!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            self.trigger_audio_alarm(current_time)
            
        if distraction_duration > 3.0: 
            cv2.putText(frame, "WARNING: DISTRACTED!", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 165, 255), 2)
            self.trigger_audio_alarm(current_time)
        elif distraction_duration > 1.5: 
            cv2.putText(frame, "Pay Attention to the Road...", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            
        if len(self.eye_closure_history) == self.perclos_window_size:
            cv2.putText(frame, f"PERCLOS: {current_perclos*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            if current_perclos >= 0.15:
                cv2.putText(frame, "WARNING: FATIGUE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.trigger_audio_alarm(current_time)
        else:
            cv2.putText(frame, f"PERCLOS: Calibrating ({len(self.eye_closure_history)}/{self.perclos_window_size})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        while self.yawn_timestamps and current_time - self.yawn_timestamps[0] > 60: self.yawn_timestamps.popleft()
        cv2.putText(frame, f"Yawns (1min): {len(self.yawn_timestamps)}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        if self.is_calibrated and (current_time - self.last_calibration_time > self.RECALIBRATION_INTERVAL):
            self.is_calibrated, self.calibration_counter = False, 0
            for lst in [self.jaw_open_baseline_scores, self.eye_blink_baseline_scores, self.nod_ratio_baseline_scores, self.turn_ratio_baseline_scores]: lst.clear()

        if not self.is_calibrated:
            cv2.putText(frame, f"Calibrating... Look Straight ({self.calibration_counter}/{self.CALIBRATION_FRAMES})", (10, 210), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if self.calibration_counter < self.CALIBRATION_FRAMES:
                self.jaw_open_baseline_scores.append(JawOpen)
                self.eye_blink_baseline_scores.append(max(eyeBlinkLeft, eyeBlinkRight))
                self.nod_ratio_baseline_scores.append(nod_ratio)
                self.turn_ratio_baseline_scores.append(turn_ratio)
                self.calibration_counter += 1
            elif self.jaw_open_baseline_scores:
                self.yawn_threshold = np.mean(self.jaw_open_baseline_scores) + 0.4 
                self.eye_close_threshold = np.mean(self.eye_blink_baseline_scores) + 0.3
                self.nod_threshold = np.mean(self.nod_ratio_baseline_scores) - 0.06
                self.turn_center_baseline = np.mean(self.turn_ratio_baseline_scores)
                self.is_calibrated = True
                self.last_calibration_time = current_time
        elif JawOpen >= self.yawn_threshold:
            if self.open_mouth_start_time is None:
                self.open_mouth_start_time, self.current_yawn_counted = current_time, False
            yawn_duration = current_time - self.open_mouth_start_time
            if yawn_duration >= 1.5:
                cv2.putText(frame, "WARNING: SEVERE YAWNING!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                self.trigger_audio_alarm(current_time)
                if not self.current_yawn_counted:
                    self.yawn_timestamps.append(current_time)
                    self.current_yawn_counted = True
            elif yawn_duration >= 0.5: cv2.putText(frame, "WARNING: YAWNING...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        else:
            self.open_mouth_start_time, self.current_yawn_counted = None, False

class AIVisionEngine:
    """將 MediaPipe 與 FatigueDetector 包裝成單一物件，供 Web 伺服器使用"""
    def __init__(self, model_path='face_landmarker.task'):
        self.latest_result = None
        self.fatigue_detector = DriverFatigueDetector()
        self.clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        self.show_landmarks = True
        
        def result_callback(result, _output_image, _timestamp_ms):
            self.latest_result = result
            
        options = FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=model_path),
            running_mode=VisionRunningMode.LIVE_STREAM,
            result_callback=result_callback,
            num_faces=1,
            output_face_blendshapes=True,
            min_face_detection_confidence=0.4,
            min_face_presence_confidence=0.4,
            min_tracking_confidence=0.4
        )
        self.detector = FaceLandmarker.create_from_options(options)

    def toggle_landmarks(self):
        self.show_landmarks = not self.show_landmarks
        return self.show_landmarks
        
    def force_recalibrate(self):
        self.fatigue_detector.force_recalibrate()
        
    def generate_report(self):
        fd = self.fatigue_detector
        generate_fatigue_report(
            fd.log_timestamps, fd.log_alertness, fd.log_perclos, 
            fd.log_nod_ratio, fd.log_jaw_open, fd.log_turn_ratio, 
            fd.is_calibrated, fd.nod_threshold, fd.yawn_threshold
        )
        
    def get_realtime_data(self, max_points=100):
        """取得最新的觀測數據供前端即時繪圖使用"""
        fd = self.fatigue_detector
        return {
            "timestamps": fd.log_timestamps[-max_points:],
            "alertness": fd.log_alertness[-max_points:],
            "perclos": fd.log_perclos[-max_points:],
            "nod": fd.log_nod_ratio[-max_points:],
            "jaw": fd.log_jaw_open[-max_points:]
        }

    def process_frame(self, frame):
        """接收原始畫面，處理完畢後回傳帶有 UI 圖層的畫面"""
        frame = apply_night_mode(frame, self.clahe)
        
        # 轉存給 MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        timestamp = int(time.time() * 1000)
        self.detector.detect_async(mp_image, timestamp)
        
        # 繪製分析結果
        if self.latest_result and self.latest_result.face_blendshapes:
            self.fatigue_detector.analyze_and_draw(frame, self.latest_result.face_blendshapes[0], self.latest_result.face_landmarks[0])
            
            # 特徵點繪製 (Mytest.py 移植)
            if self.show_landmarks and self.latest_result.face_landmarks:
                h, w, _ = frame.shape
                for face_landmarks in self.latest_result.face_landmarks:
                    for idx, landmark in enumerate(face_landmarks):
                        cx, cy = int(landmark.x * w), int(landmark.y * h)
                        cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)
                        if idx in [1, 10, 152, 234, 454]:
                            cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)
            
        return frame