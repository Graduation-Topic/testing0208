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
cap = cv2.VideoCapture(1)
print("系統啟動成功！(手動繪圖模式)")

# PERCLOS 滑動視窗設定 (視窗大小 150 幀)
perclos_window_size = 150
eye_closure_history = deque(maxlen=perclos_window_size)
open_mouth_start_time = None  # 紀錄剛開始打開嘴巴的那個瞬間
yawn_timestamps = deque()     # 紀錄每次打哈欠的時間戳記
current_yawn_counted = False  # 避免同一次打哈欠被重複計算

# [新增] 動態校正打哈欠門檻，解決鬍子或光線造成的基準值問題
CALIBRATION_FRAMES = 100  # 用前 100 幀來校正
calibration_counter = 0
jaw_open_baseline_scores = []
eye_blink_baseline_scores = []  # 紀錄睜眼時的分數
nod_ratio_baseline_scores = []  # [新增] 紀錄正常平視時的頭部比例
yawn_threshold = 0.8 # 先給一個很高的預設值，校正後會被覆蓋
eye_close_threshold = 0.40 # 預設閉眼門檻
nod_threshold = 0.30 # [新增] 預設低頭門檻
is_calibrated = False

# [新增] 定時重新校正設定
last_calibration_time = time.time()
RECALIBRATION_INTERVAL = 300  # 預設每 300 秒 (5分鐘) 自動重新校正一次，可依需求更改

# [新增] 特徵點顯示開關 (預設為開啟)
show_landmarks = True


# [新增] 紀錄數據供輸出圖表使用
log_timestamps = []
log_alertness = []
log_perclos = []
log_nod_ratio = []
log_jaw_open = []
start_record_time = time.time()

# [效能優化] 在迴圈外先建立好 CLAHE 增強器，避免每幀重複建立浪費資源
clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 鏡像並轉 RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 1. 計算整張畫面的「平均亮度」
    average_brightness = np.mean(frame)
    
    # 2. 如果環境太暗，啟動 CLAHE 專業夜間增強模式
    if average_brightness < 80:
        # [步驟 A] 將影像從 BGR 轉換到 LAB 色彩空間 (L 代表亮度，A 和 B 代表色彩)
        lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
        
        # 把亮度跟顏色拆開來
        l, a, b = cv2.split(lab)
        
        # 只針對 L (亮度) 通道進行局部對比強化！
        cl = clahe.apply(l)
        
        # [步驟 C] 把強化後的亮度，跟原本的顏色合併回去
        merged_lab = cv2.merge((cl, a, b))
        
        # 轉回 OpenCV 預設的 BGR 彩色格式
        frame = cv2.cvtColor(merged_lab, cv2.COLOR_LAB2BGR)
        
        # 在畫面上加個提示，讓你知道 CLAHE 啟動了 🌙
        cv2.putText(frame, "CLAHE Night Mode ON", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 傳送給模型
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)

    # 看看有沒有收到表情成績單
    if latest_result and latest_result.face_blendshapes:
        # 拿出畫面中第一個人的所有表情分數
        blendshapes = latest_result.face_blendshapes[0]

        eyeBlinkLeft = blendshapes[9].score
        eyeBlinkRight = blendshapes[10].score
        JawOpen = blendshapes[25].score
        
        # [新增] 抓取臉部特徵點，用來計算頭部是否垂下 (點頭打瞌睡)
        landmarks = latest_result.face_landmarks[0]
        nose_y = landmarks[1].y
        chin_y = landmarks[152].y
        forehead_y = landmarks[10].y
        
        face_height = chin_y - forehead_y
        # 正常平視時 nod_ratio 大約 0.45~0.5。低頭時下巴與鼻子距離會縮短，比例會小於 0.35
        nod_ratio = (chin_y - nose_y) / face_height if face_height > 0 else 0.5
        
        # 動態判斷低頭 (使用校正後的門檻)
        is_nodding_off = nod_ratio < nod_threshold
        nod_penalty = 40 if is_nodding_off else 0

        # 動態判斷眼睛是否閉上 (使用校正後的門檻)
        is_eyes_closed = (eyeBlinkLeft >= eye_close_threshold and eyeBlinkRight >= eye_close_threshold)
        
        # 記錄當前幀的狀態進入滑動視窗 (閉眼為 1，睜眼為 0)
        eye_closure_history.append(1 if is_eyes_closed else 0)
        
        # 計算當前的 PERCLOS 值 (供精神分數計算使用)
        current_perclos = sum(eye_closure_history) / len(eye_closure_history)
        
        # [新增] 精神分數計算 (Alertness Score)
        # 滿分 100。PERCLOS 達 15% (0.15) 時約扣 50 分；嘴巴張開 (JawOpen) 最多扣 20 分
        # 新增：點頭打瞌睡 (is_nodding_off) 扣 40 分
        spirit_score = 100 - (current_perclos / 0.15) * 50 - (JawOpen * 20) - nod_penalty
        spirit_score = max(0, min(100, int(spirit_score))) # 限制分數在 0~100 之間
        
        # [新增] 紀錄當前幀的數據供最後畫圖使用
        current_time_sec = time.time() - start_record_time
        log_timestamps.append(current_time_sec)
        log_alertness.append(spirit_score)
        log_perclos.append(current_perclos * 100)
        log_nod_ratio.append(nod_ratio)
        log_jaw_open.append(JawOpen)

        # 依分數決定顏色 (綠 -> 黃 -> 紅)
        score_color = (0, 255, 0) if spirit_score > 60 else (0, 255, 255) if spirit_score > 30 else (0, 0, 255)
        
        # 即時顯示精神分數在畫面上 (位置在 PERCLOS 下方)
        cv2.putText(frame, f"Alertness: {spirit_score}/100", (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)
        
        # 即時顯示低頭比例 (Nod Ratio) 與門檻，方便測試與微調
        cv2.putText(frame, f"Nod Ratio: {nod_ratio:.3f} (Thr: {nod_threshold:.3f})", (10, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 150, 255), 2)
        
        # 如果偵測到低頭打瞌睡，在畫面上顯示紅字警告
        if is_nodding_off:
            cv2.putText(frame, "WARNING: HEAD DROP!", (50, 200), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
        # 計算 PERCLOS 值並判斷
        if len(eye_closure_history) == perclos_window_size:
            perclos_value = current_perclos
            
            # 即時印出 PERCLOS 數值在畫面上 (青色)
            cv2.putText(frame, f"PERCLOS: {perclos_value*100:.1f}%", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            if perclos_value >= 0.15:  # 門檻 15%
                print(f"[PERCLOS 警告] 閉眼率達 {perclos_value*100:.1f}%")
                print(f"左眼 ({blendshapes[9].category_name}) 分數: {eyeBlinkLeft:.3f}")
                print(f"右眼 ({blendshapes[10].category_name}) 分數: {eyeBlinkRight:.3f}")
                print("-------------------")
                # 顯示警告
                cv2.putText(frame, "WARNING: FATIGUE DETECTED!", (50, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            # 滑動視窗未滿 150 幀時，顯示校正/收集進度
            cv2.putText(frame, f"PERCLOS: Calibrating ({len(eye_closure_history)}/{perclos_window_size})", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        
        # 移除過期的哈欠紀錄 (只保留過去 60 秒內的)
        current_time = time.time()
        while yawn_timestamps and current_time - yawn_timestamps[0] > 60:
            yawn_timestamps.popleft()
        yawn_count_1min = len(yawn_timestamps)
        
        # 即時顯示 1 分鐘內打哈欠次數在畫面上 (位置在精神分數下方)
        cv2.putText(frame, f"Yawns (1min): {yawn_count_1min}", (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 200, 0), 2)

        # [新增] 檢查是否需要定時自動重新校正
        if is_calibrated and (current_time - last_calibration_time > RECALIBRATION_INTERVAL):
            is_calibrated = False
            calibration_counter = 0
            jaw_open_baseline_scores.clear()
            eye_blink_baseline_scores.clear()
            nod_ratio_baseline_scores.clear() # [新增] 清除舊的頭部比例基準
            print(f"\n[{RECALIBRATION_INTERVAL}秒定時] 環境可能改變，啟動自動重新校正...")

        # [新增] 動態校正打哈欠門檻
        if not is_calibrated:
            cv2.putText(frame, f"Calibrating... Keep face neutral ({calibration_counter}/{CALIBRATION_FRAMES})", (10, 180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
            if calibration_counter < CALIBRATION_FRAMES:
                jaw_open_baseline_scores.append(JawOpen)
                # 記錄雙眼中較高的分數做為保守的睜眼基準
                eye_blink_baseline_scores.append(max(eyeBlinkLeft, eyeBlinkRight))
                # 記錄頭部平視時的比例
                nod_ratio_baseline_scores.append(nod_ratio)
                calibration_counter += 1
            else:
                # 校正完成，計算門檻
                if jaw_open_baseline_scores and eye_blink_baseline_scores and nod_ratio_baseline_scores:
                    avg_jaw_baseline = np.mean(jaw_open_baseline_scores)
                    avg_eye_baseline = np.mean(eye_blink_baseline_scores)
                    avg_nod_baseline = np.mean(nod_ratio_baseline_scores) # 計算正常平視比例
                    yawn_threshold = avg_jaw_baseline + 0.4 
                    eye_close_threshold = avg_eye_baseline + 0.3 # 睜眼基準 + 0.3 視為閉眼
                    nod_threshold = avg_nod_baseline - 0.06 # [調高靈敏度] 正常平視基準減去 0.06 即視為低頭
                is_calibrated = True
                last_calibration_time = current_time # [新增] 更新最後一次校正完成的時間
                print(f"校正完成！嘴巴門檻: {yawn_threshold:.3f} | 眼睛門檻: {eye_close_threshold:.3f} | 低頭門檻: {nod_threshold:.3f} (基準:{avg_nod_baseline:.3f})")
        else: # 校正完成後，才開始偵測打哈欠
            # 調整打哈欠門檻：避免講話或稍微張嘴造成誤判
            if JawOpen >= yawn_threshold:
                
                if open_mouth_start_time is None:
                    open_mouth_start_time = current_time #記錄一開始的時間
                    current_yawn_counted = False

                yawn_duration = current_time - open_mouth_start_time
                
                # 分級警告：>1.5秒亮紅燈，>0.5秒亮黃燈
                if yawn_duration >= 1.5:
                    cv2.putText(frame, "WARNING: SEVERE YAWNING!", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                    # 記錄這次打哈欠 (確保一次哈欠只算一次)
                    if not current_yawn_counted:
                        yawn_timestamps.append(current_time)
                        current_yawn_counted = True
                        print(f"[警告] 偵測到嚴重打哈欠！過去 1 分鐘內累計: {len(yawn_timestamps)} 次")
                elif yawn_duration >= 0.5:
                    cv2.putText(frame, "WARNING: YAWNING...", (50, 180), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
            else:
                open_mouth_start_time = None
                current_yawn_counted = False

    


    # 4. 手動把點畫出來 (不用官方繪圖工具，避免報錯)
    if show_landmarks and latest_result and latest_result.face_landmarks:
        h, w, _ = frame.shape
        for face_landmarks in latest_result.face_landmarks:
            
            # 遍歷 478 個特徵點
            for idx, landmark in enumerate(face_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 畫綠色小點
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # 畢業專題特效：特別標註鼻尖 (索引 1)
                if idx == 1:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('Final Project Demo', frame)
    
    key = cv2.waitKey(1) & 0xFF
    if key == 27: # 按 ESC 退出
        break
    elif key == ord('c'): # [新增] 按下小寫 'c' 鍵，強制手動觸發重新校正
        last_calibration_time = 0 # 故意把最後校正時間歸零，下個迴圈就會觸發校正
        print("\n[手動觸發] 啟動重新校正...")
    elif key == ord('l'): # [新增] 按下小寫 'l' 鍵，切換特徵點顯示狀態
        show_landmarks = not show_landmarks
        print(f"\n[切換顯示] 特徵點顯示狀態: {'開啟' if show_landmarks else '關閉'}")

detector.close()
cap.release()
cv2.destroyAllWindows()

# [新增] 程式結束後，匯出 CSV 報表與折線圖
print("\n正在生成分析報告...")
try:
    import csv
    # 1. 輸出 CSV 數據表
    with open('fatigue_data.csv', 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(['Time(s)', 'Alertness Score', 'PERCLOS(%)', 'Nod Ratio', 'Jaw Open'])
        for i in range(len(log_timestamps)):
            writer.writerow([f"{log_timestamps[i]:.2f}", log_alertness[i], f"{log_perclos[i]:.2f}", f"{log_nod_ratio[i]:.3f}", f"{log_jaw_open[i]:.3f}"])
    print("✅ 已成功匯出數據報表：fatigue_data.csv")

    # 2. 繪製並輸出折線圖
    import matplotlib.pyplot as plt
    plt.figure(figsize=(12, 10))
    plt.rcParams['font.sans-serif'] = ['Microsoft JhengHei'] # 支援中文顯示 (微軟正黑體)
    plt.rcParams['axes.unicode_minus'] = False
    
    # 精神分數圖
    plt.subplot(4, 1, 1)
    plt.plot(log_timestamps, log_alertness, label='精神分數 (Alertness)', color='green')
    plt.axhline(y=60, color='orange', linestyle='--', alpha=0.7)
    plt.axhline(y=30, color='red', linestyle='--', alpha=0.7)
    plt.ylabel('分數')
    plt.title('駕駛疲勞偵測分析報告')
    plt.legend(loc='upper right')

    # PERCLOS 圖
    plt.subplot(4, 1, 2)
    plt.plot(log_timestamps, log_perclos, label='閉眼率 PERCLOS (%)', color='blue')
    plt.axhline(y=15, color='red', linestyle='--', alpha=0.7)
    plt.ylabel('百分比 %')
    plt.legend(loc='upper right')

    # 低頭比例圖
    plt.subplot(4, 1, 3)
    plt.plot(log_timestamps, log_nod_ratio, label='低頭比例 (Nod Ratio)', color='purple')
    plt.axhline(y=nod_threshold if is_calibrated else 0.3, color='red', linestyle='--', alpha=0.7, label='低頭門檻')
    plt.ylabel('比例')
    plt.legend(loc='upper right')

    # 嘴巴張開圖
    plt.subplot(4, 1, 4)
    plt.plot(log_timestamps, log_jaw_open, label='嘴巴張開 (Jaw Open)', color='brown')
    plt.axhline(y=yawn_threshold if is_calibrated else 0.8, color='red', linestyle='--', alpha=0.7, label='打哈欠門檻')
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