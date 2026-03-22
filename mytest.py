import cv2
import time
import mediapipe as mp
import numpy as np

# 1. 既然這行之前成功過，我們就靠它了
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

# 2. 【關鍵修正】直接使用導入的模組，不要再透過 mp.tasks 呼叫
# 這樣可以避開 "has no attribute 'python'" 的錯誤
BaseOptions = python.BaseOptions
FaceLandmarker = vision.FaceLandmarker
FaceLandmarkerOptions = vision.FaceLandmarkerOptions
VisionRunningMode = vision.RunningMode

# 全域變數
latest_result = None

# 回傳函數
def print_result(result, output_image, timestamp_ms):
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
        output_face_blendshapes=True
    )
    detector = FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"錯誤：模型載入失敗！請確認 {model_path} 檔案存在。")
    print(f"詳細錯誤: {e}")
    exit()

# 開啟攝影機
cap = cv2.VideoCapture(1)
print("系統啟動成功！(手動繪圖模式)")

closed_eyes_start_time = None  # 紀錄剛開始閉上眼睛的那個瞬間
open_mouth_start_time = None  # 紀錄剛開始打開嘴巴的那個瞬間

show_score_time = None

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
        
        # [步驟 B] 建立 CLAHE 增強器 
        # (clipLimit 控制對比增強的強度，數值越大對比越強，通常設 2.0~3.0)
        clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
        
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

        #判斷眼睛是否疲勞或關閉
        if eyeBlinkLeft and eyeBlinkRight >= 0.55:
            
            if closed_eyes_start_time is None:
                closed_eyes_start_time = time.time() #記錄一開始的時間

            if time.time() - closed_eyes_start_time >= 0.5:  
                    
                print("名稱:", blendshapes[9].category_name)
                print("分數:", blendshapes[9].score)
                print("-------------------")
                print("名稱:", blendshapes[10].category_name)
                print("分數:", blendshapes[10].score)
                print("-------------------")
                #顯示警告
                cv2.putText(frame, "WARNING: SLEEPING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            closed_eyes_start_time = None
        
        #判斷嘴巴是否打開
        if JawOpen >= 0.5:
            
            if open_mouth_start_time is None:
                open_mouth_start_time = time.time() #記錄一開始的時間

            if time.time() - open_mouth_start_time >= 0.5:  
                    
                print("名稱:", blendshapes[25].category_name)
                print("分數:", blendshapes[25].score)
                print("-------------------")
                #顯示警告
                cv2.putText(frame, "WARNING: SLEEPING!", (100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
        else:
            open_mouth_start_time = None

        if show_score_time is None:
            show_score_time = time.time()

        if time.time() - show_score_time >= 10:
            print("左眼:", blendshapes[9].score)
            print("右眼:", blendshapes[10].score)
            print("嘴巴:", blendshapes[25].score)
            show_score_time = None


    # 4. 手動把點畫出來 (不用官方繪圖工具，避免報錯)
    if latest_result and latest_result.face_landmarks:
        for face_landmarks in latest_result.face_landmarks:
            h, w, _ = frame.shape
            
            # 遍歷 478 個特徵點
            for idx, landmark in enumerate(face_landmarks):
                cx, cy = int(landmark.x * w), int(landmark.y * h)
                
                # 畫綠色小點
                cv2.circle(frame, (cx, cy), 1, (0, 255, 0), -1)

                # 畢業專題特效：特別標註鼻尖 (索引 1)
                if idx == 1:
                    cv2.circle(frame, (cx, cy), 5, (0, 0, 255), -1)

    cv2.imshow('Final Project Demo', frame)
    
    if cv2.waitKey(1) & 0xFF == 27: # 按 ESC 退出
        break

detector.close()
cap.release()
cv2.destroyAllWindows()