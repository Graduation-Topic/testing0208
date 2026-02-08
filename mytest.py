import cv2
import time
import mediapipe as mp

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
        num_faces=1
    )
    detector = FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"錯誤：模型載入失敗！請確認 {model_path} 檔案存在。")
    print(f"詳細錯誤: {e}")
    exit()

# 開啟攝影機
cap = cv2.VideoCapture(0)
print("系統啟動成功！(手動繪圖模式)")

while cap.isOpened():
    success, frame = cap.read()
    if not success: break

    # 鏡像並轉 RGB
    frame = cv2.flip(frame, 1)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    # 傳送給模型
    timestamp = int(time.time() * 1000)
    detector.detect_async(mp_image, timestamp)

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