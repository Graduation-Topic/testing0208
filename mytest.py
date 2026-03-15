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
        num_faces=1,
        output_face_blendshapes=True
    )
    detector = FaceLandmarker.create_from_options(options)
except Exception as e:
    print(f"錯誤：模型載入失敗！請確認 {model_path} 檔案存在。")
    print(f"詳細錯誤: {e}")
    exit()

# 開啟攝影機
cap = cv2.VideoCapture(0)
print("系統啟動成功！(手動繪圖模式)")

closed_eyes_start_time = None  # 紀錄剛開始閉上眼睛的那個瞬間
open_mouth_start_time = None  # 紀錄剛開始打開嘴巴的那個瞬間


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

    # 看看有沒有收到表情成績單
    if latest_result and latest_result.face_blendshapes:
        # 拿出畫面中第一個人的所有表情分數
        blendshapes = latest_result.face_blendshapes[0]

        eyeBlinkLeft = blendshapes[9].score
        eyeBlinkRight = blendshapes[10].score
        JawOpen = blendshapes[25].score

        #判斷眼睛是否疲勞或關閉
        if eyeBlinkLeft and eyeBlinkRight >= 0.6:
            
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