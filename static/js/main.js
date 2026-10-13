import { FaceLandmarker, FilesetResolver } from "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3";

// 取得 HTML 元素
const video = document.getElementById('webcam');
const canvasElement = document.getElementById('output_canvas');
const canvasCtx = canvasElement.getContext('2d');

let faceLandmarker;
let lastVideoTime = -1;

// ======== 全域控制開關 ========
window.showLandmarks = true;
window.forceRecalibrate = () => { detector.forceRecalibrate(); };
window.toggleLandmarks = () => { window.showLandmarks = !window.showLandmarks; };

// ======== 環境亮度計算與夜間模式 ========
const hiddenCanvas = document.createElement('canvas');
const hiddenCtx = hiddenCanvas.getContext('2d', { willReadFrequently: true });
hiddenCanvas.width = 64; hiddenCanvas.height = 64; // 縮小尺寸以提升效能

function calculateBrightness(videoEl) {
    hiddenCtx.drawImage(videoEl, 0, 0, 64, 64);
    const data = hiddenCtx.getImageData(0, 0, 64, 64).data;
    let sum = 0;
    for (let i = 0; i < data.length; i += 4) {
        // RGB 平均值
        sum += (data[i] + data[i+1] + data[i+2]) / 3;
    }
    return sum / (64 * 64);
}

// ======== 核心疲勞偵測引擎 (補回校正邏輯) ========
class FatigueDetector {
    constructor() {
        this.windowSize = 150;
        this.eyeHistory = new Array(this.windowSize).fill(0);

        // 連續動作計時器
        this.eyeClosedStartTime = null;
        this.noddingStartTime = null;

        // 校正系統變數
        this.isCalibrated = false;
        this.calibrationCounter = 0;
        this.CALIBRATION_FRAMES = 100;
        this.baselines = { jaw: [], eye: [], nod: [], turn: [] };

        // 預設門檻
        this.yawnThreshold = 0.8;
        this.eyeCloseThreshold = 0.40;
        this.nodThreshold = 0.30;
        this.turnCenter = 0.5;
        this.turnMargin = 0.25;
    }

    forceRecalibrate() {
        this.isCalibrated = false;
        this.calibrationCounter = 0;
        this.baselines = { jaw: [], eye: [], nod: [], turn: [] };
        showNotification("重新校正中，請直視前方...", true);
    }

    analyze(blendshapes, landmarks) {
        const now = Date.now();

        const eyeLeft = blendshapes.find(s => s.categoryName === "eyeBlinkLeft").score;
        const eyeRight = blendshapes.find(s => s.categoryName === "eyeBlinkRight").score;
        const jawOpen = blendshapes.find(s => s.categoryName === "jawOpen").score;

        const noseY = landmarks[1].y;
        const chinY = landmarks[152].y;
        const foreheadY = landmarks[10].y;
        const faceHeight = chinY - foreheadY;
        const nodRatio = faceHeight > 0 ? (chinY - noseY) / faceHeight : 0.5;

        const leftX = landmarks[234].x;
        const rightX = landmarks[454].x;
        const noseX = landmarks[1].x;
        const faceWidth = rightX - leftX;
        const turnRatio = faceWidth > 0 ? (noseX - leftX) / faceWidth : 0.5;

        // --- 計算點頭「下墜速度 (Velocity)」 ---
        let nodVelocity = 0;
        if (this.lastNodRatio !== null && this.lastTimestamp !== null) {
            const dt = (now - this.lastTimestamp) / 1000; // 秒
            if (dt > 0) {
                // 如果 nodRatio 變小代表低頭，相減為正值代表「正在下墜」
                nodVelocity = (this.lastNodRatio - nodRatio) / dt; 
            }
        }
        this.lastNodRatio = nodRatio;
        this.lastTimestamp = now;

        // --- 自動校正邏輯 ---
        if (!this.isCalibrated) {
            if (this.calibrationCounter < this.CALIBRATION_FRAMES) {
                this.baselines.jaw.push(jawOpen);
                this.baselines.eye.push(Math.max(eyeLeft, eyeRight));
                this.baselines.nod.push(nodRatio);
                this.baselines.turn.push(turnRatio);
                this.calibrationCounter++;
            } else {
                const avg = (arr) => arr.reduce((a, b) => a + b, 0) / arr.length;
                this.yawnThreshold = avg(this.baselines.jaw) + 0.4;
                this.eyeCloseThreshold = avg(this.baselines.eye) + 0.3;
                this.nodThreshold = avg(this.baselines.nod) - 0.04;
                this.turnCenter = avg(this.baselines.turn);
                this.isCalibrated = true;
                showNotification("✅ 校正完成！", true);
            }
        }

        // --- 1. 連續閉眼計時 (微睡眠 Microsleep) ---
        const isEyesClosed = (eyeLeft >= this.eyeCloseThreshold && eyeRight >= this.eyeCloseThreshold);
        if (isEyesClosed) {
            if (!this.eyeClosedStartTime) this.eyeClosedStartTime = now;
        } else {
            this.eyeClosedStartTime = null;
        }
        const continuousEyeClosedTime = this.eyeClosedStartTime ? (now - this.eyeClosedStartTime) / 1000 : 0; // 換算成秒

        // --- 2. 連續低頭計時 (深沉瞌睡) ---
        const isNoddingOff = nodRatio < this.nodThreshold;
        if (isNoddingOff) {
            if (!this.noddingStartTime) this.noddingStartTime = now;
        } else {
            this.noddingStartTime = null;
        }
        const continuousNoddingTime = this.noddingStartTime ? (now - this.noddingStartTime) / 1000 : 0; // 換算成秒

        const isSnapNodding = (nodVelocity > 0.4 && nodRatio < this.nodThreshold + 0.02);

        // --- 3. 計算 PERCLOS ---
        this.eyeHistory.push(isEyesClosed ? 1 : 0);
        if (this.eyeHistory.length > this.windowSize) this.eyeHistory.shift();
        const currentPerclos = this.eyeHistory.length > 0 ? this.eyeHistory.reduce((a, b) => a + b, 0) / this.eyeHistory.length : 0;

        // --- 4. 綜合評分與狀態標記 ---
        const isLookingAway = Math.abs(turnRatio - this.turnCenter) > this.turnMargin;

        // 若連續閉眼超過 1.5 秒，精神分數直接歸零 (無視 PERCLOS 比例)
        let spiritScore = 100 - (currentPerclos / 0.15) * 50 - (jawOpen * 20) - (isNoddingOff ? 40 : 0) - (isLookingAway ? 40 : 0);
        if (continuousEyeClosedTime >= 1.5 || isSnapNodding || continuousNoddingTime >= 1.5) {
            spiritScore = 0; 
        }
        spiritScore = Math.max(0, Math.min(100, Math.floor(spiritScore)));

        // 回傳加入連續時間的指標
        return { 
            currentPerclos, nodRatio, turnRatio, spiritScore, 
            isNoddingOff, isLookingAway, isSnapNodding,
            continuousEyeClosedTime, continuousNoddingTime 
        };
    }
}

const detector = new FatigueDetector();

// 初始化 MediaPipe
async function initializeFaceLandmarker() {
    const filesetResolver = await FilesetResolver.forVisionTasks(
        "https://cdn.jsdelivr.net/npm/@mediapipe/tasks-vision@0.10.3/wasm"
    );
    faceLandmarker = await FaceLandmarker.createFromOptions(filesetResolver, {
        baseOptions: {
            modelAssetPath: "https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task",
            delegate: "GPU"
        },
        outputFaceBlendshapes: true,
        runningMode: "VIDEO",
        numFaces: 1
    });
    console.log("✅ 邊緣運算模型載入完成");
    startCamera();
}

// 啟動相機
function startCamera() {
    navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } })
        .then((stream) => {
            video.srcObject = stream;
            video.addEventListener("loadeddata", predictWebcam);
        })
        .catch((err) => console.error("❌ 鏡頭開啟失敗:", err));
}

let lastProcessTime = 0;
let frameCount = 0;

async function predictWebcam() {
    if (canvasElement.width !== video.videoWidth || canvasElement.height !== video.videoHeight) {
        canvasElement.width = video.videoWidth;
        canvasElement.height = video.videoHeight;
    }
    
    let startTimeMs = performance.now();
    if (lastVideoTime !== video.currentTime) {
        lastVideoTime = video.currentTime;

        // 算 FPS
        frameCount++;
        if (startTimeMs - lastProcessTime >= 1000) {
            document.getElementById('fps').innerText = frameCount;
            frameCount = 0;
            lastProcessTime = startTimeMs;
        }

        // 計算亮度與夜間模式 
        const brightness = calculateBrightness(video);
        const isNightMode = brightness < 70;
        document.getElementById('brightness').innerText = brightness.toFixed(1);
        document.getElementById('night-mode').style.display = isNightMode ? 'block' : 'none';
        // 若太暗，直接利用 CSS 調亮對比度與亮度
        video.style.filter = isNightMode ? "contrast(1.5) brightness(1.5)" : "none";
        
        const results = faceLandmarker.detectForVideo(video, startTimeMs);
        canvasCtx.clearRect(0, 0, canvasElement.width, canvasElement.height);
        
        if (results.faceBlendshapes && results.faceBlendshapes.length > 0 && results.faceLandmarks.length > 0) {
            const blendshapes = results.faceBlendshapes[0].categories;
            const landmarks = results.faceLandmarks[0];

            // 繪製特徵點
            if (window.showLandmarks) {
                const w = canvasElement.width;
                const h = canvasElement.height;
                const specialPoints = [1, 10, 152, 234, 454];
                
                for (let i = 0; i < landmarks.length; i++) {
                    const pt = landmarks[i];
                    canvasCtx.beginPath();
                    canvasCtx.arc(pt.x * w, pt.y * h, specialPoints.includes(i) ? 6 : 2.5, 0, 2 * Math.PI);
                    canvasCtx.fillStyle = specialPoints.includes(i) ? "#FF0000" : "#00FF00";
                    canvasCtx.fill();
                }
            }

            // 分析疲勞
            const metrics = detector.analyze(blendshapes, landmarks);

            // 更新 UI
            document.getElementById('alertness').innerText = metrics.spiritScore;
            if (!detector.isCalibrated) {
                document.getElementById('perclos').innerText = `校正中(${detector.calibrationCounter}/${detector.CALIBRATION_FRAMES})`;
            } else {
                document.getElementById('perclos').innerText = (metrics.currentPerclos * 100).toFixed(1);
            }            
            document.getElementById('nod-ratio').innerText = metrics.nodRatio.toFixed(3);
            document.getElementById('turn-ratio').innerText = metrics.turnRatio.toFixed(3);

            // 警告處理
            const warningsContainer = document.getElementById('warnings-container');
            warningsContainer.innerHTML = '';
            
            // 等級 1：最高危險 (瞬間點頭、微睡眠 > 1.5秒)
            if (metrics.continuousEyeClosedTime >= 1.5) {
                warningsContainer.innerHTML = '<div class="warning-msg" style="background: #8b0000;">🚨 致命危險：微睡眠發生！(Microsleep)</div>';
            } else if (metrics.isSnapNodding || metrics.continuousNoddingTime >= 1.5) {
                warningsContainer.innerHTML = '<div class="warning-msg" style="background: #8b0000;">🚨 致命危險：打瞌睡點頭！(Head Snap)</div>';
            } 
            // 等級 2：黃色預警 (慢速眨眼 > 0.6秒，即將斷線)
            else if (metrics.continuousEyeClosedTime >= 0.6) {
                warningsContainer.innerHTML = '<div class="warning-msg" style="background: #d48806;">⚠️ 警告：慢速眨眼，精神渙散中...</div>';
            } 
            // 等級 3：慢性疲勞 (PERCLOS 或綜合分數過低)
            else if (metrics.spiritScore < 50) {
                warningsContainer.innerHTML = '<div class="warning-msg">⚠️ 提醒：慢性疲勞，建議休息</div>';
            }
        }
    }
    window.requestAnimationFrame(predictWebcam);
}

// 供外部呼叫的提示框函式
window.showNotification = function(message, isSuccess = true) {
    const container = document.getElementById('notification-container');
    const notif = document.createElement('div');
    notif.textContent = message;
    notif.style.padding = '10px 20px';
    notif.style.backgroundColor = isSuccess ? '#28a745' : '#dc3545';
    notif.style.color = 'white';
    notif.style.borderRadius = '5px';
    container.appendChild(notif);
    setTimeout(() => container.removeChild(notif), 3000);
}

initializeFaceLandmarker();