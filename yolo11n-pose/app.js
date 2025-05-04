// DOM elements
const imageUpload = document.getElementById('imageUpload');
const captureBtn = document.getElementById('captureBtn');
const realtimeBtn = document.getElementById('realtimeBtn');
const stopRealtimeBtn = document.getElementById('stopRealtimeBtn');
const cameraModal = document.getElementById('cameraModal');
const closeCamera = document.querySelector('.close-camera');
const video = document.getElementById('video');
const takePictureBtn = document.getElementById('takePictureBtn');
const inputCanvas = document.getElementById('inputCanvas');
const outputCanvas = document.getElementById('outputCanvas');
const statusDiv = document.getElementById('status');
const infoDiv = document.getElementById('info');
const fpsDisplay = document.getElementById('fps-display');
const fpsElement = document.getElementById('fps');
const realtimeControls = document.getElementById('realtimeControls');

// Loading UI elements
const loadingSection = document.getElementById('loadingSection');
const inputSection = document.getElementById('inputSection');
const progressBar = document.getElementById('progressBar');
const progressValue = document.getElementById('progressValue');
const loadingStatus = document.getElementById('loadingStatus');

// Canvas contexts
const inputCtx = inputCanvas.getContext('2d');
const outputCtx = outputCanvas.getContext('2d');

// Default canvas size
const defaultWidth = 640;
const defaultHeight = 640;
inputCanvas.width = defaultWidth;
inputCanvas.height = defaultHeight;
outputCanvas.width = defaultWidth;
outputCanvas.height = defaultHeight;

// Model variables
let model = null;
let isModelLoaded = false;
let isRealtimeMode = false;
let realtimeStream = null;
let animationFrameId = null;

// FPS calculation
let frameCount = 0;
let lastFpsUpdateTime = 0;
let lastFrameTime = 0;

// 添加全局變量以控制性能
const FPS_UPDATE_INTERVAL = 500; // 毫秒
const TENSOR_CHECK_INTERVAL = 30; // 幀數
const MAX_TENSORS_WARNING = 500; // 張量數量警告閾值
const FORCE_GC_INTERVAL = 60; // 強制垃圾回收的幀間隔
let framesSinceLastCheck = 0;
let isMobile = false;

// Keypoint connections for drawing skeleton
const connections = [
    [0, 1], [0, 2], [1, 3], [2, 4], // Face and neck
    [5, 7], [7, 9], [6, 8], [8, 10], // Arms
    [5, 6], [5, 11], [6, 12], // Shoulders to hips
    [11, 12], [11, 13], [12, 14], // Hips to knees
    [13, 15], [14, 16] // Knees to ankles
];

// Keypoint colors
const keypointColors = [
    '#FF0000', // nose - red
    '#FF7F00', // left_eye - orange
    '#FFFF00', // right_eye - yellow
    '#00FF00', // left_ear - green
    '#0000FF', // right_ear - blue
    '#4B0082', // left_shoulder - indigo
    '#8F00FF', // right_shoulder - violet
    '#FF1493', // left_elbow - deep pink
    '#00FFFF', // right_elbow - cyan
    '#FFD700', // left_wrist - gold
    '#008000', // right_wrist - green
    '#FF00FF', // left_hip - magenta
    '#00BFFF', // right_hip - deep sky blue
    '#FF6347', // left_knee - tomato
    '#7CFC00', // right_knee - lawn green
    '#FF69B4', // left_ankle - hot pink
    '#32CD32'  // right_ankle - lime green
];

// 初始化時檢測設備類型
function detectDevice() {
    // 檢測是否為移動設備
    const userAgent = navigator.userAgent.toLowerCase();
    isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
    console.log(`檢測到${isMobile ? '移動' : '桌面'}設備`);
}

// Initialize the application
async function init() {
    updateStatus('載入 YOLO Pose 模型中...');
    updateLoadingStatus('初始化 TensorFlow.js...');
    
    // 檢測設備類型
    detectDevice();
    
    try {
        // Set the backend to WebGL for faster processing
        await tf.setBackend('webgl');
        
        // 設置WebGL參數以優化性能
        if (tf.getBackend() === 'webgl') {
            tf.env().set('WEBGL_CPU_FORWARD', false);
            tf.env().set('WEBGL_PACK', true);
            tf.env().set('WEBGL_FORCE_F16_TEXTURES', false);
            
            // 在移動設備上使用更保守的設置
            if (isMobile) {
                tf.env().set('WEBGL_RENDER_FLOAT32_CAPABLE', false);
                tf.env().set('WEBGL_FLUSH_THRESHOLD', 1);
            } else {
                tf.env().set('WEBGL_RENDER_FLOAT32_CAPABLE', true);
                tf.env().set('WEBGL_FLUSH_THRESHOLD', 5);
            }
        }
        
        updateLoadingStatus('WebGL 後端已準備就緒');
        updateProgressBar(10);
        
        // Load the model
        updateLoadingStatus('正在載入模型文件...');
        model = await tf.loadGraphModel('../models/yolo11n-pose/model.json', {
            onProgress: (fraction) => {
                // Update progress from 10% to 90%
                updateProgressBar(10 + fraction * 80);
                updateLoadingStatus(`載入模型中... ${Math.round(fraction * 100)}%`);
            }
        });
        
        updateLoadingStatus('模型已載入，正在初始化...');
        updateProgressBar(95);
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await model.predict(dummyInput).array();
        dummyInput.dispose();
        
        isModelLoaded = true;
        updateProgressBar(100);
        updateLoadingStatus('模型已完全載入，準備就緒！');
        
        // Show input section and hide loading section after a short delay
        setTimeout(() => {
            loadingSection.style.display = 'none';
            inputSection.style.display = 'flex';
            updateStatus('模型載入成功！請上傳圖片或使用相機。');
        }, 1000);
    } catch (error) {
        console.error('載入模型時發生錯誤:', error);
        updateStatus('載入模型時發生錯誤。請查看控制台獲取詳細信息。');
        updateLoadingStatus('載入失敗：' + error.message);
    }
}

// Update progress bar
function updateProgressBar(percent) {
    const value = Math.round(percent);
    progressBar.style.width = `${value}%`;
    progressValue.textContent = `${value}%`;
}

// Update loading status
function updateLoadingStatus(message) {
    loadingStatus.textContent = message;
}

// Update status message
function updateStatus(message) {
    statusDiv.textContent = message;
}

// Update info message
function updateInfo(message) {
    infoDiv.textContent = message;
}

// Event listeners
imageUpload.addEventListener('change', (e) => {
    if (e.target.files.length > 0) {
        const reader = new FileReader();
        reader.onload = (e) => {
            const img = new Image();
            img.onload = () => processImage(img);
            img.src = e.target.result;
        };
        reader.readAsDataURL(e.target.files[0]);
    }
});

captureBtn.addEventListener('click', openCamera);
closeCamera.addEventListener('click', closeVideoModal);
takePictureBtn.addEventListener('click', takePicture);
realtimeBtn.addEventListener('click', startRealtimeDetection);
stopRealtimeBtn.addEventListener('click', stopRealtimeDetection);

// Process the selected image
async function processImage(img) {
    if (!isModelLoaded) {
        updateStatus('請等待模型載入完成。');
        return;
    }
    
    // Stop realtime mode if it's running
    if (isRealtimeMode) {
        stopRealtimeDetection();
    }
    
    updateStatus('處理圖片中...');
    
    // Letterbox resize the image
    const [scaledImg, imgInfo] = letterboxImage(img);
    displayInputImage(scaledImg);
    
    try {
        // Detect poses
        const predictions = await detectPose(scaledImg);
        
        // Draw detection results
        drawDetections(predictions, imgInfo);
        
        updateStatus('檢測完成！');
        updateInfo(`在圖中檢測到 ${predictions.length} 個人。`);
    } catch (error) {
        console.error('處理圖片時發生錯誤:', error);
        updateStatus('處理圖片時發生錯誤。請再試一次。');
    }
}

// Letterbox image preprocessing as in YOLO
function letterboxImage(img) {
    try {
        // Target dimensions
        const newShape = [defaultWidth, defaultHeight];
        const stride = 32;  // YOLO's default stride
        
        // Original image dimensions - 從img元素或畫布上獲取
        let imgWidth, imgHeight;
        if (img instanceof HTMLVideoElement) {
            imgWidth = img.videoWidth;
            imgHeight = img.videoHeight;
        } else if (img instanceof HTMLCanvasElement) {
            imgWidth = img.width;
            imgHeight = img.height;
        } else {
            imgWidth = img.width;
            imgHeight = img.height;
        }
        
        const shape = [imgHeight, imgWidth];
        
        // 確保有效的尺寸
        if (shape[0] <= 0 || shape[1] <= 0) {
            console.error('無效的圖像尺寸:', shape);
            
            // 返回簡單的灰色畫布
            const canvas = document.createElement('canvas');
            canvas.width = newShape[1];
            canvas.height = newShape[0];
            const ctx = canvas.getContext('2d');
            ctx.fillStyle = 'rgb(114, 114, 114)';
            ctx.fillRect(0, 0, canvas.width, canvas.height);
            
            return [
                canvas,
                {
                    ratio: 1,
                    padding: [0, 0],
                    originalSize: [1, 1],
                    newSize: [newShape[1], newShape[0]]
                }
            ];
        }
        
        // Calculate ratio
        let ratio = Math.min(newShape[0] / shape[0], newShape[1] / shape[1]);
        
        // Only scale down, do not scale up (for better accuracy)
        ratio = Math.min(ratio, 1.0);
        
        // Calculate new unpadded dimensions
        const newUnpad = [Math.round(shape[0] * ratio), Math.round(shape[1] * ratio)];
        
        // Calculate padding
        let dw = newShape[1] - newUnpad[1];
        let dh = newShape[0] - newUnpad[0];
        
        // Center the image
        dw /= 2;
        dh /= 2;
        
        // Create a canvas with the target dimensions
        const canvas = document.createElement('canvas');
        canvas.width = newShape[1];
        canvas.height = newShape[0];
        const ctx = canvas.getContext('2d');
        
        // Fill with grey background (similar to [114, 114, 114] in YOLO)
        ctx.fillStyle = 'rgb(114, 114, 114)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        // Draw the resized image on the canvas with padding
        const left = Math.round(dw - 0.1);  // adjust as in YOLO
        const top = Math.round(dh - 0.1);   // adjust as in YOLO
        
        // 使用try-catch確保繪製錯誤不會中斷應用
        try {
            ctx.drawImage(
                img,
                0, 0, imgWidth, imgHeight,
                left, top, newUnpad[1], newUnpad[0]
            );
        } catch (e) {
            console.error('繪製圖像到信箱處理畫布時出錯:', e);
        }
        
        // Return the canvas and info about the transformation
        return [
            canvas, 
            {
                ratio: ratio,
                padding: [left, top],
                originalSize: [imgWidth, imgHeight],
                newSize: [newUnpad[1], newUnpad[0]]
            }
        ];
    } catch (error) {
        console.error('letterboxImage處理錯誤:', error);
        
        // 返回一個備用畫布
        const canvas = document.createElement('canvas');
        canvas.width = defaultWidth;
        canvas.height = defaultHeight;
        const ctx = canvas.getContext('2d');
        ctx.fillStyle = 'rgb(114, 114, 114)';
        ctx.fillRect(0, 0, canvas.width, canvas.height);
        
        return [
            canvas,
            {
                ratio: 1,
                padding: [0, 0],
                originalSize: [defaultWidth, defaultHeight],
                newSize: [defaultWidth, defaultHeight]
            }
        ];
    }
}

// Display input image on canvas
function displayInputImage(canvas) {
    // Clear canvas
    inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
    
    // Draw the image
    inputCtx.drawImage(canvas, 0, 0);
}

// Detect poses in the image
async function detectPose(canvas) {
    try {
        // 確保輸入是有效的
        if (!(canvas instanceof HTMLCanvasElement) || canvas.width <= 0 || canvas.height <= 0) {
            console.error('無效的輸入畫布:', canvas);
            return [];
        }
        
        // 使用同步tidy來處理張量
        const tensorData = tf.tidy(() => {
            // Preprocess the image
            const img = tf.browser.fromPixels(canvas);
            
            // 檢查輸入張量的形狀
            const shape = img.shape;
            if (shape.length !== 3 || shape[0] <= 0 || shape[1] <= 0 || shape[2] !== 3) {
                console.error('無效的輸入張量形狀:', shape);
                return null;
            }
            
            // Normalize image to [0, 1]
            const normalized = img.div(255.0);
            
            // Expand dimensions to match model input [1, height, width, 3]
            const batched = normalized.expandDims(0);
            
            // 檢查批處理後的形狀
            const batchedShape = batched.shape;
            if (batchedShape.length !== 4 || batchedShape[0] !== 1 || 
                batchedShape[1] !== 640 || batchedShape[2] !== 640 || batchedShape[3] !== 3) {
                console.error('批處理後的張量形狀不正確:', batchedShape);
                console.error('預期形狀: [1, 640, 640, 3]');
                return null;
            }
            
            return batched;
        });
        
        // 如果預處理失敗，返回空數組
        if (!tensorData) {
            return [];
        }
        
        // 檢查模型是否已經加載
        if (!model) {
            console.error('模型尚未加載');
            tensorData.dispose(); // 清理張量
            return [];
        }
        
        try {
            // 創建超時Promise
            const timeoutPromise = new Promise((_, reject) => {
                setTimeout(() => reject(new Error('模型預測超時')), 3000);
            });
            
            // 執行模型預測（在tidy外部）
            const predictPromise = model.predict(tensorData);
            
            // 等待預測完成或超時
            const predictions = await Promise.race([predictPromise, timeoutPromise]);
            
            // 立即處理預測結果（克隆到JavaScript數據）
            // 注意：這裡需要確保在使用predictions之後釋放它
            const outputs = processPredictionsSync(predictions);
            
            // 清理資源
            tensorData.dispose();
            predictions.dispose();
            
            return outputs;
        } catch (error) {
            // 確保在錯誤情況下也釋放資源
            tensorData.dispose();
            console.error('模型預測錯誤:', error);
            return [];
        }
    } catch (error) {
        console.error('detectPose錯誤:', error);
        return [];
    }
}

// 同步處理預測結果（不返回Promise）
function processPredictionsSync(predictions) {
    try {
        // 將張量轉換為JavaScript數據
        const output = predictions.arraySync()[0];
        
        // 常量處理
        const confThresh = 0.5;  // 置信度閾值
        const iouThresh = 0.45;  // IOU閾值
        const numKeypoints = 17;
        const results = [];
        
        // 檢查輸出是否有效
        if (!output || !output.length || !output[0] || !output[0].length) {
            return results;
        }
        
        const candidates = [];
        
        // 提取候選檢測
        for (let i = 0; i < output[0].length; i++) {
            const confidence = output[4][i];  // 置信度分數
            
            // 按置信度過濾
            if (confidence > confThresh) {
                // 提取邊界框（YOLO格式：center_x, center_y, width, height）
                const centerX = output[0][i];
                const centerY = output[1][i];
                const width = output[2][i];
                const height = output[3][i];
                
                // 轉換為xyxy格式
                const x1 = centerX - width / 2;
                const y1 = centerY - height / 2;
                const x2 = centerX + width / 2;
                const y2 = centerY + height / 2;
                
                // 提取關鍵點
                const kpts = [];
                for (let k = 0; k < numKeypoints; k++) {
                    const kpIndex = 5 + k * 3;
                    // 關鍵點格式為 [x, y, confidence]
                    if (kpIndex < output.length && output[kpIndex] && output[kpIndex][i] !== undefined) {
                        kpts.push([
                            output[kpIndex][i], 
                            output[kpIndex + 1] ? output[kpIndex + 1][i] : 0, 
                            output[kpIndex + 2] ? output[kpIndex + 2][i] : 0
                        ]);
                    } else {
                        kpts.push([0, 0, 0]); // 填充默認值
                    }
                }
                
                candidates.push({
                    bbox: [x1, y1, x2, y2],
                    confidence: confidence,
                    keypoints: kpts
                });
            }
        }
        
        // 按置信度排序（降序）
        candidates.sort((a, b) => b.confidence - a.confidence);
        
        // 應用非極大值抑制（類似YOLO的NMS）
        const selected = nonMaxSuppression(candidates, iouThresh);
        
        return selected;
    } catch (error) {
        console.error('處理預測結果時出錯:', error);
        return [];
    }
}

// Non-max suppression implementation
function nonMaxSuppression(boxes, iouThresh) {
    const selected = [];
    
    // While we still have boxes to process
    while (boxes.length > 0) {
        // Pick the box with highest confidence
        const current = boxes[0];
        selected.push(current);
        
        // Filter out boxes with high IOU overlap
        boxes = boxes.filter((box, index) => {
            if (index === 0) return false;  // Remove the current box
            
            const iou = calculateIOU(current.bbox, box.bbox);
            return iou <= iouThresh;  // Keep boxes with IOU below threshold
        });
    }
    
    return selected;
}

// Calculate Intersection over Union
function calculateIOU(box1, box2) {
    // box format: [x1, y1, x2, y2]
    const [ax1, ay1, ax2, ay2] = box1;
    const [bx1, by1, bx2, by2] = box2;
    
    // Calculate intersection area
    const x1 = Math.max(ax1, bx1);
    const y1 = Math.max(ay1, by1);
    const x2 = Math.min(ax2, bx2);
    const y2 = Math.min(ay2, by2);
    
    // Check if there is an intersection
    if (x2 < x1 || y2 < y1) return 0;
    
    const intersectionArea = (x2 - x1) * (y2 - y1);
    
    // Calculate union area
    const box1Area = (ax2 - ax1) * (ay2 - ay1);
    const box2Area = (bx2 - bx1) * (by2 - by1);
    const unionArea = box1Area + box2Area - intersectionArea;
    
    return intersectionArea / unionArea;
}

// Draw detection results on canvas
function drawDetections(predictions, imgInfo) {
    console.log("開始繪製骨架，預測數量:", predictions.length);
    
    try {
        // 直接在輸出畫布上繪製
        // 清除畫布
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        
        // 先將輸入影像以較低不透明度繪製為背景
        outputCtx.globalAlpha = 0.5; // 使背景更淡，讓骨架更明顯
        outputCtx.drawImage(inputCanvas, 0, 0);
        outputCtx.globalAlpha = 1.0;
        
        // Get padding info
        const [padX, padY] = imgInfo.padding;
        const ratio = imgInfo.ratio;
        
        // Draw each detection
        predictions.forEach((pred, idx) => {
            const [x1, y1, x2, y2] = pred.bbox;
            const confidence = pred.confidence;
            
            console.log(`繪製第${idx+1}個檢測，邊界框:`, x1, y1, x2, y2);
            
            // 繪製骨架線條（使用超鮮豔的顏色和寬線條）
            outputCtx.lineWidth = 8; // 更寬的線條
            
            connections.forEach(([i, j]) => {
                const kp1 = pred.keypoints[i];
                const kp2 = pred.keypoints[j];
                
                // 確保兩個關鍵點座標數據格式正確
                if (!kp1 || !kp2 || kp1.length < 3 || kp2.length < 3) {
                    console.warn(`關鍵點數據格式不正確: kp1=${JSON.stringify(kp1)}, kp2=${JSON.stringify(kp2)}`);
                    return;
                }
                
                // 使用更低的閾值
                if (kp1[2] > 0.1 && kp2[2] > 0.1) {
                    // 使用非常鮮艷的顏色
                    outputCtx.strokeStyle = '#FF00FF'; // 亮紫色
                    
                    // 繪製骨架線條
                    outputCtx.beginPath();
                    outputCtx.moveTo(kp1[0], kp1[1]);
                    outputCtx.lineTo(kp2[0], kp2[1]);
                    outputCtx.stroke();
                }
            });
            
            // 繪製邊界框
            outputCtx.lineWidth = 4;
            outputCtx.strokeStyle = '#FF0000'; // 純紅色
            outputCtx.fillStyle = 'rgba(255, 0, 0, 0.2)';
            outputCtx.beginPath();
            outputCtx.rect(x1, y1, x2 - x1, y2 - y1);
            outputCtx.fill();
            outputCtx.stroke();
            
            // 繪製置信度文字
            outputCtx.fillStyle = 'white';
            outputCtx.strokeStyle = 'black';
            outputCtx.lineWidth = 2;
            outputCtx.font = 'bold 18px Arial';
            outputCtx.strokeText(`置信度: ${confidence.toFixed(2)}`, x1, y1 - 8);
            outputCtx.fillText(`置信度: ${confidence.toFixed(2)}`, x1, y1 - 8);
            
            // 繪製關鍵點
            pred.keypoints.forEach((keypoint, index) => {
                const [kpX, kpY, kpConf] = keypoint;
                
                // 只繪製置信度足夠高的關鍵點
                if (kpConf > 0.1) { // 更低的閾值
                    // 繪製更大的關鍵點
                    outputCtx.fillStyle = 'yellow';
                    outputCtx.beginPath();
                    outputCtx.arc(kpX, kpY, 10, 0, 2 * Math.PI);
                    outputCtx.fill();
                    
                    // 黑色邊框使關鍵點更清晰
                    outputCtx.strokeStyle = 'black';
                    outputCtx.lineWidth = 2;
                    outputCtx.stroke();
                    
                    // 繪製關鍵點編號
                    outputCtx.fillStyle = 'black';
                    outputCtx.font = 'bold 12px Arial';
                    outputCtx.fillText(index.toString(), kpX - 4, kpY + 4);
                }
            });
        });
        
        console.log("完成繪製檢測結果");
    } catch (err) {
        console.error("繪製骨架時發生錯誤:", err);
    }
}

// Camera handling
captureBtn.addEventListener('click', openCamera);
closeCamera.addEventListener('click', closeVideoModal);
takePictureBtn.addEventListener('click', takePicture);

// Open camera modal
function openCamera() {
    cameraModal.style.display = 'block';
    startCamera();
}

// Close camera modal
function closeVideoModal() {
    cameraModal.style.display = 'none';
    stopCamera();
}

// Start camera stream
async function startCamera() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ 
            video: { facingMode: 'environment' } 
        });
        video.srcObject = stream;
    } catch (err) {
        console.error('啟動相機時發生錯誤:', err);
        alert('無法訪問相機。請確保您已授予權限。');
    }
}

// Stop camera stream
function stopCamera() {
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
        video.srcObject = null;
    }
}

// Take picture from camera
function takePicture() {
    // Create a temporary canvas to capture the image
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    const ctx = tempCanvas.getContext('2d');
    ctx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Create image from canvas
    const img = new Image();
    img.onload = () => {
        closeVideoModal();
        processImage(img);
    };
    img.src = tempCanvas.toDataURL('image/png');
}

// Handle window resize
window.addEventListener('resize', () => {
    // Maintain aspect ratio of displayed canvases
    const width = Math.min(defaultWidth, window.innerWidth - 40);
    const height = width * (defaultHeight / defaultWidth);
    
    inputCanvas.style.width = `${width}px`;
    inputCanvas.style.height = `${height}px`;
    outputCanvas.style.width = `${width}px`;
    outputCanvas.style.height = `${height}px`;
});

// Initialize the app
init();

// Process each video frame for realtime detection
async function detectFrame(video) {
    if (!isRealtimeMode) return;
    
    // 取消下一幀的請求（如果有的話），確保每次只處理一幀
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    try {
        // Calculate FPS
        const now = performance.now();
        const elapsed = now - lastFrameTime;
        lastFrameTime = now;
        
        // 更新幀計數
        frameCount++;
        framesSinceLastCheck++;
        
        // 僅在指定間隔更新FPS，減少UI更新頻率
        if (now - lastFpsUpdateTime >= FPS_UPDATE_INTERVAL) {
            const fps = Math.round((frameCount * 1000) / (now - lastFpsUpdateTime));
            fpsElement.textContent = fps;
            frameCount = 0;
            lastFpsUpdateTime = now;
        }
        
        // 定期檢查張量數量，減少CPU佔用
        if (framesSinceLastCheck >= TENSOR_CHECK_INTERVAL) {
            framesSinceLastCheck = 0;
            
            // 檢查張量數量
            const memoryInfo = tf.memory();
            const numTensors = memoryInfo.numTensors;
            
            // 如果張量數量過多，顯示警告
            if (numTensors > MAX_TENSORS_WARNING) {
                console.warn(`張量數量過多 (${numTensors})，執行清理...`);
                // 釋放未使用的張量
                tf.disposeVariables();
                
                // 每隔一段時間嘗試執行強制垃圾回收
                if (frameCount % FORCE_GC_INTERVAL === 0) {
                    if (typeof window.gc === 'function') {
                        try {
                            window.gc();
                        } catch (e) {}
                    }
                }
            }
        }
        
        // 繪製當前視頻幀到輸入畫布
        drawVideoToCanvas(video);
        
        // 等待一小段時間確保UI渲染完成
        await new Promise(resolve => setTimeout(resolve, 0));
        
        // 執行檢測
        await runDetection(video);
    } catch (error) {
        console.error('實時檢測時發生錯誤:', error);
        
        // 錯誤計數和恢復邏輯
        if (!window.errorCount) window.errorCount = 0;
        window.errorCount++;
        
        // 如果錯誤過多，嘗試重置檢測
        if (window.errorCount > 5) {
            console.warn('檢測錯誤過多，嘗試重置...');
            window.errorCount = 0;
            
            // 清理TensorFlow資源
            tf.tidy(() => {});
            tf.disposeVariables();
            tf.engine().purge();
            
            // 顯示錯誤消息
            updateStatus('檢測過程出現問題，正在恢復...');
        }
    } finally {
        // 只有在仍處於實時模式時才請求下一幀
        if (isRealtimeMode) {
            // 根據設備類型調整檢測速率
            if (isMobile) {
                // 移動設備上降低更新頻率
                setTimeout(() => {
                    animationFrameId = requestAnimationFrame(() => detectFrame(video));
                }, 50); // 加入50ms延遲
            } else {
                animationFrameId = requestAnimationFrame(() => detectFrame(video));
            }
        }
    }
}

// 處理視頻幀為realtime detection
function drawVideoToCanvas(video) {
    if (!video || video.videoWidth === 0) {
        console.error("無效的視頻輸入");
        return;
    }
    
    try {
        // 清除輸入畫布
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        
        // 計算適合的繪製尺寸和位置
        const aspectRatio = video.videoWidth / video.videoHeight;
        let drawWidth, drawHeight, offsetX = 0, offsetY = 0;
        
        if (aspectRatio > 1) {
            drawWidth = inputCanvas.width;
            drawHeight = inputCanvas.width / aspectRatio;
            offsetY = (inputCanvas.height - drawHeight) / 2;
        } else {
            drawHeight = inputCanvas.height;
            drawWidth = inputCanvas.height * aspectRatio;
            offsetX = (inputCanvas.width - drawWidth) / 2;
        }
        
        // 繪製視頻到輸入畫布
        inputCtx.drawImage(video, offsetX, offsetY, drawWidth, drawHeight);
        
        console.log("視頻繪製完成，尺寸:", drawWidth, "x", drawHeight);
    } catch (err) {
        console.error("繪製視頻到畫布時出錯:", err);
    }
}

// 執行檢測並繪製結果
async function runDetection(video) {
    try {
        console.log("開始執行檢測...");
        
        // 處理視頻幀使其適合模型輸入
        const [processedFrame, imgInfo] = letterboxImage(video);
        console.log("信箱處理完成，padding:", imgInfo.padding, "ratio:", imgInfo.ratio);
        
        // 檢測姿勢
        const predictions = await detectPose(processedFrame);
        console.log("檢測完成，找到", predictions.length, "個人");
        
        // 更新檢測信息
        updateInfo(`檢測到 ${predictions.length} 個人`);
        
        // 檢查預測結果是否包含關鍵點
        if (predictions && predictions.length > 0) {
            // 打印詳細信息供調試
            console.log("預測結果:", JSON.stringify(predictions[0].bbox));
            console.log("關鍵點數量:", predictions[0].keypoints.length);
            console.log("首個預測置信度:", predictions[0].confidence);
            
            // 直接繪製檢測結果，不再分步驟
            drawDetections(predictions, imgInfo);
            console.log("骨架繪製完成");
        } else {
            console.log("沒有檢測到人");
            // 當沒有檢測到時，直接顯示原始視頻
            outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
            outputCtx.drawImage(inputCanvas, 0, 0);
        }
        
        return predictions;
    } catch (error) {
        console.error('檢測過程中發生錯誤:', error);
        outputCtx.drawImage(inputCanvas, 0, 0); // 出錯時仍顯示原始畫面
        return [];
    }
}

// Start realtime detection
async function startRealtimeDetection() {
    if (!isModelLoaded) {
        updateStatus('請等待模型載入完成。');
        return;
    }
    
    // 如果已經在實時模式，先停止
    if (isRealtimeMode) {
        stopRealtimeDetection();
    }
    
    try {
        updateStatus('正在啟動攝像頭...');
        
        // 請求攝像頭權限並獲取流
        realtimeStream = await navigator.mediaDevices.getUserMedia({
            video: { 
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 480 } 
            }
        });
        
        // 準備視頻元素
        const realtimeVideo = document.createElement('video');
        realtimeVideo.srcObject = realtimeStream;
        realtimeVideo.autoplay = true;
        realtimeVideo.playsInline = true;
        
        // 等待視頻準備就緒
        await new Promise((resolve, reject) => {
            realtimeVideo.onloadeddata = () => {
                realtimeVideo.play()
                    .then(() => {
                        console.log("視頻開始播放，尺寸:", realtimeVideo.videoWidth, "x", realtimeVideo.videoHeight);
                        resolve();
                    })
                    .catch(err => {
                        console.error("視頻播放失敗:", err);
                        reject(err);
                    });
            };
            
            // 設置超時
            setTimeout(() => {
                if (realtimeVideo.readyState < 2) { // HAVE_CURRENT_DATA
                    reject(new Error('視頻加載超時'));
                }
            }, 5000);
        });
        
        // 清除畫布
        inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
        outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
        
        // 設置UI狀態
        isRealtimeMode = true;
        fpsDisplay.style.display = 'block';
        realtimeControls.style.display = 'flex';
        updateStatus('實時檢測中...');
        
        // 重置FPS計數器
        frameCount = 0;
        lastFpsUpdateTime = performance.now();
        lastFrameTime = performance.now();
        
        // 開始檢測循環
        animationFrameId = requestAnimationFrame(() => detectFrame(realtimeVideo));
        
    } catch (error) {
        console.error('啟動實時檢測時發生錯誤:', error);
        updateStatus('啟動實時檢測失敗: ' + error.message);
        stopRealtimeDetection();
    }
}

// Stop realtime detection
function stopRealtimeDetection() {
    // 取消下一幀的請求
    if (animationFrameId) {
        cancelAnimationFrame(animationFrameId);
        animationFrameId = null;
    }
    
    // 停止攝像頭流
    if (realtimeStream) {
        realtimeStream.getTracks().forEach(track => {
            track.stop();
        });
        realtimeStream = null;
    }
    
    // 重置UI狀態
    isRealtimeMode = false;
    fpsDisplay.style.display = 'none';
    realtimeControls.style.display = 'none';
    updateStatus('實時檢測已停止');
    updateInfo('');
    
    // 清除畫布
    inputCtx.clearRect(0, 0, inputCanvas.width, inputCanvas.height);
    outputCtx.clearRect(0, 0, outputCanvas.width, outputCanvas.height);
    
    // 執行完整的記憶體清理
    try {
        // 運行垃圾回收
        tf.tidy(() => {});
        tf.disposeVariables();
        
        // 嘗試進一步清理
        if (tf.engine().purge) {
            tf.engine().purge();
        }
        
        // 記錄當前張量信息
        const memoryInfo = tf.memory();
        console.log('記憶體清理完成，剩餘張量:', memoryInfo.numTensors);
        console.log('剩餘數據字節:', memoryInfo.numBytes);
    } catch (e) {
        console.error('清理TensorFlow資源時發生錯誤:', e);
    }
} 