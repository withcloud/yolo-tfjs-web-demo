// DOM elements
const imageUpload = document.getElementById('imageUpload');
const captureBtn = document.getElementById('captureBtn');
const cameraModal = document.getElementById('cameraModal');
const closeCamera = document.querySelector('.close-camera');
const video = document.getElementById('video');
const takePictureBtn = document.getElementById('takePictureBtn');
const originalCanvas = document.getElementById('originalCanvas');
const yoloCanvas = document.getElementById('yoloCanvas');
const yoloCustomCanvas = document.getElementById('yoloCustomCanvas');
const gdeCanvas = document.getElementById('gdeCanvas');
const gdeDistillCanvas = document.getElementById('gdeDistillCanvas');
const statusDiv = document.getElementById('status');

// Loading UI elements
const loadSection = document.getElementById('loadSection');
const inputSection = document.getElementById('inputSection');
const yoloProgress = document.getElementById('yoloProgress');
const yoloCustomProgress = document.getElementById('yoloCustomProgress');
const gdeProgress = document.getElementById('gdeProgress');
const gdeDistillProgress = document.getElementById('gdeDistillProgress');
const yoloStatus = document.getElementById('yoloStatus');
const yoloCustomStatus = document.getElementById('yoloCustomStatus');
const gdeStatus = document.getElementById('gdeStatus');
const gdeDistillStatus = document.getElementById('gdeDistillStatus');
const overallStatus = document.getElementById('overallStatus');

// Detection loading UI elements
const yoloLoading = document.getElementById('yoloLoading');
const yoloCustomLoading = document.getElementById('yoloCustomLoading');
const gdeLoading = document.getElementById('gdeLoading');
const gdeDistillLoading = document.getElementById('gdeDistillLoading');

// Performance info elements
const yoloTime = document.getElementById('yoloTime');
const yoloCustomTime = document.getElementById('yoloCustomTime');
const gdeTime = document.getElementById('gdeTime');
const gdeDistillTime = document.getElementById('gdeDistillTime');

// Canvas contexts
const originalCtx = originalCanvas.getContext('2d');
const yoloCtx = yoloCanvas.getContext('2d');
const yoloCustomCtx = yoloCustomCanvas.getContext('2d');
const gdeCtx = gdeCanvas.getContext('2d');
const gdeDistillCtx = gdeDistillCanvas.getContext('2d');

// Default canvas size
const defaultWidth = 640;
const defaultHeight = 640;
originalCanvas.width = defaultWidth;
originalCanvas.height = defaultHeight;
yoloCanvas.width = defaultWidth;
yoloCanvas.height = defaultHeight;
yoloCustomCanvas.width = defaultWidth;
yoloCustomCanvas.height = defaultHeight;
gdeCanvas.width = defaultWidth;
gdeCanvas.height = defaultHeight;
gdeDistillCanvas.width = defaultWidth;
gdeDistillCanvas.height = defaultHeight;

// Model variables
let yoloModel = null;
let yoloCustomModel = null;
let gdeModel = null;
let gdeDistillModel = null;
let modelsLoaded = false;
let inputImageData = null;
let imageInfo = null;
let realtimeStream = null;

// Device detection
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

// 回答問題1: 我們將按順序加載模型，而不是同時加載
// 這樣可以減少內存壓力，提高加載成功率

// 初始化時檢測設備類型
function detectDevice() {
    const userAgent = navigator.userAgent.toLowerCase();
    isMobile = /android|webos|iphone|ipad|ipod|blackberry|iemobile|opera mini/i.test(userAgent);
    console.log(`檢測到${isMobile ? '移動' : '桌面'}設備`);
}

// Initialize application
async function init() {
    updateStatus('正在初始化比較頁面...');
    detectDevice();
    
    try {
        // Set up TensorFlow.js with WebGL backend
        await tf.setBackend('webgl');
        console.log('使用WebGL後端');
        
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
        
        // 按順序依次加載模型，而不是同時加載
        await loadYoloModel();
        await loadYoloCustomModel();
        await loadGdeModel();
        await loadGdeDistillModel();
        
        modelsLoaded = true;
        overallStatus.textContent = '所有模型已加載完成，準備就緒！';
        
        // 顯示輸入區域
        setTimeout(() => {
            loadSection.style.display = 'none';
            inputSection.style.display = 'block';
            updateStatus('請選擇圖片或使用相機拍攝照片');
        }, 1000);
    } catch (error) {
        console.error('初始化錯誤：', error);
        overallStatus.textContent = '初始化失敗：' + error.message;
    }
}

// Load YOLO model
async function loadYoloModel() {
    yoloStatus.textContent = '正在載入中...';
    
    try {
        // Load the model
        yoloModel = await tf.loadGraphModel('../models/yolo11n-pose/model.json', {
            onProgress: (fraction) => {
                yoloProgress.style.width = `${Math.round(fraction * 100)}%`;
                yoloStatus.textContent = `載入中... ${Math.round(fraction * 100)}%`;
            }
        });
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await yoloModel.predict(dummyInput).array();
        dummyInput.dispose();
        
        yoloProgress.style.width = '100%';
        yoloStatus.textContent = '已加載完成';
        console.log('YOLO模型已加載');
        return true;
    } catch (error) {
        console.error('載入YOLO模型錯誤：', error);
        yoloStatus.textContent = '載入失敗：' + error.message;
        return false;
    }
}

// Load YOLO Custom model
async function loadYoloCustomModel() {
    yoloCustomStatus.textContent = '正在載入中...';
    
    try {
        // Load the model
        yoloCustomModel = await tf.loadGraphModel('../models/yolo11n-pose-custom-640/model.json', {
            onProgress: (fraction) => {
                yoloCustomProgress.style.width = `${Math.round(fraction * 100)}%`;
                yoloCustomStatus.textContent = `載入中... ${Math.round(fraction * 100)}%`;
            }
        });
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await yoloCustomModel.predict(dummyInput).array();
        dummyInput.dispose();
        
        yoloCustomProgress.style.width = '100%';
        yoloCustomStatus.textContent = '已加載完成';
        console.log('YOLO Custom模型已加載');
        return true;
    } catch (error) {
        console.error('載入YOLO Custom模型錯誤：', error);
        yoloCustomStatus.textContent = '載入失敗：' + error.message;
        return false;
    }
}

// Load GDE model
async function loadGdeModel() {
    gdeStatus.textContent = '正在載入中...';
    
    try {
        // Load the model
        gdeModel = await tf.loadGraphModel('../models/gde-pose-640/model.json', {
            onProgress: (fraction) => {
                gdeProgress.style.width = `${Math.round(fraction * 100)}%`;
                gdeStatus.textContent = `載入中... ${Math.round(fraction * 100)}%`;
            }
        });
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await gdeModel.predict(dummyInput).array();
        dummyInput.dispose();
        
        gdeProgress.style.width = '100%';
        gdeStatus.textContent = '已加載完成';
        console.log('GDE模型已加載');
        return true;
    } catch (error) {
        console.error('載入GDE模型錯誤：', error);
        gdeStatus.textContent = '載入失敗：' + error.message;
        return false;
    }
}

// Load GDE Distill model
async function loadGdeDistillModel() {
    gdeDistillStatus.textContent = '正在載入中...';
    
    try {
        // Load the model
        gdeDistillModel = await tf.loadGraphModel('../models/gde-pose-distill/model.json', {
            onProgress: (fraction) => {
                gdeDistillProgress.style.width = `${Math.round(fraction * 100)}%`;
                gdeDistillStatus.textContent = `載入中... ${Math.round(fraction * 100)}%`;
            }
        });
        
        // Warm up the model
        const dummyInput = tf.zeros([1, 640, 640, 3]);
        await gdeDistillModel.predict(dummyInput).array();
        dummyInput.dispose();
        
        gdeDistillProgress.style.width = '100%';
        gdeDistillStatus.textContent = '已加載完成';
        console.log('GDE Distill模型已加載');
        return true;
    } catch (error) {
        console.error('載入GDE Distill模型錯誤：', error);
        gdeDistillStatus.textContent = '載入失敗：' + error.message;
        return false;
    }
}

// Update status message
function updateStatus(message) {
    statusDiv.textContent = message;
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

// Process the input image
async function processImage(img) {
    if (!modelsLoaded) {
        updateStatus('模型尚未加載完成，請稍候...');
        return;
    }
    
    updateStatus('處理圖像中...');
    
    try {
        // Prepare image for processing
        const imgInfo = letterboxImage(img);
        imageInfo = imgInfo;
        
        // Draw original image
        originalCtx.clearRect(0, 0, originalCanvas.width, originalCanvas.height);
        originalCtx.drawImage(imgInfo.canvas, 0, 0);
        
        // Store input image data for later use by models
        inputImageData = imgInfo.canvas;
        
        // Clear all result canvases
        yoloCtx.clearRect(0, 0, yoloCanvas.width, yoloCanvas.height);
        yoloCustomCtx.clearRect(0, 0, yoloCustomCanvas.width, yoloCustomCanvas.height);
        gdeCtx.clearRect(0, 0, gdeCanvas.width, gdeCanvas.height);
        gdeDistillCtx.clearRect(0, 0, gdeDistillCanvas.width, gdeDistillCanvas.height);
        
        // Reset timing info
        yoloTime.textContent = '-';
        yoloCustomTime.textContent = '-';
        gdeTime.textContent = '-';
        gdeDistillTime.textContent = '-';
        
        // Run predictions sequentially for each model
        await runYoloPrediction();
        await runYoloCustomPrediction();
        await runGdePrediction();
        await runGdeDistillPrediction();
        
        updateStatus('所有模型處理完成！');
    } catch (error) {
        console.error('處理圖像錯誤：', error);
        updateStatus('處理圖像時發生錯誤：' + error.message);
    }
}

// YOLO Prediction
async function runYoloPrediction() {
    if (!yoloModel || !inputImageData) return;
    
    // Show loading indicator
    yoloLoading.style.display = 'block';
    yoloTime.textContent = '計算中...';
    
    try {
        // Copy the original image to YOLO canvas
        yoloCtx.clearRect(0, 0, yoloCanvas.width, yoloCanvas.height);
        yoloCtx.drawImage(inputImageData, 0, 0);
        
        // 測量開始時間
        const startTime = performance.now();
        
        // Convert image to tensor
        const imageTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(inputImageData);
            const normalized = img.div(255.0);
            return normalized.expandDims(0);
        });
        
        // Run prediction
        const predictions = await yoloModel.predict(imageTensor);
        
        // Process predictions
        const outputs = await processPredictions(predictions, 'yolo');
        
        // 測量結束時間
        const endTime = performance.now();
        const inferenceTime = Math.round(endTime - startTime);
        
        // Display time
        yoloTime.textContent = `${inferenceTime} ms`;
        
        // Draw the detections
        drawDetections(outputs, imageInfo, yoloCtx);
        
        // Clean up tensors
        imageTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('YOLO預測錯誤：', error);
        yoloTime.textContent = '錯誤';
    } finally {
        // Hide loading indicator
        yoloLoading.style.display = 'none';
    }
}

// YOLO Custom Prediction
async function runYoloCustomPrediction() {
    if (!yoloCustomModel || !inputImageData) return;
    
    // Show loading indicator
    yoloCustomLoading.style.display = 'block';
    yoloCustomTime.textContent = '計算中...';
    
    try {
        // Copy the original image to YOLO Custom canvas
        yoloCustomCtx.clearRect(0, 0, yoloCustomCanvas.width, yoloCustomCanvas.height);
        yoloCustomCtx.drawImage(inputImageData, 0, 0);
        
        // 測量開始時間
        const startTime = performance.now();
        
        // Convert image to tensor
        const imageTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(inputImageData);
            const normalized = img.div(255.0);
            return normalized.expandDims(0);
        });
        
        // Run prediction
        const predictions = await yoloCustomModel.predict(imageTensor);
        
        // Process predictions
        const outputs = await processPredictions(predictions, 'yolo-custom');
        
        // 測量結束時間
        const endTime = performance.now();
        const inferenceTime = Math.round(endTime - startTime);
        
        // Display time
        yoloCustomTime.textContent = `${inferenceTime} ms`;
        
        // Draw the detections
        drawDetections(outputs, imageInfo, yoloCustomCtx);
        
        // Clean up tensors
        imageTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('YOLO Custom預測錯誤：', error);
        yoloCustomTime.textContent = '錯誤';
    } finally {
        // Hide loading indicator
        yoloCustomLoading.style.display = 'none';
    }
}

// GDE Prediction
async function runGdePrediction() {
    if (!gdeModel || !inputImageData) return;
    
    // Show loading indicator
    gdeLoading.style.display = 'block';
    gdeTime.textContent = '計算中...';
    
    try {
        // Copy the original image to GDE canvas
        gdeCtx.clearRect(0, 0, gdeCanvas.width, gdeCanvas.height);
        gdeCtx.drawImage(inputImageData, 0, 0);
        
        // 測量開始時間
        const startTime = performance.now();
        
        // Convert image to tensor
        const imageTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(inputImageData);
            const normalized = img.div(255.0);
            return normalized.expandDims(0);
        });
        
        // Run prediction
        const predictions = await gdeModel.predict(imageTensor);
        
        // Process predictions
        const outputs = await processPredictions(predictions, 'gde');
        
        // 測量結束時間
        const endTime = performance.now();
        const inferenceTime = Math.round(endTime - startTime);
        
        // Display time
        gdeTime.textContent = `${inferenceTime} ms`;
        
        // Draw the detections
        drawDetections(outputs, imageInfo, gdeCtx);
        
        // Clean up tensors
        imageTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('GDE預測錯誤：', error);
        gdeTime.textContent = '錯誤';
    } finally {
        // Hide loading indicator
        gdeLoading.style.display = 'none';
    }
}

// GDE Distill Prediction
async function runGdeDistillPrediction() {
    if (!gdeDistillModel || !inputImageData) return;
    
    // Show loading indicator
    gdeDistillLoading.style.display = 'block';
    gdeDistillTime.textContent = '計算中...';
    
    try {
        // Copy the original image to GDE Distill canvas
        gdeDistillCtx.clearRect(0, 0, gdeDistillCanvas.width, gdeDistillCanvas.height);
        gdeDistillCtx.drawImage(inputImageData, 0, 0);
        
        // 測量開始時間
        const startTime = performance.now();
        
        // Convert image to tensor
        const imageTensor = tf.tidy(() => {
            const img = tf.browser.fromPixels(inputImageData);
            const normalized = img.div(255.0);
            return normalized.expandDims(0);
        });
        
        // Run prediction
        const predictions = await gdeDistillModel.predict(imageTensor);
        
        // Process predictions
        const outputs = await processPredictions(predictions, 'gde-distill');
        
        // 測量結束時間
        const endTime = performance.now();
        const inferenceTime = Math.round(endTime - startTime);
        
        // Display time
        gdeDistillTime.textContent = `${inferenceTime} ms`;
        
        // Draw the detections
        drawDetections(outputs, imageInfo, gdeDistillCtx);
        
        // Clean up tensors
        imageTensor.dispose();
        predictions.dispose();
    } catch (error) {
        console.error('GDE Distill預測錯誤：', error);
        gdeDistillTime.textContent = '錯誤';
    } finally {
        // Hide loading indicator
        gdeDistillLoading.style.display = 'none';
    }
}

// Process predictions based on model type
async function processPredictions(predictions, modelType) {
    try {
        // 將張量轉換為JavaScript數據
        const output = await predictions.array();
        
        // 常量處理
        const confThresh = 0.5;  // 置信度閾值
        const iouThresh = 0.45;  // IOU閾值
        const numKeypoints = 17;
        const results = [];
        
        // 檢查輸出是否有效
        if (!output || !output[0] || !output[0].length) {
            return results;
        }
        
        const candidates = [];
        const rawOutput = output[0];
        
        // 提取候選檢測
        for (let i = 0; i < rawOutput[0].length; i++) {
            const confidence = rawOutput[4][i];  // 置信度分數
            
            // 按置信度過濾
            if (confidence > confThresh) {
                // 提取邊界框（YOLO格式：center_x, center_y, width, height）
                const centerX = rawOutput[0][i];
                const centerY = rawOutput[1][i];
                const width = rawOutput[2][i];
                const height = rawOutput[3][i];
                
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
                    if (kpIndex < rawOutput.length) {
                        kpts.push([
                            rawOutput[kpIndex][i], 
                            rawOutput[kpIndex + 1][i], 
                            rawOutput[kpIndex + 2][i]
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
        
        // 應用非極大值抑制
        const selected = nonMaxSuppression(candidates, iouThresh);
        
        return selected;
    } catch (error) {
        console.error(`處理${modelType}預測結果錯誤:`, error);
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
    const xLeft = Math.max(ax1, bx1);
    const yTop = Math.max(ay1, by1);
    const xRight = Math.min(ax2, bx2);
    const yBottom = Math.min(ay2, by2);
    
    if (xRight < xLeft || yBottom < yTop) {
        return 0;  // No intersection
    }
    
    const intersectionArea = (xRight - xLeft) * (yBottom - yTop);
    
    // Calculate union area
    const box1Area = (ax2 - ax1) * (ay2 - ay1);
    const box2Area = (bx2 - bx1) * (by2 - by1);
    const unionArea = box1Area + box2Area - intersectionArea;
    
    return intersectionArea / unionArea;
}

// Letterbox image to maintain aspect ratio
function letterboxImage(img) {
    // Create a canvas to draw the letterboxed image
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = defaultWidth;
    canvas.height = defaultHeight;
    
    // Calculate scaling and padding
    const scale = Math.min(
        defaultWidth / img.width,
        defaultHeight / img.height
    );
    
    const newWidth = img.width * scale;
    const newHeight = img.height * scale;
    const offsetX = (defaultWidth - newWidth) / 2;
    const offsetY = (defaultHeight - newHeight) / 2;
    
    // Draw black background
    ctx.fillStyle = '#000';
    ctx.fillRect(0, 0, defaultWidth, defaultHeight);
    
    // Draw letterboxed image
    ctx.drawImage(
        img,
        0, 0, img.width, img.height,
        offsetX, offsetY, newWidth, newHeight
    );
    
    return {
        canvas: canvas,
        scale: scale,
        offsetX: offsetX,
        offsetY: offsetY,
        originalWidth: img.width,
        originalHeight: img.height
    };
}

// Draw pose detection results
function drawDetections(predictions, imgInfo, ctx) {
    // Clear canvas with the original image
    ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.drawImage(inputImageData, 0, 0);
    
    // Apply transparency to make skeleton more visible
    ctx.globalAlpha = 0.7;
    ctx.fillStyle = 'rgba(0, 0, 0, 0.4)';
    ctx.fillRect(0, 0, ctx.canvas.width, ctx.canvas.height);
    ctx.globalAlpha = 1.0;
    
    // Draw image again
    ctx.globalAlpha = 0.6;
    ctx.drawImage(inputImageData, 0, 0);
    ctx.globalAlpha = 1.0;
    
    // Scale settings for drawing
    ctx.lineWidth = 3;
    
    // Process each detection
    predictions.forEach(detection => {
        // Draw bounding box
        const [x1, y1, x2, y2] = detection.bbox;
        const boxWidth = x2 - x1;
        const boxHeight = y2 - y1;
        
        ctx.beginPath();
        ctx.strokeStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.rect(x1, y1, boxWidth, boxHeight);
        ctx.stroke();
        
        // Draw confidence score
        ctx.fillStyle = 'rgba(0, 255, 0, 0.8)';
        ctx.font = '16px Arial';
        ctx.fillText(`${Math.round(detection.confidence * 100)}%`, x1, y1 > 10 ? y1 - 5 : 15);
        
        // Draw keypoints
        detection.keypoints.forEach((keypoint, index) => {
            const [x, y, confidence] = keypoint;
            
            // Only draw keypoints with sufficient confidence
            if (confidence > 0.3) {  // Lower threshold to show more keypoints
                // Draw keypoint circle
                ctx.beginPath();
                ctx.arc(x, y, 5, 0, 2 * Math.PI);  // Larger circle for better visibility
                ctx.fillStyle = keypointColors[index];
                ctx.fill();
                
                // Add outline for better visibility
                ctx.strokeStyle = 'white';
                ctx.lineWidth = 1;
                ctx.stroke();
            }
        });
        
        // Draw connections (skeleton)
        ctx.lineWidth = 3;  // Thicker lines for better visibility
        connections.forEach(connection => {
            const [i, j] = connection;
            const [x1, y1, conf1] = detection.keypoints[i];
            const [x2, y2, conf2] = detection.keypoints[j];
            
            // Only draw connections when both keypoints have sufficient confidence
            if (conf1 > 0.3 && conf2 > 0.3) {  // Lower threshold for better visualization
                ctx.beginPath();
                ctx.moveTo(x1, y1);
                ctx.lineTo(x2, y2);
                
                // Use gradient color between the two keypoints
                const gradient = ctx.createLinearGradient(x1, y1, x2, y2);
                gradient.addColorStop(0, keypointColors[i]);
                gradient.addColorStop(1, keypointColors[j]);
                ctx.strokeStyle = gradient;
                
                ctx.stroke();
            }
        });
    });
}

// Camera functions
function openCamera() {
    cameraModal.style.display = 'block';
    startCamera();
}

function closeVideoModal() {
    cameraModal.style.display = 'none';
    stopCamera();
}

async function startCamera() {
    try {
        // Get user media stream
        realtimeStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: 'environment',
                width: { ideal: 640 },
                height: { ideal: 640 }
            },
            audio: false
        });
        
        // Display stream in video element
        video.srcObject = realtimeStream;
    } catch (error) {
        console.error('相機啟動錯誤：', error);
        alert('無法啟動相機：' + error.message);
    }
}

function stopCamera() {
    if (realtimeStream) {
        realtimeStream.getTracks().forEach(track => track.stop());
        realtimeStream = null;
        video.srcObject = null;
    }
}

function takePicture() {
    if (!video.srcObject) return;
    
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = video.videoWidth;
    tempCanvas.height = video.videoHeight;
    
    const tempCtx = tempCanvas.getContext('2d');
    tempCtx.drawImage(video, 0, 0, tempCanvas.width, tempCanvas.height);
    
    // Close the camera modal
    closeVideoModal();
    
    // Process the captured image
    const img = new Image();
    img.onload = () => processImage(img);
    img.src = tempCanvas.toDataURL('image/png');
}

// Initialize the application when the page loads
window.addEventListener('DOMContentLoaded', init); 