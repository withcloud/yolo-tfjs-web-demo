# YOLO TensorFlow.js Pose Detection Web Demo

This is a web application that demonstrates real-time human pose detection using a YOLO model converted to TensorFlow.js format.

## Features

- Load and run YOLO pose detection model in the browser using TensorFlow.js
- Upload images for pose detection
- Capture images from your camera for detection
- Visualize detected keypoints and skeleton connections
- Responsive design works on desktop and mobile devices

## How to Use

### Prerequisites

- A modern web browser (Chrome, Firefox, Safari, or Edge recommended)
- The YOLO TensorFlow.js model files in the `/models` directory

### Installation

1. Clone this repository or download the files
2. Ensure your model files are placed in the `/models` directory with the following structure:
   ```
   /models
   ├── group1-shard1of3.bin
   ├── group1-shard2of3.bin
   ├── group1-shard3of3.bin
   ├── metadata.yaml
   └── model.json
   ```

### Running the Application

You can run this application in several ways:

#### Method 1: Using a local web server

1. Install Node.js if you don't have it already
2. Install a simple HTTP server:
   ```
   npm install -g http-server
   ```
3. Navigate to the project directory and start the server:
   ```
   http-server -c-1
   ```
4. Open your browser and go to `http://localhost:8080`

#### Method 2: Using Visual Studio Code with Live Server extension

1. Install the "Live Server" extension in VS Code
2. Right-click on `index.html` and select "Open with Live Server"

#### Method 3: Using Python's built-in HTTP server

1. Navigate to the project directory in your terminal
2. Run one of these commands:
   ```
   # For Python 3.x
   python -m http.server
   
   # For Python 2.x
   python -m SimpleHTTPServer
   ```
3. Open your browser and go to `http://localhost:8000`

### Using the Web Application

1. Wait for the model to load (you'll see a status message when it's ready)
2. Choose one of two options:
   - Click "選擇圖片" to upload an image from your device
   - Click "使用相機" to take a picture with your device's camera
3. The application will process the image and display:
   - The original image on the left
   - Pose detection results on the right with bounding boxes, keypoints, and skeleton lines
4. The detection information will be shown below the images

## Technical Details

- The application uses TensorFlow.js to run the YOLO pose detection model in the browser
- Image processing is done on the client side using HTML5 Canvas
- The model detects 17 keypoints on each person in the image
- Connections between keypoints are drawn as skeleton lines
- The web app is responsive and works on both desktop and mobile devices

## Troubleshooting

- If the model doesn't load, check your browser console for errors
- Make sure your model files are correctly placed in the `/models` directory
- Try using a different browser if you encounter issues
- For performance issues, try using a device with a stronger GPU

## License

This project uses the YOLO model which is licensed under AGPL-3.0 License. Please check the original Ultralytics repository for more details. 