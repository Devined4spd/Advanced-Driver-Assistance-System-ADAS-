Advanced Driver Assistance System (ADAS)

Lane Detection | Lane Curvature | LDW | YOLO Object Detection | Vehicle Tracking | Steering Angle | Speed Estimation | TTC Prediction
 
	Overview

This project implements a complete Advanced Driver Assistance System (ADAS) using Python, OpenCV, and YOLOv8.

It runs in real-time and includes multiple safety-critical features found in modern autonomous vehicles.

The system integrates:
•	✔ Lane Detection (Canny + Hough)
•	✔ Lane Curvature Estimation
•	✔ Lane Departure Warning (LDW)
•	✔ Steering Angle Prediction
•	✔ YOLOv8 Object Detection
•	✔ Vehicle ID Tracking
•	✔ Speed Estimation
•	✔ Time-To-Collision (TTC) Prediction
•	✔ Real-Time Visual Overlay
 
 
	Features

1.	Lane Detection
•	Canny Edge Detection
•	Region of Interest Masking
•	Hough Line Transform
•	Separation of left/right lanes

2.	Lane Curvature Calculation

Polynomial fitting + real world scaling (m/pixel).

3.	Lane Departure Warning (LDW)
   
•	Computes vehicle offset from lane center
•	Displays red warning with severity


4.	Steering Angle Prediction
•	Predicts turning direction
•	Visual steering line overlaid on screen

5.	YOLOv8 Object Detection
Detects:
•	Cars
•	Bikes
•	Buses
•	Trucks
•	Motorcycles

6.	Vehicle ID Tracking
A simple centroid-based tracker:
•	Assigns consistent IDs
•	Tracks vehicles frame-to-frame

7.	Speed Estimation
•	Based on frame-to-frame pixel displacement
•	Converts into km/h

8.	Time-To-Collision (TTC) Prediction
•	Calculates inverse-height rate of change
•	TTC displayed per vehicle
 
	Project Structure

├── lane_detection.py        # Main ADAS pipeline

├── test_video.mp4           # Sample driving footage (add your own)

├── yolov8n.pt               # Auto-downloaded by Ultralytics

├── README.md                # Documentation

└── requirements.txt         # Dependencies
 
Installation & Setup
1. Clone the Repository
git clone https://github.com/YOUR_USERNAME/Advanced-Driver-Assistance-System-(ADAS)-Project.git

cd Advanced-Driver-Assistance-System-(ADAS)-Project

2. Create and Activate Virtual Environment
python3 -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

3. Install Dependencies
   
pip install ultralytics opencv-python numpy
 
	Run the Project
Using a video file
python lane_detection.py
Using webcam
Inside lane_detection.py,

replace:
cap = cv2.VideoCapture("test_video.mp4")

with:
cap = cv2.VideoCapture(0)
 
	Requirements
ultralytics
opencv-python
numpy
 
	Technologies Used
•	Python
•	OpenCV
•	YOLOv8 (Ultralytics)
•	NumPy
•	Custom Tracking Algorithms
 
	Tested On
•	macOS (VS Code)
•	Python 3.9+
•	YOLOv8n model
•	720p & 1080p driving videos
 
	Future Enhancements
•	ByteTrack / DeepSORT tracking
•	Accurate distance estimation (depth)
•	Bird’s-eye view transformation
•	Road segmentation using U-Net
•	PyQt dashboard UI
•	Export processed video with overlays
 
	Contributing
Pull requests are welcome!
Feel free to open issues for improvements or bugs.

