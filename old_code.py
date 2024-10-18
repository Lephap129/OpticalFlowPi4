import cv2
import numpy as np
from math import *
import time
import threading
import serial
from picamera2 import Picamera2

# Init USB communication
port = '/dev/ttyUSB0'  
baudrate = 115200  
try:
    ser = serial.Serial(port, baudrate, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)
samples = []
sample_num = 50

# Receive data from COM port
def receive_data():
    global camera_height
    while True:
        if ser.in_waiting > 0:
            read_data = ser.readline().decode('utf-8').rstrip()
            try:
                config = -(float(read_data)*7/100)
                samples.append(float(read_data)+ config)
                if len(samples) == sample_num:
                    camera_height = round( (sum(samples) / len(samples) / 10 ), 1)
                    samples.pop(0)
                else:
                    camera_height = (round( float(read_data) + config, 1) / 10)
            except:
                if len(samples) == sample_num:
                    camera_height = samples[-1]
                else:
                    continue
        time.sleep(0.01)

# Preprocessing function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Crop a region from the center of the frame
    center_x, center_y = gray.shape[0] // 2, gray.shape[1] // 2
    cropped_gray = gray[center_x - crop_size_w // 2:center_x + crop_size_w // 2,
                        center_y - crop_size_h // 2:center_y + crop_size_h // 2]
    return cropped_gray

# Frame capture thread
def capture_frames():
    global frame_buffer
    while True:
        frame = picam2.capture_array()
        with buffer_lock:
            frame_buffer = frame

# Set resolution and crop
max_resol = [4608, 2592]
div_resol = 8
width = int(max_resol[0] / div_resol)
height = int(max_resol[1] / div_resol)
div_crop = 12
crop_size_w = int(width / div_crop)
crop_size_h = int(height / div_crop)

# Parameters to find the cm per pixel 
focal_length = 0.474
sensor_pxl_w = 3072
sensor_pxl_h = 1728

# Lucas-Kanade parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Detect feature points parameters
feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)
        
# Configure cam
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (width, height)})
picam2.preview_configuration.controls.FrameRate = 120
picam2.configure(config)
picam2.start()

# Shared buffer for frames
frame_buffer = None
buffer_lock = threading.Lock()

# Timing variables
fps = 0
global_x = 0
global_y = 0
old_gray = None
p0 = None
prev_frame_time = 0
new_frame_time = 0

# Start frame capture thread
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Start data receiving thread
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()

# Main processing loop
while True:
    with buffer_lock:
        if frame_buffer is not None:
            frame = frame_buffer.copy()
        else:
            continue
    
    if old_gray is None:
        old_gray = preprocess_frame(frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    frame_gray = preprocess_frame(frame)
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame_gray, p0, None, **lk_params)
    except:
        old_gray = preprocess_frame(frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    good_new = p1[st == 1]
    good_old = p0[st == 1]

    if len(good_new) > 0:
        dx_pixels = np.mean(good_new[:, 0] - good_old[:, 0])
        dy_pixels = np.mean(good_new[:, 1] - good_old[:, 1])

        cm_per_pxl_w = ( sensor_pxl_w * 1.4 * 10**-4 * camera_height / focal_length ) / width
        cm_per_pxl_h = ( sensor_pxl_h * 1.4 * 10**-4 * camera_height / focal_length ) / height
        
        dx_cm = dx_pixels * cm_per_pxl_w
        dy_cm = dy_pixels * cm_per_pxl_h

        global_x += dx_cm
        global_y += dy_cm

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        prev_frame_time = new_frame_time
        
        print("Camera Height: ", camera_height)
        print("dx_cm: {:.2f}, dy_cm: {:.2f}".format(dx_cm, dy_cm))
        print("X: {:.2f}, Y: {:.2f}".format(global_x, global_y))
        print("FPS: {}".format(int(fps)))
    else:
        for i in range(20):
            print("Crash")
        old_gray = preprocess_frame(frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue

    #cv2.imshow("PiCam2", frame)

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
picam2.stop()
cv2.destroyAllWindows()
