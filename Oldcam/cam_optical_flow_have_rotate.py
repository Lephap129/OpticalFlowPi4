import cv2
import numpy as np
from math import *
import time
import threading
from sklearn.decomposition import PCA
import serial
from picamera2 import Picamera2
import csv

#Init Log
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
    
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

def updateLog(t,h,dx,dy,X,Y,FPS):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 'FPS': '{}'.format(FPS)}]
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerows(list_append)
        csvfile.close()

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
    #camera_height = 1000

# Preprocessing function
def preprocess_frame(frame):
    # Crop a region from the center of the frame
    center_x, center_y = frame.shape[0] // 2, frame.shape[1] // 2
    frame = frame[center_x - crop_size_w // 2:center_x + crop_size_w // 2,
                        center_y - crop_size_h // 2:center_y + crop_size_h // 2]
    frame = cv2.GaussianBlur(frame, (5, 5), 0)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    #gray = cv2.medianBlur(gray, 5)  # Applying median blur to reduce noise
    #cv2.rectangle(frame, (center_x - crop_size_w//2, center_y - crop_size_w//2),(center_x + crop_size_w//2, center_y + crop_size_w//2), (0, 255, 0), 2)
    return gray

# Frame capture thread
def capture_frames():
    global frame_buffer
    global update_task
    fps = 0
    prev_frame_time = 0
    while True:
        if update_task:
            frame = picam2.capture_array()
            with buffer_lock:
                frame_buffer = frame
            new_frame_time = time.time()
            fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time
            #update_task = False
            print("FPS_cam: ",fps)

def analyze_vectors(vectors):
    # Tính độ dịch chuyển tổng thể
    translation_vector = np.mean(vectors, axis=0)

    # Chuẩn hóa dữ liệu
    mean_vector = np.mean(vectors, axis=0)
    V_norm = vectors - mean_vector

    # Thực hiện PCA
    pca = PCA(n_components=2)
    pca.fit(V_norm)
    components = pca.components_
    explained_variance = pca.explained_variance_

    # Tính góc quay từ các thành phần chính
    angle_of_rotation = np.arctan2(components[1, 1], components[1, 0])
    return translation_vector,angle_of_rotation

# Set resolution and crop
max_resol = [4608, 2592]
div_resol = 8
width = int(max_resol[0] / div_resol)
height = int(max_resol[1] / div_resol)
div_crop = 8
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
#Kalman obsever gain
kalman = 0.9
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
count_time = 0
list_dx_cm = [0 for i in range(1)]
list_dy_cm = [0 for i in range(1)]
local_x = 0
local_y = 0
global_x = 0
global_y = 0
global_theta = 0
old_gray = None
p0 = None
prev_frame_time = 0
new_frame_time = 0

# Start frame capture thread
update_task = True
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
            frame_buffer = None
            update_task = True
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
        list_dx = good_old[:, 0] - good_new[:, 0]
        list_dy = good_old[:, 1] - good_new[:, 1]
        list_vector = np.array([[list_dx[i],list_dy[i]] for i in range(len(list_dx))])
        
        translate, d_theta = analyze_vectors(list_vector)
        
        dx_pixels = translate[0]
        dy_pixels = translate[1]
        
        cm_per_pxl_w = ( sensor_pxl_w * 1.4 * 10**-4 * camera_height / focal_length ) / width
        cm_per_pxl_h = ( sensor_pxl_h * 1.4 * 10**-4 * camera_height / focal_length ) / height
        
        
        #Add kalman obsever fillter
        local_x = (1-kalman)*local_x + kalman * dx_pixels * cm_per_pxl_w
        local_y = (1-kalman)*local_y + kalman * dy_pixels * cm_per_pxl_h
        
        T_G_B = np.array([[cos(global_theta), -sin(global_theta), global_x],
                          [sin(global_theta),  cos(global_theta), global_y],
                          [                0,                  0,        1]])
        
        r_B = np.array([local_x, local_y, 1])
        r_G = T_G_B.dot(r_B)
        global_x = r_G[0]
        global_y = r_G[1]
        global_theta += d_theta

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        if prev_frame_time > 0:
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        updateLog(count_time,camera_height, local_x, local_y, global_x, global_y, int(fps))
        #print("frame shape: ",format(frame.shape))
        #print("CM/pxl: ", 0.05*cm_per_pxl_h)
        print("Camera Height: ", camera_height)
        print("local_x: {:.2f}, local_y: {:.2f}".format(local_x, local_y))
        print("X: {:.2f}, Y: {:.2f}, theta: {:.2f}".format(global_x, global_y,global_theta))
        print("FPS: {}".format(int(fps)))
    else:
        for i in range(20):
            print("Crash")
        old_gray = preprocess_frame(frame)
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    #cv2.imshow("PiCam2", frame_gray)

    old_gray = frame_gray.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

ser.close()
picam2.stop()
cv2.destroyAllWindows()