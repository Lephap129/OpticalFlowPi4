import cv2
import numpy as np
from math import *
import time
import threading
import serial
import arducam_mipicamera as arducam
import v4l2
import csv
    
#Init Log
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()
'''    
# Init USB communication
port = '/dev/ttyUSB0'  
baudrate = 115200  
try:
    ser = serial.Serial(port, baudrate, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)
'''
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
    '''
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
        time.sleep(0.01)'''
    camera_height = 93.3
# Preprocessing function
def preprocess_frame(frame):
    fmt = camera.get_format()
    w = fmt.get("width")
    h = fmt.get("height")
    frame = arducam.remove_padding(frame.data, w, h, 10)
    frame = arducam.unpack_mipi_raw10(frame)
    frame = (frame.reshape(h, w) >> 2).astype(np.uint8)
    #print(frame.shape)
    center_x, center_y = frame.shape[0] // 2, frame.shape[1] // 2
    frame = frame[center_x - crop_size_w // 2:center_x + crop_size_w // 2,
                        center_y - crop_size_h // 2:center_y + crop_size_h // 2]
    # frame = cv2.GaussianBlur(frame, (5, 5), 0)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # frame = clahe.apply(frame)
    #print(frame.shape)
    #frame = cv2.medianBlur(frame, 5)  # Applying median blur to reduce noise
    #cv2.rectangle(frame, (center_x - crop_size_w//2, center_y - crop_size_w//2),(center_x + crop_size_w//2, center_y + crop_size_w//2), (0, 255, 0), 2)
    return frame

def pca_manual(vectors):
    # Chuẩn hóa dữ liệu
    mean_vector = np.mean(vectors, axis=0)
    V_norm = vectors - mean_vector

    # Tính toán ma trận hiệp phương sai
    covariance_matrix = np.cov(V_norm, rowvar=False)

    # Tính toán các giá trị riêng (eigenvalues) và vector riêng (eigenvectors)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)

    # Sắp xếp các giá trị riêng và vector riêng theo thứ tự giảm dần của các giá trị riêng
    sorted_indices = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[sorted_indices]
    eigenvectors = eigenvectors[:, sorted_indices]

    return eigenvectors, eigenvalues, mean_vector

def analyze_vectors(vectors):
    # Tính độ dịch chuyển tổng thể
    translation_vector = np.mean(vectors, axis=0)

    # Thực hiện PCA thủ công
    components, explained_variance, mean_vector = pca_manual(vectors)

    # Tính góc quay từ các thành phần chính
    angle_of_rotation = np.arctan2(components[1, 1], components[1, 0])

    # print(f"Kết quả phân tích cho {title}:")
    # print(f"Độ dịch chuyển tổng thể: {translation_vector}")
    # print(f"Góc quay (rad): {angle_of_rotation}")
    # print(f"Góc quay (deg): {np.degrees(angle_of_rotation)}")
    # print(f"Các thành phần chính:\n", components)
    # print(f"Giá trị riêng (explained variance):\n", explained_variance)
    return translation_vector,angle_of_rotation

exposure = 3000
rGain = 0
bGain = 0
def align_down(size, align):
    return (size & ~((align)-1))

def align_up(size, align):
    return align_down(size + align - 1, align)

def set_controls(camera):
    try:
        print("Reset the focus...")
        camera.reset_control(v4l2.V4L2_CID_FOCUS_ABSOLUTE)
    except Exception as e:
        print(e)
        print("The camera may not support this control.")

    try:
        print("Enable Auto Exposure...")
        camera.software_auto_exposure(enable = False)
        print("Handle setup...")
        camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
        camera.manual_set_awb_compensation(rGain, bGain)
        
    except Exception as e:
        print(e)
        

# Frame capture thread
def capture_frames():
    global frame_buffer
    global update_task
    global fps_cam
    t0cam = 0
    while True:
        if update_task:
            t1cam = time.time()
            fps_cam = 0.99 * fps_cam + 0.01 / (t1cam -t0cam)
            t0cam = t1cam
            frame = camera.capture(encoding = 'raw')
            with buffer_lock:
                frame_buffer = frame
            #update_task = False


# Set resolution and crop
max_resol = [480, 640]
div_resol = 1
width = int(max_resol[0] / div_resol)
height = int(max_resol[1] / div_resol)
div_crop = 6
crop_size_w = int(width / div_crop)
crop_size_h = int(height / div_crop)

# Parameters to find the cm per pixel 
focal_length = 0.13

sensor_pxl_w = 480
sensor_pxl_h = 640

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
camera = arducam.mipi_camera()
print("Open camera...")
camera.init_camera()
camera.set_mode(0) # chose a camera mode which yields raw10 pixel format, see output of list_format utility
set_controls(camera)
time.sleep(1)

# Shared buffer for frames
frame_buffer = None
buffer_lock = threading.Lock()

# Timing variables
fps = 0
fps_cam = 0
count_time = 0
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
            frame = frame_buffer
            frame = preprocess_frame(frame)
            frame_buffer = None
            update_task = True
        else:
            continue
    
    if old_gray is None:
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    try:
        p1, st, err = cv2.calcOpticalFlowPyrLK(old_gray, frame, p0, None, **lk_params)
    except:
        for i in range(10):
            print("False tracking!!!!!!!!!!!!")
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    # print("List: ",p0)
    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    sFrame = frame.copy()
    for i, (new, old) in enumerate(zip(good_new, good_old)):
        a, b = new.ravel()
        c, d = old.ravel()
        cv2.line(sFrame, (a, b), (c, d), 255, 1)
        cv2.circle(sFrame, (a, b), 3, 255, -1)

    if len(good_new) > 0:
        list_dx = good_old[:, 0] - good_new[:, 0]
        list_dy = good_old[:, 1] - good_new[:, 1]
        list_vector = np.array([[list_dx[i],list_dy[i]] for i in range(len(list_dx))])
        
        translate, d_theta = analyze_vectors(list_vector)
        
        dx_pixels = translate[0]
        dy_pixels = translate[1]
        
        cm_per_pxl_w = ( sensor_pxl_w * 3 * 10**-4 * camera_height / focal_length ) / width
        cm_per_pxl_h = ( sensor_pxl_h * 3 * 10**-4 * camera_height / focal_length ) / height
        
        #Add low pass fillter and kalman obsever fillter
        if abs(dx_pixels * cm_per_pxl_w) > 0.009: local_x = (1-kalman)*local_x + kalman * dx_pixels * cm_per_pxl_w
        else: local_x = 0
        if abs(dy_pixels * cm_per_pxl_h) > 0.009: local_y = (1-kalman)*local_y + kalman * dy_pixels * cm_per_pxl_h
        else: local_y = 0
        
        T_G_B = np.array([[cos(global_theta), -sin(global_theta), global_x],
                          [sin(global_theta),  cos(global_theta), global_y],
                          [                0,                  0,        1]])
        
        r_B = np.array([local_x, local_y, 1])
        r_G = T_G_B.dot(r_B)
        global_x = r_G[0]
        global_y = r_G[1]
        if (global_theta + d_theta) < 0:
            global_theta = global_theta + 2*pi + d_theta
        elif (global_theta + d_theta) > 2*pi:
            global_theta = global_theta - 2*pi + d_theta
        else:
            global_theta += d_theta
            

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        if prev_frame_time > 0:
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        updateLog(count_time,camera_height, local_x, local_y, global_x, global_y, int(fps))
        count_time+=1
        if count_time > 30:
            #print("frame shape: ",format(frame.shape))
            #print("CM/pxl: ", 0.05*cm_per_pxl_h)
            print("Pick point: ", len(good_new))
            print("Camera Height: ", camera_height)
            print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(local_x, local_y))
            print("X: {:.3f}, Y: {:.3f}, Theta: {:.3f}".format(global_x, global_y, np.degrees(global_theta)))
            print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            count_time = 0
    else:
        for i in range(10):
            print("Crash")
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    cv2.imshow("PiCam2", sFrame)
    old_gray = frame.copy()
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    del frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

#ser.close()
# Release memory
del frame
print("Close camera...")
camera.close_camera()
cv2.destroyAllWindows()