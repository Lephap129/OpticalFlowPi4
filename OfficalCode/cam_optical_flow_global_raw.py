import cv2
import numpy as np
import time
import threading
import serial
import arducam_mipicamera as arducam
import v4l2
import csv
    
######################### Record data ########################## 
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

def updateLog(t,h,dx,dy,X,Y,FPS):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 'FPS': '{}'.format(FPS)}]
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerows(list_append)
        csvfile.close()
################################################################

# ######################## Receive height value from STM32 #########################
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
    #camera_height = 93.3
##################################################################################

######################### Initial function and parameter #########################
def preprocess_frame(frame):
    fmt = camera.get_format()
    w = fmt.get("width")
    h = fmt.get("height")
    frame = arducam.remove_padding(frame.data, w, h, 10)
    frame = arducam.unpack_mipi_raw10(frame)
    frame = (frame.reshape(h, w) >> 2).astype(np.uint8)
    center_x, center_y = frame.shape[0] // 2, frame.shape[1] // 2
    frame = frame[center_x - crop_size_w // 2:center_x + crop_size_w // 2,
                  center_y - crop_size_h // 2:center_y + crop_size_h // 2]
    return frame

def set_controls(camera):
    exposure = 1000
    rGain = 0
    bGain = 0

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

# Initialize the camera
camera = arducam.mipi_camera()
print("Open camera...")
camera.init_camera()
camera.set_mode(0) # chose a camera mode which yields raw10 pixel format, see output of list_format utility
set_controls(camera)

# Frame capture thread
def capture_frames():
    global frame_buffer
    global update_task
    global fps_cam
    t0cam = 0
    while True:
        t1cam = time.time()
        fps_cam = 0.99 * fps_cam + 0.01 / (t1cam -t0cam)
        t0cam = t1cam
        frame = camera.capture(encoding = 'raw')
        with buffer_lock:
            frame_buffer = frame

# Lucas-Kanade parameters
lk_params = dict(winSize=(21, 21),
                 maxLevel=3,
                 criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01))

# Detect feature points parameters
feature_params = dict(maxCorners=20,
                      qualityLevel=0.3,
                      minDistance=7,
                      blockSize=7)

# Parameters to find the cm per pixel 
width_img = 480
height_img = 640
div_crop = 8
crop_size_w = int(width_img / div_crop)
crop_size_h = int(height_img / div_crop)
focal_length = 0.13
pixel_dimension = 3 * 10**-4 
sensor_pxl_w = 480
sensor_pxl_h = 640
cm_per_pxl_w_coef = (sensor_pxl_w * pixel_dimension / focal_length) / width_img 
cm_per_pxl_h_coef = (sensor_pxl_h * pixel_dimension / focal_length) / height_img 

# Timing variables
fps = 0
fps_cam = 0
count_time = 0
list_dx_cm = [0 for i in range(1)]
list_dy_cm = [0 for i in range(1)]
dx_cm = 0
dy_cm = 0
global_x = 0
global_y = 0
old_gray = None
p0 = None
prev_frame_time = 0
new_frame_time = 0
kalman = 0.99

# Shared buffer for frames
frame_buffer = None
buffer_lock = threading.Lock()

# Start frame capture thread
update_task = True
capture_thread = threading.Thread(target=capture_frames)
capture_thread.start()

# Start data receiving thread
receive_thread = threading.Thread(target=receive_data)
receive_thread.start()
##############################################################################

######################### Main processing loop ###############################
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

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    if len(good_new) > 0:
        dx_pixels = np.mean(good_old[:, 0] - good_new[:, 0])
        dy_pixels = np.mean(good_old[:, 1] - good_new[:, 1])

        cm_per_pxl_w = cm_per_pxl_w_coef * camera_height
        cm_per_pxl_h = cm_per_pxl_h_coef * camera_height
        
        if abs(dx_pixels * cm_per_pxl_w) >= cm_per_pxl_w/7: 
            dx_cm = (1 - kalman) * dx_cm + kalman * dx_pixels * cm_per_pxl_w
        else: dx_cm = 0
        if abs(dy_pixels * cm_per_pxl_h) >= cm_per_pxl_h/7: 
            dy_cm = (1 - kalman) * dy_cm + kalman * dy_pixels * cm_per_pxl_h
        else: dy_cm = 0
        
        global_x += dx_cm
        global_y += dy_cm

        new_frame_time = time.time()
        fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
        if prev_frame_time > 0:
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        updateLog(count_time,camera_height, dx_cm, dy_cm, global_x, global_y, int(fps))
        
        count_time += 1
        if count_time > 30:
            print("Pick point: ", len(good_new))
            print("Camera Height: ", camera_height)
            print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            count_time = 0
    else:
        for i in range(10):
            print("Crash")
        old_gray = frame
        p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
        continue
    
    cv2.imshow("PiCam2", frame)
    old_gray = frame
    p0 = cv2.goodFeaturesToTrack(old_gray, mask=None, **feature_params)
    del frame

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
##############################################################################

print("Close camera...")
camera.close_camera()
cv2.destroyAllWindows()
# ser.close()
