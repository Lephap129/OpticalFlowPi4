import cv2
import numpy as np
import time
import threading
import serial
import arducam_mipicamera as arducam
import v4l2
import csv
    
######################### Record data ########################## 
fields = ['t','h','dx', 'dy', 'X', 'Y','FPS','exposure', 'br', 'dr']
filename = "record_data.csv"
with open(filename, 'w') as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=fields)
    writer.writeheader()

def updateLog(t,h,dx,dy,X,Y,FPS, exposure, br_percent, dr_percent):
    list_append = [{'t':'{:.04f}'.format(t),'h':'{:.02f}'.format(h),
                    'dx': '{:.02f}'.format(dx), 'dy': '{:.02f}'.format(dy), 
                    'X': '{:.02f}'.format(X), 'Y': '{:.02f}'.format(Y), 
                    'FPS': '{}'.format(FPS), 'exposure': '{}'.format(exposure),
                    'br': '{}'.format(br_percent), 'dr': '{}'.format(dr_percent)}]
    with open(filename, 'a') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fields)
        writer.writerows(list_append)
        csvfile.close()
################################################################

######################### Receive height value from STM32 #########################
# Init USB communication
port = '/dev/ttyUSB0'  
baudrate = 115200  
try:
    ser = serial.Serial(port, baudrate, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)
samples = []
sample_num = 5

# Receive data from COM port
def receive_data():
    global camera_height
    while True:
        if ser.in_waiting > 0:
            read_data = ser.readline().decode('utf-8').rstrip()
            try:
                config = -(float(read_data)*14/100) #Relative error = 14%
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
    # camera_height = 93.3
##################################################################################

######################### Camera setting function #########################
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
    global exposure, rGain, bGain

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

# AutoLight
def check_brightness():
    global frame, old_gray, exposure, gain, br_percent, dr_percent, cB_count,fps,dx_cm,dy_cm,global_x,global_y
    br_hold = 240
    dr_hold = 30
    cB_count += 1 
    try:
        n = frame.shape[0]*frame.shape[1]        
        # Count pixels with intensity greater than the threshold
        bright_pixels = np.sum(frame > br_hold)
        dark_pixels = np.sum(frame < dr_hold)
        br_percent = round(bright_pixels*100/n,2)
        dr_percent = round(dark_pixels*100/n,2)
        if (cB_count > 10):
            cB_count = 0
            if (br_percent > 60):
                print("br-{},dr-{}".format(br_percent,dr_percent))
                if (gain > 30):
                    print("Decrease gain...")
                    gain -= 30
                    camera.set_control(v4l2.V4L2_CID_GAIN, gain)
                elif (exposure > 500):
                    print("Decrease exposule...")
                    exposure -= 500
                    gain = 255
                    camera.set_control(v4l2.V4L2_CID_GAIN, gain)
                    camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
                    time.sleep(0.2)
                return True
            if (dr_percent > 60):
                print("br-{},dr-{}".format(br_percent,dr_percent))
                if (gain < 255 - 30):
                    print("Increase gain...")
                    gain += 30
                    camera.set_control(v4l2.V4L2_CID_GAIN, gain) 
                elif (exposure < 5500):
                    print("Increase exposule...")
                    exposure += 500
                    gain = 0
                    camera.set_control(v4l2.V4L2_CID_GAIN, gain)
                    camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
                    time.sleep(0.2)
                return True
        return False
    except:
        return False
    
    
# Frame capture thread
def capture_frames():
    global frame_buffer
    global fps_cam
    t0cam = 0
    fps_limit = 60
    while True:
        t1cam = time.time()
        fps_cam = 0.99 * fps_cam + 0.01 / (t1cam -t0cam)
        t0cam = t1cam
        frame = camera.capture(encoding = 'raw')
        with buffer_lock:
            frame_buffer = frame
        if (1/fps_limit)-1/fps_cam > 0:
            time.sleep(round((1/fps_limit)-1/fps_cam, 5))


##################################################################################

######################### Initial function and parameter #########################

# Initialize the camera
camera = arducam.mipi_camera()
print("Open camera...")
camera.init_camera()
exposure = 525
gain = 0
rGain = 0
bGain = 0
camera.set_mode(0) # chose a camera mode which yields raw10 pixel format, see output of list_format utility
set_controls(camera)

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
count_print = 0
cB_count = 0
dx_cm = 0
dy_cm = 0
global_x = 0
global_y = 0
old_gray = None
p0 = None
prev_frame_time = 0
new_frame_time = 0
kalman = 0.7

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

prev_frame_time = time.time()
check_light = True
while True:
    with buffer_lock:
        if frame_buffer is not None:
            frame = frame_buffer
            frame = preprocess_frame(frame)
            check_light = check_brightness()
            frame_buffer = None
    new_frame_time = time.time()
    if (new_frame_time - prev_frame_time > 10) and not check_light:
        break

##############################################################################

######################### Main processing loop ###############################
prev_frame_time = time.time()
while True:
    with buffer_lock:
        if frame_buffer is not None:
            frame = frame_buffer
            frame = preprocess_frame(frame)
            check_light = check_brightness()
            frame_buffer = None
            if check_light:
                dx_cm = 0
                dy_cm = 0
                old_gray = None
                continue
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
        global_x += dx_cm
        global_y += dy_cm
        continue

    good_new = p1[st == 1]
    good_old = p0[st == 1]
    
    # # Show detail of flow if need
    # sFrame = frame.copy()
    # for i, (new, old) in enumerate(zip(good_new, good_old)):
    #     a, b = new.ravel()
    #     c, d = old.ravel()
    #     cv2.line(sFrame, (a, b), (c, d), 255, 1)
    #     cv2.circle(sFrame, (a, b), 3, 255, -1)
    
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
        updateLog(count_time,camera_height, dx_cm, dy_cm, global_x, global_y, int(fps), exposure, br_percent, dr_percent)
        
        count_print+=1
        if count_print > 10:
            # print("frame shape: ",format(frame.shape))
            # print("CM/pxl: ", round(cm_per_pxl_w/7,3))
            # print("Br = {:.2f}%, Dr = {:.2f}% ".format(br_percent,dr_percent))
            # print("Exposule: ", exposure)
            # print("Gain: ", gain)
            # print("Pick point: ", len(good_new))
            print("Camera Height: ", camera_height)
            print("dx_cm: {:.3f}, dy_cm: {:.3f}".format(dx_cm, dy_cm))
            print("X: {:.3f}, Y: {:.3f}".format(global_x, global_y))
            print("FPS: {}".format(int(fps)))
            print("FPS_cam: {}".format(int(fps_cam)))
            count_print = 0
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
ser.close()
