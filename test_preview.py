import cv2
import fcntl
import v4l2
import time
import numpy as np
import os

# Function to set frame rate
def set_frame_rate(device, fps):
    fd = open(device, 'rb+', buffering=0)
    parm = v4l2.v4l2_streamparm()
    parm.type = v4l2.V4L2_BUF_TYPE_VIDEO_CAPTURE
    parm.parm.capture.timeperframe.numerator = 1
    parm.parm.capture.timeperframe.denominator = fps
    
    # Set the frame rate
    fcntl.ioctl(fd, v4l2.VIDIOC_S_PARM, parm)
    
    # Verify the frame rate
    fcntl.ioctl(fd, v4l2.VIDIOC_G_PARM, parm)
    actual_fps = parm.parm.capture.timeperframe.denominator / parm.parm.capture.timeperframe.numerator
    print(f"Frame rate set to {actual_fps} FPS")
    
    fd.close()

def set_control_parameter(device, control_id, value):
    fd = open(device, 'rb+', buffering=0)
    control = v4l2.v4l2_control()
    control.id = control_id
    control.value = value
    try:
        fcntl.ioctl(fd, v4l2.VIDIOC_S_CTRL, control)
        print(f'Successfully set control {control_id} to {value}')
    except IOError as e:
        print(f'Error setting control {control_id}: {e}')
    finally:
        fd.close()

# Device path for the camera (usually /dev/video0)
device_path = "/dev/video0"
desired_fps = 100

# Set the frame rate
set_frame_rate(device_path, desired_fps)

# V4L2 control IDs for brightness and exposure
V4L2_CID_GAIN = v4l2.V4L2_CID_GAIN
V4L2_CID_EXPOSURE = v4l2.V4L2_CID_EXPOSURE

# Optionally, set Exposure
# exposure_value = 500
# set_control_parameter(device_path, V4L2_CID_EXPOSURE, exposure_value)

# Open the video capture device using OpenCV
cap = cv2.VideoCapture(device_path)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open video device")
    exit()

# Optionally, set properties using OpenCV if needed
cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
cap.set(cv2.CAP_PROP_FPS, desired_fps)

fps = 0
prev_frame_time = 0
frame_counter = 0
count_time = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break

    frame_counter += 1

    new_frame_time = time.time()
    if prev_frame_time > 0:
        fps = 0.9 * fps + 0.1 / (new_frame_time - prev_frame_time)
        count_time += new_frame_time - prev_frame_time
    prev_frame_time = new_frame_time

    print(f"FPS: {fps:.2f}")

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
