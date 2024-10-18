import cv2
import fcntl
import v4l2
import time
import numpy as np
import os
import asyncio
fps_Cam = 0
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

async def read_frame(cap):
    global fps_Cam
    prev_frame_time = 0
    frame_counter = 0
    count_time = 0
    while True:
        ret, frame = cap.read()
        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps_Cam = 0.9 * fps_Cam + 0.1 / (new_frame_time - prev_frame_time)
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time
        if not ret:
            print("Failed to grab frame")
            break
        yield frame

async def process_frame(cap):
    fps = 0
    prev_frame_time = 0
    frame_counter = 0
    count_time = 0

    async for frame in read_frame(cap):
        frame_counter += 1

        new_frame_time = time.time()
        if prev_frame_time > 0:
            fps = 0.9 * fps + 0.1 / (new_frame_time - prev_frame_time)
            count_time += new_frame_time - prev_frame_time
        prev_frame_time = new_frame_time

        print(f"FPS: {fps:.2f}")
        print(f"FPS_Cam: {fps_Cam:.2f}")

        #cv2.imshow("Frame", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

async def main():
    # Device path for the camera (usually /dev/video0)
    device_path = "/dev/video0"
    desired_fps = 100

    # Set the frame rate
    set_frame_rate(device_path, desired_fps)

    # Open the video capture device using OpenCV
    cap = cv2.VideoCapture(device_path)

    # Check if the camera opened successfully
    if not cap.isOpened():
        print("Error: Could not open video device")
        return

    # Optionally, set properties using OpenCV if needed
    cap.set(cv2.CAP_PROP_CONVERT_RGB, 0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, desired_fps)

    await process_frame(cap)

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    asyncio.run(main())
