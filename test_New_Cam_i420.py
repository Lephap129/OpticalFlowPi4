import arducam_mipicamera as arducam
import v4l2
import time
import numpy as np
import cv2

exposure = 552
rGain = 0
bGain = 0

def align_down(size, align):
    return (size & ~((align)-1))

def align_up(size, align):
    return align_down(size + align - 1, align)

def set_controls(camera):
    try:
        print("Reset the focus...")
        camera.reset_control(v4l2.V4L2_CID_EXPOSURE_ABSOLUTE)
    except Exception as e:
        print(e)
        print("The camera may not support this control.")

    try:
        print("Enable Auto Exposure...")
        camera.software_auto_exposure(enable = False)
        print("Set exposure...")
        camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
        print(camera.get_control(v4l2.V4L2_CID_EXPOSURE))
        print("Set White Balance...")
        camera.manual_set_awb_compensation(rGain, bGain)
    except Exception as e:
        print(e)

if __name__ == "__main__":
    try:
        camera = arducam.mipi_camera()
        print("Open camera...")
        camera.init_camera()
        print("Setting the resolution...")
        fmt = camera.set_resolution(1920, 1080)
        print("Current resolution is {}".format(fmt))
        set_controls(camera)
        print(camera.get_format())
        fps = 0
        prev_frame_time = 0
        while True:
            frame = camera.capture(encoding = 'i420')
            height = int(align_up(fmt[1], 16))
            width = int(align_up(fmt[0], 32))
            image = frame.as_array.reshape(int(height * 1.5), width)
            y_plane = image[:height, :]
            brightness_factor = 2.5  # Adjust this factor to change brightness
            brightened_image = np.clip(y_plane * brightness_factor, 0, 255).astype(np.uint8)
            new_frame_time = time.time()
            fps = 0.99 * fps + 0.01 / (new_frame_time - prev_frame_time)
            prev_frame_time = new_frame_time

            cv2.imshow("Arducam", brightened_image)

            key = cv2.waitKey(10)
            if key == 27:  # ESC key to exit
                break
            elif key == ord('w'):
                exposure += 10
                camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
                print(f"Exposure increased to {camera.get_control(v4l2.V4L2_CID_EXPOSURE)}")
            elif key == ord('s'):
                exposure -= 10
                camera.set_control(v4l2.V4L2_CID_EXPOSURE, exposure)
                print(f"Exposure decreased to {camera.get_control(v4l2.V4L2_CID_EXPOSURE)}")
            elif key == ord('e'):
                rGain += 1
                camera.manual_set_awb_compensation(rGain, bGain)
                print(f"Red Gain increased to {rGain}")
            elif key == ord('d'):
                rGain -= 1
                camera.manual_set_awb_compensation(rGain, bGain)
                print(f"Red Gain decreased to {rGain}")
            elif key == ord('r'):
                bGain += 1
                camera.manual_set_awb_compensation(rGain, bGain)
                print(f"Blue Gain increased to {bGain}")
            elif key == ord('f'):
                bGain -= 1
                camera.manual_set_awb_compensation(rGain, bGain)
                print(f"Blue Gain decreased to {bGain}")
        
        # Release memory
        del frame
        print("Close camera...")
        camera.close_camera()
    except Exception as e:
        print(e)
