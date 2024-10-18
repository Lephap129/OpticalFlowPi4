import time
from picamera2 import Picamera2
import cv2

# Khởi tạo Picamera2
picam2 = Picamera2()
config = picam2.create_preview_configuration(main={"size": (576, 324), "format": "XBGR8888"})
picam2.configure(config)

crop_size_w = 288
crop_size_h = 162

# Preprocessing function
def preprocess_frame(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    
    # Crop a region from the center of the frame
    center_x, center_y = gray.shape[0] // 2, gray.shape[1] // 2
    #print("cx,cy:",center_y - crop_size_h // 2,center_y)
    cropped_gray = gray[center_x - crop_size_h // 2:center_x + crop_size_h // 2,
                        center_y - crop_size_h // 2:center_y + crop_size_h // 2]
    #cv2.rectangle(frame, (center_x - crop_size_w//2, center_y - crop_size_w//2),(center_x + crop_size_w//2, center_y + crop_size_w//2), (0, 255, 0), 2)
    return cropped_gray


# Hàm tính FPS
def calculate_fps():
    start_time = time.time()
    frame_count = 0

    # Bắt đầu camera
    picam2.start()

    
    while True:
        # Đọc khung hình
        frame = picam2.capture_array()
        frame = preprocess_frame(frame)
        frame_count += 1
        cv2.imshow("PiCamCrop", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    picam2.stop()

# Gọi hàm tính FPS
calculate_fps()
