import time
from picamera2 import Picamera2

# Khởi tạo Picamera2
picam2 = Picamera2()

# Cấu hình chế độ xem trước với kích thước và các thông số tương ứng
config = picam2.create_preview_configuration(main={"size": (576, 324), "format": "XBGR8888"})
picam2.configure(config)

# Hàm tính FPS
def calculate_fps():
    start_time = time.time()
    frame_count = 0

    # Bắt đầu camera
    picam2.start()

    try:
        while True:
            # Đọc khung hình
            view_frame = picam2.capture_array()
            frame_count += 1
            # Tính toán FPS mỗi giây
            elapsed_time = time.time() - start_time
            if elapsed_time >  1.0:
                fps = frame_count / elapsed_time
                print(f"FPS: {fps:.2f}")
                # Đặt lại thời gian bắt đầu và số lượng khung hình
                start_time = time.time()
                frame_count = 0

    except KeyboardInterrupt:
        # Ngắt khi nhấn Ctrl+C
        pass
    finally:
        # Dừng camera
        picam2.stop()

# Gọi hàm tính FPS
calculate_fps()
