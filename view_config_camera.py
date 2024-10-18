from picamera2 import Picamera2

picam2 = Picamera2()

config = picam2.create_preview_configuration(main={"size": (640, 480), "format": "XBGR8888"})

print("-------------------------------")
print(config)
print("-------------------------------")

picam2.configure(config)

print("-------------------------------")
print(picam2.camera_controls)
print("-------------------------------")
