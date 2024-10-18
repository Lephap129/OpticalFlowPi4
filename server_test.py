import socket 
import time
# Định nghĩa host và port mà server sẽ chạy và lắng nghe
host = "192.168.1.47"
port = 4000
s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
s.bind((host, port))

s.listen(1) # 1 ở đây có nghĩa chỉ chấp nhận 1 kết nối
print("Server listening on port", port)
        
c, addr = s.accept()
print("Connect from ", str(addr))

count = 0
while(1):
    count+= 1
    but = b"Hello, how are you"
    c.send(but)
    time.sleep(1)
    if (count == 5):
        c.send(b"Bye")
        c.close()
        break