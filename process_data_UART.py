import serial
import time

port = '/dev/ttyUSB0'  
baudrate = 115200  

try:
    ser = serial.Serial(port, baudrate, timeout=1)
except serial.SerialException as e:
    print(f"Error opening serial port: {e}")
    exit(1)

try:
    samples = []
    sample_num = 50
    while True:
        if ser.in_waiting > 0:
            read_data = ser.readline().decode('utf-8').rstrip()
            try:
                samples.append(float(read_data))
                if len(samples) == sample_num:
                    data = round( (sum(samples) / len(samples) / 10 ), 1)
                    samples.pop(0)
                else:
                    data = round( float(read_data), 1) / 10
            except:
                data = read_data
            print(f"Received: {data}")
        time.sleep(0.01)
except KeyboardInterrupt:
    print("Program stopped by user")
finally:
    ser.close()
