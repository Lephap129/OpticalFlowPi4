import socket
import cv2

def receive_image_from_client(ip, port, output_file):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    
    try:
        # Bind the socket to the IP address and port
        server_socket.bind((ip, port))
        server_socket.listen(1)  # Listen for a single connection
        print(f"Listening on {ip}:{port}")
        
        # Accept the connection from the client
        client_socket, client_address = server_socket.accept()
        print(f"Connection from {client_address}")
        while True:
            # Open the output file to write the received image
            with open(output_file, 'wb') as image_file:
                while True:
                    # Receive data in chunks
                    data = client_socket.recv(1024)  # Buffer size of 1024 bytes
                    if not data:
                        break
                    if data == "Finish!".encode():
                        display_image(output_file)
                        break  # Exit the loop if no data is received
                    image_file.write(data)  # Write the received chunk to the file
                    print(f"Received a chunk of {len(data)} bytes")
            print(f"Image received and saved as {output_file}")

    except socket.error as e:
        print(f"Socket error: {e}")

def display_image(image_path):
    # Load the image using OpenCV
    image = cv2.imread(image_path)

    if image is None:
        print(f"Failed to load the image from {image_path}")
        return

    # Display the image in a window
    cv2.imshow('Received Image', image)

    # Wait for a key press and close the window
    cv2.waitKey(0)
    cv2.destroyAllWindows()

# Example usage
server_ip = '192.168.1.47'  # Listen on all available network interfaces
server_port = 5000      # Port number to listen on
output_image_file = 'received_image.jpg'  # File to save the received image

receive_image_from_client(server_ip, server_port, output_image_file)
