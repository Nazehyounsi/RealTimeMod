
import struct
import time

import socket

def start_server(port, address="localhost"):
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_socket.bind((address, port))
    server_socket.listen(1)
    print(f"Server listening on {address}:{port}")

    while True:
        conn, addr = server_socket.accept()
        print(f"Connected by {addr}")

        while True:
            data = conn.recv(1024)
            if not data:
                break
            print("Received data structure:")
            print(data.decode('utf-8'))

        conn.close()
        print(f"Connection closed by {addr}")

if __name__ == "__main__":
    start_server(port=50151)

#
#
# class Client:
#     def __init__(self, port, address):
#         self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
#         self.port = port
#         self.address = address
#         self.frames = []  # To accumulate frames
#         self.batch_size = 16  # Define the batch size
#
#     def connect_to_server(self):
#         while True:
#             try:
#                 self.client_socket.connect((self.address, self.port))
#                 print("Connected to server at", self.address, "on port", self.port)
#                 break
#             except ConnectionRefusedError:
#                 print("Server not active yet. Retrying in 2 seconds...")
#                 time.sleep(2)
#
#     def receive_aus(self):
#         # Receive the data
#         au1 = self.client_socket.recv(8)
#         au2 = self.client_socket.recv(8)
#         au12 = self.client_socket.recv(8)
#         au6 = self.client_socket.recv(8)
#         au25 = self.client_socket.recv(8)
#         au9 = self.client_socket.recv(8)
#         au10 = self.client_socket.recv(8)
#         au14 = self.client_socket.recv(8)
#         au15 = self.client_socket.recv(8)
#
#         au1 = struct.unpack('!d', au1)[0]
#         au2 = struct.unpack('!d', au2)[0]
#         au12 = struct.unpack('!d', au12)[0]
#         au6 = struct.unpack('!d', au6)[0]
#         au25 = struct.unpack('!d', au25)[0]
#         au9 = struct.unpack('!d', au9)[0]
#         au10 = struct.unpack('!d', au10)[0]
#         au14 = struct.unpack('!d', au14)[0]
#         au15 = struct.unpack('!d', au15)[0]
#
#         # Append the received frame to the frames list
#         frame = [au1, au2, au12, au6, au25, au9, au10, au14, au15]
#         self.frames.append(frame)
#
#         if len(self.frames) == self.batch_size:
#             self.process_and_send_batched_frames()
#             self.frames = []  # Clear the frames list after sending
#
#     def process_and_send_batched_frames(self):
#         processed_frames = [self.process_frame(frame) for frame in self.frames]
#
#         # Flatten the processed frames
#         flat_frames = [item for frame in processed_frames for item in frame]
#         print(flat_frames)
#         return flat_frames
#
#     def process_frame(self, frame):
#         # Check for the highest value in the frame
#         max_value = max(frame)
#         if max_value <= 0.5:
#             return [0]
#
#         highest_au_index = frame.index(max_value)
#         if highest_au_index in [2, 3, 4]:  # au12, au6, au25
#             return [1]
#         elif highest_au_index in [5, 6]:  # au9, au10
#             return [2]
#         elif highest_au_index in [7, 8]:  # au14, au15
#             return [3]
#         return [0]  # Default case, should not be reached
#
#
# def main():
#     client = Client(50150, "localhost")
#     client.connect_to_server()
#
#     while True:
#         try:
#             input_vector = client.receive_aus()
#             if not input_vector:
#                 continue
#
#             print(input_vector)
#         except Exception as e:
#             print("Error:", e)
#             break
#
#
# if __name__ == "__main__":
#     main()
#
#
