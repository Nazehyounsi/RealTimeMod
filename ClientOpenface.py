
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

def interpolate_activations_betweenchunks(df, columns, N, K):
    """
    Interpolates activation values between two successive chunks to handle merging of activations
    based on the K parameter, including cases where the previous chunk ends with an activation
    and the current chunk starts with a new activation within less than K frames.

    :param df: DataFrame containing the last two chunks (assumed to be 128 rows).
    :param columns: List of columns to perform interpolation on.
    :param N: Number of frames for interpolation before and after activations.
    :param K: Maximum gap between activations to consider merging.
    :return: DataFrame with merged activations across chunks.
    """
    # Number of frames in each chunk
    chunk_size = 64

    # Ensure the DataFrame has exactly two chunks
    if df.shape[0] != 2 * chunk_size:
        raise ValueError("DataFrame must contain exactly two chunks of size 64 frames (total 128 frames).")

    # Indices for previous and current chunks
    prev_chunk_indices = range(0, chunk_size)
    curr_chunk_indices = range(chunk_size, 2 * chunk_size)

    for col in columns:
        values = df[col].values  # Extract the column's values as a numpy array

        # Previous chunk values and activation indices
        prev_values = values[prev_chunk_indices]
        prev_non_zero_indices = np.where(prev_values > 0)[0]
        last_prev_activation_idx = prev_non_zero_indices[-1] if len(prev_non_zero_indices) > 0 else None

        # Current chunk values and activation indices
        curr_values = values[curr_chunk_indices]
        curr_non_zero_indices = np.where(curr_values > 0)[0]
        first_curr_activation_idx = curr_non_zero_indices[0] + chunk_size if len(curr_non_zero_indices) > 0 else None

        # Determine if previous activation is ongoing (does not end)
        prev_activation_ongoing = False
        if last_prev_activation_idx == chunk_size - 1:
            # Check if there is no falling edge in the previous chunk after the last rising edge
            prev_rising_indices, prev_falling_indices = identify_activation_indices(prev_values)
            if not prev_falling_indices or prev_falling_indices[-1] < prev_rising_indices[-1]:
                prev_activation_ongoing = True

        # Calculate the gap between activations
        if prev_activation_ongoing and first_curr_activation_idx is not None:
            gap = first_curr_activation_idx - (chunk_size - 1) - 1  # Gap after the last frame of prev chunk
            if gap < K:
                # Merge the activations

                # Start of merged activation is the first rising index in the previous chunk
                prev_rising_indices = [idx for idx in prev_rising_indices]
                if prev_rising_indices:
                    merged_start = prev_rising_indices[0]
                else:
                    merged_start = 0  # If no rising edge, start from the beginning

                # End of merged activation is the last falling index in the current chunk
                curr_rising_indices, curr_falling_indices = identify_activation_indices(curr_values)
                curr_falling_indices = [idx + chunk_size for idx in curr_falling_indices]

                if curr_falling_indices:
                    merged_end = curr_falling_indices[-1]
                else:
                    merged_end = len(values) - 1  # If no falling edge, go to the end

                # Determine activation value (since activation values are constant)
                activation_value = max(values[merged_start:merged_end + 1])

                # Interpolate before the activation starts
                start_idx = max(0, merged_start - N)
                for idx in range(start_idx, merged_start):
                    step = activation_value / (merged_start - start_idx)
                    values[idx] = round((idx - start_idx + 1) * step, 3)

                # Set activation values within the merged segment
                values[merged_start:merged_end + 1] = activation_value

                # Interpolate after the activation ends
                end_idx = min(len(values), merged_end + N + 1)
                for idx in range(merged_end + 1, end_idx):
                    step = activation_value / (end_idx - merged_end - 1)
                    values[idx] = round(activation_value - (idx - merged_end) * step, 3)

            else:
                # Do not merge; activations remain separate
                pass

        else:
            # Original logic for when the previous activation has ended
            # Recalculate the gap between last_prev_activation_idx and first_curr_activation_idx
            if last_prev_activation_idx is not None and first_curr_activation_idx is not None:
                gap = first_curr_activation_idx - last_prev_activation_idx - 1
                if gap < K:
                    # Merge activations as before (code from previous implementation)
                    # [Include the merging logic here]
                    # ...
                    pass  # For brevity, not repeating the code here
                else:
                    # Do not merge; activations remain separate
                    pass
            else:
                # No activations to merge
                pass

        # Update the column in the DataFrame
        df[col] = values

    return df


# Read the last 128 rows from the adjusted CSV file
df = pd.read_csv(extended_csv_path)
columns_to_interpolate = ['AU06_r', 'AU25_r', 'AU12_r', 'AU10_r', 'AU09_r', 'AU14_r', 'AU15_r']

if df.shape[0] >= 128:
    # Interpolate between chunks using K
    df_last_two_chunks = df.tail(128)
    df_interpolated = interpolate_activations_betweenchunks(df_last_two_chunks, columns_to_interpolate, N, K)

    # Replace the last 128 rows in the original DataFrame with the interpolated DataFrame
    df.iloc[-128:] = df_interpolated.values

# Extract the last 64 rows to send
last_buffer_rows = df.tail(64).to_csv(index=False, header=False)
sender.queue_data(last_buffer_rows.splitlines())
#
#
