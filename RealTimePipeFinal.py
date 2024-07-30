
import numpy as np
import struct
import socket
import torch
import time
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import random
from sklearn.neighbors import KernelDensity
import threading
import os


n_hidden = 512
n_T = 1000
num_event_types =13
event_embedding_dim =64
embed_output_dim =128
guide_w =0
num_facial_types = 7
facial_embed_dim = 32
cnn_output_dim = 512  # Output dimension after passing through CNN layers
lstm_hidden_dim = 256
sequence_length = 137
x_dim = 137
y_dim = 137


def extend_non_zero_sequences(input_file, output_file, column_names, extend_by=2):
    # Load the CSV file
    df = pd.read_csv(input_file)

    for column_name in column_names:
        # Ensure the column for operation exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in the CSV file.")

        # Apply gap filling
        df[column_name] = fill_gaps(df[column_name], 10)
        # Working with the specified column
        series = df[column_name]


        # Create a copy to manipulate and later replace in the df
        modified_series = series.copy()

        # Identify indices where series is non-zero
        non_zero_indices = series[series != 0].index

        # Group consecutive indices
        groups = []
        group = []
        for idx in non_zero_indices:
            if not group or idx == group[-1] + 1:
                group.append(idx)
            else:
                groups.append(group)
                group = [idx]
        if group:
            groups.append(group)

        # Extend the non-zero sequences and interpolate
        for group in groups:
            start_idx, end_idx = group[0], group[-1]
            extend_start = max(0, start_idx - extend_by)
            extend_end = min(len(series), end_idx + extend_by + 1)

            # Interpolate upwards to the first non-zero
            if extend_start < start_idx:
                start_value = 0
                end_value = series.iloc[start_idx]
                modified_series[extend_start:start_idx] = np.linspace(start_value, end_value, start_idx - extend_start)

            # Interpolate downwards from the last non-zero
            if end_idx + 1 < extend_end:
                start_value = series.iloc[end_idx]
                end_value = 0
                modified_series[end_idx + 1:extend_end] = np.linspace(start_value, end_value,
                                                                      extend_end - (end_idx + 1))

        # Replace the modified column in the DataFrame
        df[column_name] = modified_series

    # Save the modified DataFrame back to a CSV file
    df.to_csv(output_file, index=False)

def fill_gaps(series, gap_length):
    # Iterate through the series with a buffer of gap_length on each side
    for i in range(gap_length, len(series) - gap_length):
        # Check for zeros and non-zero surroundings within gap_length
        if series[i] == 0:
            for gap in range(1, gap_length + 1):
                if series[i - gap] > 0 and series[i + gap] > 0:
                    series[i] = (series[i - gap] + series[i + gap]) / 2
                    break  # Exit the loop once a gap has been filled

def append_sequence_to_csv(sequence, output_csv_path, frame_rate=25):
    duration_per_frame = 1.0 / frame_rate

    if os.path.exists(output_csv_path):
        existing_df = pd.read_csv(output_csv_path)
        last_timestamp = existing_df['timestamp'].iloc[-1]
    else:
        last_timestamp = 0.0
        existing_df = pd.DataFrame(columns=['timestamp'] + [str(i) for i in range(4)])
        existing_df.to_csv(output_csv_path, index=False)

    new_data = []
    for category in sequence:
        new_row = [last_timestamp + duration_per_frame] + [1 if i == category else 0 for i in range(4)]
        new_data.append(new_row)
        last_timestamp += duration_per_frame

    new_df = pd.DataFrame(new_data, columns=['timestamp'] + [str(i) for i in range(4)])
    new_df.to_csv(output_csv_path, mode='a', header=False, index=False)

    return new_df

def restructure_to_baseline(transformed_csv_path, baseline_csv_path, output_csv_path):
    transformed_df = pd.read_csv(transformed_csv_path)
    baseline_df = pd.read_csv(baseline_csv_path)

    transformed_df.columns = transformed_df.columns.str.strip().astype(str)
    baseline_df.columns = baseline_df.columns.str.strip().astype(str)

    structured_df = pd.DataFrame(0, index=transformed_df.index, columns=baseline_df.columns)
    structured_df['timestamp'] = transformed_df['timestamp']

    for col in transformed_df.columns:
        if col in structured_df.columns:
            structured_df[col] = transformed_df[col]

    structured_df.to_csv(output_csv_path, index=False)

    return structured_df

def coactivate_aus(input_csv_path, output_csv_path):
    df = pd.read_csv(input_csv_path)

    df['AU06_r'] = df['AU12_r'] * 0.5
    df['AU25_r'] = df['AU12_r'] * 0.25
    df['AU10_r'] = df['AU09_r'] * 5
    df['AU09_r'] = df['AU10_r'] * 0.5
    df['AU14_r'] = df['AU15_r'] * 0.5

    df.to_csv(output_csv_path, index=False)

def process_and_save_to_csv(reprojected_sequence, intermed_csv_path, transformed_csv_path, ground_csv_path, final_csv_path, adjusted_csv_path, extended_csv_path, frame_rate=25):
    # Step 1: Append the reprojected sequence to the intermediate CSV
    append_sequence_to_csv(reprojected_sequence, intermed_csv_path, frame_rate)

    # Step 2: Rename headers and drop unnecessary columns
    df = pd.read_csv(intermed_csv_path)
    rename_dict = {'1': 'AU12_r', '2': 'AU09_r', '3': 'AU15_r'}
    df.rename(columns=rename_dict, inplace=True)
    df.drop(columns='0', inplace=True)
    df.to_csv(transformed_csv_path, index=False)

    # Step 3: Restructure the CSV to match the baseline structure
    restructure_to_baseline(transformed_csv_path, ground_csv_path, final_csv_path)

    # Step 4: Coactivate AUs and save the final output
    coactivate_aus(final_csv_path, adjusted_csv_path)
    # column_names = ['AU06_r', 'AU25_r', 'AU12_r', 'AU10_r', 'AU09_r', 'AU14_r', 'AU15_r']
    # extend_non_zero_sequences(adjusted_csv_path, extended_csv_path, column_names, extend_by=25)

import socket
import threading

class Sender:
    def __init__(self, forward_port, forward_address):
        self.forward_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.forward_port = forward_port
        self.forward_address = forward_address
        self.lock = threading.Lock()
        self.queue = []
        self.connect_to_forwarding_server()
        self.sending_thread = threading.Thread(target=self.send_data_continuously)
        self.sending_thread.daemon = True
        self.sending_thread.start()  # Start the thread

    def connect_to_forwarding_server(self):
        while True:
            try:
                self.forward_socket.connect((self.forward_address, self.forward_port))
                print("Connected to forwarding server at", self.forward_address, "on port", self.forward_port)
                break
            except ConnectionRefusedError:
                print("Forwarding server not active yet. Retrying in 2 seconds...")
                time.sleep(2)

    def queue_data(self, data):
        with self.lock:
            self.queue.extend(data)

    def send_data_continuously(self):
        while True:
            with self.lock:
                if self.queue:
                    row = self.queue.pop(0)
                    self.send_to_forwarding_server(row)
                    print(f"Sent data: {row}")
            time.sleep(0.04)  # Sleep briefly to avoid busy-waiting #ROBINET A AJUSTER

    def send_to_forwarding_server(self, data):
        try:
            self.forward_socket.sendall((data + "\n").encode('utf-8'))
        except Exception as e:
            print("Failed to send data to forwarding server:", e)

class Client:
    def __init__(self, port, address, batch_size=17):
        self.client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.port = port
        self.address = address
        self.frames = []  # To accumulate frames
        self.batch_size = batch_size  # Define the batch size
        self.latest_processed_batch = None
        self.lock = threading.Lock()

    def connect_to_server(self):
        # Connect to the server
        self.client_socket.connect((self.address, self.port))
        print("Connected to server at", self.address, "on port", self.port)

    def connect_to_server(self):
        while True:
            try:
                self.client_socket.connect((self.address, self.port))
                print("Connected to server at", self.address, "on port", self.port)
                break
            except ConnectionRefusedError:
                print("Server not active yet. Retrying in 2 seconds...")
                time.sleep(2)

    def start_receiving(self):
        thread = threading.Thread(target=self.receive_aus_continuously)
        thread.daemon = True  # Daemonize thread
        thread.start()

    def receive_aus_continuously(self):
        while True:
            self.receive_aus()

    def receive_aus(self):
        # Receive the data
        au1 = self.client_socket.recv(8)
        au2 = self.client_socket.recv(8)
        au12 = self.client_socket.recv(8)
        au6 = self.client_socket.recv(8)
        au25 = self.client_socket.recv(8)
        au9 = self.client_socket.recv(8)
        au10 = self.client_socket.recv(8)
        au14 = self.client_socket.recv(8)
        au15 = self.client_socket.recv(8)

        au1 = struct.unpack('!d', au1)[0]
        au2 = struct.unpack('!d', au2)[0]
        au12 = struct.unpack('!d', au12)[0]
        au6 = struct.unpack('!d', au6)[0]
        au25 = struct.unpack('!d', au25)[0]
        au9 = struct.unpack('!d', au9)[0]
        au10 = struct.unpack('!d', au10)[0]
        au14 = struct.unpack('!d', au14)[0]
        au15 = struct.unpack('!d', au15)[0]

        # Append the received frame to the frames list
        frame = [au1, au2, au12, au6, au25, au9, au10, au14, au15]
        self.frames.append(frame)

        if len(self.frames) == self.batch_size:
            self.process_and_send_batched_frames()
            self.frames = []  # Clear the frames list after sending

    def process_and_send_batched_frames(self):
        processed_frames = [self.process_frame(frame) for frame in self.frames]
        # Flatten the processed frames
        flat_frames = [item for frame in processed_frames for item in frame]
        with self.lock:
            self.latest_processed_batch = flat_frames
        print(flat_frames)
        return flat_frames

    def process_frame(self, frame):
        # Check for the highest value in the frame
        max_value = max(frame)
        if max_value <= 0.5:
            return [0]
        highest_au_index = frame.index(max_value)
        if highest_au_index in [2, 3, 4]:  # au12, au6, au25
            return [4]
        elif highest_au_index in [5, 6]:  # au9, au10
            return [5]
        elif highest_au_index in [7, 8]:  # au14, au15
            return [6]
        return [0]  # Default case, should not be reached

class RealTimeProcessor:
    def __init__(self, buffer_size=68, target_size=137):
        self.buffer_size = buffer_size
        self.target_size = target_size
        #self.buffer = buffer_sequence

        self.buffer = [0] * buffer_size

    def update_buffer(self, new_data):
        if len(new_data) != 68:
            raise ValueError("New data must be exactly 16 frames long.")
        #self.buffer = self.buffer[len(new_data):] + new_data
        self.buffer = new_data

    def project_to_target(self):
        buffer_length = len(self.buffer)
        projected = [0] * self.target_size

        # Calculate the projection ratio
        ratio = self.target_size / buffer_length

        # Iterate through the buffer and fill the projected array
        pos = 0
        for i in range(buffer_length):
            value = self.buffer[i]
            count = round((i + 1) * ratio) - round(i * ratio)
            for _ in range(count):
                if pos < self.target_size:
                    projected[pos] = value
                    pos += 1
                else:
                    break
        print("projected client sequence ")
        print(projected)
        return projected

    def reproject_to_buffer(self, projected, buffer_size):
        target_length = len(projected)
        reprojected = [0] * buffer_size

        # Calculate the reprojection ratio
        ratio = round(target_length / buffer_size)

        # Initialize the position in the projected array
        pos = 0.0

        for i in range(buffer_size):
            value_count = 0
            value_sum = 0
            next_pos = pos + ratio

            # Sum the values in the range corresponding to the current buffer position
            while pos < next_pos and int(pos) < target_length:
                value_sum += projected[int(pos)]
                value_count += 1
                pos += 1

            # Average the values and assign to the reprojected buffer
            if value_count > 0:
                reprojected[i] = round(value_sum / value_count)
            else:
                reprojected[i] = 0

        return reprojected

    def get_buffer(self):
        return self.buffer

    def get_projected_buffer(self):
        return self.project_to_target()


def preprocess_real_time(input_frames, processor):
    processor.update_buffer(input_frames)
    return processor.get_projected_buffer()


def generate_random_series_sequence(sequence_length=137, max_series=4):
    sequence = [0] * sequence_length
    possible_values = [1, 2, 3]
    max_series_length = sequence_length // 2
    series_count = 0

    while series_count < max_series:
        value = random.choice(possible_values)
        series_length = random.randint(5, max_series_length)
        start_position = random.randint(0, sequence_length - series_length)

        for i in range(start_position, start_position + series_length):
            sequence[i] = value

        series_count += 1
    return sequence



def generate_random_tensors_numpy(batch_size=1):
    z_tensor = np.zeros((batch_size, 4))
    z_tensor[:, 0] = 12
    z_tensor[:, 1] = 2
    z_tensor[:, 2] = 12
    z_tensor[:, 3] = 0

    chunk_descriptor_tensor = np.zeros((batch_size, 3))
    chunk_descriptor_tensor[:, 0] = np.random.random(size=batch_size)
    chunk_descriptor_tensor[:, 1:] = np.random.choice([-1, 0, 1], size=(batch_size, 2))

    z_tensor = torch.tensor(z_tensor, dtype=torch.float32)
    chunk_descriptor_tensor = torch.tensor(chunk_descriptor_tensor, dtype=torch.float32)

    return z_tensor, chunk_descriptor_tensor


def real_time_inference_loop(model, device, client, sender, processor, guide_weight=0.0):
    model.eval()
    batch_size = 1
    z_tensor, chunk_descriptor_tensor = generate_random_tensors_numpy(batch_size)
    chunk_descriptor_tensor[:, 0] = 0.151

    intermed_csv_path = 'intermed.csv'
    transformed_csv_path = 'csv_file.csv'
    final_csv_path = 'Finalcsv.csv'
    adjusted_csv_path = 'Ajusted_Final_csv_file.csv'
    ground_csv_path = 'GroundTruth.csv'
    extended_csv_path = 'FinalInterpolated.csv'

    while True:

        try:
            with client.lock:
                input_vector = client.latest_processed_batch

            if not input_vector:
                time.sleep(0.01)
                continue

            start_time = time.time()
            chunk_descriptor_tensor[:, 0] += 0.001
            chunk_descriptor_tensor[:, 1:] = 0


            updated_buffer = preprocess_real_time(input_vector, processor)
            input_tensor = torch.tensor(updated_buffer, dtype=torch.float32).unsqueeze(0).to(device)
            z_tensor = z_tensor.to(device)
            chunk_descriptor_tensor = chunk_descriptor_tensor.to(device)

            # with torch.no_grad():
            #     model.guide_w = guide_weight
            #     start_time2 = time.time()
            #     y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()
            #     end_time2 = time.time()
            #     inference_time = end_time2 - start_time2
            #     print(f"Inference time for the current batch: {inference_time: .4f} seconds")


            # best_prediction = np.round(y_pred)
            # best_prediction[best_prediction == 4] = 3
            # best_prediction[best_prediction >= 5] = 0
            # best_prediction[best_prediction < 0] = 0


            #reprojected_output = processor.reproject_to_buffer(best_prediction[0], 68)

            reprojected_output = generate_random_series_sequence(68, 4)

            process_and_save_to_csv(reprojected_output, intermed_csv_path, transformed_csv_path, ground_csv_path, final_csv_path, adjusted_csv_path, extended_csv_path)

            # Read the last 16 rows from the adjusted CSV file and queue them for the sender
            df = pd.read_csv(adjusted_csv_path)
            last_16_rows = df.tail(68).to_csv(index=False, header=False)
            sender.queue_data(last_16_rows.splitlines())

            print("Client sequence:")
            print(input_tensor)
            print("Reprojected Output:")
            print(reprojected_output)

            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.3 - elapsed_time))
        except Exception as e:
            print("Error:", e)
            break


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, cnn_output_dim, lstm_hidden_dim,
                                               sequence_length)
    mi_embedder = SpeakingTurnDescriptorEmbedder(num_event_types, event_embedding_dim, embed_output_dim)
    chunk_embedder = ChunkDescriptorEmbedder(continious_embedding_dim=16, valence_embedding_dim=8, output_dim=64)

    model_path = 'saved_model_NewmodelChunkd1000.pth'
    nn_model = Model_mlp_diff(observation_embedder, mi_embedder, chunk_embedder, sequence_length,
                              net_type="transformer")
    model = Model_Cond_Diffusion(nn_model, observation_embedder, mi_embedder, chunk_embedder, betas=(1e-4, 0.02),
                                 n_T=n_T, device=device, x_dim=x_dim, y_dim=y_dim, drop_prob=0, guide_w=guide_w)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    processor = RealTimeProcessor(buffer_size=68, target_size=137)

    client = Client(50150, "localhost", 68)
    client.connect_to_server()
    client.start_receiving()
    sender = Sender(50151, "localhost")

    real_time_inference_loop(model, device, client, sender,  processor, guide_weight=guide_w)


if __name__ == "__main__":
    main()