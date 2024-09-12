
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



class Miror:
    def __init__(self, consecutive_zero_threshold=4):
        self.stored_sequences = []
        self.consecutive_zero_threshold = consecutive_zero_threshold
        self.mirroring = False
        self.conversion_dict = {4: 1, 5: 2, 6: 3, 0: 0}  # Conversion mapping

    def store_sequence(self, sequence):
        self.stored_sequences.append(sequence)

        # Keep only the last 4 sequences
        if len(self.stored_sequences) > self.consecutive_zero_threshold:
            self.stored_sequences.pop(0)

        # Check if the last 4 sequences are fully zero
        if all(np.all(seq == 0) for seq in self.stored_sequences):
            self.mirroring = True  # Activate mirroring mode
        else:
            self.mirroring = False  # Deactivate mirroring mode

    def mirror_sequence(self, input_tensor):
        input_values = input_tensor.cpu().numpy().squeeze()  # Convert tensor to numpy array
        mirrored_output = np.vectorize(self.conversion_dict.get)(input_values)  # Apply conversion
        return mirrored_output.tolist()

    def should_mirror(self):
        return self.mirroring


def identify_activation_indices(values, N):
    """
    Identify the indices of rising and falling edges in the sequence of values.
    This function returns two lists:
    1. rising_indices: Indices where the value changes from 0 to a non-zero value.
    2. falling_indices: Indices where the value changes from a non-zero value to 0.

    :param values: The array of values (AU activation values).
    :param N: The number of frames for interpolation (for reference).
    :return: A list of rising_indices and falling_indices.
    """
    rising_indices = []
    falling_indices = []

    length = len(values)

    for i in range(1, length):
        # Rising edge: from 0 to non-zero
        if values[i] > 0 and values[i - 1] == 0:
            rising_indices.append(i)

        # Falling edge: from non-zero to 0
        if values[i] == 0 and values[i - 1] > 0:
            falling_indices.append(i)

    return rising_indices, falling_indices


def interpolate_activations(df, columns, N):
    """
    Interpolates N frames before and after an activation for the specified columns.
    Interpolates only over the registered indices to avoid overlapping interpolations.

    :param df: DataFrame with activation data.
    :param columns: List of columns to perform interpolation on.
    :param N: Number of frames for interpolation.
    :return: DataFrame with interpolated values.
    """
    for col in columns:
        values = df[col].values  # Extract the column's values as a numpy array

        # Step 1: Identify rising and falling edges
        rising_indices, falling_indices = identify_activation_indices(values, N)

        # Step 2: Perform interpolation only on registered indices
        for idx in rising_indices:
            start_idx = max(0, idx - N)
            for j in range(start_idx, idx):
                step = (values[idx] - values[start_idx]) / (idx - start_idx)
                values[j] = round(values[start_idx] + (j - start_idx) * step, 3)

        for idx in falling_indices:
            end_idx = min(len(values), idx + N)
            for j in range(idx, end_idx):
                step = values[idx - 1] / (end_idx - idx)
                values[j] = round(values[idx - 1] - (j - idx) * step, 3)

        df[col] = values  # Update the DataFrame column with the interpolated values

    return df


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

    df['AU06_r'] = df['AU12_r'] * 0.8
    df['AU25_r'] = df['AU12_r'] * 0.5
    df['AU10_r'] = df['AU09_r'] * 0.8
    df['AU14_r'] = df['AU15_r'] * 0.8

    df['AU12_r'] = df['AU06_r'] * 2
    df['AU15_r'] = df['AU14_r'] * 2
    df['AU09_r'] = df['AU10_r'] * 2

    df.to_csv(output_csv_path, index=False)

def process_and_save_to_csv(reprojected_sequence, intermed_csv_path, transformed_csv_path, ground_csv_path, final_csv_path, adjusted_csv_path, extended_csv_path, N,frame_rate=25):
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

    # Step 5: Perform interpolation on the newly added rows in adjusted_csv_path
    df_adjusted = pd.read_csv(adjusted_csv_path)

    # Specify the columns to interpolate (AU columns)
    columns_to_interpolate = ['AU06_r', 'AU25_r', 'AU12_r', 'AU10_r', 'AU09_r', 'AU14_r', 'AU15_r']  # Add more columns if needed

    # Perform interpolation
    interpolated_df = interpolate_activations(df_adjusted, columns_to_interpolate, N)

    # Save the interpolated results
    interpolated_df.to_csv(extended_csv_path, index=False)


def generate_chunk_descriptor_tensor(storer, previous_input_tensor):
    """
    Generates the chunk_descriptor_tensor based on the previous reprojected output
    and the previous input tensor stored in the Storer.

    :param storer: A Storer object that stores the previous reprojected output.
    :param previous_input_tensor: The previous input tensor.
    :return: chunk_descriptor_tensor (PyTorch tensor)
    """
    value1, value2 = 0

    # Extract the previous reprojected output and previous input tensor
    previous_reprojected_output = storer.stored_sequences[
        -1] if storer.stored_sequences else None  # Last stored sequence

    # Check for previous reprojected output
    if previous_reprojected_output is not None:
        count_1s = previous_reprojected_output.count(1)
        count_2s_and_3s = previous_reprojected_output.count(2) + previous_reprojected_output.count(3)

        if all(value == 0 for value in previous_reprojected_output):
            value1 = 0
        elif count_2s_and_3s > count_1s:
            value1 = -1
        else:
            value1 = 1

    # Check for previous input tensor
    input_values = previous_input_tensor.cpu().numpy().flatten().tolist()  # Convert to list for easy counting
    count_1s = input_values.count(1)
    count_2s_and_3s = input_values.count(2) + input_values.count(3)

    if all(value == 0 for value in input_values):
        value2 = 0
    elif count_2s_and_3s > count_1s:
        value2 = -1
    else:
        value2 = 1

    return value1, value2

def generate_z_tensor_from_string(string_value, conversion_table):
    # Retrieve the corresponding values from the conversion table
    value_one, value_two = conversion_table.get(string_value, (0.0, 0.0))  # Default to (0.0, 0.0) if string not found

    # Third value is either 0 or the same as value_one (based on your explanation)
    value_three = value_one  # You can set this to value_one if needed, e.g., value_three = value_one
    #value_four = 0.0 si nécessaire

    # Create the z_tensor
    z_tensor = torch.tensor([[value_one, value_two, value_three]], dtype=torch.float32)

    return z_tensor

class ExternalTensorClient:
    def __init__(self, server1_address, server1_port, server2_address, server2_port, conversion_table):
        self.server1_address = server1_address
        self.server1_port = server1_port
        self.server2_address = server2_address
        self.server2_port = server2_port
        self.conversion_table = conversion_table

        self.z_tensor = None  # Initialize z_tensor as None before any updates
        self.lock = threading.Lock()  # Use a lock to protect access to z_tensor
        self.server1_socket = None
        self.server2_socket = None

    def connect(self):
        """Attempt to connect to both servers, retrying if they are not available."""
        threading.Thread(target=self.connect_to_server1, daemon=True).start()
        threading.Thread(target=self.connect_to_server2, daemon=True).start()

    def connect_to_server1(self):
        """Keep trying to connect to server1."""
        while True:
            try:
                self.server1_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server1_socket.connect((self.server1_address, self.server1_port))
                print(f"Connected to server1 at {self.server1_address}:{self.server1_port}")
                self.receive_from_server1()  # Start receiving data once connected
                break
            except ConnectionRefusedError:
                print(f"Server1 not available, retrying in 2 seconds...")
                time.sleep(2)  # Retry every 2 seconds

    def connect_to_server2(self):
        """Keep trying to connect to server2."""
        while True:
            try:
                self.server2_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
                self.server2_socket.connect((self.server2_address, self.server2_port))
                print(f"Connected to server2 at {self.server2_address}:{self.server2_port}")
                self.receive_from_server2()  # Start receiving data once connected
                break
            except ConnectionRefusedError:
                print(f"Server2 not available, retrying in 2 seconds...")
                time.sleep(2)  # Retry every 2 seconds

    def receive_from_server1(self):
        """Receive strings from server1 and update z_tensor."""
        while True:
            try:
                string_value = self._receive_string(self.server1_socket)
                z_tensor = self.generate_z_tensor_from_string(string_value)
                with self.lock:
                    self.z_tensor = z_tensor  # Update z_tensor safely
                print(f"Updated z_tensor from server1: {z_tensor}")
            except Exception as e:
                print(f"Error in server1: {e}")
                self.connect_to_server1()  # Reconnect if the connection drops
                break

    def receive_from_server2(self):
        """Receive strings from server2 and update z_tensor."""
        while True:
            try:
                string_value = self._receive_string(self.server2_socket)
                z_tensor = self.generate_z_tensor_from_string(string_value)
                with self.lock:
                    self.z_tensor = z_tensor  # Update z_tensor safely
                print(f"Updated z_tensor from server2: {z_tensor}")
            except Exception as e:
                print(f"Error in server2: {e}")
                self.connect_to_server2()  # Reconnect if the connection drops
                break

    def _receive_string(self, sock):
        """Receive one string from a given socket."""
        data = sock.recv(8)  # Adjust byte size as per your protocol
        string_value = data.decode('utf-8').strip()  # Decode and strip any padding or whitespace
        return string_value

    def generate_z_tensor_from_string(self, string_value):
        """Convert the received string into a z_tensor using the conversion table."""
        value_one, value_two = self.conversion_table.get(string_value, (0.0, 0.0))  # Default to (0.0, 0.0)
        value_three = 0.0  # Third value as 0.0 (you can change this if needed)
        z_tensor = torch.tensor([[value_one, value_two, value_three]], dtype=torch.float32)
        return z_tensor

    def get_z_tensor(self):
        """Safely get the current z_tensor."""
        with self.lock:
            return self.z_tensor

    def close(self):
        """Close the connections to both servers."""
        if self.server1_socket:
            self.server1_socket.close()
        if self.server2_socket:
            self.server2_socket.close()
        print("Connections to both servers closed.")

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
    def __init__(self, buffer_size=32, target_size=137):
        self.buffer_size = buffer_size
        self.target_size = target_size
        #self.buffer = buffer_sequence

        self.buffer = [0] * buffer_size

    def update_buffer(self, new_data):
        if len(new_data) != 32:
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
        # print("projected client sequence ")
        # print(projected)
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


def real_time_inference_loop(model, device, client, sender, processor, miror, tensor_client, guide_weight=0.0):
    model.eval()

    # A commmenter for server reciving MI dialogs
    batch_size = 1
    previous_z_tensor, _ = generate_random_tensors_numpy(batch_size)
    z_tensor, chunk_descriptor_tensor = generate_random_tensors_numpy(batch_size)
    previous_input_tensor = generate_random_series_sequence(32, 4)
    chunk_descriptor_tensor[:, 0] = 0


    intermed_csv_path = 'intermed.csv'
    transformed_csv_path = 'csv_file.csv'
    final_csv_path = 'Finalcsv.csv'
    adjusted_csv_path = 'Ajusted_Final_csv_file.csv'
    ground_csv_path = 'GroundTruth.csv'
    extended_csv_path = 'FinalInterpolated.csv'
    N = 10

    while True:
        try:
            with client.lock:
                input_vector = client.latest_processed_batch

            if not input_vector:
                time.sleep(0.01)
                continue

            start_time = time.time()


            updated_buffer = preprocess_real_time(input_vector, processor)
            input_tensor = torch.tensor(updated_buffer, dtype=torch.float32).unsqueeze(0).to(device)

            # Get the latest z_tensor from the tensor client
            z_tensor = tensor_client.get_z_tensor()

            # If no new z_tensor is received, continue using the previous value
            if z_tensor is None:
                print("No new z_tensor received, using previous z_tensor.")
                z_tensor = previous_z_tensor
            else:
                previous_z_tensor = z_tensor  # Update the previous_z_tensor with the latest received value

            z_tensor = z_tensor.to(device)

            #For the chunk descriptor it should be computed given the loop iteration number, and the previous projected outputs using the miroring storer
            chunk_descriptor_tensor[:, 1], chunk_descriptor_tensor[:, 2] = generate_chunk_descriptor_tensor(miror, previous_input_tensor)
            chunk_descriptor_tensor[:, 0] += 0.0001


            z_tensor = z_tensor.to(device)
            chunk_descriptor_tensor = chunk_descriptor_tensor.to(device)

            with torch.no_grad():
                model.guide_w = guide_weight
                start_time2 = time.time()
                y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()
                end_time2 = time.time()
                inference_time = end_time2 - start_time2
                print(f"Inference time for the current batch: {inference_time: .4f} seconds")


            best_prediction = np.round(y_pred)
            best_prediction[best_prediction == 4] = 3
            best_prediction[best_prediction >= 5] = 0
            best_prediction[best_prediction < 0] = 0


            reprojected_output = processor.reproject_to_buffer(best_prediction[0], 32)

            #reprojected_output = generate_random_series_sequence(32, 4)

            # Store the reprojected output sequence in the Storer
            miror.store_sequence(reprojected_output)

            if miror.should_mirror():
                print("Mirroring Mode Activated: Using input tensor as output.")
                reprojected_output = miror.mirror_sequence(input_tensor)

            print("Client sequence:")
            print(input_tensor)
            print("Reprojected Output:")
            print(reprojected_output)

            process_and_save_to_csv(reprojected_output, intermed_csv_path, transformed_csv_path, ground_csv_path, final_csv_path, adjusted_csv_path, extended_csv_path, N)

            # Read the last 16 rows from the adjusted CSV file and queue them for the sender
            df = pd.read_csv(extended_csv_path)
            last_16_rows = df.tail(32).to_csv(index=False, header=False)
            sender.queue_data(last_16_rows.splitlines())

            print("Client sequence:")
            print(input_tensor)
            print("Reprojected Output:")
            print(reprojected_output)

            previous_input_tensor = input_tensor

            elapsed_time = time.time() - start_time
            time.sleep(max(0, 0.3 - elapsed_time))
        except Exception as e:
            print("Error:", e)
            break


def main():
    conversion_table = {
        "Ask for consent": (1.0, 8.0),
        "Medical Education and Guidance": (1.0, 8.0),
        "Planning with the Patient": (1.0, 5.0),
        "Give Solutions": (1.0, 5.0),
        "Ask about current emotions": (1.0, 8.0),
        "Reflections": (1.0, 7.0),
        "Ask for information": (1.0, 8.0),
        "Empathic reactions": (1.0, 7.0),
        "Acknowledge Progress and Encourage": (1.0, 6.0),
        "Backchannel": (3.0, 5.0),
        "Greeting or Closing" : (1.0, 9.0),
        "Experience Normalization and Reassurance": (1.0, 9.0),
        "Changing unhealthy behavior": (2.0, 12.0),
        "Sustaining unhealthy behavior": (2.0, 12.0),
        "Sharing negative feeling or emotion": (2.0,10.0),
        "Sharing positive feeling or emotion": (2.0, 10.0),
        "Realization or Understanding": (2.0, 10.0),
        "Sharing personal information": (2.0, 10.0),
        "Asking for medical information": (2.0, 11.0)
    }

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

    processor = RealTimeProcessor(buffer_size=32, target_size=137)
    mirror = Miror(consecutive_zero_threshold=4)

    client = Client(50150, "localhost", 32)
    client.connect_to_server()
    client.start_receiving()

    # # Initialize the external tensor client
    tensor_client = ExternalTensorClient(conversion_table, "localhost", 50167, "localhost", 50168)
    tensor_client.connect()

    sender = Sender(50151, "localhost")

    real_time_inference_loop(model, device, client, sender,  processor, mirror, tensor_client, guide_weight=guide_w)


def create_server(message_to_send, host='127.0.0.1', port=65432):
    # Create a socket object
    server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

    # Bind the socket to an address and port
    server_socket.bind((host, port))

    # Enable the server to accept connections (listen)
    server_socket.listen(1)
    print(f"Server listening on {host}:{port}")

    while True:
        try:
            # Wait for a connection
            conn, addr = server_socket.accept()
            print(f"Connected by {addr}")

            with conn:
                # Continuously send messages as long as the client is connected
                while True:
                    try:
                        # Send the string message to the client
                        conn.sendall(message_to_send.encode('utf-8'))
                        print(f"Message sent to client: {message_to_send}")

                        # You can add a sleep here to wait before sending another message, for example:
                        time.sleep(5)  # Sends every 5 seconds, adjust as needed

                    except BrokenPipeError:
                        print("Client disconnected.")
                        break  # Exit the inner loop if the client disconnects

        except ConnectionError:
            print("Connection failed. Retrying...")
            time.sleep(2)  # Wait before trying to accept another connection

if __name__ == "__main__":
    main()