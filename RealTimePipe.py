import numpy as np
import torch
import time
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import random
from sklearn.neighbors import KernelDensity

buffer_sequence = [0., 5., 5., 5., 6., 6., 4., 4., 4., 4., 4., 4., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.,
        0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0.]
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

def generate_random_series_sequence(sequence_length=137, max_series=4):
    sequence = [0] * sequence_length
    possible_values = [4, 5, 6]
    max_series_length = sequence_length // 2
    series_count = 0

    while series_count < max_series:
        value = random.choice(possible_values)
        series_length = random.randint(5, max_series_length)
        start_position = random.randint(0, sequence_length - series_length)

        for i in range(start_position, start_position + series_length):
            sequence[i] = value

        series_count += 1
    print(sequence)

    return sequence

def send_frames_in_batches(sequence, batch_size=8):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i + batch_size]


def generate_random_tensors_numpy(batch_size=1):
    # Generate random z_tensor
    z_tensor = np.zeros((batch_size, 4))
    #z_tensor[:, 0] = np.random.randint(5, 13, size=batch_size)  # First value: int from 5 to 12
    #z_tensor[:, 1] = np.random.randint(1, 5, size=batch_size)   # Second value: int from 1 to 4
    z_tensor[:,0] = 12
    z_tensor[:, 1] = 2
    z_tensor[:, 2] = 12  # Last value always 0
    z_tensor[:, 3] = 0  # Last value always 0

    # Generate random chunk_descriptor_tensor
    chunk_descriptor_tensor = np.zeros((batch_size, 3))
    chunk_descriptor_tensor[:, 0] = np.random.random(size=batch_size)  # First value: float from 0 to 1
    chunk_descriptor_tensor[:, 1:] = np.random.choice([-1, 0, 1], size=(batch_size, 2))  # Last two values: -1, 0, or 1

    # Convert to PyTorch tensors
    z_tensor = torch.tensor(z_tensor, dtype=torch.float32)
    chunk_descriptor_tensor = torch.tensor(chunk_descriptor_tensor, dtype=torch.float32)

    return z_tensor, chunk_descriptor_tensor
class RealTimeProcessor:
    def __init__(self, buffer_size=137):
        self.buffer_size = buffer_size
        self.buffer = buffer_sequence

    def update_buffer(self, new_data):
        if len(new_data) != 8:
            raise ValueError("New data must be exactly 8 frames long.")

        self.buffer = np.roll(self.buffer, -8)
        self.buffer[-8:] = new_data

    def get_buffer(self):
        return self.buffer

def preprocess_real_time(input_frames, processor):
    processor.update_buffer(input_frames)
    return processor.get_buffer()



def real_time_inference_loop(model, device, processor, guide_weight=0.0):
    model.eval()
    # Generate a random sequence of 137 frames
    random_sequence = generate_random_series_sequence()

    # Create a generator for 8-frame batches
    frame_batches = send_frames_in_batches(random_sequence)
    batch_size = 1
    z_tensor, chunk_descriptor_tensor = generate_random_tensors_numpy(batch_size)
    chunk_descriptor_tensor[:, 0] = 0.151
    n = 0
    kde_samples = 3
    while n < 100:
        start_time = time.time()

        chunk_descriptor_tensor[:, 0] = chunk_descriptor_tensor[:, 0] + 0.001
        chunk_descriptor_tensor[:, 1] = 0
        chunk_descriptor_tensor[:, 2] = 0

        try:
            input_vector = next(frame_batches)  # Get the next 8-frame batch
            while len(input_vector) < 8 :
                input_vector.append(0)

        except StopIteration:
            # If the sequence is exhausted, generate a new sequence and create a new generator
            random_sequence = generate_random_series_sequence()
            frame_batches = send_frames_in_batches(random_sequence)
            input_vector = next(frame_batches)


        updated_buffer = preprocess_real_time(input_vector, processor)
        input_tensor = torch.tensor(updated_buffer, dtype=torch.float32).unsqueeze(0).to(device)
        z_tensor = z_tensor.to(device)
        chunk_descriptor_tensor = chunk_descriptor_tensor.to(device)
        print("la client sequence entière")
        print(input_tensor)
        print("Z:")
        print(z_tensor)
        print("C:")
        print(chunk_descriptor_tensor)
# #KDE####################################################################
#         all_predictions = []
#         for _ in range(kde_samples):
#             with torch.no_grad():
#                 model.guide_w = guide_weight
#                 start_time = time.time()
#                 y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()
#                 end_time = time.time()
#                 inference_time = end_time - start_time
#                 print(f"Inference time for the current batch: {inference_time: .4f} seconds")
#                 all_predictions.append(y_pred)
#
#         # Apply KDE for the chunk and determine the best prediction
#         best_prediction = np.zeros_like(y_pred)
#         single_pred_samples = np.array(all_predictions).squeeze()
#         kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(single_pred_samples)
#         log_density = kde.score_samples(single_pred_samples)
#         best_idx = np.argmax(log_density)
#         best_prediction = single_pred_samples[best_idx]
#
#         ####################################KDE###########"""
        with torch.no_grad():
            model.guide_w = guide_weight
            y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()
        best_prediction = np.round(y_pred)
        best_prediction[best_prediction == 4] = 3
        best_prediction[best_prediction >= 5] = 0
        best_prediction[best_prediction < 0] = 0

        print("la prediction entière")
        print(best_prediction)
        #agent_output = best_prediction[-8:]
        agent_output = best_prediction[0, -8:] #NON KDE

        print("la outputed sequence")
        print(agent_output)

        n=n+1
        elapsed_time = time.time() - start_time
        time.sleep(max(0, 0.3 - elapsed_time))

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

    processor = RealTimeProcessor()

    real_time_inference_loop(model, device, processor, guide_weight=guide_w)


if __name__ == "__main__":
    main()


def interpolate_activations_betweenchunks(df_prev, df_curr, columns, N, K):
    """
    Interpolates activations between two successive chunks.

    :param df_prev: DataFrame of the previous chunk.
    :param df_curr: DataFrame of the current chunk.
    :param columns: List of columns to process.
    :param N: Number of frames for interpolation.
    :param K: Maximum gap to consider merging activations.
    :return: Modified df_prev and df_curr DataFrames.
    """
    for col in columns:
        # Get the values for the column in both chunks
        values_prev = df_prev[col].values.copy()
        values_curr = df_curr[col].values.copy()

        # Find the last non-zero index in the previous chunk
        non_zero_indices_prev = np.nonzero(values_prev)[0]
        if len(non_zero_indices_prev) > 0:
            last_non_zero_prev = non_zero_indices_prev[-1]
            activation_value_prev = values_prev[last_non_zero_prev]
        else:
            last_non_zero_prev = None
            activation_value_prev = 0

        # Find the first non-zero index in the current chunk
        non_zero_indices_curr = np.nonzero(values_curr)[0]
        if len(non_zero_indices_curr) > 0:
            first_non_zero_curr = non_zero_indices_curr[0]
            activation_value_curr = values_curr[first_non_zero_curr]
        else:
            first_non_zero_curr = None
            activation_value_curr = 0

        # Calculate the gap between the two activations
        if last_non_zero_prev is not None and first_non_zero_curr is not None:
            gap = first_non_zero_curr + (len(values_prev) - last_non_zero_prev - 1)
            if gap < K:
                # Merge the activations by filling the gap
                # Adjust previous chunk if needed
                # No need to adjust previous chunk in this case
                # Adjust current chunk
                values_curr[:first_non_zero_curr] = activation_value_curr
        elif last_non_zero_prev is not None and first_non_zero_curr is None:
            # Previous chunk ends with activation, current chunk is zeros
            # Interpolate decrease over N frames in current chunk
            end_idx = min(N, len(values_curr))
            for i in range(end_idx):
                values_curr[i] = activation_value_prev * (1 - (i + 1) / N)
        elif last_non_zero_prev is None and first_non_zero_curr is not None:
            # Previous chunk ends with zeros, current chunk starts with activation
            # Interpolate increase over N frames at the end of previous chunk
            start_idx = max(0, len(values_prev) - N)
            for i in range(start_idx, len(values_prev)):
                values_prev[i] = activation_value_curr * ((i - start_idx + 1) / N)

        # Update the DataFrames
        df_prev[col] = values_prev
        df_curr[col] = values_curr

    return df_prev, df_curr

df = pd.read_csv(extended_csv_path)
columns_to_interpolate = ['AU06_r', 'AU25_r', 'AU12_r', 'AU10_r', 'AU09_r', 'AU14_r', 'AU15_r']

# Ensure enough data is available
if df.shape[0] > 127:
    # Get the previous and current chunks
    df_prev = df.iloc[-256:-128].reset_index(drop=True)
    df_curr = df.iloc[-128:].reset_index(drop=True)

    # Apply the interpolation between chunks
    df_prev, df_curr = interpolate_activations_betweenchunks(
        df_prev, df_curr, columns_to_interpolate, N, K
    )

    # Update the main DataFrame with modified chunks
    df.update(df_prev)
    df.update(df_curr)

# Get the last 64 rows to send
last_buffer_rows = df.tail(64).to_csv(index=False, header=False)
sender.queue_data(last_buffer_rows.splitlines())