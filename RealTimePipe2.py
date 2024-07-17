
import numpy as np
import torch
import time
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import random
from sklearn.neighbors import KernelDensity

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

buffer_sequence = [0., 5., 5., 5., 6., 6., 4., 4., 4., 4., 4., 4., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 0., 6., 6., 6., 6., 6]
class RealTimeProcessor:
    def __init__(self, buffer_size=34, target_size=137):
        self.buffer_size = buffer_size
        self.target_size = target_size
        self.buffer = buffer_sequence

        #self.buffer = [0] * buffer_size

    def update_buffer(self, new_data):
        if len(new_data) != 8:
            raise ValueError("New data must be exactly 8 frames long.")
        self.buffer = self.buffer[len(new_data):] + new_data

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
            if next_pos > target_length:
                reprojected[i] = reprojected[i-1]

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
    return sequence


def send_frames_in_batches(sequence, batch_size=8):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i + batch_size]


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


def real_time_inference_loop(model, device, processor, guide_weight=0.0):
    model.eval()
    random_sequence = generate_random_series_sequence()
    frame_batches = send_frames_in_batches(random_sequence)
    batch_size = 1
    z_tensor, chunk_descriptor_tensor = generate_random_tensors_numpy(batch_size)
    chunk_descriptor_tensor[:, 0] = 0.151
    n = 0

    while n < 100:
        start_time = time.time()
        chunk_descriptor_tensor[:, 0] += 0.001
        chunk_descriptor_tensor[:, 1:] = 0

        try:
            input_vector = next(frame_batches)
            while len(input_vector) < 8:
                input_vector.append(0)
        except StopIteration:
            random_sequence = generate_random_series_sequence()
            frame_batches = send_frames_in_batches(random_sequence)
            input_vector = next(frame_batches)

        updated_buffer = preprocess_real_time(input_vector, processor)
        input_tensor = torch.tensor(updated_buffer, dtype=torch.float32).unsqueeze(0).to(device)
        z_tensor = z_tensor.to(device)
        chunk_descriptor_tensor = chunk_descriptor_tensor.to(device)

        with torch.no_grad():
            model.guide_w = guide_weight
            y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()

        best_prediction = np.round(y_pred)
        best_prediction[best_prediction == 4] = 3
        best_prediction[best_prediction >= 5] = 0
        best_prediction[best_prediction < 0] = 0


        reprojected_output = processor.reproject_to_buffer(best_prediction[0], 34)
        agent_output = reprojected_output[-8:]

        print("Client sequence:")
        print(input_tensor)
        print("Prediction:")
        print(best_prediction)
        print("Reprojected Output:")
        print(reprojected_output)
        print("Output sequence:")
        print(agent_output)

        n += 1
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

    processor = RealTimeProcessor(buffer_size=16, target_size=137)

    real_time_inference_loop(model, device, processor, guide_weight=guide_w)


if __name__ == "__main__":
    main()