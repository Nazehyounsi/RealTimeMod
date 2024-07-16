import numpy as np
import torch
import time
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import random

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

    while True:
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
        print("la client sequence entière")
        print(input_tensor)
        print("Z:")
        print(z_tensor)
        print("C:")
        print(chunk_descriptor_tensor)
        with torch.no_grad():
            model.guide_w = guide_weight
            y_pred = model.sample(input_tensor, z_tensor, chunk_descriptor_tensor).detach().cpu().numpy()

        best_prediction = np.round(y_pred)
        best_prediction[best_prediction == 4] = 3
        best_prediction[best_prediction >= 5] = 0
        best_prediction[best_prediction < 0] = 0

        print("la prediction entière")
        print(best_prediction)
        agent_output = best_prediction[0, -8:]
        print("la outputed sequence")
        print(agent_output)

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
