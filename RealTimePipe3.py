
import numpy as np
import torch
import time
import pandas as pd
from Model import Model_mlp_diff, Model_Cond_Diffusion, ObservationEmbedder, SpeakingTurnDescriptorEmbedder, ChunkDescriptorEmbedder
import random
from sklearn.neighbors import KernelDensity
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

def calculate_metrics(y_pred_list, y_target_list):
    correct_activations = 0
    correct_classless_activations = 0
    correct_activations_for_class_1 = 0
    correct_activations_for_class_2 = 0
    correct_activations_for_class_3 = 0
    total_activations_ground_truth = 0
    total_activations_ground_truth_1 = 0
    total_activations_ground_truth_2 = 0
    total_activations_ground_truth_3 = 0
    correct_non_activations = 0
    total_non_activations_ground_truth = 0

    for pred, target in zip(y_pred_list, y_target_list):
        pred_array = np.array(pred)
        target_array = np.array(target)

        is_active_pred = pred_array > 0
        is_active_target = target_array > 0
        is_active_1 = target_array == 1
        is_active_2 = target_array == 2
        is_active_3 = target_array == 3

        correct_activations += np.sum((pred_array == target_array) & is_active_target)
        correct_classless_activations += np.sum(is_active_pred & is_active_target)
        correct_activations_for_class_1 += np.sum((pred_array == target_array) & is_active_1)
        correct_activations_for_class_2 += np.sum((pred_array == target_array) & is_active_2)
        correct_activations_for_class_3 += np.sum((pred_array == target_array) & is_active_3)

        total_activations_ground_truth += np.sum(is_active_target)
        total_activations_ground_truth_1 += np.sum(is_active_1)
        total_activations_ground_truth_2 += np.sum(is_active_2)
        total_activations_ground_truth_3 += np.sum(is_active_3)

        correct_non_activations += np.sum((pred_array == target_array) & ~is_active_target)
        total_non_activations_ground_truth += np.sum(~is_active_target)

    ahr_mouthup = correct_activations_for_class_1 / total_activations_ground_truth_1 if total_activations_ground_truth_1 > 0 else 0
    ahr_nosewrinkle = correct_activations_for_class_2 / total_activations_ground_truth_2 if total_activations_ground_truth_2 > 0 else 0
    ahr_mouthdown = correct_activations_for_class_3 / total_activations_ground_truth_3 if total_activations_ground_truth_3 > 0 else 0
    ahr = correct_activations / total_activations_ground_truth if total_activations_ground_truth > 0 else 0
    achr = correct_classless_activations / total_activations_ground_truth if total_activations_ground_truth > 0 else 0
    nhr = correct_non_activations / total_non_activations_ground_truth if total_non_activations_ground_truth > 0 else 0

    return {
        "ahr_mouthup": ahr_mouthup,
        "ahr_nosewrinkle": ahr_nosewrinkle,
        "ahr_mouthdown": ahr_mouthdown,
        "ahr": ahr,
        "achr": achr,
        "nhr": nhr
    }

def is_valid_chunk(chunk):
    if not chunk[0]:  # Check if the chunk[0] is an empty list
        return False
    for event in chunk[0]:
        if event[0] is float or event[1] is None or event[2] is None:
            return False
    return True
def transform_action_to_sequence(events, sequence_length):
    # This remains the same as the transform_to_sequence function
    sequence = [0] * sequence_length
    for event in events:

        event_type, start_time, duration = event
        start_sample = int(start_time * sequence_length)
        end_sample = int(start_sample + (duration * sequence_length))
        for i in range(start_sample, min(end_sample, sequence_length)):
            sequence[i] = event_type
    return sequence

def transform_obs_to_sequence(events, sequence_length):
    facial_expression_events = [21, 27, 31]  # Define facial expression event types
    sequence = [0] * sequence_length
    mi_behaviors = []  # To store MI behaviors
    for event in events:
        event_type, start_time, duration = event
        if event_type not in facial_expression_events and event_type == round(event_type):
            mi_behaviors.append(event_type)
        else:
            start_sample = int(start_time * sequence_length)
            end_sample = int(start_sample + (duration * sequence_length))
            for i in range(start_sample, min(end_sample, sequence_length)):
                if event_type == round(event_type):
                    sequence[i] = event_type
                else:
                    sequence[i] = 0
    return sequence, mi_behaviors


def process_single_file(file_path):
    facial_expression_mapping = {0: 0, 16: 1, 26: 2, 30: 3, 21: 4, 27: 5, 31: 6}
    mi_behavior_mapping = {39: 1, 38: 2, 40: 3, 41: 4, 3: 5, 4: 6, 5: 7, 6: 8, 8: 9, 11: 10, 13: 11, 12: 12}

    # Step 1: Load and preprocess the sample
    raw_data = load_data_from_file(file_path)  # Assumes this function is correctly adapted for single files
    processed_data, sequence_length = preprocess_data(raw_data)  # Wrap in list, assuming preprocessing expects a list

    # Initialize a container for results
    evaluation_results = []
    all_chunks = []
    max_z_len = 3

    # Step 2: Transform data to sequences, including the chunk_descriptor
    for (observation, action, chunk_descriptor), speaking_turn_duration in processed_data:
        x, z = transform_obs_to_sequence(observation, sequence_length)
        y = transform_action_to_sequence(action, sequence_length)

        if len(z) < max_z_len:
            z = z + [0] * (max_z_len - len(z))  # Assuming 0 is an appropriate padding value

        # Reassign event values based on the new mappings
        x = [facial_expression_mapping.get(item, item) for item in x]
        y = [facial_expression_mapping.get(item, item) for item in y]
        z = [mi_behavior_mapping.get(item, item) for item in z]  # Assuming 'z' needs mapping as well

        # Convert to tensors or your model's required input format, ensuring a batch dimension
        x_tensor = torch.tensor([x], dtype=torch.float32)
        y_tensor = torch.tensor([y], dtype=torch.float32)
        z_tensor = torch.tensor([z], dtype=torch.float32)
        chunk_descriptor_tensor = torch.tensor([chunk_descriptor], dtype=torch.float32)
        speaking_turn_duration_tensor = torch.tensor([speaking_turn_duration], dtype=torch.float32)

        # Append a tuple for each chunk to the list
        all_chunks.append((x_tensor, y_tensor, z_tensor, chunk_descriptor_tensor, speaking_turn_duration_tensor))
    return all_chunks


def load_data_from_file(file_path):
    if not os.path.exists(file_path) or not file_path.endswith(".txt"):
        raise ValueError("File does not exist or is not a '.txt' file.")

    with open(file_path, 'r') as f:
        lines = f.readlines()
        non_empty_lines = [line.strip() for line in lines if line.strip() != ""]
        chunks = [eval(line) for line in non_empty_lines]


    observation_chunks = [chunk[:-1] for chunk in chunks[::2]]  # get all tuples except the last one
    action_chunks = chunks[1::2]  # extract every second element starting from 1
    chunk_descriptors = [chunk[-1] for chunk in chunks[::2]]


    for i in range(len(chunk_descriptors)):
        event = list(chunk_descriptors[i])
        if event[2] is None:
            event[2] = -1
        if event[1] is None:
            event[1] = -1
        chunk_descriptors[i] = tuple(event)

    all_data = list(zip(observation_chunks, action_chunks, chunk_descriptors))

    return all_data

def preprocess_data(data):
    filtered_data = [chunk for chunk in data if is_valid_chunk(chunk)]
    sequence_length = 137  # Assuming 25 FPS

    processed_chunk = []

    for i, chunk in enumerate(filtered_data):
        if not chunk[0]:  # Skip if the observation vector is empty
            continue

        # Combine and filter start times and durations for observation and action events
        combined_events = chunk[0] + chunk[1]
        valid_events = [(event[1], event[2]) for event in combined_events if
                        isinstance(event[1], float) and event[1] > 0]

        if not valid_events:  # Skip if no valid start times
            continue

        # Calculate the minimum start time and the maximum end time
        min_start_time = min(event[0] for event in valid_events)
        max_end_time = max((event[0] + event[1] for event in valid_events), default=min_start_time)
        speaking_turn_duration = max_end_time - min_start_time

        # Adjust speaking turn duration based on next chunk's starting time
        if i + 1 < len(filtered_data):
            next_chunk = filtered_data[i + 1]
            next_chunk_events = next_chunk[0] + next_chunk[1]
            next_chunk_start_times = [event[1] for event in next_chunk_events if
                                      isinstance(event[1], float) and event[1] > 0]

            if next_chunk_start_times:
                next_chunk_start_time = min(next_chunk_start_times)
                current_interaction_time = min_start_time + speaking_turn_duration
                if current_interaction_time > next_chunk_start_time:
                    speaking_turn_duration = next_chunk_start_time - min_start_time


        if speaking_turn_duration <= 0: # Skip turns with non-positive duration
            continue

        # Normalize start times and durations within each chunk
        for vector in [0, 1]:  # 0 for observation, 1 for action
            for i, event in enumerate(chunk[vector]):
                event_type, start_time, duration = event
                if start_time == 0.0:
                    continue
                if start_time<min_start_time:
                    start_time = min_start_time
                # Standardize the starting times relative to the speaking turn's start
                normalized_start_time = (start_time - min_start_time)


                # Normalize start times and durations against the speaking turn duration
                normalized_start_time = normalized_start_time / speaking_turn_duration
                normalized_duration = duration / speaking_turn_duration


                # Update the event with normalized values
                chunk[vector][i] = (event_type, round(normalized_start_time, 3), round(normalized_duration, 3))

        processed_chunk.append((chunk, speaking_turn_duration))
    return processed_chunk, sequence_length
class RealTimeProcessor:
    def __init__(self, buffer_size=68, target_size=137):
        self.buffer_size = buffer_size
        self.target_size = target_size
        #self.buffer = np.zeros((buffer_size,))

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



def send_frames_in_batches(sequence, batch_size=68):
    for i in range(0, len(sequence), batch_size):
        yield sequence[i:i + batch_size]


def convert_x_tensor(x_tensor, speaking_turn_duration, original_size=137):
    ratio = speaking_turn_duration / 5.48
    new_size = int(original_size * ratio)

    projected = [0] * new_size
    buffer_length = x_tensor.shape[1]
    x_array = x_tensor.squeeze().cpu().numpy().tolist()

    pos = 0
    for i in range(buffer_length):
        value = x_array[i]
        count = round((i + 1) * ratio) - round(i * ratio)
        for _ in range(count):
            if pos < new_size:
                projected[pos] = value
                pos += 1
            else:
                break

    return projected


def real_time_inference_loop(model, device, processor, all_chunks, guide_weight=0.0):
    model.eval()
    batch_size = 1
    chunk_index = 0
    y_pred_list = []
    y_target_list = []

    while chunk_index < len(all_chunks):
        x_tensor, y_tensor, z_tensor, chunk_descriptor_tensor, speaking_turn_duration_tensor = all_chunks[chunk_index]

        speaking_turn_duration = speaking_turn_duration_tensor.item()
        x_sequence = convert_x_tensor(x_tensor, speaking_turn_duration)
        y_sequence = convert_x_tensor(y_tensor, speaking_turn_duration)

        # Create a generator for 8-frame batches
        frame_batches = send_frames_in_batches(x_sequence)
        target_batches =send_frames_in_batches(y_sequence)
        z_tensor = z_tensor.to(device)
        chunk_descriptor_tensor = chunk_descriptor_tensor.to(device)

        try:
            while True:
                start_time = time.time()
                target_vector = next(target_batches)
                input_vector = next(frame_batches)
                while len(input_vector) < 68:
                    input_vector.append(0)
                while len(target_vector) < 68:
                    target_vector.append(0)

                print("the client 17 frames tel quel")
                print(input_vector)
                print("the therapist  17 target frames")
                print(target_vector)

                updated_buffer = preprocess_real_time(input_vector, processor)
                input_tensor = torch.tensor(updated_buffer, dtype=torch.float32).unsqueeze(0).to(device)

                print("Z:")
                print(z_tensor)
                print("C:")
                print(chunk_descriptor_tensor)

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

                reprojected_output = processor.reproject_to_buffer(best_prediction[0], 68)

                print("Reprojected Output prediction :")
                print(reprojected_output)

                y_pred_list.append(reprojected_output)
                y_target_list.append(target_vector)

                elapsed_time = time.time() - start_time
                print(f"Inference time for the current loop: {inference_time: .4f} seconds")
                time.sleep(max(0, 0.3 - elapsed_time))

        except StopIteration:
            chunk_index += 1

    metrics = calculate_metrics(y_pred_list, y_target_list)
    print("Metrics:", metrics)

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    observation_embedder = ObservationEmbedder(num_facial_types, facial_embed_dim, cnn_output_dim, lstm_hidden_dim, sequence_length)
    mi_embedder = SpeakingTurnDescriptorEmbedder(num_event_types, event_embedding_dim, embed_output_dim)
    chunk_embedder = ChunkDescriptorEmbedder(continious_embedding_dim=16, valence_embedding_dim=8, output_dim=64)

    model_path = 'saved_model_NewmodelChunkd1000.pth'
    nn_model = Model_mlp_diff(observation_embedder, mi_embedder, chunk_embedder, sequence_length, net_type="transformer")
    model = Model_Cond_Diffusion(nn_model, observation_embedder, mi_embedder, chunk_embedder, betas=(1e-4, 0.02), n_T=n_T, device=device, x_dim=x_dim, y_dim=y_dim, drop_prob=0, guide_w=guide_w)

    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    processor = RealTimeProcessor(buffer_size=16, target_size=137)

    # Process the text file to get all chunks
    file_path = 'intervieww_36_606.txt'
    all_chunks = process_single_file(file_path)

    real_time_inference_loop(model, device, processor, all_chunks, guide_weight=guide_w)

if __name__ == "__main__":
    main()
