import os
import h5py
import numpy as np

def preprocess_single_sample(hdf5_path, output_bin_path, sample_idx=0, time_step=0.002, total_ticks=500):
    if not os.path.exists(hdf5_path):
        print(f"Error: Could not find raw dataset at {hdf5_path}")
        return

    print(f"Extracting sample {sample_idx} from {hdf5_path}...")
    
    with h5py.File(hdf5_path, 'r') as f:
        # Extract spike times and neuron IDs for the sample
        times = f['spikes']['times'][sample_idx]
        units = f['spikes']['units'][sample_idx]
    
    # 700 input channels for the SHD artificial cochlea
    num_channels = 700
    dense_matrix = np.zeros((total_ticks, num_channels), dtype=bool)
    
    # Map continuous timestamps into discrete simulation clock ticks
    for t, u in zip(times, units):
        tick_idx = int(t / time_step)
        if tick_idx < total_ticks and 0 <= u < num_channels:
            dense_matrix[tick_idx, u] = True
            
    # 700 booleans to bytes (700 bits -> 88 bytes)
    packed_data = np.packbits(dense_matrix, axis=-1)
    
    os.makedirs(os.path.dirname(output_bin_path), exist_ok=True)
    
    with open(output_bin_path, 'wb') as out_file:
        out_file.write(packed_data.tobytes())
        
    print(f"Successfully generated bit-packed binary: {output_bin_path} ({packed_data.nbytes} bytes)")

if __name__ == "__main__":
    raw_dataset = "data/shd_train.h5" 
    output_binary = "data/shd_binned_input.bin"
    
    preprocess_single_sample(raw_dataset, output_binary, sample_idx=0)
