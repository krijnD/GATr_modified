import numpy as np
from pathlib import Path

# Define paths to your data files
data_dir = Path('/home/krijn/Documents/DL2/geometric-algebra-transformer/data_conversion/npy_files')
data_files = {
    'charges_test': data_dir / 'charges_test_charged5_initvel1small.npy',
    'charges_train': data_dir / 'charges_train_charged5_initvel1small.npy',
    'charges_valid': data_dir / 'charges_valid_charged5_initvel1small.npy',
    'loc_test': data_dir / 'loc_test_charged5_initvel1small.npy',
    'loc_train': data_dir / 'loc_train_charged5_initvel1small.npy',
    'loc_valid': data_dir / 'loc_valid_charged5_initvel1small.npy',
    'vel_test': data_dir / 'vel_test_charged5_initvel1small.npy',
    'vel_train': data_dir / 'vel_train_charged5_initvel1small.npy',
    'vel_valid': data_dir / 'vel_valid_charged5_initvel1small.npy'
}

# Load the data from the .npy files
data = {key: np.load(file) for key, file in data_files.items()}

# extract the mass, initial position, initial velocity, and final position data
charges_train = data['charges_train'].reshape(-1, 5, 1)
x_initial_train = data['loc_train'][:, 0, :, :].reshape(-1, 5, 3)
v_initial_train = data['vel_train'][:, 0, :, :].reshape(-1, 5, 3)
x_final_train = data['loc_train'][:, -1, :, :].reshape(-1, 5, 3)

charges_val = data['charges_valid'].reshape(-1, 5, 1)
x_initial_val = data['loc_valid'][:, 0, :, :].reshape(-1, 5, 3)
v_initial_val = data['vel_valid'][:, 0, :, :].reshape(-1, 5, 3)
x_final_val = data['loc_valid'][:, -1, :, :].reshape(-1, 5, 3)

charges_test = data['charges_test'].reshape(-1, 5, 1)
x_initial_test = data['loc_test'][:, 0, :, :].reshape(-1, 5, 3)
v_initial_test = data['vel_test'][:, 0, :, :].reshape(-1, 5, 3)
x_final_test = data['loc_test'][:, -1, :, :].reshape(-1, 5, 3)

# Define the output directory
output_dir = data_dir.parent / 'converted_data'
output_dir.mkdir(parents=True, exist_ok=True)

# Save the data in .npz format in the specified directory
np.savez(output_dir / 'train.npz', charges=charges_train, x_initial=x_initial_train, v_initial=v_initial_train, x_final=x_final_train)
np.savez(output_dir / 'val.npz', charges=charges_val, x_initial=x_initial_val, v_initial=v_initial_val, x_final=x_final_val)
np.savez(output_dir / 'test.npz', charges=charges_test, x_initial=x_initial_test, v_initial=v_initial_test, x_final=x_final_test)

# Function to print a sample data point for verification
def print_sample_data(file):
    with np.load(file) as data:
        print(f"File: {file}")
        for key, value in data.items():
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
            print(f"Sample data from {key}:", value[0])
        print()

# Print a sample from each dataset to check if the conversion was successful
print_sample_data(output_dir / 'train.npz')
print_sample_data(output_dir / 'val.npz')
print_sample_data(output_dir / 'test.npz')
