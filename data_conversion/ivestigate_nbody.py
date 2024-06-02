import numpy as np
from pathlib import Path

# Define the path to your data directory
data_dir = Path('/home/krijn/Documents/DL2/geometric-algebra-transformer/data_conversion/converted_data/')

# List of filenames to inspect
data_files = [
    'test.npz',
    'train.npz',
    'val.npz'
]

# Function to inspect and print the contents and shapes of the data files
def inspect_data(file_path):
    with np.load(file_path) as data:
        print(f"Inspecting file: {file_path.name}")
        for key, value in data.items():
            print(f"{key}: shape {value.shape}, dtype {value.dtype}")
        print("\n")


# Function to print a sample data point for verification
def print_sample_data(file):
    with np.load(file) as data:
        print(f"Inspecting file: {file.name}")
        x_initial = data['x_initial']
        x_final = data['x_final']
        charges = data['charges']

        # Print the shapes
        print(f"x_initial shape: {x_initial.shape}")
        print(f"x_final shape: {x_final.shape}\n")
        print(f"charges shape: {charges.shape}\n")

        # # Print the first sample for verification
        # print("First sample x_initial:")
        # print(x_initial[0])
        # print("\nFirst sample x_final:")
        # print(x_final[0])
        # print("\n")

# # Inspect each data file
for file_name in data_files:
    file_path = data_dir / file_name
    inspect_data(file_path)

# Inspect each data file
# for file_name in data_files:
#     file_path = data_dir / file_name
#     print_sample_data(file_path)
