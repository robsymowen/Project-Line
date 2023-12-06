# -*- coding: utf-8 -*-
"""line_drawing_conversion_redux.py

GAA's update of cluster_linedrawing_converstions.py

Original file is located at
    https://colab.research.google.com/github/robsymowen/Project-Line/blob/main/Collab/Cluster_LineDrawing_Conversions.ipynb
    
    Example Usage:
    python cluster_linedrawing_converstions.py $SHARED_DATA_DIR/imagenet1k-256/val/n01440764 /n/alvarez_lab_tier1/Users/alvarez/datasets/imagenet1k-line/val/n01440764
    
    python cluster_linedrawing_converstions.py $SHARED_DATA_DIR/imagenet1k-256/val /n/alvarez_lab_tier1/Users/alvarez/datasets/imagenet1k-line/val
    
"""

import argparse
from glob import glob
import os
import subprocess
import shutil

# Create the parser
parser = argparse.ArgumentParser(description='Process line drawing conversion arguments.')

# Add arguments
parser.add_argument('root_dir', type=str, help='The root directory of the ImageNet dataset')
parser.add_argument('output_dir', type=str, help='The directory where output will be stored')

# Parse the arguments
args = parser.parse_args()

# Use the parsed arguments
IMAGENET_ROOT_DIR = args.imagenet_root_dir
OUTPUT_DIR = args.output_dir

"""# **Set up Line Drawing Library**

The biggest development for me here was parsing through the test.py file to edit the data directory being used and the number of files we would iterate through.
- The edited area of the test.py file is labeled
- The files are then copied over to the google drive.
"""

# Git clone command
git_clone_command = "git clone https://github.com/carolineec/informative-drawings.git"

# Run the command
subprocess.run(git_clone_command, shell=True, check=True)

# URL to model download.
checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'

def download_and_unzip_model(checkpoint_url, destination_folder):
    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    try:
        # Download the model checkpoint zip file
        subprocess.run(['wget', '-c', checkpoint_url, '-O', os.path.join(destination_folder, 'model.zip')], check=True)

        # Unzip the model checkpoint
        subprocess.run(['unzip', os.path.join(destination_folder, 'model.zip'), '-d', destination_folder], check=True)

        # Move the contents of the 'model' folder to the destination
        model_folder = os.path.join(destination_folder, 'model')
        for item in os.listdir(model_folder):
            s = os.path.join(model_folder, item)
            d = os.path.join(destination_folder, item)
            if os.path.isdir(s):
                shutil.move(s, d)

        # Remove the now-empty 'model' folder
        os.rmdir(model_folder)

    except subprocess.CalledProcessError as e:
        # Handle any errors (non-zero exit code)
        print(f"Error executing command: {e}")
        print(f"Command output (stderr): {e.stderr}")

# Example: Download and unzip the model
checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'
destination_folder = '/content/informative-drawings/checkpoints'
download_and_unzip_model(checkpoint_url, destination_folder)

"""# **Setting up Arguments for Test.py**

This involves iterating through each folder in the dataset, and creating a matching folder of the converted line drawings in the results folder.
"""

# Check if a results directory exists. If not, create one.
results_dir_name = "AugmentedDataset"
results_dir_path = os.path.join(OUTPUT_DIR, results_dir_name)
os.makedirs(results_dir_path, exist_ok=True)

results_root = os.path.join(OUTPUT_DIR, results_dir_name)

def convert_images(data_dir, results_root, trainval):

    # Create data directory to pull from (either train or val datset)
    data = os.path.join(data_dir, trainval)

    # Create list of folders in dataset to iterate through.
    folders = os.listdir(data)
    print('Folders to iterate through:', folders)

    # Iterate through folders, running test.py on each.
    for folder in folders:
        print('\nConverting images in folder:', folder)
        dataroot = os.path.join(data, folder)
        files = glob(os.path.join(dataroot, "*.JPEG"))

        results_root = os.path.join(data_dir, 'AugmentedDataset_2', trainval)

        # Create results directory with same name
        folder_path = os.path.join(results_root, folder)
        os.makedirs(folder_path)

        # Set results_dir to the folder that matches the subfolder from the dataset
        results_dir = os.path.join(results_root, folder)

        # Set number of files to convert to line drawings
        how_many = len(files)-1

        # Run test.py with these parameters
        script_path = 'informative-drawings/test.py'
        subprocess.run(["python", script_path, "--name", "anime_style", "--dataroot", dataroot, "--results_dir", results_dir, "--how_many", str(how_many)])

    return 0

# Convert Val dataset
convert_images(IMAGENET_ROOT_DIR, results_root, 'val')

# Convert train dataset
convert_images(IMAGENET_ROOT_DIR, results_root, 'train')

