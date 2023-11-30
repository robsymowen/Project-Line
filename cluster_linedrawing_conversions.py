# -*- coding: utf-8 -*-
"""Cluster_LineDrawing_Conversions.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1ACvnxs0A8I7eK5tNuPxLZoTXjMckl3Ls
"""

# Input Directory
IMAGENET_ROOT_DIR = "/content/drive/MyDrive/Classes/Third/PSY2356R/ThomasGarity/LineDrawing/datasets/imagenette2-320"

# Output Directory
OUTPUT_DIR = '/content/drive/MyDrive/Classes/Third/PSY2356R/ThomasGarity/LineDrawing'

"""# **Load Dataset**
- Director to dataset is stored under 'IMAGENET_ROOT_DIR'
"""

from glob import glob
import os

!ls $IMAGENET_ROOT_DIR

"""# **Set up Line Drawing Library**

The biggest development for me here was parsing through the test.py file to edit the data directory being used and the number of files we would iterate through.
- The edited area of the test.py file is labeled
- The files are then copied over to the google drive.
"""

# Clone repositry of line drawing models.
!git clone https://github.com/carolineec/informative-drawings.git

# URL to model download.
checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'

# Get model from URL, move into correct direcotry, remove the directory
!mkdir -p informative-drawings/checkpoints
!wget -c {checkpoint_url} -O informative-drawings/checkpoints/model.zip
!unzip /content/informative-drawings/checkpoints/model.zip -d /content/informative-drawings/checkpoints
!mv /content/informative-drawings/checkpoints/model/* /content/informative-drawings/checkpoints
!rmdir /content/informative-drawings/checkpoints/model

print("Testing that we can convert to line drawings")
!cd informative-drawings && python test.py --name anime_style --dataroot examples/test

"""# **Setting up Arguments for Test.py**

This involves iterating through each folder in the dataset, and creating a matching folder of the converted line drawings in the results folder.
"""

# Check if a results directory exists. If not, create one.
results_dir_name = "AugmentedDataset"
!cd $OUTPUT_DIR && mkdir -p $results_dir_name

results_root = os.path.join(OUTPUT_DIR, results_dir_name)

# Navigate to results_root and show what exists there
!cd $results_root && ls

def convert_images(data_dir, results_root, trainval):

    # Create data directory to pull from (either train or val datset)
    data = os.path.join(data_dir, trainval)
    !ls $data


    # Create list of folders in dataset to iterate through.
    folders = os.listdir(data)
    print('Folders to iterate through:', folders)

    # Iterate through folders, running test.py on each.
    for folder in folders:
        print('\nConverting images in folder:', folder)
        dataroot = os.path.join(data, folder)
        files = glob(os.path.join(dataroot, "*.JPEG"))

        results_root = os.path.join(root_dir, 'AugmentedDataset_2', trainval)

        # Create results directory with same name
        !cd $results_root && mkdir -p $folder

        # Set results_dir to the folder that matches the subfolder from the dataset
        results_dir = os.path.join(results_root, folder)

        # Set number of files to convert to line drawings
        how_many = len(files)-1

        # Run test.py with these parameters
        !cd informative-drawings && python test.py --name anime_style --dataroot $dataroot --results_dir $results_dir --how_many $how_many

    return 0

# Convert train dataset
convert_images(IMAGENET_ROOT_DIR, results_root, 'train')

# Convert Val dataset
convert_images(IMAGENET_ROOT_DIR, results_root, 'val')