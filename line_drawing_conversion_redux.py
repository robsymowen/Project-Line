# -*- coding: utf-8 -*-
"""line_drawing_conversion_redux.py

GAA's update of cluster_linedrawing_converstions.py

https://github.com/carolineec/informative-drawings.git

Original file is located at
    https://colab.research.google.com/github/robsymowen/Project-Line/blob/main/Collab/Cluster_LineDrawing_Conversions.ipynb
    
    Example Usage:
    python line_drawing_conversion_redux.py anime_style $SHARED_DATA_DIR/imagenet1k-256/val/n01440764 /n/alvarez_lab_tier1/Users/alvarez/datasets/imagenet1k-line/anime_style/val/n01440764
    
    python line_drawing_conversion_redux.py anime_style $SHARED_DATA_DIR/imagenet1k-256/val /n/alvarez_lab_tier1/Users/alvarez/datasets/imagenet1k-line/anime_style/val
    
    styles = ['anime_style', 'contour_style', 'opensketch_style']
    
"""
import os
import argparse
from urllib.parse import urlparse
from glob import glob
from pdb import set_trace
import subprocess
import shutil
from fastprogress import progress_bar

styles = ['anime_style', 'contour_style', 'opensketch_style']

def parse_arguments():
    """ Parse command line arguments for line drawing conversion. """
    parser = argparse.ArgumentParser(description='Process line drawing conversion arguments.')
    parser.add_argument('style', type=str, choices=styles, default='anime_style', 
                        help='The style for line drawing conversion')
    parser.add_argument('image_dir', type=str, help='The root directory of the ImageNet dataset')
    parser.add_argument('output_dir', type=str, help='The directory where output will be stored')

    return parser.parse_args()

def get_filename_from_url(url):
    """ Extract the filename from a given URL. """
    parsed_url = urlparse(url)
    return os.path.basename(parsed_url.path)

def download_and_unzip_model(checkpoint_url, destination_folder):
    """ Download and unzip the line drawing model if not already present. """
    # Check if the model or a key file already exists to avoid re-downloading
    model_file_check = get_filename_from_url(checkpoint_url)
    model_zip_path = os.path.join(destination_folder, model_file_check)
    if os.path.exists(model_zip_path):
        print(f"Model file {model_file_check} already exists, skipping download.")
        return

    # Create the destination folder if it doesn't exist
    os.makedirs(destination_folder, exist_ok=True)

    try:
        # Download the model checkpoint zip file
        subprocess.run(['wget', '-c', checkpoint_url, '-O', model_zip_path], check=True)

        # Unzip the model checkpoint
        subprocess.run(['unzip', model_zip_path, '-d', destination_folder], check=True)

        # Move contents from 'model' folder, if exists, to destination folder
        model_folder = os.path.join(destination_folder, 'model')
        if os.path.exists(model_folder):
            for item in os.listdir(model_folder):
                shutil.move(os.path.join(model_folder, item), destination_folder)
            os.rmdir(model_folder)

    except subprocess.CalledProcessError as e:
        print(f"Error downloading or unzipping model: {e}")
        exit(1)
        
def get_folders_or_use_base_dir(data_dir):
    """ Get a list of subdirectories in data_dir. If there are none, return a list with only data_dir. """
    folders = [f for f in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, f))]

    # If there are no subdirectories, use data_dir itself
    if not folders:
        return [data_dir]
    return folders

def convert_images(style, data_dir, results_root):
    """ Convert images to line drawings. """
    conda_env_path = "/n/home02/alvarez/.conda/envs/workshop/bin/python"
    script_path = 'test.py'
    repo_root_path = './submodules/informative-drawings'
    folders = get_folders_or_use_base_dir(data_dir)
    for folder in progress_bar(folders):
        dataroot = os.path.join(data_dir, folder)
        files = sorted(glob(os.path.join(dataroot, "*.JPEG")))
        results_dir = os.path.join(results_root, folder.replace(data_dir,""))
        assert results_dir != data_dir, f"Oops, {results_dir} is same as source {data_dir}"
        os.makedirs(results_dir, exist_ok=True)
        
        # make sure these files haven't already been generated
        existing_files_count = len(glob(os.path.join(results_dir, style, "*.png")))
        how_many = len(files)
        if existing_files_count >= how_many:
            print(f"Skipping {folder} as it already contains the required number of files.")
            continue
        
        print(f"==> Generating {how_many} drawings from {folder}")
        subprocess.run([conda_env_path, script_path, "--name", style, "--dataroot", dataroot, "--results_dir", results_dir, "--how_many", str(how_many)], cwd=repo_root_path)
        
def main():
    """ Main function to run the script. """
    args = parse_arguments()    
    checkpoint_url = 'https://s3.us-east-1.wasabisys.com/visionlab-projects/transfer/informative-drawings/model.zip'
    download_and_unzip_model(checkpoint_url, './submodules/informative-drawings/checkpoints')
    convert_images(args.style, args.image_dir, args.output_dir)

if __name__ == "__main__":
    main()