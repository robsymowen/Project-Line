'''
    
    cluster_linedrawing_training.py generates line-drawing copies of up to 10000 images in a directory
    by calling informative-drawings.
    
    Unfortunately the generated files have this name format:
    
        anime_style/val/n02895154/anime_style/<filename>.png
        ...
        
    But I want them to be named
        imagenet1k-anime_style/val/n02895154/<filename>.png
        ...
    
    To fix the filenames, run:
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/anime_style/val anime_style
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/contour_style/val contour_style
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/opensketch_style/val opensketch_style
    
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/anime_style/train anime_style
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/contour_style/train contour_style
    python fix_filenames.py $SHARED_DATA_DIR/imagenet1k-line/opensketch_style/train opensketch_style
    
    mv anime_style imagenet1k-anime_style
    mv contour_style imagenet1k-contour_style
    mv opensketch_style imagenet1k-opensketch_style
'''
import os
import shutil
import argparse
import contextlib
import joblib
from joblib import Parallel, delayed
from tqdm import tqdm

@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""
    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()

def rename_file(path, new_path):
    """ Rename a file by moving it to the new path. """
    try:
        shutil.move(path, new_path)
    except Exception as e:
        print(f"Error moving file {path} to {new_path}: {e}")

def process_directory(directory, style_name):
    """ Process each subdirectory in the given directory. """
    for root, dirs, files in os.walk(directory):
        parts = root.split(os.sep)
        
        for file in files:
            full_path = os.path.join(root, file)
            # if the root for this file ends with style_name, drop it from filename
            if len(parts) > 1 and parts[-1] == style_name:
                new_root = os.sep.join(parts[:-1])
                new_full_path = os.path.join(new_root, file)
                rename_file(full_path, new_full_path)
        
        # Check if the current root directory ends with style_name and is now empty
        if len(parts) > 1 and parts[-1] == style_name:
            if not os.listdir(root):  # The directory is empty
                os.rmdir(root)
                print(f"Removed empty directory: {root}")
            else:
                print(f"Directory not empty: {root}")
                
def main(root_directory, style_name):
    """ Main function to process all directories under the root. """
    subdirs = [os.path.join(root_directory, d) for d in os.listdir(root_directory) if os.path.isdir(os.path.join(root_directory, d))]
    assert len(subdirs)==1000, f"Expected 1000 subdirs, got {len(subdirs)}"
    
    with tqdm_joblib(tqdm(desc="Processing Files", total=len(subdirs))) as progress_bar:
        Parallel(n_jobs=-1)(delayed(process_directory)(d, style_name) for d in subdirs)

def parse_arguments():
    """ Parse command line arguments for the script. """
    parser = argparse.ArgumentParser(description='Fix filenames in a specified directory.')
    parser.add_argument('root_dir', type=str, help='Root directory to fix filenames in')
    parser.add_argument('style_name', type=str, help='Name of the style to be fixed')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_arguments()
    main(args.root_dir, args.style_name)