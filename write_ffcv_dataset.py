'''
    Write an ffcv dataset (https://github.com/libffcv/ffcv)
    lightly edited from here: https://github.com/libffcv/ffcv-imagenet/blob/main/write_imagenet.py
    
    One significant difference is that we use a custom RGBImageField, which sets
    the length of the shortest-side, as opposed to longest-side, to keep consistent
    with our other datasets. We also set a max on the longest side, because some
    datasets (e.g., imagenet-21k), have some freakishly large images.
    
    Args:
        dataset: name of dataset (loaded by name from `datasets` folder)
        
    Example Use:
    python write_ffcv_dataset.py \
        --cfg.dataset=imagenet1k \
        --cfg.split=val \
        --cfg.data_dir=$SHARED_DATA_DIR/imagenet1k-line/imagenet1k-anime_style \
        --cfg.write_path=$SHARED_DATA_DIR/imagenet1k-line-ffcv/imagenet1k-anime_style_val_jpg_q100_s256_lmax512_crop.ffcv \
        --cfg.write_mode=jpg \
        --cfg.min_resolution=256 \
        --cfg.max_resolution=512 \
        --cfg.max_enforced_with=center_crop \
        --cfg.num_workers=16 \
        --cfg.chunk_size=100 \
        --cfg.shuffle_indices=1 \
        --cfg.jpeg_quality=100 \
        --cfg.compress_probability=1.0
'''
import os
import cv2
import torch

from torch.utils.data import Subset

from typing import Any, Callable, cast, Dict, List, Optional, Tuple

from argparse import ArgumentParser
from fastargs import Section, Param
from fastargs.validation import And, OneOf
from fastargs.decorators import param, section
from fastargs import get_current_config

from pathlib import Path

from pdb import set_trace

from .utils import datasets
from .utils import writers 

dataset_names = sorted(name for name in datasets.__dict__
                       if name.islower() and not name.startswith("__")
                       and callable(datasets.__dict__[name]))

writer_types = sorted(name for name in writers.__dict__
                      if name.islower() and not name.startswith("__")
                      and callable(writers.__dict__[name]))

Section('cfg', 'arguments to give the writer').params(
    dataset=Param(And(str, OneOf(dataset_names)), 'Which dataset to write', default='imagenet1k'),
    split=Param(And(str, OneOf(['train', 'val', 'test'])), 'Train or val set', required=True),
    data_dir=Param(str, 'Where to find the PyTorch dataset', required=True),
    write_path=Param(str, 'Where to write the new dataset', required=True),
    write_mode=Param(str, 'Mode: jpg, raw (uint8 pixel values), smart (JPEG compressing based on image size) or proportion (JPEG compress a random subset of the data with size specified by the `compress probability` argument)', required=False, default='jpg'),        
    smart_threshold=Param(int, 'When write_mode=’smart, will compress an image if it would take more than smart_threshold bytes to use RAW instead of jpeg.', default=0),
    min_resolution=Param(int, 'Length of shortest side of image', required=True),
    max_resolution=Param(int, 'Max length of the longest side', required=True),
    max_enforced_with=Param(And(str, OneOf(['center_crop', 'squeeze_resize'])), 'Whether we enforce by center_crop or squeeze_resize', default='center_crop'),
    num_workers=Param(int, 'Number of workers to use', default=16),
    chunk_size=Param(int, 'Chunk size for writing', default=100),
    shuffle_indices=Param(int, 'Shuffle order of the dataset', required=True),
    jpeg_quality=Param(float, 'Quality of jpeg images', default=100),
    subset=Param(int, 'How many images to use (-1 for all)', default=-1),
    compress_probability=Param(float, "compress probability; Ignored unless ``write_mode='proportion'``; in the latter case it is the probability with which image is JPEG-compressed", default=None),
    writer_type=Param(And(str,OneOf(writer_types)), 'Which writer to use', default='image_label')
)

@section('cfg')
@param('dataset')
@param('split')
@param('data_dir')
@param('subset')
def get_dataset(dataset, split, data_dir, subset):
    my_dataset = datasets.__dict__[dataset](root=data_dir, split=split)
    if subset > 0: my_dataset = Subset(my_dataset, range(subset))
    return my_dataset

@section('cfg')
@param('writer_type')
@param('write_path')
@param('min_resolution')
@param('max_resolution')
@param('max_enforced_with')
@param('jpeg_quality')
@param('write_mode')
@param('smart_threshold')
@param('compress_probability')
@param('num_workers')
def get_writer(writer_type, write_path, min_resolution, max_resolution, max_enforced_with,
               jpeg_quality, write_mode, smart_threshold, compress_probability,
               num_workers):
    
    if compress_probability > 0.0 and compress_probability < 1.0: 
        assert write_mode=='proportion', f"Write mode_must be `proportion` for compress_probability > 0, got {write_mode}"
    elif compress_probability == 0.0:
        assert write_mode=='raw', f"Write mode_must be `raw` for compress_probability==1.0, got {write_mode}"
    
    Path(write_path).parent.mkdir(parents=True, exist_ok=True)
    
    writer = writers.__dict__[writer_type](write_path, 
                                           write_mode=write_mode, 
                                           smart_threshold=smart_threshold,
                                           min_resolution=min_resolution,
                                           max_resolution=max_resolution,
                                           max_enforced_with=max_enforced_with,
                                           compress_probability=compress_probability,
                                           jpeg_quality=jpeg_quality)
    
    return writer
    
@section('cfg')
@param('chunk_size')
@param('shuffle_indices')
def main(chunk_size, shuffle_indices):
    
    my_dataset = get_dataset()
    print(my_dataset)
    
    writer = get_writer() 
    print(writer)
        
    writer.from_indexed_dataset(my_dataset, chunksize=chunk_size, shuffle_indices=shuffle_indices)

if __name__ == '__main__':
    config = get_current_config()
    parser = ArgumentParser()
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')
    config.summary()
    main()