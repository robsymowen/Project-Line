'''
    Adapted from https://github.com/facebookresearch/FFCV-SSL/blob/main/examples/train_ssl.py

    This version of the training script is modified to work only for supervised training,
    and to have separate val and test datasets.

    This script is intended to be run with submitit on a slurm cluster with
    a bunch of environment variables set for remote storage of model
    results.
    
    See train.sh for example usage.
'''
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

# Adapted from https://github.com/libffcv/ffcv-imagenet to support SSL
import sys
import torch
import torch as ch
from torch.cuda.amp import GradScaler
from torch.cuda.amp import autocast
from torch import nn
import torch.nn.functional as F
import torch.distributed as dist
ch.backends.cudnn.benchmark = True
ch.autograd.profiler.emit_nvtx(False)
ch.autograd.profiler.profile(False)
import torchvision.transforms as transforms
import torchmetrics
import numpy as np
from tqdm import tqdm
import subprocess
import os
import time
import json
import uuid
import ffcv
import submitit
from uuid import uuid4
from typing import List
from pathlib import Path
from argparse import ArgumentParser

from fastargs import get_current_config, set_current_config
from fastargs.decorators import param
from fastargs import Param, Section
from fastargs.validation import And, OneOf

from ffcv.pipeline.operation import Operation
from ffcv.loader import Loader, OrderOption
from ffcv.transforms import ToTensor, ToDevice, Squeeze, NormalizeImage, \
    RandomHorizontalFlip, ToTorchImage
from ffcv.fields.rgb_image import CenterCropRGBImageDecoder
from ffcv.fields.basics import IntDecoder
from ffcv.fields import IntField, RGBImageField

from utils.utils import LARS, cosine_scheduler, learning_schedule
from utils.remote_storage import RemoteStorage
from utils.transforms import InvertTensorImageColors, InvertImageColorsFFCV

from torchvision import models

from pdb import set_trace

import inspect

DEFAULT_CROP_RATIO = 224/256

IMAGE_STATS = dict(
    imagenet_rgb=dict(
        mean=np.array([0.485, 0.456, 0.406]) * 255,
        std=np.array([0.229, 0.224, 0.225]) * 255
    ),
    imagenet_rgb_avg=dict(
        mean=np.array([0.449, 0.449, 0.449]) * 255,
        std=np.array([0.226, 0.226, 0.226]) * 255
    ),
    imagenet_rgb_avg_stdonly=dict(
        mean=np.array([0.0, 0.0, 0.0]) * 255,
        std=np.array([0.226, 0.226, 0.226]) * 255
    ),
    imagenet_line_stdonly=dict(
        mean=np.array([0.0, 0.0, 0.0]) * 255,
        std=np.array([0.181, 0.181, 0.181]) * 255
    ),
    no_op=dict(
        mean=np.array([0.0, 0.0, 0.0]) * 255,
        std=np.array([1.0, 1.0, 1.0]) * 255
    )
)

image_stats_options = list(IMAGE_STATS.keys())

Section('model', 'model details').params(
    arch=Param(str, 'model to use', default='alexnet'),  
)

Section('resolution', 'resolution scheduling').params(
    min_res=Param(int, 'the minimum (starting) resolution', default=160),
    max_res=Param(int, 'the maximum (starting) resolution', default=192),
    start_ramp=Param(int, 'when to start interpolating resolution', default=65),
    end_ramp=Param(int, 'when to stop interpolating resolution', default=76),
)

Section('data', 'data related stuff').params(
    train_dataset=Param(str, '.ffcv file to use for training', required=True),
    val_dataset=Param(str, '.ffcv file to use for validation', required=True),
    test_dataset=Param(str, '.ffcv file to use for testing', default=""),
    num_classes=Param(int, 'The number of image classes', default=1000),
    num_workers=Param(int, 'The number of workers', default=10),
    in_memory=Param(int, 'does the dataset fit in memory? (1/0)', required=True),    
    image_stats_rgb=Param(And(str, OneOf(image_stats_options)), 
                          'imagenet stats to use on rgb images', default='imagenet_rgb_avg'),
    image_stats_line=Param(And(str, OneOf(image_stats_options)), 
                           'imagenet stats to use on line drawing images', default='imagenet_line_stdonly'),
)

Section('logging', 'how to log stuff').params(
    incubator=Param(str, 'literally just the name of this file', default='train_ssl.py'),
    folder=Param(str, 'log location', required=True),
    bucket_name=Param(str, 's3 bucket storage location', default=''),
    bucket_subfolder=Param(str, 's3 subfolder for storing logs and weights', default=''),
    log_level=Param(int, '0 if only at end 1 otherwise', default=2),
    checkpoint_freq=Param(int, 'When saving checkpoints', default=5)
)

Section('validation', 'Validation parameters stuff').params(
    batch_size=Param(int, 'The batch size for validation', default=256),
    resolution=Param(int, 'final resized validation image size', default=224),
)

Section('training', 'training hyper param stuff').params(
    eval_only=Param(int, 'eval only?', default=0),
    eval_freq=Param(float, 'number of epochs', default=1),
    batch_size=Param(int, 'The batch size', default=512),
    num_crops=Param(int, 'number of crops?', default=1),
    optimizer=Param(And(str, OneOf(['sgd', 'adamw', 'lars'])), 'The optimizer', default='adamw'),
    momentum=Param(float, 'SGD momentum', default=0.9),
    weight_decay=Param(float, 'weight decay', default=0.0001),
    epochs=Param(int, 'number of epochs', default=100),
    base_lr=Param(float, 'number of epochs', default=0.0001),
    end_lr_ratio=Param(float, 'number of epochs', default=0.001),
    label_smoothing=Param(float, 'label smoothing parameter', default=0),
    distributed=Param(int, 'is distributed?', default=0),
    clip_grad=Param(float, 'sign the weights of last residual block', default=0),
    stop_early_epoch=Param(int, 'For debugging, stop afer this many epochs (ignored if less than 1)', default=0),    
)

Section('dist', 'distributed training options').params(
    use_submitit=Param(int, 'enable submitit', default=0),
    world_size=Param(int, 'number gpus', default=1),
    ngpus=Param(int, 'number of gpus per node', default=4),
    nodes=Param(int, 'number of nodes', default=1),
    comment=Param(str, 'comment for slurm', default=''),
    timeout=Param(int, 'timeout', default=2800),
    partition=Param(str, 'partition', default="kempner"),
    account=Param(str, 'account', default="kempner_Alvarez_Lab"),
    address=Param(str, 'address', default='localhost'),
    port=Param(str, 'port', default='58492'),
    exclude_nodes=Param(str, 'nodes to exclude when submitting (iff there are issues)', default='')
)




################################
##### Some Miscs functions #####
################################

def has_argument(func, arg_name):
    """
    Check if a function or method has a specific argument.
    
    :param func: The function or method to inspect.
    :param arg_name: Name of the argument to check for.
    :return: Boolean indicating whether the argument is in the function's signature.
    """
    signature = inspect.signature(func)
    return arg_name in signature.parameters

def get_shared_folder() -> Path:
    user = os.getenv("USER")
    
    # Get the full path to the current script.
    current_script_path = Path(__file__).resolve()

    # Get the directory containing the current script.
    current_script_directory = current_script_path.parent

    # Construct the path to the 'checkpoint' directory in the same location as the current script.
    checkpoint_path = current_script_directory / "checkpoint"
    print("checkpoint_path: ", checkpoint_path, Path(checkpoint_path).is_dir())
    
    if Path(checkpoint_path).is_dir():
        p = Path(f"{checkpoint_path}/{user}/experiments")
        p.mkdir(exist_ok=True, parents=True)
        return p
    raise RuntimeError("No shared folder available")

def get_init_file():
    # Init file must not exist, but it's parent dir must exist.
    os.makedirs(str(get_shared_folder()), exist_ok=True)
    init_file = get_shared_folder() / f"{uuid.uuid4().hex}_init"
    if init_file.exists():
        os.remove(str(init_file))
    return init_file

def exclude_bias_and_norm(p):
    return p.ndim == 1

def off_diagonal(x):
    n, m = x.shape
    assert n == m
    return x.flatten()[:-1].view(n - 1, n + 1)[:, 1:].flatten()

def gather_center(x):
    x = batch_all_gather(x)
    x = x - x.mean(dim=0)
    return x

def batch_all_gather(x):
    x_list = GatherLayer.apply(x.contiguous())
    return ch.cat(x_list, dim=0)

class GatherLayer(ch.autograd.Function):
    """
    Gather tensors from all process and support backward propagation
    for the gradients across processes.
    """

    @staticmethod
    def forward(ctx, x):
        output = [ch.zeros_like(x) for _ in range(dist.get_world_size())]
        dist.all_gather(output, x)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        all_gradients = ch.stack(grads)
        dist.all_reduce(all_gradients)
        return all_gradients[dist.get_rank()]

################################
##### Main Trainer ############
################################

class ImageNetTrainer:
    @param('training.distributed')
    @param('training.batch_size')
    @param('training.label_smoothing')
    @param('training.epochs')
    @param('data.train_dataset')
    @param('data.val_dataset')
    @param('data.test_dataset')
    @param('data.num_classes')
    def __init__(self, gpu, ngpus_per_node, world_size, dist_url, distributed, batch_size, label_smoothing, 
                 epochs, train_dataset, val_dataset, test_dataset, num_classes):
        
        self.all_params = get_current_config()
        ch.cuda.set_device(gpu)
        self.gpu = gpu
        self.rank = self.gpu + int(os.getenv("SLURM_NODEID", "0")) * ngpus_per_node
        self.world_size = world_size
        self.dist_url = dist_url
        self.batch_size = batch_size
        self.uid = str(uuid4())
        if distributed:
            self.setup_distributed()
        self.start_epoch = 0
        print("CURRENT GPU:", self.gpu)
        
        # Create DataLoaders
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset

        self.index_labels = 1
        self.train_loader = self.create_train_loader(train_dataset)
        self.num_train_examples = self.train_loader.indices.shape[0]
        self.num_classes = num_classes

        self.val_loader = self.create_val_loader(val_dataset)
        self.test_loader = self.create_val_loader(test_dataset)
        print("NUM TRAINING EXAMPLES:", self.num_train_examples)
        
        # Create SSL model, scaler, loss, and optimizer
        self.model, self.scaler = self.create_model_and_scaler()
        print(self.model)
        self.classif_loss = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.create_optimizer()

        # Initialize Logger
        self.initialize_logger()
        self.initialize_remote_logger()
        
        # Load models if checkpoint exists
        self.load_checkpoint()

    # resolution tools
    @param('resolution.min_res')
    @param('resolution.max_res')
    @param('resolution.end_ramp')
    @param('resolution.start_ramp')
    def get_resolution(self, epoch, min_res, max_res, end_ramp, start_ramp):
        assert min_res <= max_res

        if epoch <= start_ramp:
            return min_res

        if epoch >= end_ramp:
            return max_res

        # otherwise, linearly interpolate to the nearest multiple of 32
        interp = np.interp([epoch], [start_ramp, end_ramp], [min_res, max_res])
        final_res = int(np.round(interp[0] / 32)) * 32
        return final_res

    def setup_distributed(self):
        dist.init_process_group("nccl", init_method=self.dist_url, rank=self.rank, world_size=self.world_size)

    def cleanup_distributed(self):
        dist.destroy_process_group()

    @param('training.momentum')
    @param('training.optimizer')
    @param('training.weight_decay')
    @param('training.label_smoothing')
    def create_optimizer(self, momentum, optimizer, weight_decay,
                         label_smoothing):
        assert optimizer == 'sgd' or optimizer == 'adamw' or optimizer == "lars"

        # Only do weight decay on non-batchnorm parameters
        all_params = list(self.model.named_parameters())
        bn_params = [v for k, v in all_params if ('bn' in k)]
        other_params = [v for k, v in all_params if not ('bn' in k)]
        param_groups = [{
            'params': bn_params,
            'weight_decay': 0.
        }, {
            'params': other_params,
            'weight_decay': weight_decay
        }]
        if optimizer == 'sgd':
            self.optimizer = ch.optim.SGD(param_groups, lr=1, momentum=momentum)
        elif optimizer == 'adamw':
            # We use a big eps value to avoid instabilities with fp16 training
            self.optimizer = ch.optim.AdamW(param_groups, lr=1e-4)
        elif optimizer == "lars":
            self.optimizer = LARS(param_groups)  # to use with convnet and large batches
        self.optim_name = optimizer

    @param('data.num_workers')
    @param('training.batch_size')
    @param('data.image_stats_rgb')
    @param('data.image_stats_line')
    @param('training.distributed')
    @param('data.in_memory')
    def create_train_loader(self, train_dataset, num_workers, batch_size,
                            image_stats_rgb, image_stats_line,
                            distributed, in_memory):
        this_device = f'cuda:{self.gpu}'
        train_path = Path(train_dataset)
        assert train_path.is_file()

        is_line_drawing_dataset = '_style_' in train_dataset
        
        if is_line_drawing_dataset:
            mean = IMAGE_STATS[image_stats_line]['mean']
            std = IMAGE_STATS[image_stats_line]['std']
        else:
            mean = IMAGE_STATS[image_stats_rgb]['mean']
            std = IMAGE_STATS[image_stats_rgb]['std']
        
        # image pipeline
        self.decoder = ffcv.transforms.RandomResizedCrop((224, 224))
        image_pipeline: List[Operation] = [
            self.decoder,
            RandomHorizontalFlip(),
            ffcv.transforms.RandomGrayscale(0.2),
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
        ]
        
        if is_line_drawing_dataset:
            image_pipeline += [InvertImageColorsFFCV()]
            
        image_pipeline += [NormalizeImage(mean, std, np.float16)]
        
        label_pipeline: List[Operation] = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device), non_blocking=True)
        ]

        pipelines={
            'image': image_pipeline,
            'label': label_pipeline,
            'index': None,    # drop index
            'rel_path': None, # drop rel_path
        }

        order = OrderOption.RANDOM if distributed else OrderOption.QUASI_RANDOM
        custom_field_mapper={}

        # Create data loader
        loader = ffcv.Loader(train_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        os_cache=in_memory,
                        drop_last=True,
                        pipelines=pipelines,
                        distributed=distributed,
                        custom_fields={
                            'image': RGBImageField,
                            'label': IntField,
                        },
                        custom_field_mapper=custom_field_mapper)

        return loader

    @param('data.num_workers')
    @param('validation.batch_size')
    @param('data.image_stats_rgb')
    @param('data.image_stats_line')
    @param('validation.resolution')
    @param('training.distributed')
    def create_val_loader(self, val_dataset, num_workers, batch_size,
                          image_stats_rgb, image_stats_line,
                          resolution, distributed):
        this_device = f'cuda:{self.gpu}'
        val_path = Path(val_dataset)
        assert val_path.is_file()
        
        is_line_drawing_dataset = '_style_' in val_dataset
        
        if is_line_drawing_dataset:
            mean = IMAGE_STATS[image_stats_line]['mean']
            std = IMAGE_STATS[image_stats_line]['std']
        else:
            mean = IMAGE_STATS[image_stats_rgb]['mean']
            std = IMAGE_STATS[image_stats_rgb]['std']
        
        res_tuple = (resolution, resolution)
        cropper = CenterCropRGBImageDecoder(res_tuple, ratio=DEFAULT_CROP_RATIO)
        image_pipeline = [
            cropper,
            ToTensor(),
            ToDevice(ch.device(this_device), non_blocking=True),
            ToTorchImage(),
        ]
        
        if is_line_drawing_dataset:
            image_pipeline += [InvertImageColorsFFCV()]
            
        image_pipeline += [NormalizeImage(mean, std, np.float16)]
        
        label_pipeline = [
            IntDecoder(),
            ToTensor(),
            Squeeze(),
            ToDevice(ch.device(this_device),
            non_blocking=True)
        ]

        order = OrderOption.SEQUENTIAL

        loader = Loader(val_dataset,
                        batch_size=batch_size,
                        num_workers=num_workers,
                        order=order,
                        drop_last=False,
                        pipelines={
                            'image': image_pipeline,
                            'label': label_pipeline,
                            'index': None,    # drop index
                            'rel_path': None, # drop rel_path
                        },
                        custom_fields={
                            'image': RGBImageField,
                            'label': IntField,
                        },
                        distributed=distributed)
        return loader

    @param('training.epochs')
    @param('training.stop_early_epoch')
    @param('logging.log_level')
    def train(self, epochs, stop_early_epoch, log_level):
        # We scale the number of max steps w.t the number of examples in the training set
        self.max_steps = epochs * self.num_train_examples // (self.batch_size * self.world_size)
        for epoch in range(self.start_epoch, epochs):
            res = self.get_resolution(epoch)
            self.res = res
            self.decoder.output_size = (res, res)
            train_loss, stats = self.train_loop(epoch)
            if log_level > 0:
                extra_dict = {
                    'epoch': epoch,
                    'resolution': res,
                    'lr': self.optimizer.param_groups[0]['lr'],
                }
                self.log(dict(stats, **extra_dict, phase='train'), 'train')
            self.eval_and_log(dict(**extra_dict, phase='val'))
            self.test_and_log(dict(**extra_dict, phase='test'))
            # Run checkpointing
            self.checkpoint(epoch + 1)
            
            # debugging
            if stop_early_epoch>0 and (epoch+1)>=stop_early_epoch: break
            
        if self.rank == 0:
            self.save_checkpoint(epoch + 1)
            ch.save(dict(
                epoch=epoch,
                state_dict=self.model.state_dict(),
                params=self.params_dict()
            ), self.log_folder / 'final_weights.pth')
            
            if self.remote_store is not None:
                self.remote_store.upload_final_results()
            
    def params_dict(self):
        params = {
            '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
        }
        return params
    
    def eval_and_log(self, extra_dict={}):
        stats = self.val_loop(self.val_loader, self.val_meters)
        self.log(dict(stats, **extra_dict), 'val')
        return stats

    def test_and_log(self, extra_dict={}):
        stats = self.val_loop(self.test_loader, self.test_meters)
        self.log(dict(stats, **extra_dict), 'test')
        return stats
    
    @param('model.arch')
    @param('data.num_classes')
    def create_model_and_scaler(self, arch, num_classes):
        scaler = GradScaler()
        model = models.__dict__[arch](num_classes=num_classes)
        model = model.to(memory_format=ch.channels_last)
        model = model.to(self.gpu)

        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        model = ch.nn.parallel.DistributedDataParallel(model, device_ids=[self.gpu])
              
        return model, scaler

    def load_checkpoint(self):
        if (self.log_folder / "model.pth").is_file():
            if self.rank == 0:
                print("resuming from checkpoint")
            ckpt = ch.load(self.log_folder / "model.pth", map_location="cpu")
            self.start_epoch = ckpt["epoch"]
            self.model.load_state_dict(ckpt["model"])
            self.optimizer.load_state_dict(ckpt["optimizer"])

    @param('logging.checkpoint_freq')
    def checkpoint(self, epoch, checkpoint_freq):
        if self.rank != 0 or epoch % checkpoint_freq != 0:
            return
        self.save_checkpoint(epoch)
    
    def save_checkpoint(self, epoch):
        params = self.params_dict()
        state = dict(
            epoch=epoch, 
            model=self.model.state_dict(), 
            optimizer=self.optimizer.state_dict(),
            params=params
        )
        save_name = f"model.pth"
        ch.save(state, self.log_folder / save_name)
    
    @param('logging.log_level')
    @param('training.base_lr')
    @param('training.end_lr_ratio')
    def train_loop(self, epoch, log_level, base_lr, end_lr_ratio):
        """
            Main training loop for training with SSL or Supervised criterion.
        """
        model = self.model
        model.train()
        losses = []

        iterator = tqdm(self.train_loader)
        for ix, loaders in enumerate(iterator, start=epoch * len(self.train_loader)):
            self.optimizer.zero_grad(set_to_none=True)

            # Get lr
            lr = learning_schedule(
                global_step=ix,
                batch_size=self.batch_size * self.world_size,
                base_lr=base_lr,
                end_lr_ratio=end_lr_ratio,
                total_steps=self.max_steps,
                warmup_steps=10 * self.num_train_examples // (self.batch_size * self.world_size),
            )
            for g in self.optimizer.param_groups:
                 g["lr"] = lr

            # Get data
            images = loaders[0]
            labels = loaders[1]
            
            #print("images device:", images.dtype, images.device)
            #print("model device:", next(model.parameters()).device)
            #print("bias device:", model.module.features[0].bias.device)
        
            # forward pass
            with autocast():
                outputs = model(images)

            # compute loss 
            loss = self.classif_loss(outputs, labels)
            
            # backward + optimizer step
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            # update meters
            self.train_meters['loss'](loss.detach())
            self.train_meters['top1'](outputs.detach(), labels)
            self.train_meters['top5'](outputs.detach(), labels)

            # Logging
            if log_level > 0:
                self.train_meters['loss'](loss.detach())
                self.train_meters['top1'](outputs.detach(), labels)
                self.train_meters['top5'](outputs.detach(), labels)

                losses.append(loss.detach())
                group_lrs = []
                for _, group in enumerate(self.optimizer.param_groups):
                    group_lrs.append(f'{group["lr"]:.5f}')

                names = ['ep', 'iter', 'shape', 'lrs']
                values = [epoch, ix, tuple(images.shape), group_lrs]
                if log_level > 1:
                    names += ['loss']
                    values += [f'{loss.item():.3f}']

                msg = ', '.join(f'{n}={v}' for n, v in zip(names, values))
                iterator.set_description(msg)

        # Return epoch's log
        if log_level > 0:
            self.train_meters['time'](ch.tensor(iterator.format_dict["elapsed"]))
            loss = ch.stack(losses).mean().cpu()
            assert not ch.isnan(loss), 'Loss is NaN!'
            stats = {k: m.compute().item() for k, m in self.train_meters.items()}
            [meter.reset() for meter in self.train_meters.values()]
            return loss.item(), stats

    @torch.no_grad()
    def val_loop(self, dataloader, meters):
        model = self.model
        model.eval()
        with autocast():
            for images, target in tqdm(dataloader):
                outputs = model(images)
                loss = self.classif_loss(outputs, target)
                meters['loss'](loss.detach())
                meters['top1'](outputs.detach(), target)
                meters['top5'](outputs.detach(), target)
                            
        stats = {k: m.compute().item() for k, m in meters.items()}
        [meter.reset() for meter in meters.values()]
        return stats

    @param('logging.folder')
    def initialize_logger(self, folder):
        folder = folder.replace("//","/")
        self.train_meters = {
            'loss': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
            'top1': torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu),
            'top5': torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu),
            'time': torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu),
        }

        self.val_meters = {}
        self.val_meters['loss'] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
        self.val_meters['top1'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
        self.val_meters['top5'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)        

        self.test_meters = {}
        self.test_meters['loss'] = torchmetrics.MeanMetric(compute_on_step=False).to(self.gpu)
        self.test_meters['top1'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=1, compute_on_step=False).to(self.gpu)
        self.test_meters['top5'] = torchmetrics.Accuracy('multiclass', num_classes=self.num_classes, top_k=5, compute_on_step=False).to(self.gpu)        

        if self.rank == 0:
            if Path(folder + 'final_weights.pth').is_file():
                self.uid = ""
                folder = Path(folder)
            else:
                folder = Path(folder)
            self.log_folder = folder
            self.start_time = time.time()

            print(f'=> Logging in {self.log_folder}')
            Path(self.log_folder).mkdir(parents=True, exist_ok=True)

            params = {
                '.'.join(k): self.all_params[k] for k in self.all_params.entries.keys()
            }
            
            with open(folder / 'params.json', 'w+') as handle:
                json.dump(params, handle)
        self.log_folder = Path(folder)
    
    @param('logging.folder')
    @param('logging.bucket_name')
    @param('logging.bucket_subfolder')
    def initialize_remote_logger(self, folder, bucket_name, bucket_subfolder):
        folder = folder.replace("//","/")
        if bucket_name == '' or bucket_subfolder == '': 
            self.remote_store = None
        else:
            self.remote_store = RemoteStorage(folder, bucket_name, bucket_subfolder)
            # next two steps will test whether user has write permissions to local and remote
            self.remote_store.init_logs(verbose=True)
            self.remote_store.upload_logs(verbose=True)            
            print(f'=> Remote Storage in {self.remote_store.bucket_path}')                
                
    def log(self, content, phase):        
        print(f'=> Log (rank={self.rank}): {content}')
        if self.rank != 0: return        
        cur_time = time.time()
        name_file = f'log_{phase}.txt'
        with open(self.log_folder / name_file, 'a+') as fd:
            fd.write(json.dumps({
                'timestamp': cur_time,
                'relative_time': cur_time - self.start_time,
                **content
            }) + '\n')
            fd.flush()
        
        # stream results to remote storage for live monitoring
        if self.remote_store is not None:
            self.remote_store.upload_logs()

    @classmethod
    @param('training.distributed')
    @param('dist.world_size')
    @param('dist.port')
    def launch_from_args(cls, distributed, world_size, port):
        if distributed:
            ngpus_per_node = ch.cuda.device_count()
            world_size = int(os.getenv("SLURM_NNODES", "1")) * ngpus_per_node
            if "SLURM_JOB_NODELIST" in os.environ:
                cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
                host_name = subprocess.check_output(cmd).decode().splitlines()[0]
                dist_url = f"tcp://{host_name}:"+port
            else:
                dist_url = "tcp://localhost:"+port
            ch.multiprocessing.spawn(cls._exec_wrapper, nprocs=ngpus_per_node, join=True, args=(None, ngpus_per_node, world_size, dist_url))
        else:
            cls.exec(0)

    @classmethod
    def _exec_wrapper(cls, *args, **kwargs):
        if args[1] is not None:
            set_current_config(args[1])
        make_config(quiet=True)
        cls.exec(*args, **kwargs)

    @classmethod
    @param('training.distributed')
    @param('training.eval_only')
    def exec(cls, gpu, config, ngpus_per_node, world_size, dist_url, distributed, eval_only):
        trainer = cls(gpu=gpu, ngpus_per_node=ngpus_per_node, world_size=world_size, dist_url=dist_url)
        if eval_only:
            trainer.eval_and_log()
        else:
            trainer.train()

        if distributed:
            trainer.cleanup_distributed()

class Trainer(object):
    def __init__(self, config, num_gpus_per_node, dump_path, dist_url, port):
        self.num_gpus_per_node = num_gpus_per_node
        self.dump_path = dump_path
        self.dist_url = dist_url
        self.config = config
        self.port = port

    def __call__(self):
        self._setup_gpu_args()

    def checkpoint(self):
        self.dist_url = get_init_file().as_uri()
        print("Requeuing ")
        empty_trainer = type(self)(self.config, self.num_gpus_per_node, self.dump_path, self.dist_url, self.port)
        return submitit.helpers.DelayedSubmission(empty_trainer)

    def _setup_gpu_args(self):
        from pathlib import Path
        job_env = submitit.JobEnvironment()
        self.dump_path = Path(str(self.dump_path).replace("%j", str(job_env.job_id)))
        gpu = job_env.local_rank
        world_size = job_env.num_tasks
        if "SLURM_JOB_NODELIST" in os.environ:
            cmd = ["scontrol", "show", "hostnames", os.getenv("SLURM_JOB_NODELIST")]
            host_name = subprocess.check_output(cmd).decode().splitlines()[0]
            dist_url = f"tcp://{host_name}:"+self.port
        else:
            dist_url = "tcp://localhost:"+self.port
        print(f"Process group: {job_env.num_tasks} tasks, rank: {job_env.global_rank}")
        ImageNetTrainer._exec_wrapper(gpu, config, self.num_gpus_per_node, world_size, dist_url)

# Running
def make_config(quiet=False):
    config = get_current_config()
    parser = ArgumentParser(description='Fast SSL training')
    parser.add_argument("folder", type=str)
    config.augment_argparse(parser)
    config.collect_argparse_args(parser)
    config.validate(mode='stderr')

    if not quiet:
        config.summary()
    return config

@param('logging.folder')
@param('logging.bucket_name')
@param('logging.bucket_subfolder')
@param('dist.ngpus')
@param('dist.nodes')
@param('dist.timeout')
@param('dist.partition')
@param('dist.account')
@param('dist.comment')
@param('dist.port')
@param('dist.exclude_nodes')
def run_submitit(config, folder, bucket_name, bucket_subfolder, ngpus, nodes,  timeout, partition, account, comment, port, exclude_nodes):
    folder = folder.replace("//","/")
    Path(folder).mkdir(parents=True, exist_ok=True)
    executor = submitit.AutoExecutor(folder=folder, slurm_max_num_timeout=30)

    num_gpus_per_node = ngpus
    nodes = nodes
    timeout_min = timeout

    # Cluster specifics: To update accordingly to your cluster
    kwargs = {}
    kwargs['slurm_comment'] = comment
    executor.update_parameters(
        mem_gb=220 * num_gpus_per_node,
        gpus_per_node=num_gpus_per_node,
        tasks_per_node=num_gpus_per_node, 
        cpus_per_task=16,
        nodes=nodes,
        timeout_min=timeout_min,
        slurm_partition=partition,
        slurm_account=account,
        slurm_signal_delay_s=120,
        slurm_exclude=exclude_nodes,
        **kwargs
    )

    executor.update_parameters(name="lines")

    dist_url = get_init_file().as_uri()

    trainer = Trainer(config, num_gpus_per_node, folder, dist_url, port)
    job = executor.submit(trainer)

    print(f"Submitted job_id: {job.job_id}")
    print(f"Logs and checkpoints will be saved at local_dir: {folder}")
    
    if bucket_name is not None and bucket_subfolder is not None:
        url = f"{bucket_name}.s3.wasabisys.com/{bucket_subfolder}".replace("//","/")
        base_url = f"https://{url}";
        train_url = f"{base_url}/log_train.txt";
        val_url = f"{base_url}/log_val.txt";
        test_url = f"{base_url}/log_test.txt";
        print(f"Remote Storage Location: s3://{bucket_name}/{bucket_subfolder}")
        print(f"train_log = {train_url}")
        print(f"val_log = {val_url}")
        print(f"test_url = {test_url}")

@param('dist.use_submitit')
def main(config, use_submitit):
    if use_submitit:
        run_submitit(config)
    else:
        ImageNetTrainer.launch_from_args()

if __name__ == "__main__":
    config = make_config()
    main(config)
