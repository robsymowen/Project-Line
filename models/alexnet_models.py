import torch
import torch.nn as nn
import numpy as np
from torchvision import transforms
import torchvision.models as tv_models
from torch.hub import load_state_dict_from_url

class InvertTensorImageColors(nn.Module):
    def __init__(self):
        super(InvertTensorImageColors, self).__init__()

    def __call__(self, img):
        # Check if the image is in uint8 format
        if img.dtype == torch.uint8 or img.dtype == np.uint8:
            return 255 - img
        # assume the image is in float format
        else:
            return 1 - img

checkpoint_urls = {
    "alexnet_rgb_stats_rgb_avg_adamw": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_144920/final_weights-5e38ba62c7.pth',
    "alexnet_anime_stats_line_stdonly_adamw": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_145131/final_weights-84ec35d2fc.pth',

    "alexnet_rgb_stats_rgb_avg_adamw_div10": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_151752/final_weights-9a6765a655.pth',
    "alexnet_rgb_stats_rgb_avg_adamw_div2": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_165145/final_weights-280e078e07.pth',

    "alexnet_rgb_stats_rgb_avg_sgd_lr0.01": 'https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_181051/final_weights-76c7036df6.pth',
    "alexnet_rgb_stats_rgb_avg_sgd_lr0.001": 'https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183236/final_weights-12e535cb08.pth',
    "alexnet_rgb_stats_rgb_avg_sgd_lr0.005": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_rgb/alexnet/20231208_183615/final_weights-d0058cb517.pth',

    "alexnet_anime_stats_line_stdonly_sgd_lr0.005": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_201507/final_weights-591181e55e.pth',
    "alexnet_anime_stats_line_stdonly_sgd_lr0.01": "https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231209_043147/final_weights-f086d90f0a.pth",
    "alexnet_anime_stats_rgb_avg_sgd_lr0.005": 'https://visionlab-members.s3.wasabisys.com/alvarez/Projects/Project-Line/models/in1k_anime_style/alexnet/20231208_184720/final_weights-4c9a4fe9c9.pth',
}

IMAGE_STATS = dict(
    rgb=dict(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    ),
    rgb_avg=dict(
        mean=[0.449, 0.449, 0.449],
        std=[0.226, 0.226, 0.226]
    ),
    rgb_avg_stdonly=dict(
        mean=[0.0, 0.0, 0.0],
        std=[0.226, 0.226, 0.226]
    ),
    line_stdonly=dict(
        mean=[0.0, 0.0, 0.0],
        std=[0.181, 0.181, 0.181]
    ),
    no_op=dict(
        mean=[0.0, 0.0, 0.0],
        std=[1.0, 1.0, 1.0]
    )
)

def get_image_stats(model_name):
  if 'rgb_avg_stdonly' in model_name:
    stats = IMAGE_STATS['rgb_avg_stdonly']
  elif 'rgb_avg' in model_name:
    stats = IMAGE_STATS['rgb_avg']
  elif 'line_stdonly' in model_name:
    stats = IMAGE_STATS['line_stdonly']
  else:
    stats = IMAGE_STATS['no_op']

  return stats['mean'], stats['std']

def get_transform(model_name, img_size=256, crop_size=224):

  mean, std = get_image_stats(model_name)

  if model_name.startswith("alexnet_rgb"):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])
  elif model_name.startswith("alexnet_anime"):
    transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.CenterCrop(crop_size),
        transforms.ToTensor(),
        InvertTensorImageColors(),
        transforms.Normalize(mean=mean, std=std)
    ])

  return transform

def print_top1_acc(checkpoint_url, cache_dir='./logs'):

  train_log_url = checkpoint_url.replace("final_weights-", "log_train-")
  train_log_url = train_log_url.replace(".pth.tar", ".txt").replace(".pth", ".txt")
  train_df = load_remote_log(train_log_url, cache_dir=cache_dir)
  train_top1 = train_df.iloc[-1].top1 * 100

  val_log_url = checkpoint_url.replace("final_weights-", "log_val-")
  val_log_url = val_log_url.replace(".pth.tar", ".txt").replace(".pth", ".txt")
  val_df = load_remote_log(val_log_url, cache_dir=cache_dir)
  val_top1 = val_df.iloc[-1].top1 * 100

  print(f"==> train top1={train_top1:3.2f}%, val top1={val_top1:3.2f}%")

def print_model_accuracy(model_name):
  checkpoint_url = checkpoint_urls[model_name]
  print_top1_acc(checkpoint_url)

def load_model(model_name):
  checkpoint_url = checkpoint_urls[model_name]
  model = tv_models.alexnet()
  checkpoint = load_state_dict_from_url(checkpoint_url, check_hash=True, map_location='cpu')
  print(checkpoint.keys())
  if 'state_dict' in checkpoint:
    print(f"alexnet trained for {checkpoint['epoch']} epochs")
    state_dict = {k.replace("module.",""):v for k,v in checkpoint['state_dict'].items()}
  else:
    state_dict = checkpoint
  msg = model.load_state_dict(state_dict)
  print(msg)
  print_top1_acc(checkpoint_url)

  transform = get_transform(model_name)

  return model, transform

def show_conv1(model, nrow=16):
    # find first conv
    first_conv = None
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            first_conv = m
            break

    if first_conv is not None:
        kernels = first_conv.weight.detach().clone().cpu()
        kernels = kernels - kernels.min()
        kernels = kernels / kernels.max()
        img = make_grid(kernels, nrow=nrow)
        plt.imshow(img.permute(1, 2, 0))
    else:
        print("failed to find first conv layer")
