import os
import importlib.util

import torch, torchvision

dependencies = ['torch', 'torchvision']

# ================================================
#  informative-drawings models
# ================================================

# add informative models module
hub_dir = torch.hub.get_dir()
repo_root = os.path.dirname(os.path.abspath(__file__))
module_file_path = os.path.join(repo_root, 'models', 'informative_drawings_models.py')
module_name = 'informative_drawings'
spec = importlib.util.spec_from_file_location(module_name, module_file_path)
models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(models)

def _load_generator(style, img_size=256):
  urls = dict(
      anime_style="https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/Project-Line/models/informative-drawings/anime_netG_A_latest-c686ced2.pth",
      contour_style="https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/Project-Line/models/informative-drawings/contour_netG_A_latest-e8b6ec6d.pth",
      opensketch_style="https://s3.us-east-1.wasabisys.com/visionlab-members/alvarez/Projects/Project-Line/models/informative-drawings/opensketch_netG_A_latest-30a53478.pth"
  )
  style_names = list(urls.keys())
  
  class ConvertToRGB:
      def __call__(self, img):
          return img.convert('RGB')
      def __repr__(self):
          return self.__class__.__name__ + '()'
      
  assert style in style_names, f"Expected style to be in {style_names}, got {style}"
  url = urls[style]
  print(f"==> Loading Generator from: {url}")

  net_G = models.Generator(input_nc=3, output_nc=1, n_residual_blocks=3)
  
  state_dict = torch.hub.load_state_dict_from_url(url, map_location='cpu')
  
  msg = net_G.load_state_dict(state_dict)
  print(msg)

  transforms_r = torchvision.transforms.Compose([
      ConvertToRGB(),
      torchvision.transforms.Resize(img_size, 3), # Image.BICUBIC = 3
      torchvision.transforms.ToTensor()
  ])

  return net_G, transforms_r

def anime_style(img_size=256):
  return _load_generator("anime_style", img_size=img_size)

def contour_style(img_size=256):
  return _load_generator("contour_style", img_size=img_size)

def opensketch_style(img_size=256):
  return _load_generator("opensketch_style", img_size=img_size)

# ================================================
#  alexnet models trained on informative drawings
# ================================================

# add informative models module
hub_dir = torch.hub.get_dir()
repo_root = os.path.dirname(os.path.abspath(__file__))
module_file_path = os.path.join(repo_root, 'models', 'alexnet_models.py')
module_name = 'alexnet_models'
spec = importlib.util.spec_from_file_location(module_name, module_file_path)
alexnet_models = importlib.util.module_from_spec(spec)
spec.loader.exec_module(alexnet_models)

def alexnet_rgb_stats_rgb_avg_adamw():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_adamw")

def alexnet_anime_stats_line_stdonly_adamw():
  return alexnet_models.load_model("alexnet_anime_stats_line_stdonly_adamw")

def alexnet_rgb_stats_rgb_avg_adamw_div10():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_adamw_div10")

def alexnet_rgb_stats_rgb_avg_adamw_div2():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_adamw_div2")

def alexnet_rgb_stats_rgb_avg_sgd_lr0_01():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_sgd_lr0.01")

def alexnet_rgb_stats_rgb_avg_sgd_lr0_001():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_sgd_lr0.001")

def alexnet_rgb_stats_rgb_avg_sgd_lr0_005():
  return alexnet_models.load_model("alexnet_rgb_stats_rgb_avg_sgd_lr0.005")

def alexnet_anime_stats_line_stdonly_sgd_lr0_005():
  return alexnet_models.load_model("alexnet_anime_stats_line_stdonly_sgd_lr0.005")

def alexnet_anime_stats_line_stdonly_sgd_lr0_01():
  return alexnet_models.load_model("alexnet_anime_stats_line_stdonly_sgd_lr0.01")

def alexnet_anime_stats_rgb_avg_sgd_lr0_005():
  return alexnet_models.load_model("alexnet_anime_stats_rgb_avg_sgd_lr0.005")

def print_top1_acc():
  return alexnet_models.print_top1_acc
  
def show_conv1():
  return alexnet_models.show_conv1
  
  
  

