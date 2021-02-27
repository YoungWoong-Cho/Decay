# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture

import argparse
import random
import time

import os
from PIL import Image
import torch.backends.cudnn as cudnn
import torch.utils.data.distributed
import torchvision.transforms as transforms
from torchvision.utils import save_image
from shutil import copyfile

from models import Generator

parser = argparse.ArgumentParser(description="This is a pytorch implementation of CycleGAN. Please refer to the following arguments.")
parser.add_argument("--data_root", type=str, default="./test", help="Root directory to the test dataset. Default: ./test")
parser.add_argument("--model", type=str, default="weights/fruit2rotten/G_A2B.pth", help="Generator model to use. Default: weights/fruit2rotten/G_A2B.pth")
parser.add_argument('--cuda', action="store_true", help="Turn on the cuda option.")
parser.add_argument('--image_size', type=int, default=256, help='Size of the image. Default: 256')
args = parser.parse_args()

print('****Preparing training with following options****')
time.sleep(0.2)

# Cuda option
if torch.cuda.is_available() and not args.cuda:
    print("Cuda device foud. Turning on cuda...")
    args.cuda = True
    time.sleep(0.2)
device = torch.device("cuda:0" if args.cuda else "cpu")

# Random seed to initialize the random state
seed = random.randint(1, 10000)
torch.manual_seed(seed)
print(f'Random Seed: {seed}')

print(f'Cuda: {args.cuda}')
print(f'Image size: {args.image_size}')
print(f'Testing dataset: {args.data_root}')
print(f'Testing model: {args.model}')
time.sleep(0.2)

# create model
model = Generator().to(device)

# Load state dicts
model.load_state_dict(torch.load(args.model))

# Set model mode
model.eval()

# Load image
def translate(image_dir):
  image = Image.open(image_dir)
  transform = transforms.Compose([
    transforms.Resize(int(args.image_size), Image.BICUBIC),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
  image = transform(image).unsqueeze(0)
  image = image.to(device)
  return image


print("Begin translation...", end='')
for image in os.listdir(args.data_root):
  if not image.startswith('translated_'):
    translated_filename = image[:image.find('.')]
    translated_image = translate(os.path.join(args.data_root, image))
    save_image(translated_image, f'test/translated_{translated_filename}.png')
print("Done.")