# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture


# Module dependencies 
import argparse
import itertools
import os
import random
import time

from PIL import Image
from tqdm import tqdm
import torch.backends.cudnn as cudnn
import torch.utils.data
import torchvision.transforms as transforms
import torchvision.utils as vutils

from cyclegan_pytorch import DecayLR
from cyclegan_pytorch import Discriminator
from cyclegan_pytorch import Generator
from cyclegan_pytorch import LoadDataset
from cyclegan_pytorch import ReplayBuffer
from cyclegan_pytorch import weights_init

import image_preprocessing as ip

# Command-line parser
parser = argparse.ArgumentParser(description="This is a pytorch implementation of CycleGAN. Please refer to the following arguments.")
parser.add_argument('--batch_size', default=10, type=int, help='Size of a mini-batch. Default: 1')
parser.add_argument('--cuda', action="store_true", help="Turn on the cuda option.")
parser.add_argument('--dataset', type=str, default="fruit2rotten", help='Name of the dataset. Default: fruit2rotten)')
parser.add_argument('--decay_epochs', type=int, default=80, help="epoch to start linearly decaying the learning rate to 0. Default: 80")
parser.add_argument('--epochs', default=100, type=int, help="Number of epochs. Default: 100")
parser.add_argument('--image_size', type=int, default=256, help='Size of the image. Default: 256')
parser.add_argument('--input_root', type=str, default='./input', help='Root directory to the input dataset. Default: ./input')
parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate. Default: 0.0002')
parser.add_argument('--output_root', default='./output', help='Root directory to the output dataset. Default: ./output')
parser.add_argument('--verbose_freq', default=100, type=int, help='Frequency of printing the information. Defatul: 100')
args = parser.parse_args()
print('****Preparing training with following options****')
time.sleep(0.2)

# Cuda option
if torch.cuda.is_available() and not args.cuda:
    print("Cuda device foud. Turning on cuda...")
    args.cuda = True
    time.sleep(0.2)

# Random seed to initialize the random state
seed = random.randint(1, 10000)
torch.manual_seed(seed)
print(f'Random Seed: {seed}')

print(f'Batch size: {args.batch_size}')
print(f'Cuda: {args.cuda}')
print(f'Dataset: {args.dataset}')
print(f'Decay epochs: {args.decay_epochs}')
print(f'Epochs: {args.epochs}')
print(f'Image size: {args.image_size}')
print(f'Input root: {args.input_root}')
print(f'Learning rate: {args.learning_rate}')
print(f'Output root: {args.output_root}')
print(f'Verbose frequency: {args.verbose_freq}')
time.sleep(0.2)

# Create directory
required_dir = [args.input_root, args.output_root]
for dir in required_dir:
    if not os.path.exists(dir):
        print(f'Following director is not found: \'{dir}\'. Creating the directory...')
        os.makedirs(dir)

# Dataset
dataset = LoadDataset(input_root=os.path.join(args.input_root, args.dataset),
                       transform=transforms.Compose([
                           transforms.Resize(int(args.image_size * 1.12), Image.BICUBIC),
                           transforms.RandomCrop(args.image_size),
                           transforms.RandomHorizontalFlip(),
                           transforms.ToTensor(),
                           transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]),
                       unaligned=True)

dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

try:
    os.makedirs(os.path.join(args.output_root, args.dataset, "A"))
    os.makedirs(os.path.join(args.output_root, args.dataset, "B"))
except OSError:
    pass

try:
    os.makedirs(os.path.join("weights", args.dataset))
except OSError:
    pass

device = torch.device("cuda:0" if args.cuda else "cpu")

# create model
netG_A2B = Generator().to(device)
netG_B2A = Generator().to(device)
netD_A = Discriminator().to(device)
netD_B = Discriminator().to(device)

netG_A2B.apply(weights_init)
netG_B2A.apply(weights_init)
netD_A.apply(weights_init)
netD_B.apply(weights_init)

if args.netG_A2B != "":
    netG_A2B.load_state_dict(torch.load(args.netG_A2B))
if args.netG_B2A != "":
    netG_B2A.load_state_dict(torch.load(args.netG_B2A))
if args.netD_A != "":
    netD_A.load_state_dict(torch.load(args.netD_A))
if args.netD_B != "":
    netD_B.load_state_dict(torch.load(args.netD_B))

# define loss function (adversarial_loss) and optimizer
cycle_loss = torch.nn.L1Loss().to(device)
identity_loss = torch.nn.L1Loss().to(device)
adversarial_loss = torch.nn.MSELoss().to(device)

# Optimizers
optimizer_G = torch.optim.Adam(itertools.chain(netG_A2B.parameters(), netG_B2A.parameters()),
                               lr=args.lr, betas=(0.5, 0.999))
optimizer_D_A = torch.optim.Adam(netD_A.parameters(), lr=args.lr, betas=(0.5, 0.999))
optimizer_D_B = torch.optim.Adam(netD_B.parameters(), lr=args.lr, betas=(0.5, 0.999))

lr_lambda = DecayLR(args.epochs, 0, args.decay_epochs).step
lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=lr_lambda)
lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=lr_lambda)
lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=lr_lambda)

g_losses = []
d_losses = []

identity_losses = []
gan_losses = []
cycle_losses = []

fake_A_buffer = ReplayBuffer()
fake_B_buffer = ReplayBuffer()

for epoch in range(0, args.epochs):
    progress_bar = tqdm(enumerate(dataloader), total=len(dataloader))
    for i, data in progress_bar:
        # get batch size data
        real_image_A = data["A"].to(device)
        real_image_B = data["B"].to(device)
        batch_size = real_image_A.size(0)

        # real data label is 1, fake data label is 0.
        real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)
        fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)

        ##############################################
        # (1) Update G network: Generators A2B and B2A
        ##############################################

        # Set G_A and G_B's gradients to zero
        optimizer_G.zero_grad()

        # Identity loss
        # G_B2A(A) should equal A if real A is fed
        identity_image_A = netG_B2A(real_image_A)
        loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
        # G_A2B(B) should equal B if real B is fed
        identity_image_B = netG_A2B(real_image_B)
        loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

        # GAN loss
        # GAN loss D_A(G_A(A))
        fake_image_A = netG_B2A(real_image_B)
        fake_output_A = netD_A(fake_image_A)
        loss_GAN_B2A = adversarial_loss(fake_output_A, real_label)
        # GAN loss D_B(G_B(B))
        fake_image_B = netG_A2B(real_image_A)
        fake_output_B = netD_B(fake_image_B)
        loss_GAN_A2B = adversarial_loss(fake_output_B, real_label)

        # Cycle loss
        recovered_image_A = netG_B2A(fake_image_B)
        loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

        recovered_image_B = netG_A2B(fake_image_A)
        loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

        # Combined loss and calculate gradients
        errG = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

        # Calculate gradients for G_A and G_B
        errG.backward()
        # Update G_A and G_B's weights
        optimizer_G.step()

        ##############################################
        # (2) Update D network: Discriminator A
        ##############################################

        # Set D_A gradients to zero
        optimizer_D_A.zero_grad()

        # Real A image loss
        real_output_A = netD_A(real_image_A)
        errD_real_A = adversarial_loss(real_output_A, real_label)

        # Fake A image loss
        fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
        fake_output_A = netD_A(fake_image_A.detach())
        errD_fake_A = adversarial_loss(fake_output_A, fake_label)

        # Combined loss and calculate gradients
        errD_A = (errD_real_A + errD_fake_A) / 2

        # Calculate gradients for D_A
        errD_A.backward()
        # Update D_A weights
        optimizer_D_A.step()

        ##############################################
        # (3) Update D network: Discriminator B
        ##############################################

        # Set D_B gradients to zero
        optimizer_D_B.zero_grad()

        # Real B image loss
        real_output_B = netD_B(real_image_B)
        errD_real_B = adversarial_loss(real_output_B, real_label)

        # Fake B image loss
        fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
        fake_output_B = netD_B(fake_image_B.detach())
        errD_fake_B = adversarial_loss(fake_output_B, fake_label)

        # Combined loss and calculate gradients
        errD_B = (errD_real_B + errD_fake_B) / 2

        # Calculate gradients for D_B
        errD_B.backward()
        # Update D_B weights
        optimizer_D_B.step()

        progress_bar.set_description(
            f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
            f"Loss_D: {(errD_A + errD_B).item():.4f} "
            f"Loss_G: {errG.item():.4f} "
            f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
            f"loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
            f"loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

        if i % args.print_freq == 0:
            vutils.save_image(real_image_A,
                              f"{args.output_root}/{args.dataset}/A/real_samples.png",
                              normalize=True)
            vutils.save_image(real_image_B,
                              f"{args.output_root}/{args.dataset}/B/real_samples.png",
                              normalize=True)

            fake_image_A = 0.5 * (netG_B2A(real_image_B).data + 1.0)
            fake_image_B = 0.5 * (netG_A2B(real_image_A).data + 1.0)

            vutils.save_image(fake_image_A.detach(),
                              f"{args.output_root}/{args.dataset}/A/fake_samples_epoch_{epoch}.png",
                              normalize=True)
            vutils.save_image(fake_image_B.detach(),
                              f"{args.output_root}/{args.dataset}/B/fake_samples_epoch_{epoch}.png",
                              normalize=True)

    # do check pointing
    torch.save(netG_A2B.state_dict(), f"weights/{args.dataset}/netG_A2B_epoch_{epoch}.pth")
    torch.save(netG_B2A.state_dict(), f"weights/{args.dataset}/netG_B2A_epoch_{epoch}.pth")
    torch.save(netD_A.state_dict(), f"weights/{args.dataset}/netD_A_epoch_{epoch}.pth")
    torch.save(netD_B.state_dict(), f"weights/{args.dataset}/netD_B_epoch_{epoch}.pth")

    # Update learning rates
    lr_scheduler_G.step()
    lr_scheduler_D_A.step()
    lr_scheduler_D_B.step()

# save last check pointing
torch.save(netG_A2B.state_dict(), f"weights/{args.dataset}/netG_A2B.pth")
torch.save(netG_B2A.state_dict(), f"weights/{args.dataset}/netG_B2A.pth")
torch.save(netD_A.state_dict(), f"weights/{args.dataset}/netD_A.pth")
torch.save(netD_B.state_dict(), f"weights/{args.dataset}/netD_B.pth")
