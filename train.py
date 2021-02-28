# Copyright 2021 by YoungWoon Cho, Danny Hong
# The Cooper Union for the Advancement of Science and Art
# ECE471 Machine Learning Architecture
# https://www.geeksforgeeks.org/cycle-generative-adversarial-network-cyclegan-2/

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

from optimization import QuadraticLR
from models import *
from datasets import LoadDataset
from utils import ReplayBuffer
from utils import weights_init

def main():
    # Command-line parser
    parser = argparse.ArgumentParser(description="This is a pytorch implementation of CycleGAN. Please refer to the following arguments.")
    parser.add_argument('--batch_size', default=1, type=int, help='Size of a mini-batch. Default: 1')
    parser.add_argument('--cuda', action="store_true", help="Turn on the cuda option.")
    parser.add_argument('--data_root', type=str, default='./train', help='Root directory to the input dataset. Default: ./train')
    parser.add_argument('--dataset', type=str, default="fruit2rotten", help='Name of the dataset. Default: fruit2rotten)')
    parser.add_argument('--decay_epochs', type=int, default=80, help="epoch to start linearly decaying the learning rate to 0. Default: 80")
    parser.add_argument('--epochs', default=100, type=int, help="Number of epochs. Default: 100")
    parser.add_argument('--image_size', type=int, default=256, help='Size of the image. Default: 256')
    parser.add_argument('--learning_rate', type=float, default=0.0002, help='Learning rate. Default: 0.0002')
    args = parser.parse_args()
    print('****Preparing training with following options****')
    time.sleep(0.2)

    # Cuda option
    if torch.cuda.is_available() and not args.cuda:
        print("Cuda device found. Turning on cuda...")
        args.cuda = True
        time.sleep(0.2)
    device = torch.device("cuda:0" if args.cuda else "cpu")

    # Random seed to initialize the random state
    seed = random.randint(1, 10000)
    torch.manual_seed(seed)
    print(f'Random Seed: {seed}')

    print(f'Batch size: {args.batch_size}')
    print(f'Cuda: {args.cuda}')
    print(f'Data root: {args.data_root}')
    print(f'Dataset: {args.dataset}')
    print(f'Decay epochs: {args.decay_epochs}')
    print(f'Epochs: {args.epochs}')
    print(f'Image size: {args.image_size}')
    print(f'Learning rate: {args.learning_rate}')
    time.sleep(0.2)

    # Create directory
    try: os.makedirs(args.data_root)
    except OSError: pass

    # Weights
    try: os.makedirs(os.path.join("weights", args.dataset))
    except OSError: pass

    # Load dataset
    dataset = LoadDataset(data_root=os.path.join(args.data_root, args.dataset), img_size = args.image_size)

    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True, pin_memory=True)

    # Create models
    G_A2B = Generator().to(device)
    G_B2A = Generator().to(device)
    D_A = Discriminator().to(device)
    D_B = Discriminator().to(device)

    G_A2B.apply(weights_init)
    G_B2A.apply(weights_init)
    D_A.apply(weights_init)
    D_B.apply(weights_init)

    # Loss function
    gan_loss = torch.nn.MSELoss().to(device)
    cycle_loss = torch.nn.L1Loss().to(device)
    identity_loss = torch.nn.L1Loss().to(device)

    discriminator_losses = []
    generator_losses = []
    cycle_losses = []
    gan_losses = []
    identity_losses = []

    # Optimizers
    optimizer_G = torch.optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=args.learning_rate)
    optimizer_D_A = torch.optim.Adam(D_A.parameters(), lr=args.learning_rate)
    optimizer_D_B = torch.optim.Adam(D_B.parameters(), lr=args.learning_rate)

    # Learning rate schedulers that will implement quadratic decreasing of the learning rate
    quadraticLR = QuadraticLR(args.epochs, 0, args.decay_epochs).step
    lr_scheduler_G = torch.optim.lr_scheduler.LambdaLR(optimizer_G, lr_lambda=quadraticLR)
    lr_scheduler_D_A = torch.optim.lr_scheduler.LambdaLR(optimizer_D_A, lr_lambda=quadraticLR)
    lr_scheduler_D_B = torch.optim.lr_scheduler.LambdaLR(optimizer_D_B, lr_lambda=quadraticLR)

    fake_A_buffer = ReplayBuffer()
    fake_B_buffer = ReplayBuffer()

    for epoch in range(0, args.epochs):
        progress = tqdm(enumerate(dataloader), total=len(dataloader))
        for i, data in progress:
            # get batch size data
            real_image_A = data["A"].to(device)
            real_image_B = data["B"].to(device)
            batch_size = real_image_A.size(0)

            # Fake: 0, Real: 1
            fake_label = torch.full((batch_size, 1), 0, device=device, dtype=torch.float32)
            real_label = torch.full((batch_size, 1), 1, device=device, dtype=torch.float32)

            #***********************
            # 1. UPDATE GENERATORS *
            #***********************
            optimizer_G.zero_grad()

            # Identity loss
            # A = G_B2A(A)
            identity_image_A = G_B2A(real_image_A)
            loss_identity_A = identity_loss(identity_image_A, real_image_A) * 5.0
            # B = G_A2B(B)
            identity_image_B = G_A2B(real_image_B)
            loss_identity_B = identity_loss(identity_image_B, real_image_B) * 5.0

            # GAN loss
            # D_A(G_A(A))
            fake_image_A = G_B2A(real_image_B)
            fake_output_A = D_A(fake_image_A)
            loss_GAN_B2A = gan_loss(fake_output_A, real_label)
            # D_B(G_B(B))
            fake_image_B = G_A2B(real_image_A)
            fake_output_B = D_B(fake_image_B)
            loss_GAN_A2B = gan_loss(fake_output_B, real_label)

            # Cycle loss
            recovered_image_A = G_B2A(fake_image_B)
            loss_cycle_ABA = cycle_loss(recovered_image_A, real_image_A) * 10.0

            recovered_image_B = G_A2B(fake_image_A)
            loss_cycle_BAB = cycle_loss(recovered_image_B, real_image_B) * 10.0

            # Net loss
            loss_G = loss_identity_A + loss_identity_B + loss_GAN_A2B + loss_GAN_B2A + loss_cycle_ABA + loss_cycle_BAB

            # Update Generator
            loss_G.backward()
            optimizer_G.step()

            #***************************
            # 2. UPDATE DISCRIMINATORS *
            #***************************
            optimizer_D_A.zero_grad()
            optimizer_D_B.zero_grad()

            # Real image loss
            real_output_A = D_A(real_image_A)
            loss_D_real_A = gan_loss(real_output_A, real_label)

            real_output_B = D_B(real_image_B)
            loss_D_real_B = gan_loss(real_output_B, real_label)

            # Fake image loss
            fake_image_A = fake_A_buffer.push_and_pop(fake_image_A)
            fake_output_A = D_A(fake_image_A.detach())
            loss_D_fake_A = gan_loss(fake_output_A, fake_label)

            fake_image_B = fake_B_buffer.push_and_pop(fake_image_B)
            fake_output_B = D_B(fake_image_B.detach())
            loss_D_fake_B = gan_loss(fake_output_B, fake_label)

            # Net loss
            loss_D_A = (loss_D_real_A + loss_D_fake_A) / 2
            loss_D_B = (loss_D_real_B + loss_D_fake_B) / 2

            # Update Discriminator
            loss_D_A.backward()
            optimizer_D_A.step()

            loss_D_B.backward()
            optimizer_D_B.step()

            #*************
            # 3. Verbose *
            #*************
            progress.set_description(
                f"[{epoch}/{args.epochs - 1}][{i}/{len(dataloader) - 1}] "
                f"Loss_D: {(loss_D_A + loss_D_B).item():.4f} "
                f"Loss_G: {loss_G.item():.4f} "
                f"Loss_G_identity: {(loss_identity_A + loss_identity_B).item():.4f} "
                f"Loss_G_GAN: {(loss_GAN_A2B + loss_GAN_B2A).item():.4f} "
                f"Loss_G_cycle: {(loss_cycle_ABA + loss_cycle_BAB).item():.4f}")

        # Update learning rates
        lr_scheduler_G.step()
        lr_scheduler_D_A.step()
        lr_scheduler_D_B.step()

    # save last check pointing
    torch.save(G_A2B.state_dict(), f"weights/{args.dataset}/G_A2B.pth")
    torch.save(G_B2A.state_dict(), f"weights/{args.dataset}/G_B2A.pth")
    torch.save(D_A.state_dict(), f"weights/{args.dataset}/D_A.pth")
    torch.save(D_B.state_dict(), f"weights/{args.dataset}/D_B.pth")

if __name__ == "__main__":
    main()