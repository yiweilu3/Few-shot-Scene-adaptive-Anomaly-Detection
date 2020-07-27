from __future__ import print_function
import matplotlib.pyplot as plt
import argparse
import torch
import torch.utils.data
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import Dataset,DataLoader
from torchvision import datasets, transforms, models
from PIL import Image
import numpy as np
import ast
from torch.nn import functional as F
import os
import random
import torch.utils.data
import torchvision.utils as vutils
import torch.backends.cudnn as cudn
from torch.nn import functional as F
from unet_parts import *
from scipy.misc import imsave
from torch.nn import BCELoss as adversarial_loss
import ast

from rGAN import Generator, Discriminator
from dataset import TrainingDataset
from utils import createEpochData, roll_axis, loss_function, create_folder, prep_data

def Load_Dataloader(train_path_list, tf, batch_size):
    train_data = TrainingDataset(train_path_list, tf)
    train_dataloader = DataLoader(train_data,batch_size=batch_size)
    return train_dataloader

def overall_generator_pass(generator, discriminator, img, gt, valid):
    recon_batch = generator(img)
    msssim, f1, psnr = loss_function(recon_batch, gt)

    imgs = recon_batch.data.cpu().numpy()[0, :]
    imgs = roll_axis(imgs)
    loss= msssim+f1
    G_loss = adversarial_loss(discriminator(recon_batch),valid)
    g_loss = adversarial_loss(discriminator(recon_batch),valid) + loss
    return imgs, g_loss, recon_batch, loss, msssim

def overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake):
    real_loss = adversarial_loss(discriminator(gt), valid)
    fake_loss = adversarial_loss(discriminator(recon_batch.detach()), fake)
    d_loss = (real_loss + fake_loss) / 2
    return d_loss

def meta_update_model(model, optimizer, loss, gradients):
    # Register a hook on each parameter in the net that replaces the current dummy grad
    # with our grads accumulated across the meta-batch
    # GENERATOR
    hooks = []
    for (k,v) in model.named_parameters():
        def get_closure():
            key = k
            def replace_grad(grad):
                return gradients[key]
            return replace_grad
        hooks.append(v.register_hook(get_closure()))

    # Compute grads for current step, replace with summed gradients as defined by hook
    optimizer.zero_grad()
    loss.backward()

    # Update the net parameters with the accumulated gradient according to optimizer
    optimizer.step()

    # Remove the hooks before next training phase
    for h in hooks:
        h.remove()

"""MAIN TRAINING SCRIPT"""
def main(k_shots, num_tasks, adam_betas, gen_lr, dis_lr, total_epochs, model_folder_path):
    torch.manual_seed(1)
    # Initialize generator and discriminator
    batch_size = 1
    generator = Generator(batch_size=batch_size) 
    discriminator = Discriminator()
    generator.cuda()
    discriminator.cuda()

    # Training the Model

    # optimizer
    optimizer_G = optim.Adam (generator.parameters(), lr= gen_lr,  betas=adam_betas)
    optimizer_D = optim.Adam (discriminator.parameters(), lr= dis_lr,  betas=adam_betas)   

    # define dataloader
    tf = transforms.Compose([transforms.Resize((256,256)),transforms.ToTensor()])

    create_folder(model_folder_path)
        
    generator_path = os.path.join(model_folder_path, str.format("Generator_previous.pt"))
    discriminator_path = os.path.join(model_folder_path, str.format("Discriminator_previous.pt"))

    torch.save(generator.state_dict(), generator_path)
    torch.save(discriminator.state_dict(), discriminator_path)
    previous_generator = generator_path
    previous_discriminator = discriminator_path

    frame_path = '/mnt/creeper/grad/luy2/Meta-Learning/data/shanghaitech-5tasks/training/frames/' 

    # Set Up Training Loop
    for epoch in range(total_epochs):
        train_path_list = createEpochData(frame_path, num_tasks, k_shots)
        train_dataloader = Load_Dataloader(train_path_list, tf, batch_size)
        for _, epoch_of_tasks in enumerate(train_dataloader):
            
            # Create folder for saving results
            epoch_results = 'results'.format(epoch+1)
            create_folder(epoch_results)

            gen_epoch_grads = []
            dis_epoch_grads = []

            print("Epoch: ", epoch+1)

                                    # Meta-Training
            for tidx, task in enumerate(epoch_of_tasks):
                # Copy rGAN
                print ('\n Meta Training \n')
                print("Memory Allocated: ",torch.cuda.memory_allocated()/1e9)
                generator.load_state_dict(torch.load(previous_generator))
                discriminator.load_state_dict(torch.load(previous_discriminator))
                inner_optimizer_G = optim.Adam(generator.parameters(), lr=1e-4)
                inner_optimizer_D = optim.Adam(discriminator.parameters(), lr=1e-4)
                print("Task: ", tidx)
                for kidx, frame_sequence in enumerate(task[:k_shots]):
                    print('k-Shot Training: ', kidx)
                    # Configure input
                    img = frame_sequence[0]
                    gt = frame_sequence[1]
                    img, gt, valid, fake = prep_data(img, gt)

                    # Train Generator
                    inner_optimizer_G.zero_grad()
                    imgs, g_loss, recon_batch, loss, msssim = overall_generator_pass(generator, discriminator, img, gt, valid)
                    img_path = os.path.join(epoch_results,'{}-fig-train{}.png'.format(tidx+1, kidx+1))
                    imsave(img_path , imgs)
                    g_loss.backward()
                    inner_optimizer_G.step()

                    # Train Discriminator
                    inner_optimizer_D.zero_grad()
                    # Measure discriminator's ability to classify real from generated samples
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    d_loss.backward()
                    inner_optimizer_D.step()
                    print ('Epoch [{}/{}], Step [{}/{}], Reconstruction_Loss: {:.4f}, G_Loss: {:.4f}, D_loss: {:.4f},  msssim:{:.4f} '.format(epoch+1, total_epochs, tidx+1, 5, loss.item(), g_loss, d_loss, msssim))
            
                                        # Meta-Validation
                print ('\n Meta Validation \n')
                # Store Loss Values
                gen_validation_loss_store = 0.0
                dis_validation_loss_store = 0.0
                gen_validation_loss = 0.0
                dis_validation_loss = 0.0
                
                dummy_frame_sequence = []
                # forward pass
                for vidx, val_frame_sequence in enumerate(task[-k_shots:]):
                    print(vidx)
                    if vidx == 0:
                        dummy_frame_sequence = val_frame_sequence
                    
                    img = val_frame_sequence[0]
                    gt = val_frame_sequence[1]
                    img, gt, valid, fake = prep_data(img, gt)
                                    
                    # k-Validation Generator
                    imgs, g_loss, recon_batch, loss, msssim = overall_generator_pass(generator, discriminator, img, gt, valid)
                    img_path = os.path.join(epoch_results,'{}-fig-val{}.png'.format(tidx+1, vidx+1))
                    imsave(img_path , imgs)

                    # k-Validation Discriminator
                    d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)
                    
                    # Store Loss Items to reduce memory usage
                    gen_validation_loss_store += g_loss.item()
                    dis_validation_loss_store += d_loss.item()

                    if (vidx == k_shots-1):
                        # Store the loss
                        gen_validation_loss = g_loss
                        dis_validation_loss = d_loss
                        gen_validation_loss.data = torch.FloatTensor([gen_validation_loss_store/k_shots]).cuda()
                        dis_validation_loss.data = torch.FloatTensor([dis_validation_loss_store/k_shots]).cuda()


                    print("Generator Validation Loss: ", gen_validation_loss_store)
                    print("Discriminator Validation Loss: ", dis_validation_loss_store)
                    print ('Epoch [{}/{}], Step [{}/{}], G_Loss: {:.4f}, D_loss: {:.4f}'.format(epoch+1, total_epochs, tidx+1, 5, loss.item(), g_loss, d_loss))
                    print("Memory Allocated: ",torch.cuda.memory_allocated()/1e9)

                # Compute Validation Grad
                print("Memory Allocated: ",torch.cuda.memory_allocated()/1e9)

                generator.load_state_dict(torch.load(previous_generator))
                discriminator.load_state_dict(torch.load(previous_discriminator))

                gen_grads = torch.autograd.grad(gen_validation_loss, generator.parameters())
                dis_grads = torch.autograd.grad(dis_validation_loss, discriminator.parameters())
                
                gen_meta_grads = {name:g for ((name, _), g) in zip(generator.named_parameters(), gen_grads)}
                dis_meta_grads = {name:g for ((name, _), g) in zip(discriminator.named_parameters(), dis_grads)}
                
                gen_epoch_grads.append(gen_meta_grads)
                dis_epoch_grads.append(dis_meta_grads)


            # Meta Update
            print('\n Meta update \n')

            generator.load_state_dict(torch.load(previous_generator))
            discriminator.load_state_dict(torch.load(previous_discriminator))
            
            # Configure input
            img = dummy_frame_sequence[0]
            gt = dummy_frame_sequence[1]
            img, gt, valid, fake = prep_data(img, gt)

            # Dummy Forward Pass
            imgs, g_loss, recon_batch, loss, msssim = overall_generator_pass(generator, discriminator, img, gt, valid)
            d_loss = overall_discriminator_pass(discriminator, recon_batch, gt, valid, fake)

            # Unpack the list of grad dicts
            gen_gradients = {k: sum(d[k] for d in gen_epoch_grads) for k in gen_epoch_grads[0].keys()}
            dis_gradients = {k: sum(d[k] for d in dis_epoch_grads) for k in dis_epoch_grads[0].keys()}
            
            meta_update_model(generator, optimizer_G, g_loss, gen_gradients)
            meta_update_model(discriminator, optimizer_D, d_loss, dis_gradients)

            # Save the Model
            torch.save(generator.state_dict(), previous_generator)
            torch.save(discriminator.state_dict(), previous_discriminator)
            if (epoch % 10 == 0):
                gen_path = os.path.join(model_folder_path, str.format("Generator_{}.pt", epoch+1))
                dis_path = os.path.join(model_folder_path, str.format("Discriminator_{}.pt", epoch+1))
                torch.save(generator.state_dict(), gen_path)
                torch.save(discriminator.state_dict(), dis_path)
            

    print("Training Complete")
            
    gen_path = os.path.join(model_folder_path, str.format("Generator_Final.pt"))
    dis_path = os.path.join(model_folder_path, str.format("Discriminator_Final.pt"))
    torch.save(generator.state_dict(), gen_path)
    torch.save(discriminator.state_dict(), dis_path)


if __name__ == "__main__":
    if (len(sys.argv) == 8):
        """SYS ARG ORDER: 
        K_shots, num_tasks, adam_betas, generator lr, discriminator lr, total epochs, save model path
        """
        k_shots = int(sys.argv[1])
        num_tasks =  int(sys.argv[2])
        adam_betas = ast.literal_eval(sys.argv[3])
        gen_lr = float(sys.argv[4])
        dis_lr = float(sys.argv[5])
        total_epochs = int(sys.argv[6])
        model_folder_path = sys.argv[7]
    else:
        k_shots = 1 
        num_tasks = 6
        adam_betas = (0.5, 0.999)
        gen_lr = 2e-4
        dis_lr = 1e-5
        total_epochs = 2000
        model_folder_path = "model"
    main(k_shots, num_tasks, adam_betas, gen_lr, dis_lr, total_epochs, model_folder_path)
