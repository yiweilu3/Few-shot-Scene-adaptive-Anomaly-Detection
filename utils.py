import os
import random
import torch
import pytorch_msssim
import numpy as np
from math import log10
from torch.nn import functional as F
from torch.cuda import FloatTensor as Tensor
from torch.autograd import Variable


def generate_k_shot_frames(video_folder, k_shots):
    k_video_sequences = []

    all_frames = list(os.walk(video_folder))[0][2]

    all_frames = sorted(all_frames, key=lambda x: int(x.replace('frame', '').replace('.jpg', '')))        
    video_length = len(all_frames)

    frame_samples = random.sample(range(3,video_length), k_shots * 2) # first k are meta-training, rest are meta-testing

    k_frame_sequences = [[all_frames[v_index - before] for before in reversed(range(0,4))] for v_index in frame_samples]
    return video_folder, k_frame_sequences

def createEpochData(frame_path, numTasks, k_shots):
    dirs = os.listdir(frame_path)
    dirs = list(dirs)
    dirs.sort(key=int)
    
    # Selected Tasks (videos that are being used)
    selected_videos = []

    for task in range(numTasks):
        sample = random.sample(list(os.listdir(os.path.join(frame_path, dirs[task]))), 1)
        selected_videos.append(os.path.join(frame_path, dirs[task], sample[0]))

    train_path_list = []

    # task_order = [0, 2, 3, 4, 7, 12]
    train_curr_paths = []
    # for task in range(len(task_order)):
    for task in range(numTasks):
        video = selected_videos[task]
        video_folder, k_shot_frames = generate_k_shot_frames(video, k_shots)        
        train_curr_paths.append([[os.path.join(frame_path, str(video_folder), ind) for ind in frame] for frame in k_shot_frames])
    train_path_list.append(train_curr_paths)

    return train_path_list

def loss_function(recon_x, x):
    msssim = ((1-pytorch_msssim.msssim(x,recon_x)))/2
    f1 =  F.l1_loss(recon_x, x)
    # psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_batch))))))
    psnr_error=(10 * log10( 65025/ ((torch.abs(torch.sum(x) - torch.sum(recon_x))))))

    return msssim, f1, psnr_error

def roll_axis(img):
    img = np.rollaxis(img, -1, 0)
    img = np.rollaxis(img, -1, 0)
    return img

def create_folder(path):
    if not os.path.exists(path):
        os.makedirs(path)
        return True
    return False

def prep_data(img, gt, gen_labels=True):
    if gen_labels:
        # Adversarial ground truths
        valid = Variable(Tensor(1, 1).fill_(0.9), requires_grad=False)
        fake = Variable(Tensor(1, 1).fill_(0.1), requires_grad=False)
        valid.cuda()
        fake.cuda()
    for x in range(len(img)):
        img[x] = Variable(img[x].cuda())
    gt = Variable(gt.cuda())
    return img, gt, valid, fake
