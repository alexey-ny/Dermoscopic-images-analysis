# -*- coding: utf-8 -*-
"""
Created on Tue Apr  9 20:08:18 2024

@author: VHABHSKotliA
"""

import sys
import os
import glob
import random
import time
import uuid

import numpy as np
import pandas as pd

import cv2
import matplotlib.pyplot as plt
import simplejpeg 
    

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

import albumentations as A
from albumentations.pytorch import ToTensorV2

from sklearn.model_selection import train_test_split


plt.style.use("dark_background")

cur_uuid = str(uuid.uuid4())
SEED = 1970
random.seed(SEED)
# tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

params = {
    'IMG_SIZES' :  512,
    'BATCH_SIZES' : 5,
    'EPOCHS' : 10,
    # 'EPOCHS' : 100,
    # 'EPOCHS' : 1,
    'ES_PATIENCE' : 10,
    # 'LR_START'  : 0.000005,
    # 'LR_MAX'    : 0.00000125,
    'LR_START'    : 0.00005,
    'LR_MAX'      : 0.000125,
    # 'LR_MIN'   : 0.00005,  
    'LR_MIN'    : 0.000001,
    # 'LR_RAMP_EP' : 10,
    'LR_RAMP_EP' : 5,
    # 'LR_SUS_EP' : 0, # cycle step
    'LR_SUS_EP' : 0, # cycle step
    'LR_DECAY' : 0.8,
    # 'GPU' : 0, 
    'GPU' : 1,
    'UUID' : cur_uuid,
    'VER' : 1
    }

IMG_PATH = '.\\skin_segmented_images\\originals\\'
MSK_PATH = '.\\skin_segmented_images\\masks\\'

org_files = [x.split('\\')[3] for x in glob.glob(IMG_PATH + '*')]
# org_files = [x.split('\\')[3].split('.')[0] for x in glob.glob(IMG_PATH + '*')]
# org_files = [x.split('\\')[3][:-4] for x in glob.glob(IMG_PATH + '*')]
msk_files = [x.split('\\')[3] for x in glob.glob(MSK_PATH + '*')]
# msk_files = [x.split('\\')[3].split('.')[0] + ".png" for x in glob.glob(IMG_PATH + '*')]
df_files = pd.DataFrame(org_files, columns = ['filename'])
df_files['mask'] =  msk_files

device = torch.device(f'cuda:{params["GPU"]}' if torch.cuda.is_available() else 'cpu')

PATCH_SIZE = 256
# PATCH_SIZE = 128
# PATCH_SIZE = 2048
transforms = A.Compose([
    A.Resize(width = PATCH_SIZE, height = PATCH_SIZE, p=1.0),
    # A.HorizontalFlip(p=0.5),
    # A.VerticalFlip(p=0.5),
    # A.RandomRotate90(p=0.5),
    # A.Transpose(p=0.5),
    # A.ShiftScaleRotate(shift_limit=0.01, scale_limit=0.04, rotate_limit=0, p=0.25),
    # A.Normalize(p=1.0),
    ToTensorV2(),
])


class SkinDataset(Dataset):
    def __init__(self, df, transforms):
        
        self.df = df
        self.transforms = transforms
        
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        image = cv2.imread(f"{IMG_PATH}{self.df.iloc[idx, 0]}")
        mask = cv2.imread(f"{MSK_PATH}{self.df.iloc[idx, 1]}", 0)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB) 
        # image = cv2.imread(f"{IMG_PATH}{self.df.iloc[idx, 1]}.jpg")
        # mask = cv2.imread(f"{MSK_PATH}{self.df.iloc[idx, 1]}.png", 0)
        # image = cv2.imread(self.df.iloc[idx, 1])
        # mask = cv2.imread(self.df.iloc[idx, 2], 0)
        # plt.imshow(image)

        augmented = self.transforms(image=image, mask=mask)
 
        image = augmented['image'].to(torch.float32)
        mask = augmented['mask'].to(torch.float32)   
        # image = augmented['image']
        # mask = augmented['mask']   
        
        return image, mask
    
    
# Split df into train_df and val_df
train_df, val_df = train_test_split(df_files, test_size=0.1)
train_df = train_df.reset_index(drop=True)
val_df = val_df.reset_index(drop=True)

# Split train_df into train_df and test_df
train_df, test_df = train_test_split(train_df, test_size=0.15)
train_df = train_df.reset_index(drop=True)

#train_df = train_df[:1000]
print(f"Train: {train_df.shape} \nVal: {val_df.shape} \nTest: {test_df.shape}")

# train
train_dataset = SkinDataset(df = train_df, transforms = transforms)
# train_dataloader = DataLoader(train_dataset, batch_size = params['BATCH_SIZES'], num_workers=4, shuffle=True)
train_dataloader = DataLoader(train_dataset, batch_size = params['BATCH_SIZES'], num_workers = 0, shuffle=True)

# val
# val_dataset = SkinDataset(df = val_df[:50], transforms = transforms)
val_dataset = SkinDataset(df = val_df, transforms = transforms)
# val_dataloader = DataLoader(val_dataset, batch_size = params['BATCH_SIZES'], num_workers = 4, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size = params['BATCH_SIZES'], num_workers = 0, shuffle=True)

#test
test_dataset = SkinDataset(df = test_df, transforms = transforms)
# test_dataloader = DataLoader(test_dataset, batch_size = params['BATCH_SIZES'], num_workers=4, shuffle=True)
test_dataloader = DataLoader(test_dataset, batch_size = params['BATCH_SIZES'], num_workers = 0, shuffle=True)

def show_aug(inputs, nrows=5, ncols=5, image=True):
    plt.figure(figsize=(10, 10))
    plt.subplots_adjust(wspace=0., hspace=0.)
    i_ = 0
    
    if len(inputs) > 25:
        inputs = inputs[:25]
        
    for idx in range(len(inputs)):
    
        # normalization
        if image is True:           
            img = inputs[idx].numpy().transpose(1,2,0)
            # img = inputs[idx].numpy().transpose(1,2,0)
            # mean = [0.485, 0.456, 0.406]
            # std = [0.229, 0.224, 0.225] 
            # img = (img*std+mean).astype(np.float32)
        else:
            img = inputs[idx].numpy().astype(np.float32)
            # img = img[:,:]
            # img = img[0,:,:]
        
        #plot
        #print(img.max(), len(np.unique(img)))
        plt.subplot(nrows, ncols, i_+1)
        plt.imshow(img); 
        plt.axis('off')
 
        i_ += 1
        
    return plt.show()

    
images, masks = next(iter(val_dataloader))
# images, masks = next(iter(train_dataloader))
print(f'Shapes: {images.shape}, {masks.shape}')

show_aug(images)
show_aug(masks, image=False)

def dice_coef_metric(inputs, target):
    intersection = 2.0 * (target * inputs).sum()
    union = target.sum() + inputs.sum()
    if target.sum() == 0 and inputs.sum() == 0:
        return 1.0

    return intersection / union

def dice_coef_loss(inputs, target):
    smooth = 1.0
    intersection = 2.0 * ((target * inputs).sum()) + smooth
    union = target.sum() + inputs.sum() + smooth

    return 1 - (intersection / union)


def bce_dice_loss(inputs, target):
    dicescore = dice_coef_loss(inputs, target)
    bcescore = nn.BCELoss()
    bceloss = bcescore(inputs, target)

    return bceloss + dicescore


def double_conv(in_channels, out_channels):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True),
        nn.Conv2d(out_channels, out_channels, 3, padding=1),
        nn.ReLU(inplace=True))

class UNet(nn.Module):

    def __init__(self, n_classes):
        super().__init__()
                
        self.conv_down1 = double_conv(3, 64)
        self.conv_down2 = double_conv(64, 128)
        self.conv_down3 = double_conv(128, 256)
        self.conv_down4 = double_conv(256, 512)        

        self.maxpool = nn.MaxPool2d(2)
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)        
        
        self.conv_up3 = double_conv(256 + 512, 256)
        self.conv_up2 = double_conv(128 + 256, 128)
        self.conv_up1 = double_conv(128 + 64, 64)
        
        self.last_conv = nn.Conv2d(64, n_classes, kernel_size=1)
        
        
    def forward(self, x):
        # Batch - 1d tensor.  N_channels - 1d tensor, IMG_SIZE - 2d tensor.
        # Example: x.shape >>> (10, 3, 256, 256).
        
        conv1 = self.conv_down1(x)  # <- BATCH, 3, IMG_SIZE  -> BATCH, 64, IMG_SIZE..
        x = self.maxpool(conv1)     # <- BATCH, 64, IMG_SIZE -> BATCH, 64, IMG_SIZE 2x down.
        conv2 = self.conv_down2(x)  # <- BATCH, 64, IMG_SIZE -> BATCH,128, IMG_SIZE.
        x = self.maxpool(conv2)     # <- BATCH, 128, IMG_SIZE -> BATCH, 128, IMG_SIZE 2x down.
        conv3 = self.conv_down3(x)  # <- BATCH, 128, IMG_SIZE -> BATCH, 256, IMG_SIZE.
        x = self.maxpool(conv3)     # <- BATCH, 256, IMG_SIZE -> BATCH, 256, IMG_SIZE 2x down.
        x = self.conv_down4(x)      # <- BATCH, 256, IMG_SIZE -> BATCH, 512, IMG_SIZE.
        x = self.upsample(x)        # <- BATCH, 512, IMG_SIZE -> BATCH, 512, IMG_SIZE 2x up.
        
        #(Below the same)                                 N this       ==        N this.  Because the first N is upsampled.
        x = torch.cat([x, conv3], dim=1) # <- BATCH, 512, IMG_SIZE & BATCH, 256, IMG_SIZE--> BATCH, 768, IMG_SIZE.
        
        x = self.conv_up3(x) #  <- BATCH, 768, IMG_SIZE --> BATCH, 256, IMG_SIZE. 
        x = self.upsample(x)  #  <- BATCH, 256, IMG_SIZE -> BATCH,  256, IMG_SIZE 2x up.   
        x = torch.cat([x, conv2], dim=1) # <- BATCH, 256,IMG_SIZE & BATCH, 128, IMG_SIZE --> BATCH, 384, IMG_SIZE.  

        x = self.conv_up2(x) # <- BATCH, 384, IMG_SIZE --> BATCH, 128 IMG_SIZE. 
        x = self.upsample(x)   # <- BATCH, 128, IMG_SIZE --> BATCH, 128, IMG_SIZE 2x up.     
        x = torch.cat([x, conv1], dim=1) # <- BATCH, 128, IMG_SIZE & BATCH, 64, IMG_SIZE --> BATCH, 192, IMG_SIZE.  
        
        x = self.conv_up1(x) # <- BATCH, 128, IMG_SIZE --> BATCH, 64, IMG_SIZE.
        
        out = self.last_conv(x) # <- BATCH, 64, IMG_SIZE --> BATCH, n_classes, IMG_SIZE.
        out = torch.sigmoid(out)
        
        return out
    

def train_model(model_name, model, train_loader, val_loader, train_loss, optimizer, lr_scheduler, num_epochs):  
    
    print(model_name)
    loss_history = []
    train_history = []
    val_history = []

    for epoch in range(num_epochs):
        model.train() # Enter train mode
        
        losses = []
        train_iou = []
                
        if lr_scheduler:
            
            warmup_factor = 1.0 / 100
            warmup_iters = min(100, len(train_loader) - 1)
            lr_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        
        
        print(f'Epoch {epoch}')
        for i_step, (data, target) in enumerate(train_loader):
            data = data.to(device)
            # np_target = target.numpy()
            target = target.to(device)
                      
            outputs = model(data)
            outputs  = torch.squeeze(outputs)
            
            out_cut = np.copy(outputs.data.cpu().numpy())
            # np_out = np.squeeze(out_cut)

            out_cut[np.nonzero(out_cut < 0.5)] = 0.0
            out_cut[np.nonzero(out_cut >= 0.5)] = 1.0
            
            train_dice = dice_coef_metric(out_cut, target.data.cpu().numpy())
            
            # loss = train_loss(np_out, np_target)
            loss = train_loss(outputs, target)
            
            losses.append(loss.item())
            train_iou.append(train_dice)

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
            if lr_scheduler:
                lr_scheduler.step()
 
        #torch.save(model.state_dict(), f'{model_name}_{str(epoch)}_epoch.pt')
        val_mean_iou = compute_iou(model, val_loader)
        
        loss_history.append(np.array(losses).mean())
        train_history.append(np.array(train_iou).mean())
        val_history.append(val_mean_iou)
        
        print("Epoch [%d]" % (epoch))
        print("Mean loss on train:", np.array(losses).mean(), 
              "\nMean DICE on train:", np.array(train_iou).mean(), 
              "\nMean DICE on validation:", val_mean_iou)
        
    return loss_history, train_history, val_history


def compute_iou(model, loader, threshold=0.3):
    """
    Computes accuracy on the dataset wrapped in a loader
    
    Returns: accuracy as a float value between 0 and 1
    """
    #model.eval()
    valloss = 0
    
    with torch.no_grad():

        for i_step, (data, target) in enumerate(loader):
            
            data = data.to(device)
            target = target.to(device)
            #prediction = model(x_gpu)
            
            outputs = model(data)
           # print("val_output:", outputs.shape)

            out_cut = np.copy(outputs.data.cpu().numpy())
            out_cut[np.nonzero(out_cut < threshold)] = 0.0
            out_cut[np.nonzero(out_cut >= threshold)] = 1.0

            picloss = dice_coef_metric(out_cut, target.data.cpu().numpy())
            valloss += picloss

        #print("Threshold:  " + str(threshold) + "  Validation DICE score:", valloss / i_step)

    return valloss / i_step


unet = UNet(n_classes = 1).to(device)
temp_tensor = torch.randn(5, 3, 256, 256)
output = unet(temp_tensor.to(device))
print("T output:", output.shape)

out_np = output.cpu().detach().numpy()
out_np1 = np.squeeze(out_np)

plt.imshow(out_np1[0])
plt.show()


# Optimizers
unet_optimizer = torch.optim.Adamax(unet.parameters(), lr=1e-3)
# fpn_optimizer = torch.optim.Adamax(fpn.parameters(), lr=1e-3)
# rx50_optimizer = torch.optim.Adam(rx50.parameters(), lr=5e-4)

# lr_scheduler
def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)



# num_ep = 10                                                                                                  
# Train UNet
unet_lh, unet_th, unet_vh = train_model("Vanila_UNet", unet, train_dataloader, val_dataloader, bce_dice_loss, unet_optimizer, False, params['EPOCHS']) 

# Train FPN
#fpn_lh, fpn_th, fpn_vh = train_model("FPN", fpn, train_dataloader, val_dataloader, bce_dice_loss, fpn_optimizer, False, 20)#

# Train ResNeXt50
# rx50_lh, rx50_th, rx50_vh = train_model("ResNeXt50", rx50, train_dataloader, val_dataloader, bce_dice_loss, rx50_optimizer, False, num_ep)