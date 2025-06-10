import time # measure time
import torch # Neural Network Framework
import pandas as pd # dataframes
from modules.loss_functions.loss_function import loss_function
from tqdm import tqdm  # progress bar
import matplotlib.pyplot as plt # plots
from pathlib import Path # build paths
import os
from torch.amp import autocast, GradScaler
from datetime import datetime
import numpy as np

def train_model(model, train_loader, val_loader, epochs, learning_rate, alpha, device, early_stopping_patience):
    model.to(device) # use device (GPU or CPU)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate) # choose optimizer AdamW instead of Adam because of weight decay. Default value is 0.1. In future versions you can choose in config.

    #scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs, eta_min=1e-5)                 #<-------------------------- UPDATE

    metrics = pd.DataFrame(columns=["Epoch", "Loss", "Mae", "Msssim", "Val-Loss", "Val-Mae", "Val-Msssim", "Learning-Rate", "Time"]) # initialize pandas df to create the csv later
    lrs = [] # list of learning rates
    #[epoch + 1, avg_loss, avg_mae, avg_msssim, avg_val_loss, avg_val_mae, avg_val_msssim, duration]
    scaler = GradScaler('cuda') ##################################################

    print("Training started!") # print to see the start


    scheduler = None
    epochs_no_improve = 0
    best_avg_val_loss = float('inf')
    cosine = False

    warmup_epochs = 10
    warmup_lrs = np.linspace(1e-7, learning_rate, num=warmup_epochs)
    print(warmup_lrs)
    for epoch in range(epochs): # loop over the epochs
        start = time.time() # starttime
        
        ###############################################################################
        if epoch < warmup_epochs:                                                    ##
            warmup_lr =  warmup_lrs[epoch]                                           ##
            for param_group in optimizer.param_groups:                               ##
                param_group['lr'] = warmup_lr                                        ##
        ###############################################################################

        model.train() # go to train mode, not eval mode. Model can change weights and biases now!
        total_loss = 0.0 # initialize total_loss for calculating the sum
        total_mae = 0.0
        total_msssim = 0.0
        progress = tqdm(train_loader, desc=f"Epoch [{epoch+1}/{epochs}]", leave=False) # now its:  for batch in tqdm(dataloader):   instead of:  for batch in dataloader

        current_lr = optimizer.param_groups[0]["lr"] # take current lr from param_groups[0]
        lrs.append(current_lr) # add it to lrs

        for low_res_image, high_res_image in progress:
            low_res_image, high_res_image = low_res_image.to(device), high_res_image.to(device) # tensors to the right device
            optimizer.zero_grad() # reset gradients

            ###########################################################################################
            with autocast(device_type='cuda'):  # ✅ Mixed Precision aktiv
                output = model(low_res_image)
                loss, mae, msssim = loss_function(output, high_res_image, alpha)
            ###########################################################################################
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            ###########################################################################################



            total_loss += loss.item() # add loss to list
            progress.set_postfix(loss=loss.item()) # add loss to progress bar
            total_mae += mae.item()
            total_msssim += msssim.item()

        duration = time.time() - start # total time for a epoch
        ###################


        avg_loss = total_loss / len(train_loader) # calculation of the average loss: sum of total loss of the batch / Batchsize
        avg_mae = total_mae / len(train_loader)
        avg_msssim = total_msssim / len(train_loader)

        # Validation!
        model.eval()
        val_loss = 0.0
        val_mae = 0.0
        val_msssim = 0.0
        with torch.no_grad():
            for low_res_image, high_res_image in val_loader:
                low_res_image, high_res_image = low_res_image.to(device), high_res_image.to(device)
                with autocast(device_type='cuda'):
                    output = model(low_res_image)
                    loss, mae, msssim = loss_function(output, high_res_image, alpha)
                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)
        avg_val_mae = val_mae / len(val_loader)
        avg_val_msssim = val_msssim / len(val_loader)

        ###########################################################################################################


        if cosine:
            cosine_counter += 1
            metrics.loc[len(metrics)] = [epoch + 1, avg_loss, avg_mae, avg_msssim, avg_val_loss, avg_val_mae, avg_val_msssim, current_lr, duration] # add loss and duration information
            print(f"[{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f} | Val-Loss: {avg_val_loss:.4f} | Time: {duration:.2f}s | Learning-Rate: {param_group['lr']:.2e} | CosineAnnealing [{cosine_counter}/{cosine_epochs}]")
            scheduler.step()
            if cosine_counter >= cosine_epochs: ####### entferne "="
                print("Ready with CosineAnnealing!")

                break

        if not cosine:
            if avg_val_loss < best_avg_val_loss - 1e-4: #-4
                best_avg_val_loss = avg_val_loss
                epochs_no_improve = 0
                metrics.loc[len(metrics)] = [epoch + 1, avg_loss, avg_mae, avg_msssim, avg_val_loss, avg_val_mae, avg_val_msssim, current_lr, duration] # add loss and duration information
                print(f"[{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f} | Val-Loss: {avg_val_loss:.4f} | Time: {duration:.2f}s | Learning-Rate: {param_group['lr']:.2e} | Patience: {epochs_no_improve}")
            else:
                if epoch >= warmup_epochs:
                    epochs_no_improve += 1
                metrics.loc[len(metrics)] = [epoch + 1, avg_loss, avg_mae, avg_msssim, avg_val_loss, avg_val_mae, avg_val_msssim, current_lr, duration] # add loss and duration information
                print(f"[{epoch+1:3d}/{epochs}] Loss: {avg_loss:.4f} | Val-Loss: {avg_val_loss:.4f} | Time: {duration:.2f}s | Learning-Rate: {param_group['lr']:.2e} | Patience: {epochs_no_improve}")

            if epochs_no_improve >= early_stopping_patience:
                cosine_epochs = max(int(0.3 * (epoch + 1)), 10)
                scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                    optimizer, T_max=cosine_epochs-1, eta_min=1e-7)
                cosine = True
                print(f'Switching to CosineAnnealing with {cosine_epochs} epochs!')
                cosine_counter = 0
        
        
        ############################################################################################################


    # Plot for lr-analysis
    plot_dir = "results/lr_finder"
    Path(plot_dir).mkdir(parents=True, exist_ok=True)

    # Plot settings
    plt.figure(figsize=(10, 4))
    plt.plot(lrs)
    plt.xlabel("Trainingsteps")
    plt.ylabel("Learningrate")
    plt.yscale("log")
    plt.title("Learningrate Plot")
    plt.grid(True)
    plt.tight_layout()

    
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    mask = f"lr_plot_{timestamp}"
    plot_path = os.path.join(plot_dir, f"lr_plot_{mask}.png")
    plt.savefig(plot_path)
    plt.close()

    print(f"Learningrate Plot is saved in {plot_path}")

    return model, metrics
