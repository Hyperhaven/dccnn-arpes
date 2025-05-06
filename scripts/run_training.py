import yaml # structure of the config
import glob # search for paths
import torch # Neural Network Framework
import gc # Garbage Collector
from datetime import datetime # gives current time for datanames
from pathlib import Path # build paths
from torch.utils.data import DataLoader, Subset # class of PyTorch for seperation of the Dataset in batches, shuffling, multithreading and iteration of batches

# Import externs
from modules.datasets.dataset import CustomDataset
from modules.models.ccnn import CCNN

# import trainer
from train.trainer import train_model

def load_config(path):
    with open(path, "r") as f: # r is read mode, after usage it gets closed because of "with", python object is loaded as f
        return yaml.safe_load(f) # safe_load does not activate code in this file

def run_all_configs(config_dir):
    configs = sorted(glob.glob(f"{config_dir}/*.yaml")) # searching for all .yaml files and gives back a list with all of them

    for cfg_path in configs: # for every .yaml file

        # initialize all config values 
        cfg = load_config(cfg_path)
        model_cfg = cfg["model"]
        train_cfg = cfg["training"]
        path_cfg = cfg["paths"]
        output_dir = Path(path_cfg["output_dir"])
        csv_dir = output_dir / path_cfg.get("csv_subdir")
        model_dir = output_dir / "models"

        # create result folder and csv, model subfolder
        output_dir.mkdir(exist_ok=True)
        csv_dir.mkdir(exist_ok=True)
        model_dir.mkdir(exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S") # current time for filename
        mask = f"{timestamp}_layers{model_cfg['num_layers']}_batch{train_cfg['batch_size']}_kernel{model_cfg['kernel_size']}" # mask for resulting model filename 
        csv_path = csv_dir / f"{mask}.csv" # path of the .csv
        model_path = model_dir / f"{mask}.pt" # path of the .pt (model)

        device = torch.device("cuda" if train_cfg["use_gpu"] and torch.cuda.is_available() else "cpu") # use GPU-RAM for cuda calculations
        dataset = CustomDataset(path_cfg["high_res_dir"], path_cfg["low_res_dir"]) # load dataset

        train_dataset = Subset(dataset, list(set(range(len(dataset))) - set(range(500))))
        val_dataset = Subset(dataset, list(range(500)))



        #dataloader = DataLoader(dataset, batch_size=train_cfg["batch_size"], shuffle=True, drop_last=True) # load dataloader

        train_loader = DataLoader(train_dataset, batch_size=train_cfg["batch_size"], shuffle=True,drop_last=True)
        val_loader = DataLoader(val_dataset, batch_size=train_cfg["batch_size"], shuffle=False,drop_last=True)

        model = CCNN(model_cfg["kernel_size"], model_cfg["num_layers"]) # load model


        ##########################################################################################################################################
        ##########################################################################################################################################
        model, metrics = train_model(model, train_loader, val_loader, train_cfg["epochs"], train_cfg["learning_rate"], train_cfg["alpha"], device) #############
        ##########################################################################################################################################
        ##########################################################################################################################################


        metrics.to_csv(csv_path, index=False) # save the .csv
        torch.save(model.state_dict(), model_path) # save model weights and biases


        del model, train_loader, val_loader, dataset # delete references for gc.collect()
        torch.cuda.empty_cache() # clears GPU-RAM
        gc.collect() # Python Garbage Collector clears CPU-RAM


        # NOW COMES THE NEXT YAML!

if __name__ == "__main__":
    run_all_configs("config")
