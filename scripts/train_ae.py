import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import pickle
import sys
import os
import random
from tqdm.auto import tqdm, trange
from torch.utils.data import Dataset, DataLoader
import MDAnalysis as mda
import pytorch_lightning as pl
import argparse
import warnings
from compressTraj import *

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed()

parser = argparse.ArgumentParser(description="Process input files and output prefix.")

parser.add_argument('-t', '--train_loader', type=str, required=True, help='Path to the train loader file (PKL)')
parser.add_argument('-v', '--val_loader', type=str, required=True, help='Path to the validation loader file (PKL)')
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for output files')
parser.add_argument('-e', '--epochs', type=str, required=True, help='Number of epochs to train')
parser.add_argument('-l', '--latent', type=str, help='Number of latent dims [default=128]', default=128)

# Parse the arguments
args = parser.parse_args()

# Access the arguments
train_loader_file = args.train_loader
val_loader_file = args.val_loader
prefix = args.prefix
latent = args.latent

train_loader = pickle.load(open(train_loader_file, "rb"))
val_loader = pickle.load(open(val_loader_file, "rb"))

N = train_loader.dataset.get_N()

# Initialize the LightningModule
ae = DenseAutoEncoder(N=N, latent=latent)
model = LightAutoEncoder(model=ae, learning_rate=3e-4)

# Initialize the Trainer with the logger
trainer = pl.Trainer(max_epochs=int(args.epochs))

# Train the model
trainer.fit(model, train_loader, val_loader)

# saving
torch.save(model, prefix+"_model.pt")
pickle.dump({"train_loss":model.train_loss, "val_loss":model.val_loss},
              open(prefix+"_losses.pkl", "wb"))

print(f"""AutoEncoder model saved at {prefix + '_model.pt'}
Losses saved at {prefix + '_losses.pkl'}""")
