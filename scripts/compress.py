import warnings
warnings.filterwarnings("ignore")

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
from MDAnalysis import Universe, Writer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
import pytorch_lightning as pl
import argparse 
import shutil
from glob import glob
from Classes import *
from Helpers import *

set_seed()

parser = argparse.ArgumentParser(description="Process input files and model files for compression.")
    
# Add arguments for model, reffile, trajfile, scaler, and output with short and long options
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t', '--trajfile', type=str, required=True, help='Path to the trajectory file (xtc/trr/dcd/xyz)')
parser.add_argument('-p', '--prefix', type=str, required = True, help='prefix to to the files to be generated.')
parser.add_argument('-e', '--epochs', type=str, help='Number of epochs to train [default=200]', default=200)
parser.add_argument('-l', '--latent', type=str, help='Number of latent dims', default=None)
parser.add_argument('-c', '--compression', type=str, help='Extent of compression to achieve if latent dimension is not specified. [default = 20]', default=20)
parser.add_argument('-sel', '--selection', type=str, default="not element H",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default=0)

args = parser.parse_args()

reffile = args.reffile
trajfile = args.trajfile
output_deffnm = args.prefix
latent = args.latent
compression = float(args.compression)
selection = args.selection
gpu_id = int(args.gpuID)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using GPU:{gpu_id}.")

# processing trajectory
scaler = MinMaxScaler()
pos = pool(reffile, trajfile, selection=selection)
pos = pos.astype("float32")

X_train = scaler.fit_transform(pos)

train_loader = DataLoader(TrajLoader(X_train), shuffle=True, batch_size=256, num_workers=16)
del X_train

pickle.dump(scaler, open(output_deffnm+"_scaler.pkl", "wb"))
print(f"""Trajectories processed.
scaler saved at {output_deffnm}_scaler.pkl""")

with Writer(output_deffnm+"_select.pdb", "w") as w:
    u = Universe(reffile)
    # w.write(u.select_atoms("not element H"))
    w.write(u.select_atoms(selection))
print(f"selected atom coordinates saved at {output_deffnm}_select.pdb")

# training AE
N = train_loader.dataset.get_N()
latent = int(np.ceil(N/compression)) if latent is None else int(args.latent)

## Initialize the LightningModule
ae = DenseAutoEncoder(N=N, latent=latent)
model = LightAutoEncoder(model=ae, learning_rate=3e-4)

## Initialize the Trainer with the logger
trainer = pl.Trainer(max_epochs=int(args.epochs), accelerator='gpu', devices=1)

## Train the model
trainer.fit(model, train_loader)

## saving model and losses
torch.save(model, output_deffnm+"_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(output_deffnm+"_losses.pkl", "wb"))

print(f"""AutoEncoder model saved at {output_deffnm}_model.pt
Losses saved at {output_deffnm}_losses.pkl""")

del train_loader

# compress
loader = DataLoader(TrajLoader(scaler.transform(pos)), shuffle=False, batch_size=256, num_workers=8)
latent = []

encoder = model.model.encoder
del model
encoder = encoder.to(device)

with torch.no_grad():
    encoder.eval()
    for Pos in loader:
        Pos = Pos.to(device=device, dtype=torch.float32)
        lat = encoder(Pos)
        latent.append(lat)

pickle.dump(latent, open(output_deffnm+"_compressed.pkl", "wb"))

print(f"saved compressed data at {output_deffnm}_compressed.pkl")

compression = 100*(1 - os.path.getsize(f"{output_deffnm}_compressed.pkl")/os.path.getsize(trajfile))
print(f"compression  = {compression:.2f} %")

if len(glob("lightning_logs")):
    shutil.rmtree("lightning_logs")
if len(glob("*rmsfit*")):
    os.system("rm *rmsfit*")
