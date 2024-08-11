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
import pytorch_lightning as pl
import argparse 
import shutil
from glob import glob
sys.path.append("/data/wabdul/compressTraj/lib")
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
parser.add_argument('-lig', '--ligand', type=str, required=True, help='selection string for ligand')
parser.add_argument('-sel', '--select', type=str, required=True,
                    help="""selection string for parts of the system except the ligand.
the program will append to this selection to exclude Hydrogens. Do not add that here.""")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default=0)

args = parser.parse_args()

reffile = args.reffile
trajfile = args.trajfile
output_deffnm = args.prefix
latent = args.latent
compression = float(args.compression)
ligand_selection = args.ligand
selection = args.select
gpu_id = int(args.gpuID)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using GPU:{gpu_id}.")

assert Universe(reffile).select_atoms(ligand_selection).n_atoms > 0, f"no ligand with selection '{ligand_selection}' found."

# processing trajectory
scaler = MinMaxScaler()
pos_prt, pos_lig, lig_com = pool(reffile, trajfile, selection=selection, ligand=ligand_selection)
pos_prt = pos_prt.astype("float32")
pos_lig = pos_lig.astype("float32")

# protein compression
pos = pos_prt.copy()
X_train, X_val, _, _ = train_test_split(pos, pos, test_size=0.1)

X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

train_loader = DataLoader(TrajLoader(X_train), shuffle=True, batch_size=256, num_workers=8)
val_loader = DataLoader(TrajLoader(X_val), shuffle=False, batch_size=256, num_workers=8)
del X_train, X_val

pickle.dump(scaler, open(output_deffnm+"_prt_scaler.pkl", "wb"))
print(f"""Trajectory for protein processed.
scaler for protein saved at {output_deffnm}_prt_scaler.pkl""")

with Writer(output_deffnm+"_select.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms(f"({selection}) or ({ligand_selection})"))
print(f"selected atom coordinates for the protein saved at {output_deffnm}_prt_select.pdb")

## training AE
N = train_loader.dataset.get_N()
latent = int(np.ceil(N/compression)) if latent is None else int(args.latent)

### Initialize the LightningModule
ae = DenseAutoEncoder(N=N, latent=latent)
model = LightAutoEncoder(model=ae, learning_rate=3e-4)

### Initialize the Trainer with the logger
trainer = pl.Trainer(max_epochs=int(args.epochs))

### Train the model
trainer.fit(model, train_loader, val_loader)

### saving model and losses
torch.save(model, output_deffnm+"_prt_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(output_deffnm+"_prt_losses.pkl", "wb"))

print(f"""AutoEncoder model for protein saved at {output_deffnm}_prt_model.pt
Losses saved at {output_deffnm}_prt_losses.pkl""")

del train_loader, val_loader

## compress
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

pickle.dump(latent, open(output_deffnm+"_prt_compressed.pkl", "wb"))

print(f"saved compressed data for protein at {output_deffnm}_prt_compressed.pkl")
del pos_prt

# ligand
pos = pos_lig.copy()

X_train = scaler.fit_transform(pos)

train_loader = DataLoader(TrajLoader(X_train), shuffle=True, batch_size=256, num_workers=16)
del X_train

pickle.dump(scaler, open(output_deffnm+"_lig_scaler.pkl", "wb"))
print(f"""Trajectories processed.
scaler for ligand saved at {output_deffnm}_prt_scaler.pkl""")

with Writer(output_deffnm+"_lig_select.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms(ligand_selection))
print(f"selected atom coordinates for the ligand saved at {output_deffnm}_lig_select.pdb")

## training AE
N = train_loader.dataset.get_N()
latent = int(np.ceil(N/compression))
if latent <= 1:
    latent = 2

### Initialize the LightningModule
ae = DenseAutoEncoder(N=N, latent=latent)
model = LightAutoEncoder(model=ae, learning_rate=3e-4)

### Initialize the Trainer with the logger
trainer = pl.Trainer(max_epochs=int(args.epochs))

### Train the model
trainer.fit(model, train_loader)

### saving model and losses
torch.save(model, output_deffnm+"_lig_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(output_deffnm+"_lig_losses.pkl", "wb"))

print(f"""AutoEncoder model for ligand saved at {output_deffnm}_lig_model.pt
Losses saved at {output_deffnm}_lig_losses.pkl""")

del train_loader

## compress
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

pickle.dump(latent, open(output_deffnm+"_lig_compressed.pkl", "wb"))

print(f"saved compressed data for protein at {output_deffnm}_lig_compressed.pkl")
del pos_lig
del pos

np.save(f"{output_deffnm}_lig_com.npy", lig_com)
print(f"saved the ligand COM in {output_deffnm}_lig_com.npy")

prt_size = os.path.getsize(f"{output_deffnm}_prt_compressed.pkl")
lig_size = os.path.getsize(f"{output_deffnm}_lig_compressed.pkl")
compression = 100*(1 - (prt_size + lig_size)/os.path.getsize(trajfile))
print(f"compression  = {compression:.2f} %")
    
    

# remove unwanted stuff generated
if len(glob("lightning_logs")):
    shutil.rmtree("lightning_logs")
if len(glob("*rmsfit*")):
    os.system("rm *rmsfit*")
