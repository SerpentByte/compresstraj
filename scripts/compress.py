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
from pytorch_lightning.callbacks import ModelCheckpoint
import argparse 
import shutil
from natsort import natsorted
from glob import glob
import json

from compresstraj.classes import *
from compresstraj.helpers import *

# some stuff that effects the whole code
torch.set_float32_matmul_precision('medium')
set_seed()
# some stuff that effects the whole code

# argument parsing
parser = argparse.ArgumentParser(description="Process input files and model files for compression.")
    
# Add arguments for model, reffile, trajfile, scaler, and output with short and long options
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t', '--trajfile', type=str, required=True, help='Path to the trajectory file (xtc/trr/dcd/xyz)')
parser.add_argument('-p', '--prefix', type=str, required = True, help='prefix to to the files to be generated.')
parser.add_argument('-e', '--epochs', type=str, help='Number of epochs to train [default=200]', default=200)
parser.add_argument('-b', '--batch', type=str, help='batch size [default=128]', default=128)
parser.add_argument('-l', '--latent', type=str, help='Number of latent dims', default=None)
parser.add_argument('-c', '--compression', type=str, help='Extent of compression to achieve if latent dimension is not specified. [default = 20]', default=20)
parser.add_argument('-sel', '--selection', type=str, default="not element H",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default=0)
parser.add_argument('-ckpt', '--ckpt', type=str, help="checkpoint from where to resume training", default=None)
parser.add_argument('--layers', '-layers', type=str, default="2048,1024",
                    help="Comma-separated list of hidden layer sizes for the autoencoder, e.g., '2048,1024'")
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")

args = parser.parse_args()

reffile = args.reffile
trajfile = args.trajfile
prefix = args.prefix
latent = args.latent
compression = int(args.compression)
selection = args.selection
gpu_id = int(args.gpuID)
layers = list(map(int, args.layers.strip().split(",")))
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

# processing trajectory
traj = Universe(reffile, trajfile)
scaler = StandardScaler()
scaler.fit(traj, selection=selection)

indices = traj.select_atoms(selection).indices
train_loader = DataLoader(TrajLoader(traj, indices, scaler), batch_size=int(args.batch), shuffle=True)

pickle.dump(scaler, open(f"{outdir}/{prefix}_scaler.pkl", "wb"))
print(f"""scaler saved at {outdir}/{prefix}_scaler.pkl""", flush=True)

with Writer(f"{outdir}/{prefix}_select.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms(selection))
print(f"selected atom coordinates saved at {outdir}/{prefix}_select.pdb", flush=True)

# training AE
# check for checkpoint
N = train_loader.dataset.get_N()
latent = int(np.ceil(N/compression)) if latent is None else int(args.latent)

u = Universe(reffile)

ae = DenseAutoEncoder(N=N, latent=latent, layers=layers)

# loss_fn = CombinedCoordinateLoss(l_rmsd=1.0, l_dist=1.0, indices=u.select_atoms("element O").indices)
loss_fn = RMSDLoss()

model = LightAutoEncoder(model=ae, loss_fn=loss_fn, learning_rate=3e-4)

restart_ckpt_callback = ModelCheckpoint(
    dirpath=outdir,
    filename="restart",
    every_n_epochs=10,            # <<< save every e epochs
    save_top_k=1,               # <<< save the last one
    save_weights_only=False,     # <<< save full model + optimizer
)


accelerator="gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(max_epochs=int(args.epochs),
                     accelerator=accelerator,
                     devices=[gpu_id if accelerator == "gpu" else 0],
                     precision="32-true", enable_checkpointing=True, logger=None,
                     callbacks=[restart_ckpt_callback])
    
trainer.fit(model, train_loader, ckpt_path=args.ckpt if args.ckpt else None)

## saving model and losses
torch.save(model.model, f"{outdir}/{prefix}_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(f"{outdir}/{prefix}_losses.pkl", "wb"))

print(f"""Final model saved at {outdir}/{prefix}_model.pt
Losses saved at {outdir}/{prefix}_losses.pkl""", flush=True)

del train_loader

# compress
loader = DataLoader(TrajLoader(traj, indices, scaler), batch_size=int(args.batch), shuffle=False)

latent = []

model = torch.load(f"{outdir}/{prefix}_model.pt", weights_only=False)
encoder = model.encoder
del model
encoder = encoder.to(device)

with torch.no_grad():
    encoder.eval()
    for Pos in loader:
        Pos = Pos.to(device=device, dtype=torch.float32)
        lat = encoder(Pos).to(device="cpu")
        latent.append(lat)

pickle.dump(latent, open(f"{outdir}/{prefix}_compressed.pkl", "wb"))

print(f"saved compressed data at {outdir}/{prefix}_compressed.pkl", flush=True)

with open(f"{outdir}/config.json", "w") as f:
    json.dump({"selection": selection}, f, indent=4)

print(f"Saved selections to {outdir}/config.json", flush=True)

compression = 100*(1 - os.path.getsize(f"{outdir}/{prefix}_compressed.pkl")/os.path.getsize(trajfile))
print(f"compression  = {compression:.2f} %", flush=True)

if len(glob(f"lightning_logs")):
    shutil.rmtree(f"lightning_logs")
if len(glob(f"{outdir}/*rmsfit*")):
    os.system(f"rm {outdir}/*rmsfit*")
if len(glob(f"{outdir}/frame*.pkl")):
    os.system(f"rm {outdir}/frame*.pkl")
