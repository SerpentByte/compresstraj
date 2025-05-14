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
from glob import glob
import json

from compresstraj.classes import *
from compresstraj.helpers import *

torch.set_float32_matmul_precision('medium')
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
parser.add_argument('-ckpt', '--ckpt', type=str, help="checkpoint from where to resume training", default=None)
parser.add_argument('-b', '--batch', type=str, help='batch size [default=128]', default=128)
parser.add_argument('--layers', '-layers', type=str, default="2048,1024",
                    help="Comma-separated list of hidden layer sizes for the autoencoder, e.g., '2048,1024'")
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")

args = parser.parse_args()

reffile = args.reffile
trajfile = args.trajfile
prefix = args.prefix
latent = args.latent
compression = float(args.compression)
ligand_selection = args.ligand
selection = args.select
gpu_id = int(args.gpuID)
layers = list(map(int, args.layers.strip().split(",")))
outdir = args.outdir

os.makedirs(outdir, exist_ok=True)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

assert Universe(reffile).select_atoms(ligand_selection).n_atoms > 0, f"no ligand with selection '{ligand_selection}' found."

# PROTEIN COMPRESSION
## processing trajectory
traj = Universe(reffile, trajfile)

scaler_prt = StandardScaler()
scaler_prt.fit(traj, selection=selection)

indices_prt = traj.select_atoms(selection).indices
train_loader_prt = DataLoader(TrajLoader(traj, indices_prt, scaler_prt), batch_size=int(args.batch), shuffle=True)

pickle.dump(scaler_prt, open(f"{outdir}/{prefix}_prt_scaler.pkl", "wb"))
print(f"""Trajectory for protein processed.
scaler for protein saved at {outdir}/{prefix}_prt_scaler.pkl""", flush=True)

with Writer(f"{outdir}/{prefix}_select.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms(f"({selection}) or ({ligand_selection})"))
print(f"selected atom coordinates for the protein saved at {outdir}/{prefix}_prt_select.pdb", flush=True)

## training AE
N = train_loader_prt.dataset.get_N()
latent = int(np.ceil(N/compression)) if latent is None else int(args.latent)
ae = DenseAutoEncoder(N=N, latent=latent, layers=layers)

### Initialize the LightningModule
loss_fn = RMSDLoss()

model = LightAutoEncoder(model=ae, loss_fn=loss_fn, learning_rate=3e-4)

restart_ckpt_callback = ModelCheckpoint(
    dirpath=outdir,
    filename="restart_prt",
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

trainer.fit(model, train_loader_prt, ckpt_path=args.ckpt if args.ckpt else None)

### saving model and losses
torch.save(model.model, f"{outdir}/{prefix}_prt_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(f"{outdir}/{prefix}_prt_losses.pkl", "wb"))

print(f"""Final model for protein saved at {outdir}/{prefix}_prt_model.pt
Losses saved at {outdir}/{prefix}_prt_losses.pkl""", flush=True)

## compress
train_loader_prt = DataLoader(TrajLoader(traj, indices_prt, scaler_prt), batch_size=int(args.batch), shuffle=False)
latent = []

model = torch.load(f"{outdir}/{prefix}_prt_model.pt", weights_only=False)
encoder = model.encoder
del model
encoder = encoder.to(device)

with torch.no_grad():
    encoder.eval()
    for Pos in train_loader_prt:
        Pos = Pos.to(device=device, dtype=torch.float32)
        lat = encoder(Pos)
        latent.append(lat)

pickle.dump(latent, open(f"{outdir}/{prefix}_prt_compressed.pkl", "wb"))

print(f"saved compressed data for protein at {outdir}/{prefix}_prt_compressed.pkl", flush=True)

# LIGAND COMPRESSION
scaler_lig = MinMaxScaler()
scaler_lig.fit(traj, selection=ligand_selection)

indices_lig = traj.select_atoms(ligand_selection).indices
train_loader_lig = DataLoader(TrajLoader(traj, indices_lig, scaler_lig), batch_size=int(args.batch), shuffle=True)

pickle.dump(scaler_lig, open(f"{outdir}/{prefix}_lig_scaler.pkl", "wb"))
print(f"""Trajectory processed.
scaler for ligand saved at {outdir}/{prefix}_lig_scaler.pkl""", flush=True)

with Writer(f"{outdir}/{prefix}_lig_select.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms(ligand_selection))
print(f"selected atom coordinates for the ligand saved at {outdir}/{prefix}_lig_select.pdb", flush=True)

## training AE
N = train_loader_lig.dataset.get_N()
latent = int(np.ceil(N/compression))
if latent <= 1:
    latent = 2

### Initialize the LightningModule
ae = DenseAutoEncoder(N=N, latent=latent, layers=layers)
loss_fn = RMSDLoss()

model = LightAutoEncoder(model=ae, loss_fn=loss_fn, learning_rate=3e-4)

restart_ckpt_callback = ModelCheckpoint(
    dirpath=outdir,
    filename="restart_lig",
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

trainer.fit(model, train_loader_lig, ckpt_path=args.ckpt if args.ckpt else None)

### saving model and losses
torch.save(model.model, f"{outdir}/{prefix}_lig_model.pt")
pickle.dump({"train_loss":model.train_loss},
              open(f"{outdir}/{prefix}_lig_losses.pkl", "wb"))

print(f"""Final model for ligand saved at {outdir}/{prefix}_lig_model.pt
Losses saved at {outdir}/{prefix}_lig_losses.pkl""", flush=True)

## compress
train_loader_lig = DataLoader(TrajLoader(traj, indices_lig, scaler_lig), batch_size=int(args.batch), shuffle=False)
latent = []

model = torch.load(f"{outdir}/{prefix}_lig_model.pt", weights_only=False)
encoder = model.encoder
del model
encoder = encoder.to(device)

with torch.no_grad():
    encoder.eval()
    for Pos in train_loader_lig:
        Pos = Pos.to(device=device, dtype=torch.float32)
        lat = encoder(Pos)
        latent.append(lat)

pickle.dump(latent, open(f"{outdir}/{prefix}_lig_compressed.pkl", "wb"))

print(f"saved compressed data for ligand at {outdir}/{prefix}_lig_compressed.pkl", flush=True)

lig_com = np.array([traj.select_atoms(ligand_selection).center_of_geometry() - traj.select_atoms(selection).center_of_geometry() for t in traj.trajectory])

np.save(f"{outdir}/{prefix}_lig_com.npy", lig_com)
print(f"saved the ligand COM in {outdir}/{prefix}_lig_com.npy", flush=True)

with open(f"{outdir}/config.json", "w") as f:
    json.dump({"protein": selection,
               "ligand": ligand_selection}, f, indent=4)

print(f"Saved selections to {outdir}/config.json", flush=True)

prt_size = os.path.getsize(f"{outdir}/{prefix}_prt_compressed.pkl")
lig_size = os.path.getsize(f"{outdir}/{prefix}_lig_compressed.pkl")
compression = 100*(1 - (prt_size + lig_size)/os.path.getsize(trajfile))
print(f"compression  = {compression:.2f} %", flush=True)
    
    

# remove unwanted stuff generated
if len(glob(f"lightning_logs")):
    shutil.rmtree(f"lightning_logs")
if len(glob(f"{outdir}/*rmsfit*")):
    os.system(f"rm {outdir}/*rmsfit*")
