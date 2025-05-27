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
from MDAnalysis.analysis import align
import MDAnalysis as mda
import pytorch_lightning as pl
import argparse
import warnings
import shutil
from glob import glob

from compresstraj.classes import *
from compresstraj.helpers import *

warnings.filterwarnings("ignore")

set_seed()

parser = argparse.ArgumentParser(description="Process input parameters.")
    
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model file.")
parser.add_argument("-s", "--scaler", type=str, required=True, help="Path to the scaler file.")
parser.add_argument("-r", "--reffile", type=str, required=True, help="Path to the reference file.")
parser.add_argument('-t', '--trajfile', type=str, help='Path to the trajectory file 1 (xtc)')
parser.add_argument("-c", "--compressed", type=str, required=True, help="Path to the compressed file.")
parser.add_argument("-p", "--prefix", type=str, required=True, help="output file prefix.")
parser.add_argument('-sel', '--selection', type=str, default="all",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default="0")
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")
parser.add_argument('-cog', '--cog', type=str, default=None, help='center of geometry.')

args = parser.parse_args()

modelpath = args.model
scalerpath = args.scaler
reffile = args.reffile
trajfile = args.trajfile
compressed = args.compressed
prefix = args.prefix
selection = args.selection
gpu_id = [int(idx) for idx in args.gpuID.split(",")]
outdir = args.outdir
cog_path = args.cog

device = torch.device(f"cuda:{'.'.join([str(idx) for idx in gpu_id])}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

ref = Universe(reffile)
select = ref.select_atoms(selection)

model = torch.load(modelpath, weights_only=False)
decoder = model.decoder.to(device)
decoder.eval()  
del model

latent = pickle.load(open(compressed, "rb"))
scaler = pickle.load(open(scalerpath, "rb"))

if cog_path:
    cog = np.load(cog_path)
    batch_size = latent[0].shape[0]

print("Decompressing.....", flush=True)

with mda.Writer(f"{outdir}/{prefix}_decompressed.xtc", "w") as w:
    with torch.no_grad():
        for frame_idx, lat in enumerate(latent):
            pos = decoder(lat.to(device, dtype=torch.float32)).cpu().numpy()
            pos = pos.reshape((pos.shape[0], pos.shape[1]//3, 3))
            pos = scaler.inverse_transform(pos)
            pos -= pos.mean(axis=1, keepdims=True)

            if cog_path:
                cog_batch = cog[frame_idx*batch_size:(frame_idx + 1)*batch_size]
                pos += np.expand_dims(cog_batch, axis=1)

            for p in pos:
                select.positions = p
                w.write(select)

print(f"decompressed trajectory saved to {outdir}/{prefix}_decompressed.xtc.", flush=True)

if trajfile is None:
    exit(0)
    
ref = Universe(reffile)
traj1 = Universe(reffile, trajfile)

with Writer("temp.pdb", "w") as w:
    w.write(ref.select_atoms(selection))
traj2 = Universe("temp.pdb", f"{outdir}/{prefix}_decompressed.xtc")

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

# calcaulating and saving RMSD
print("calculating RMSD between original and decompressed trajectory.", flush=True)

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory[:len(traj2.trajectory)], traj2.trajectory), desc="Computing RMSD", total=len(traj2.trajectory)):
    pos1 = traj1.select_atoms(selection).positions - traj1.select_atoms(selection).center_of_geometry()
    pos2 = traj2.select_atoms(selection).positions - traj2.select_atoms(selection).center_of_geometry()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(f"{outdir}/{prefix}_rmsd.txt", rmsd, fmt="%.3f", header="RMSD in nm")
print(f"RMSD values saved in {outdir}/{prefix}_rmsd.txt", flush=True)

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm", flush=True)

os.system("rm -rf temp.pdb")
