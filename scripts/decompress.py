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
parser.add_argument('-sel', '--selection', type=str, default="not element H",
                    help="a list of selections. the training will treat each selection as a separated entity.")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default=0)
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")

args = parser.parse_args()

modelpath = args.model
scalerpath = args.scaler
reffile = args.reffile
trajfile = args.trajfile
compressed = args.compressed
prefix = args.prefix
selection = args.selection
gpu_id = int(args.gpuID)
outdir = args.outdir

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

ref = Universe(reffile)
select = ref.select_atoms(selection)

model = torch.load(modelpath, weights_only=False)
decoder = model.decoder.to(device)
decoder.eval()  
del model

latent = pickle.load(open(compressed, "rb"))
scaler = pickle.load(open(scalerpath, "rb"))

print("Decompressing.....", flush=True)

with mda.Writer(f"{outdir}/{prefix}_decompressed.xtc", "w") as w:
    with torch.no_grad():
        for lat in latent:
            pos = decoder(lat.to(device, dtype=torch.float32)).cpu().numpy()

            pos = pos.reshape((pos.shape[0], pos.shape[1]//3, 3))
            pos = scaler.inverse_transform(pos)

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

if len(glob(f"lightning_logs")):
    shutil.rmtree(f"lightning_logs")
if len(glob(f"{outdir}/*rmsfit*")):
    os.system(f"rm {outdir}/*rmsfit*")
os.system("rm -rf temp.pdb")
os.system(f"rm .{prefix}*")
