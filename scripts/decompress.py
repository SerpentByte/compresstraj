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
import pytorch_lightning as pl
import argparse
import warnings
import shutil
from glob import glob
sys.path.append("/data/wabdul/compressTraj/lib")
from Classes import *
from Helpers import *

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

args = parser.parse_args()

modelpath = args.model
scalerpath = args.scaler
reffile = args.reffile
trajfile = args.trajfile
compressed = args.compressed
prefix = args.prefix
selection = args.selection
gpu_id = int(args.gpuID)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using GPU:{gpu_id}.")

model = torch.load(modelpath).to(device)
decoder = model.model.decoder
del model
decoder = decoder.to(device)
latent = pickle.load(open(compressed, "rb"))
scaler = pickle.load(open(scalerpath, "rb"))

print("Decompressing.....")
pos = []
with torch.no_grad():
    decoder.eval()
    for lat in latent:
        Pos = decoder(lat.to(device, dtype=torch.float32))
        pos.append(Pos.cpu().numpy())

pos = np.concatenate(pos, axis=0)

pos = auto_reverse(pos, scaler=scaler)
pos = pos.reshape((pos.shape[0], pos.shape[1]//3, 3))

ref = mda.Universe(reffile, reffile)
select = ref.select_atoms(selection)

with mda.Writer(prefix+"_decompressed.xtc", "w") as w:
    for p in pos:
        select.positions = p
        w.write(select)

print(f"decompressed trajectory saved to {prefix}_decompressed.xtc.")

if trajfile is None:
    exit(0)
    
ref = Universe(reffile)
traj1 = Universe(reffile, trajfile)

with Writer("temp.pdb", "w") as w:
    w.write(ref.select_atoms(selection))
traj2 = Universe("temp.pdb", prefix+"_decompressed.xtc")
os.system("rm -rf temp.pdb")

fit1 = align.AlignTraj(traj1, ref, select=selection)
fit1.run()

fit2 = align.AlignTraj(traj2, ref, select=selection)
fit2.run()

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

# calcaulating and saving RMSD
print("calculating RMSD between original and decompressed trajectory.")

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory, traj2.trajectory), desc="Computing RMSD", total=len(traj1.trajectory)):
    pos1 = traj1.select_atoms(selection).positions - traj1.select_atoms(selection).center_of_mass()
    pos2 = traj2.select_atoms(selection).positions - traj2.select_atoms(selection).center_of_mass()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(prefix+"_rmsd.txt", rmsd, fmt="%.3f", header="RMSD in nm")
print(f"RMSD values saved in {prefix}_rmsd.txt")

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm")

if len(glob("lightning_logs")):
    shutil.rmtree("lightning_logs")
if len(glob("*rmsfit*")):
    os.system("rm *rmsfit*")
