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
import pytorch_lightning as pl
import argparse
import shutil
from sklearn.preprocessing import MinMaxScaler
from glob import glob
from Classes import *
from Helpers import *

set_seed()

parser = argparse.ArgumentParser(description="Process input parameters.")
    
parser.add_argument("-mp", "--model-prt", type=str, required=True, help="Path to the model file for protein.")
parser.add_argument("-ml", "--model-lig", type=str, required=True, help="Path to the model file for ligand.")
parser.add_argument("-sp", "--scaler-prt", type=str, required=True, help="Path to the scaler file for protein.")
parser.add_argument("-sl", "--scaler-lig", type=str, required=True, help="Path to the scaler file for ligand.")
parser.add_argument("-r", "--reffile", type=str, required=True, help="Path to the reference file.")
parser.add_argument('-t', '--trajfile', type=str, help='Path to the actual trajectory file')
parser.add_argument("-cp", "--compressed-prt", type=str, required=True, help="Path to the compressed file for protein.")
parser.add_argument("-cl", "--compressed-lig", type=str, required=True, help="Path to the compressed file for ligand.")
parser.add_argument("-p", "--prefix", type=str, required=True, help="output file prefix.")
parser.add_argument('-lig', '--ligand', type=str, help='selection string for ligand, if present.', default=None)
parser.add_argument('-lcom', '--ligand-com', type=str, required=True, help='center of mass file for ligand.')
parser.add_argument('-sel', '--select', type=str, required=True,
                    help="""selection string for parts of the system except the ligand.
the program will append to this selection to exclude Hydrogens. Do not add that here.""")
parser.add_argument('-c', '--center', help="center system inside box. [default=off]", action="store_true")
parser.add_argument('-gid', '--gpuID', type=str, help="select GPU to use [default=0]", default=0)

args = parser.parse_args()

modelpath_prt = args.model_prt
modelpath_lig = args.model_lig
scalerpath_prt = args.scaler_prt
scalerpath_lig = args.scaler_lig
reffile = args.reffile
trajfile = args.trajfile
compressed_prt = args.compressed_prt
compressed_lig = args.compressed_lig
prefix = args.prefix
ligand_selection = args.ligand
ligand_com_path = args.ligand_com
selection = args.select
center = args.center
gpu_id = int(args.gpuID)

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using GPU:{gpu_id}.")

assert Universe(reffile).select_atoms(ligand_selection).n_atoms > 0, f"no ligand with selection '{ligand_selection}' found in the reference file."

# protein
model = torch.load(modelpath_prt).to(device)
decoder = model.model.decoder
del model
decoder = decoder.to(device)
latent = pickle.load(open(compressed_prt, "rb"))
scaler = pickle.load(open(scalerpath_prt, "rb"))

print("Decompressing protein.....")
pos_prt = []
with torch.no_grad():
    decoder.eval()
    for lat in latent:
        Pos = decoder(lat.to(device, dtype=torch.float32))
        pos_prt.append(Pos.cpu().numpy())

pos_prt = np.concatenate(pos_prt, axis=0)

pos_prt = scaler.inverse_transform(pos_prt)
pos_prt = pos_prt.reshape((pos_prt.shape[0], pos_prt.shape[1]//3, 3))

# ligand
model = torch.load(modelpath_lig).to(device)
decoder = model.model.decoder
del model
decoder = decoder.to(device)
latent = pickle.load(open(compressed_lig, "rb"))
scaler = pickle.load(open(scalerpath_lig, "rb"))

print("Decompressing ligand.....")
pos_lig = []
with torch.no_grad():
    decoder.eval()
    for lat in latent:
        Pos = decoder(lat.to(device, dtype=torch.float32))
        pos_lig.append(Pos.cpu().numpy())

pos_lig = np.concatenate(pos_lig, axis=0)

pos_lig = scaler.inverse_transform(pos_lig)
pos_lig = pos_lig.reshape((pos_lig.shape[0], pos_lig.shape[1]//3, 3))
com_lig = np.load(ligand_com_path)

ref = mda.Universe(reffile, reffile)
select = ref.select_atoms(f"({selection}) or ({ligand_selection})")

assert select.n_atoms == pos_prt.shape[1] + pos_lig.shape[1], "mismatch in the number of atoms in actual and decompressed trajectory."

if center:
    pos_prt += 0.5*ref.trajectory[0]._unitcell[3] # shifting center to center of box
    pos_lig += 0.5*ref.trajectory[0]._unitcell[3] # shifting center to center of box

with mda.Writer(prefix+"_decompressed.xtc", "w") as w:
    for prt, lig, lcom in zip(tqdm(pos_prt, desc="Writing decompressed trajectory"), pos_lig, com_lig):
        select.positions = np.vstack((prt, lig + lcom))
        w.write(select)

print(f"decompressed trajectory saved to {prefix}_decompressed.xtc.")

if trajfile is None:
    exit(0)

print("loading trajectories for RMSD calculation.")

ref = Universe(reffile)
traj1 = Universe(reffile, trajfile)

with Writer("temp.pdb", "w") as w:
    w.write(ref.select_atoms(f"({selection}) or ({ligand_selection})"))

traj2 = Universe("temp.pdb", prefix+"_decompressed.xtc")
os.system("rm -rf temp.pdb")

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

# calcaulating and saving RMSD
print("calculating RMSD between original and decompressed trajectory.")

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory, traj2.trajectory), desc="Computing RMSD", total=len(traj1.trajectory)):
    pos1 = traj1.select_atoms(f"{selection} or {ligand_selection}").positions - traj1.select_atoms(f"{selection} or {ligand_selection}").center_of_geometry()
    pos2 = traj2.select_atoms(f"{selection} or {ligand_selection}").positions - traj2.select_atoms(f"{selection} or {ligand_selection}").center_of_geometry()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(prefix+"_rmsd.txt", rmsd, fmt="%.3f", header="RMSD in nm")
print(f"RMSD values saved in {prefix}_rmsd.txt")

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm")

# removing unwanted files
if len(glob("lightning_logs")):
    shutil.rmtree("lightning_logs")
if len(glob("*rmsfit*")):
    os.system("rm *rmsfit*")
