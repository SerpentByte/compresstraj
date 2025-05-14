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

sys.path.append(os.environ["COMPRESSTRAJ_LIB"])
from classes import *
from helpers import *

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
parser.add_argument('-o', '--outdir', type=str, help='output directory', default=".")

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
outdir = args.outdir

device = torch.device(f"cuda:{gpu_id}") if torch.cuda.is_available() else torch.device("cpu")
print(f"Using device: {device}.", flush=True)

assert Universe(reffile).select_atoms(ligand_selection).n_atoms > 0, f"no ligand with selection '{ligand_selection}' found in the reference file."

# protein
model = torch.load(modelpath_prt, weights_only=False).to(device)
decoder_prt = model.decoder
decoder_prt.eval()

del model
decoder_prt = decoder_prt.to(device)
latent_prt = pickle.load(open(compressed_prt, "rb"))
scaler_prt = pickle.load(open(scalerpath_prt, "rb"))

# ligand
model = torch.load(modelpath_lig, weights_only=False).to(device)
decoder_lig = model.decoder
decoder_lig.eval() 

del model
decoder_lig = decoder_lig.to(device)
latent_lig = pickle.load(open(compressed_lig, "rb"))
scaler_lig = pickle.load(open(scalerpath_lig, "rb"))
com_lig = np.load(ligand_com_path)

batch_size = latent_lig[0].shape[0]

print("Decompressing .....", flush=True)

with Writer(f"{outdir}/{prefix}_decompressed.xtc", "w") as w:
    ref = Universe(reffile)
    select = ref.select_atoms(f"({selection}) or ({ligand_selection})")
    
    with torch.no_grad():
        decoder_prt.eval()
        decoder_lig.eval()
    
        for frame_idx, (lat_prt, lat_lig) in enumerate(zip(latent_prt, latent_lig)):
            pos_prt = decoder_prt(lat_prt.to(device, dtype=torch.float32)).detach().cpu().numpy()
            pos_lig = decoder_lig(lat_lig.to(device, dtype=torch.float32)).detach().cpu().numpy()
            lig_com_batch = com_lig[frame_idx*batch_size:(frame_idx + 1)*batch_size]

            pos_prt = pos_prt.reshape((pos_prt.shape[0], pos_prt.shape[1]//3, 3))
            pos_prt = scaler_prt.inverse_transform(pos_prt)
            
            pos_lig = pos_lig.reshape((pos_lig.shape[0], pos_lig.shape[1]//3, 3))
            pos_lig = scaler_lig.inverse_transform(pos_lig) 
            pos_lig += np.expand_dims(lig_com_batch, axis=1)

            positions = np.concatenate((pos_prt, pos_lig), axis=1)

            assert select.n_atoms == positions.shape[1], "mismatch in the number of atoms in actual and decompressed trajectory."
            
            for pos in positions:
                select.positions = pos
                w.write(select)

print(f"decompressed trajectory saved to {outdir}/{prefix}_decompressed.xtc.", flush=True)

if trajfile is None:
    exit(0)

print("loading trajectories for RMSD calculation.", flush=True)

ref = Universe(reffile)
traj1 = Universe(reffile, trajfile)

with Writer("temp.pdb", "w") as w:
    w.write(ref.select_atoms(f"({selection}) or ({ligand_selection})"))

traj2 = Universe("temp.pdb", f"{outdir}/{prefix}_decompressed.xtc")

assert len(traj1.trajectory) == len(traj2.trajectory), "Trajectories must be of the same length."

# calcaulating and saving RMSD
print("calculating RMSD between original and decompressed trajectory.", flush=True)

rmsd = []
for t1, t2 in tqdm(zip(traj1.trajectory, traj2.trajectory), desc="Computing RMSD", total=len(traj1.trajectory)):
    pos1 = traj1.select_atoms(f"{selection} or {ligand_selection}").positions - traj1.select_atoms(f"{selection} or {ligand_selection}").center_of_geometry()
    pos2 = traj2.select_atoms(f"{selection} or {ligand_selection}").positions - traj2.select_atoms(f"{selection} or {ligand_selection}").center_of_geometry()
    rmsd.append(np.mean((pos1 - pos2)**2))

rmsd = np.sqrt(rmsd)*0.1 # to nm

np.savetxt(f"{outdir}/{prefix}_rmsd.txt", rmsd, fmt="%.3f", header="RMSD in nm")
print(f"RMSD values saved in {outdir}/{prefix}_rmsd.txt", flush=True)

print(f"RMSD = ({rmsd.min():.2f}, {rmsd.max():.2f}) nm", flush=True)

# removing unwanted files
if len(glob(f"lightning_logs")):
    shutil.rmtree(f"lightning_logs")
if len(glob(f"{outdir}/*rmsfit*")):
    os.system(f"rm {outdir}/*rmsfit*")
os.system("rm -rf temp.pdb")
