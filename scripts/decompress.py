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

parser = argparse.ArgumentParser(description="Process input parameters.")
    
parser.add_argument("-m", "--model", type=str, required=True, help="Path to the model file.")
parser.add_argument("-s", "--scaler", type=str, required=True, help="Path to the scaler file.")
parser.add_argument("-r", "--reffile", type=str, required=True, help="Path to the reference file.")
parser.add_argument("-o", "--output", type=str, required=True, help="Path to the output file.")
parser.add_argument("-c", "--compressed", type=str, required=True, help="Path to the compressed file.")
parser.add_argument("-p", "--padding", type=str, required=True, help="Path to the padding file.")

args = parser.parse_args()

modelpath = args.model
scalerpath = args.scaler
reffile = args.reffile
output = args.output
compressed = args.compressed
padfile = args.padding

model = torch.load(modelpath).to(device)
latent = pickle.load(open(compressed, "rb"))
# padding = int(np.loadtxt(padfile).astype("int"))
scaler = pickle.load(open(scalerpath, "rb"))

pos = []
with torch.no_grad():
    model.eval()
    for lat in latent:
        Pos = model.model.decoder(lat.to(device, dtype=torch.float32))
        pos.append(Pos.cpu().numpy())

pos = np.concatenate(pos, axis=0)

pos = auto_reverse(pos, scaler=scaler)
pos = pos.reshape((pos.shape[0], pos.shape[1]//3, 3))

ref = mda.Universe(reffile, reffile)
select = ref.select_atoms("not name H*")

with mda.Writer(output, "w") as w:
    for p in pos:
        select.positions = p
        w.write(select)

print(f"decompressed trajectory saved to {output}")
