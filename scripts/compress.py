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
import gzip as gz
import warnings 
sys.path.append("lib")
from Classes import *
from Helpers import *

warnings.filterwarnings("ignore")

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed()

parser = argparse.ArgumentParser(description="Process input files and model files for compression.")
    
# Add arguments for model, reffile, trajfile, scaler, and output with short and long options
parser.add_argument('-m', '--model', type=str, required=True, help='Path to the model file (pt)')
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (pdb/gro)')
parser.add_argument('-t', '--trajfile', type=str, required=True, help='Path to the trajectory file (xtc)')
parser.add_argument('-o', '--output', type=str, required=True, help='Path to the output compressed file (pkl)')

args = parser.parse_args()

model = args.model
reffile = args.reffile
trajfile = args.trajfile
output = args.output

scaler = MinMaxScaler()
pos = auto_process(reffile, trajfile, scaler)

loader = DataLoader(TrajLoader(pos), shuffle=False, batch_size=256, num_workers=8)
latent = []
model = torch.load(model).to(device)

with torch.no_grad():
    model.eval()
    for Pos in loader:
        Pos = Pos.to(device=device, dtype=torch.float32)
        lat = model.model.encoder(Pos)
        latent.append(lat)

pickle.dump(latent, open(output, "wb"))

print(f"saving compressed data at {output}")