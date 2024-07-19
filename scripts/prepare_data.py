import numpy as np
import torch
from MDAnalysis import Universe, Writer
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import os
import sys
import pickle
import argparse
import warnings
from compressTraj import *

warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser(description="Process input files and output prefix.")
    
# Add arguments for reffile, trajfile, and output_deffnm
parser.add_argument('-r', '--reffile', type=str, required=True, help='Path to the reference file (PDB)')
parser.add_argument('-t', '--trajfile', type=str, required=True, help='Path to the trajectory file (XTC)')
parser.add_argument('-p', '--prefix', type=str, required=True, help='Prefix for the output files')

# Parse the arguments
args = parser.parse_args()

# Access the arguments
reffile = args.reffile
trajfile = args.trajfile
output_deffnm = args.prefix

train_output = output_deffnm + "_train.pkl"
val_output = output_deffnm + "_val.pkl"

scaler = MinMaxScaler()
pos = auto_process(reffile, trajfile, scaler=scaler)
pos = pos.astype("float32")

X_train, X_val, _, _ = train_test_split(pos, pos, test_size=0.1)

train_loader = DataLoader(TrajLoader(X_train), shuffle=True, batch_size=256, num_workers=8)
val_loader = DataLoader(TrajLoader(X_val), shuffle=False, batch_size=256, num_workers=8)

pickle.dump(train_loader, open(output_deffnm+"_train.pkl", "wb"))
pickle.dump(val_loader, open(output_deffnm+"_val.pkl", "wb"))
pickle.dump(scaler, open(output_deffnm+"_scaler.pkl", "wb"))
# with open(output_deffnm+"_padding.txt", "w") as w:
#     w.write(f"{padding}\n")

with Writer(output_deffnm+"_heavy.pdb", "w") as w:
    u = Universe(reffile)
    w.write(u.select_atoms("not name H*"))

print(f"""heavy atom pdb saved at {output_deffnm}_heavy.pdb
training data loader saved at {output_deffnm+"_train.pkl"}
validation data loader saved at {output_deffnm+"_val.pkl"}
padding value saved at {output_deffnm+"_padding.txt"}""")

if scaler:
    print(f"""scaler saved at {output_deffnm+"_scaler.pkl"}""")
