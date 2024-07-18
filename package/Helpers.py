import numpy as np
import torch
import random
import os
import MDAnalysis as mda
from tqdm.auto import tqdm
from MDAnalysis.analysis import align

def set_seed(seed=42):
    r""" A function to set all seeds.
seed : the value of the seed. [default = 42]
"""
    random.seed(seed)
    
    # Set seed for NumPy
    np.random.seed(seed)
    
    # Set seed for PyTorch (both CPU and GPU)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # If using multi-GPU.
        
    # Ensure that operations are deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


    # Set seed for system randomness (used in some libraries)
    os.environ['PYTHONHASHSEED'] = str(seed)


def pool(reffile, trajfile, start=0, stop=-1, step=1, selection=None):
    r"""A function to pool all coordinates from starting frame to end frame, skipping the supplied number of steps
for the selection provided.
reffile : reference structure file. [pdb,gro]
trajfile : trajectory file. [xtc,trr]
start : starting frame [default = 0]
stop : end frame [default = last frame]
selection : atom selection [default = all]
"""
    if selection is None:
        selection = "not name H*"
        
    traj = mda.Universe(reffile, trajfile)
    ref = mda.Universe(reffile)

    fit = align.AlignTraj(traj, ref, select="all")
    fit.run()
    
    if stop == -1:
        stop = len(traj.trajectory)

    pos = []
    for t in tqdm(traj.trajectory[start:stop:step], desc="processing trajectory"):
        coords = traj.select_atoms(selection).positions - traj.select_atoms(selection).center_of_mass()
        pos.append(coords.flatten())
    return np.array(pos, dtype="float32")


def zero_pad(pos):
    r"""to zero pad number of particles to the nearest integer divisible by 8.
frame : a single frame of a trajectory. it should be of the shape (N, 3).
"""
    N = pos.shape[1]
    nearest = np.ceil(N/8)*8
    padding = int(nearest - N)

    return np.pad(pos, ((0, 0), (0, padding)), mode="constant"), padding


def auto_process(reffile, trajfile, scaler):
    r"""automatically process the trajectories.
reffile : reference structure file. [pdb,gro]
trajfile : trajectory file. [xtc,trr]
scaler : the class used to scale the coordinates"""
    pos = pool(reffile, trajfile)
    if scaler:
        pos = scaler.fit_transform(pos)

    return pos


def auto_reverse(pos, scaler):
    r"""automatically obtain the real coordinates from scaled positions.
pos : coodinates in (frames, N, 3) format.
scaler : the class used to scale the coordinates"""
    if scaler:
        pos = scaler.inverse(pos)

    return pos