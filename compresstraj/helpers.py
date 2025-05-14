import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import MDAnalysis
from MDAnalysis import Universe
from MDAnalysis.analysis import align
import pickle
from glob import glob
import os
from tqdm.auto import tqdm
import random
import sys

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
    # seed_everything(seed, workers=True)


class MinMaxScaler:
    r"""A class that implements a global min-max scaler."""
    def __init__(self):
        self.data_min = np.zeros(3)
        self.data_max = np.ones(3)
        self.scale = 1.0
        self.inv_scale = 1.0
        self.min_ = 0.0

    def fit(self, traj, selection, remove_cog=True):
        minim, maxim = np.ones(3)*100000.0, np.zeros(3)

        atoms = traj.select_atoms(selection)
        for t in traj.trajectory:
            pos = atoms.positions.copy()
            
            if remove_cog:
                pos -= atoms.center_of_geometry()
            # pos = pos.flatten()
    
            minim = np.minimum(pos.min(axis=0), minim)
            maxim = np.maximum(pos.max(axis=0), maxim)

        self.data_min = minim
        self.data_max = maxim
        self.scale = 1/(self.data_max - self.data_min)
        self.inv_scale = (self.data_max - self.data_min)

    def transform(self, X):
        X = np.array(X, dtype=float)
        return (X - self.data_min)* self.scale

    def inverse_transform(self, X_scaled):
        X_scaled = np.array(X_scaled, dtype=float)
        return X_scaled*self.inv_scale + self.data_min


class StandardScaler:
    r"""A class that implements a global standard-scaler."""
    def __init__(self):
        self.mean = 0.0
        self.std = 0.0
        
    def fit(self, traj, selection, remove_cog=True):
        sum_, sqsum_ = 0.0, 0.0

        atoms = traj.select_atoms(selection)
        for t in traj.trajectory:
            pos = atoms.positions.copy()
            if remove_cog:
                pos -= atoms.center_of_geometry()
            
            pos = pos.flatten()
    
            sum_ += pos.sum()
            sqsum_ += (pos*pos).sum()

        sqsum_ /= (traj.select_atoms(selection).n_atoms*len(traj.trajectory)*3)
        self.mean = sum_/(traj.select_atoms(selection).n_atoms*len(traj.trajectory)*3)
        self.std = np.sqrt(max(sqsum_ - self.mean**2, 1e-16))

    def transform(self, X):
        return (X - self.mean) / self.std

    def inverse_transform(self, X):
        return X * self.std + self.mean


    
