import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl
from MDAnalysis import Universe
from MDAnalysis.analysis import align
import pickle
from glob import glob
import random
import os

class RMSDLoss(nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, predictions, targets):
        # Compute RMSD
        rmsd_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        return rmsd_loss
        

class DenseAutoEncoder(nn.Module):
    def __init__(self, N, latent, layers=[4096, 1024]):
        super().__init__()

        self.latent = latent
        self.N = N
        self.layers = layers

        # encoder
        encoder_layers = []
        in_dim = self.N
        for out_dim in self.layers:
            encoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ELU()
            ])
            in_dim = out_dim
        encoder_layers.extend([
            nn.Linear(in_dim, self.latent),
            nn.ELU()
        ])
        self.encoder = nn.Sequential(*encoder_layers)
    
        # decoder
        decoder_layers = []
        in_dim = self.latent
        for out_dim in reversed(self.layers):
            decoder_layers.extend([
                nn.Linear(in_dim, out_dim),
                nn.BatchNorm1d(out_dim),
                nn.ELU()
            ])
            in_dim = out_dim
        decoder_layers.append(nn.Linear(in_dim, self.N))
        self.decoder = nn.Sequential(*decoder_layers)

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)

        
class LightAutoEncoder(pl.LightningModule):
    def __init__(self, model, loss_fn, scaler=None, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.loss_fn = loss_fn
        self.learning_rate = learning_rate
        self.training_outputs = []
        self.train_loss = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.float()
        
        x_hat = self.model(x)

        loss = self.loss_fn(x_hat, x)  # Assuming you want to use X as target",

        self.log('train_loss', loss, on_epoch=True)
        self.training_outputs.append(loss.detach().cpu().item())
        
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = np.mean(self.training_outputs)
        self.log('avg_training_loss', avg_train_loss, prog_bar=True, logger=True)
        self.train_loss.append(avg_train_loss)
        self.training_outputs.clear()
        torch.cuda.empty_cache()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer



class TrajLoader:
    def __init__(self, traj, indices, scaler, remove_cog=True):
        self.traj = traj
        self.indices = indices
        self.scaler = scaler
        self.remove_cog = remove_cog

    def __len__(self):
        return len(self.traj.trajectory)

    def __getitem__(self, idx):
        pos = self.traj.trajectory[idx]._pos[self.indices]

        if self.remove_cog:
            pos -= pos.mean(axis=0, keepdims=True)
        
        pos = self.scaler.transform(pos).flatten()
        return pos

    def get_N(self):
        return 3*len(self.indices)
