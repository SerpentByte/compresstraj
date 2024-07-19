import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np

class DenseAutoEncoder(nn.Module):
    def __init__(self, N, latent):
        super().__init__()

        self.latent = latent
        self.N = N

        self.encoder = nn.Sequential(
            nn.Linear(self.N, 4096),
            nn.BatchNorm1d(4096),
            nn.ELU(),
            nn.Linear(4096, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, self.latent),
            nn.ELU()
        )


        self.decoder = nn.Sequential(
            nn.Linear(self.latent, 1024),
            nn.BatchNorm1d(1024),
            nn.ELU(),
            nn.Linear(1024, 4096),
            nn.BatchNorm1d(4096),
            nn.ELU(),
            nn.Linear(4096, self.N),
            nn.Sigmoid()
            
        )

    def forward(self, x):
        latent = self.encoder(x)
        return self.decoder(latent)


class RMSDLoss(nn.Module):
    def __init__(self, l2_penalty=0.01):
        super(RMSDLoss, self).__init__()
        self.l2_penalty = l2_penalty

    def forward(self, predictions, targets, model_parameters=None):
        # Calculate the Root Mean Square Deviation
        rmsd_loss = torch.sqrt(torch.mean((predictions - targets) ** 2))
        
        # L2 Regularization (if model_parameters are provided)
        if model_parameters is not None:
            l2_loss = sum(torch.sum(param ** 2) for param in model_parameters)
            rmsd_loss += self.l2_penalty * l2_loss
        
        return rmsd_loss

class LightAutoEncoder(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4):
        super().__init__()
        self.model = model
        self.loss_fn = RMSDLoss()
        self.learning_rate = learning_rate
        self.training_outputs = []
        self.validation_outputs = []
        self.train_loss = []
        self.val_loss = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, x)  # Assuming you want to use X as target",
        self.log('train_loss', loss, on_epoch=True)
        self.training_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.training_outputs).mean()
        self.log('avg_training_loss', avg_train_loss.cpu().item(), prog_bar=True, logger=True)
        self.train_loss.append(avg_train_loss.cpu().item())
        self.training_outputs.clear()

    def validation_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat = self.model(x)
        loss = self.loss_fn(x_hat, x)  # Assuming you want to use X as target
        self.log('val_loss', loss, on_epoch=True)
        self.validation_outputs.append(loss)
        return loss

    def on_validation_epoch_end(self):
        avg_val_loss = torch.stack(self.validation_outputs).mean()
        self.log('avg_val_loss', avg_val_loss.cpu().item(), prog_bar=True, logger=True)
        self.val_loss.append(avg_val_loss.cpu().item())
        self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer


class MinMaxScaler:
    r"""A class for scaling and unscaling inputs that have been scaled between (0, 1)"""
    def __init__(self):
        self.minim = 0
        self.maxim = 0

    def fit(self, X):
        r"""calculate the minimum and maximum and store it.
X : input."""
        self.minim = X.min(axis=tuple([i for i in range(len(X.shape[:-1]))]))
        self.maxim = X.max(axis=tuple([i for i in range(len(X.shape[:-1]))]))

        self.minim = np.expand_dims(self.minim, axis=tuple([i for i in range(len(X.shape[:-1]))]))
        self.maxim = np.expand_dims(self.maxim, axis=tuple([i for i in range(len(X.shape[:-1]))]))

    def transform(self, X):
        r"""scale the input between (0, 1).
X : input"""
        return (X - self.minim)/(self.maxim - self.minim)

    def fit_transform(self, X):
        r"""combines fit and transform.
X : input"""
        self.fit(X)
        return self.transform(X)

    def inverse(self, X):
        r"""unscales back to real values using the minimum and maximum values stored in the class instance.
X : input"""
        return X*(self.maxim - self.minim) + self.minim



class TrajLoader:
    r"""class for PyTorch dataloader to load coordinates."""
    def __init__(self, pos):
        self.pos = pos

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return self.pos[i]

    def get_N(self):
        return self.pos.shape[-1]