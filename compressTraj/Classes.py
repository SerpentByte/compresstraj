import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import pytorch_lightning as pl


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


class LightAutoEncoder(pl.LightningModule):
    def __init__(self, model, learning_rate=1e-4, idx=None):
        super().__init__()
        self.model = model
        self.loss_fn = RMSDLoss() 
        self.learning_rate = learning_rate
        self.idx = idx
        self.training_outputs = []
        # self.validation_outputs = []
        self.train_loss = []
        # self.val_loss = []

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x = batch.to(self.device)
        x_hat = self.model(x)
        if self.idx is None:
            loss = self.loss_fn(x_hat, x)  # Assuming you want to use X as target",
        else:
            loss = 0.0
            for i in self.idx:
                _ = self.loss_fn(x[:, i[0]:i[1]], x_hat[:, i[0]:i[1]])
                loss += _

        self.log('train_loss', loss, on_epoch=True)
        self.training_outputs.append(loss)
        return loss

    def on_train_epoch_end(self):
        avg_train_loss = torch.stack(self.training_outputs).mean()
        self.log('avg_training_loss', avg_train_loss.cpu().item(), prog_bar=True, logger=True)
        self.train_loss.append(avg_train_loss.cpu().item())
        self.training_outputs.clear()

    # def validation_step(self, batch, batch_idx):
    #     with torch.no_grad():
    #         x = batch.to(self.device)
    #         x_hat = self.model(x)
    #         if self.idx is None:
    #             loss = self.loss_fn(x_hat, x)  # Assuming you want to use X as target",
    #         else:
    #             loss = 0.0
    #             for i in self.idx:
    #                 _ = self.loss_fn(x[:, i[0]:i[1]], x_hat[:, i[0]:i[1]])
    #                 loss += _
  
    #     self.log('val_loss', loss, on_epoch=True)
    #     self.validation_outputs.append(loss)
    #     return loss

    # def on_validation_epoch_end(self):
    #     avg_val_loss = torch.stack(self.validation_outputs).mean()
    #     self.log('avg_val_loss', avg_val_loss.cpu().item(), prog_bar=True, logger=True)
    #     self.val_loss.append(avg_val_loss.cpu().item())
    #     self.validation_outputs.clear()

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.learning_rate)
        return optimizer



class TrajLoader:
    def __init__(self, pos):
        self.pos = pos

    def __len__(self):
        return len(self.pos)

    def __getitem__(self, i):
        return self.pos[i]

    def get_N(self):
        return self.pos.shape[-1]