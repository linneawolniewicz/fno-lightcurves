# Imports
from lightning import LightningModule
import math
import torch
import neuralop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv1d(nn.Module):
    def __init__(self,
                 in_channels,   # Number of input channels
                 out_channels,  # Number of output channels
                 modes,         # Number of Fourier modes to multiply, at most floor(N/2) + 1
                 debug=False):  # If true, print the shape of every parameter      
        super(SpectralConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.modes = modes
        self.debug = debug

        # Initialize weights
        self.weights = torch.empty(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        nn.init.xavier_normal_(self.weights, gain=1.0)
        self.weights = nn.Parameter(self.weights)

        # Intialize bias
        self.bias = torch.empty(out_channels, dtype=torch.float)
        nn.init.constant_(self.bias, 0.0)
        self.bias = nn.Parameter(self.bias)

    def forward(self, x):
        batchsize = x.shape[0]

        #Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes] = \
            self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        out = torch.fft.irfft(out_ft, n=x.size(-1))

        # Add bias
        out = out + self.bias.view(1, -1, 1)

        if self.debug:
            print(f"Shape, mean, median, and mode of x: {x.shape}, {x.mean()}, {x.median()}, {x.mode()}")
            print(f"Shape, mean, median, and mode of x_ft: {x_ft.shape}, {x_ft.mean()}, {x_ft.median()}, {x_ft.mode()}")
            print(f"Shape, mean, median, and mode of weights: {self.weights.shape}, {self.weights.mean()}, {self.weights.median()}, {self.weights.mode()}")
            print(f"Shape, mean, median, and mode of out_ft: {out_ft.shape}, {out_ft.mean()}, {out_ft.median()}, {out_ft.mode()}")
            print(f"Shape, mean, median, and mode of out: {out.shape}, {out.mean()}, {out.median()}, {out.mode()}")
        
        return out

    def compl_mul1d(self, input, weights):
        # (batch, in_channel, x), (in_channel, out_channel, x) -> (batch, out_channel, x)
        return torch.einsum("bix,iox->box", input, weights)

class FNOClassifier(LightningModule):
    def __init__(self, 
                 modes, 
                 lr=1e-3, 
                 channels=[64, 64, 64], 
                 pooling=500, 
                 optimizer="adam", 
                 scheduler="reducelronplateau", 
                 momentum=0.9,
                 pool_type="max",
                 seq_length=500,
                 proj_dim=128,
                 p_dropout=0,
                 add_noise=False
        ):
        super().__init__()

        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.momentum = momentum
        self.num_channels = len(channels)
        self.proj_dim = proj_dim
        self.loss = nn.BCELoss()
        self.add_noise = add_noise
        self.example_input_array = torch.rand(1, 1, seq_length)

        # Projection layer
        self.project = nn.Sequential(
            nn.Linear(seq_length, proj_dim * seq_length),
            nn.Dropout(p_dropout)
        )
        
        # Initialize the linear layer with Xavier normal initialization
        for module in self.project.modules():
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)

        # FNO layers
        for i in range(self.num_channels):
            if i == 0:
                in_channels = proj_dim
            else:
                in_channels = channels[i-1]

            out_channels = channels[i]

            setattr(self, f"fno_layer_{i}", nn.Sequential(
                # SpectralConv1d(in_channels, out_channels, modes),
                neuralop.layers.spectral_convolution.SpectralConv1d(in_channels, out_channels, modes),
                nn.BatchNorm1d(out_channels)
            ))

        # Dropout layer
        self.dropout = nn.Dropout(p_dropout)

        # TODO: change this back
        # Fully connected layer for final classification
        # self.fc = nn.Linear(channels[-1] * seq_length, seq_length) # converts from input number of channels to one channel
        # self.fc.weight.data.fill_(float(0.5))

        self.fc_post_dropout = nn.Linear(channels[-1] * int((seq_length/pooling)), 1) # output number of channels of final fno_block * (3rd input dimension / maxpool size)
        self.fc_post_dropout.weight.data.fill_(float(0.5))

        # Pooling layer
        if pool_type == "max":
            self.pool = nn.MaxPool1d(pooling)
        else:
            self.pool = nn.AvgPool1d(pooling)

    def forward(self, x):
        # Project
        x = x.float()
        x = self.project(x)
        x = x.view(x.size(0), self.proj_dim, -1)
        x = x.permute(0, 1, 2)

        # FNO layers
        for i in range(self.num_channels):
            x = getattr(self, f"fno_layer_{i}")(x)
            x = F.relu(x)
        # x = x.view(x.size(0), -1) # Flatten

        # Dropout
        # x = self.dropout(x)

        # Final classification layer
        # x = self.fc(x)

        # Add noise
        # if self.add_noise:
        #     noise = torch.randn_like(x)
        #     x = x + noise

        # Pool
        x = self.pool(x)
        x = x.view(x.size(0), -1) # Flatten

        # Add noise
        if self.add_noise:
            noise = torch.randn_like(x)
            x = x + noise

        # Dropout
        x = self.dropout(x)

        # Final classification layer
        x = self.fc_post_dropout(x)
        
        # Sigmoid activation
        x = F.sigmoid(x)
        x = x.squeeze(1)

        return x    

    def training_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y.float()) # No need for softmax, as it is included in nn.CrossEntropyLoss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True)

        # Log the accuracy
        binary_preds = (preds > 0.5).float()
        acc = (binary_preds == y).float().mean()
        self.log("train_acc", acc, on_step=True, on_epoch=True, prog_bar=True)

        # collapse flag 
        collapse_flg = torch.unique(preds).size(dim=0)
        self.log("collapse_flg_train", collapse_flg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        # mean, median, and mode flag
        mean_flg = preds.mean().item()
        std_flag = preds.std().item()
        self.log("mean_flg_train", mean_flg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log("std_flag_train", std_flag, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y.float())
        self.log("val_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        
        # Log the accuracy
        binary_preds = (preds > 0.5).float()
        acc = (binary_preds == y).float().mean()
        self.log("val_acc", acc, on_step=False, on_epoch=True, prog_bar=True)

        # collapse flag 
        collapse_flg = torch.unique(preds).size(dim=0)
        self.log("collapse_flg_val", collapse_flg, sync_dist=True, on_epoch=True, prog_bar=True)

        # mean, median, and mode flag
        mean_flg = preds.mean().item()
        std_flg = preds.std().item()
        self.log("mean_flg_val", mean_flg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)
        self.log("std_flg_val", std_flg, sync_dist=True, on_step=False, on_epoch=True, prog_bar=True)

        return loss
    
    def test_step(self, batch, batch_idx):
        x, y = batch
        preds = self.forward(x)

        # Log the loss
        loss = self.loss(preds, y.float())
        self.log("test_loss", loss)

        # Log the accuracy
        binary_preds = (preds > 0.5).float()
        acc = (binary_preds == y).float().mean()
        self.log("test_acc", acc)

        return loss
    
    def configure_optimizers(self):
        if self.optimizer == "sgd":
            optimizer = torch.optim.SGD(self.parameters(), lr=self.lr, momentum=self.momentum)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.lr)

        if self.scheduler == "reducelronplateau":
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.9, patience=100, min_lr=1e-6, cooldown=50)
        elif self.scheduler == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-7)
        elif self.scheduler == "cosineannealingwarmrestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=5, eta_min=1e-7)
        elif self.scheduler == "steplr":
            scheduler = StepLR(optimizer, step_size=1000, gamma=0.5)
        else:
            scheduler = ExponentialLR(optimizer, gamma=0.98)
        
        return [optimizer], [{"scheduler": scheduler, "monitor": "train_loss"}]