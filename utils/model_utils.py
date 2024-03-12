# Imports
from lightning import LightningModule
import math
import torch
from torch.optim.lr_scheduler import LambdaLR, SequentialLR, ReduceLROnPlateau, CosineAnnealingLR, ExponentialLR, CosineAnnealingWarmRestarts
import torch.nn as nn
import torch.nn.functional as F

# Define a linear warm up schedule
def LinearWarmup(optimizer, warmup_epochs=5, warmup_start_lr=1e-6, warmup_end_lr=1e-2):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (warmup_end_lr - warmup_start_lr) / (warmup_epochs - 1) * epoch + warmup_start_lr
        else:
            return warmup_end_lr
    return LambdaLR(optimizer, lr_lambda)

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

        self.weights = torch.empty(in_channels, out_channels, self.modes, dtype=torch.cfloat)
        nn.init.xavier_normal_(self.weights, gain=1.0)
        self.weights = nn.Parameter(self.weights)

    def forward(self, x):
        batchsize = x.shape[0]

        #Compute Fourier coefficients
        x_ft = torch.fft.rfft(x)

        # Multiply relevant Fourier modes
        out_ft = torch.zeros(batchsize, self.out_channels,  x.size(-1)//2 + 1, dtype=torch.cfloat, device=x.device)

        out_ft[:, :, :self.modes] = \
            self.compl_mul1d(x_ft[:, :, :self.modes], self.weights)

        #Return to physical space
        out = torch.fft.irfft(out_ft)

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
                 p_dropout=0,
                 add_noise=False
        ):
        super().__init__()

        self.lr = lr
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.momentum = momentum
        self.num_channels = len(channels)
        self.loss = nn.BCELoss()
        self.add_noise = add_noise

        self.example_input_array = torch.rand(32, 1, 500)

        for i in range(self.num_channels):
            if i == 0:
                in_channels = 1
            else:
                in_channels = channels[i-1]

            out_channels = channels[i]

            setattr(self, f"fno_layer_{i}", nn.Sequential(
                SpectralConv1d(in_channels, out_channels, modes),
                nn.BatchNorm1d(out_channels)
            ))
        
        if pool_type == "max":
            self.pool = nn.MaxPool1d(pooling)
        else:
            self.pool = nn.AvgPool1d(pooling)

        self.dropout = nn.Dropout(p_dropout)

        self.fc = nn.Linear(channels[-1] * int((500/pooling)), 1) # output number of channels of final fno_block * (3rd input dimension 500 / maxpool size)
        self.fc.weight.data.fill_(float(0.5))


    def forward(self, x):
        for i in range(self.num_channels):
            x = getattr(self, f"fno_layer_{i}")(x)
            x = F.tanh(x)

        x = self.pool(x)
        x = x.view(x.size(0), -1)

        if self.add_noise:
            noise = torch.randn_like(x)
            x = x + noise

        x = self.dropout(x) # dropping out seems to help a little, but not to the extent we need it to 

        x = self.fc(x)
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
            scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-7)
        elif self.scheduler == "cosineannealinglr":
            scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=1e-6)
        elif self.scheduler == "cosineannealingwarmrestarts":
            scheduler = CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=3, eta_min=1e-6)
        elif self.scheduler == "linearwarmupcosineannealingwarmrestarts":
            warmup_epochs=5
            scheduler = SequentialLR(optimizer, schedulers=[
                LinearWarmup(optimizer, warmup_epochs=warmup_epochs, warmup_start_lr=1e-6, warmup_end_lr=1e-2),
                CosineAnnealingWarmRestarts(optimizer, T_0=50, T_mult=3, eta_min=1e-6)
                ],
                milestones=[warmup_epochs]
            )
        else:
            scheduler = ExponentialLR(optimizer, gamma=0.95)
        
        return [optimizer], [{"scheduler": scheduler, "monitor": "train_loss"}]