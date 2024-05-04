
import importlib
import model_utils
import data_utils
importlib.reload(model_utils)
importlib.reload(data_utils)
from model_utils import FNOClassifier
from data_utils import CustomDataset, RandomSample, RandomTimeTranslateFill05, RandomTimeTranslateReflect, RandomNoise

import os
import optuna
import datetime
import numpy as np
import matplotlib.pyplot as plt
from numpy.random import seed, shuffle
from sklearn.metrics import roc_curve, auc

import torch
import torch.utils.data as data
from torch.utils.data import DataLoader
import torchvision.transforms.v2 as transforms

import pytorch_lightning as pl
from lightning import Trainer
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
pl.__version__

# Check if CUDA is available
if torch.cuda.is_available():
    device = torch.device("cuda")
    print(f"GPU is available on device {torch.cuda.get_device_name(0)} with device count: {torch.cuda.device_count()}")
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
else:
    device = torch.device("cpu")
    print("GPU is not available")

def get_data(batch_size=32, data_augmentation=None):
        # Kepler Lightcurve dataset
        directory = "../../kepler_lightcurve_data/"

        # Load KeplerLightCurves.csv
        data = np.genfromtxt(directory + "KeplerLightCurves.csv", delimiter=',')
        X = data[:, :-1]
        y = data[:, -1]

        # Reshape the data to be ready for multivariate time-series data (multiple channels)
        # Shape is (samples, channels, sequence length)
        X = X.reshape((X.shape[0], 1, X.shape[1]))

        # Prepare labels
        y = y - 1 # Shift the labels to start from 0

        # Canonicalize the data (pass through 0 at the origin)
        X -= X[:, :, 0].reshape(-1, 1, 1)

        # Create a list of indices for your dataset
        indices = list(range(len(X)))
        seed(1)
        shuffle(indices)

        # Split the indices into train, val, and test
        train_size = int(0.7 * len(X))
        val_size = int(0.2 * len(X))
        train_indices, val_indices, test_indices = np.array(indices[:train_size]), np.array(indices[train_size:train_size+val_size]), np.array(indices[train_size+val_size:])

        # Scale the data to be between 0 and 1
        x_train = X[train_indices]
        min_val = np.min(x_train)
        max_val = np.max(x_train)
        X = (X - min_val) / (max_val - min_val)

        # Create datasets
        train_x = torch.tensor(X[train_indices], dtype=torch.float32)
        train_y = torch.tensor(y[train_indices], dtype=torch.long)
        valid_x = torch.tensor(X[val_indices], dtype=torch.float32)
        valid_y = torch.tensor(y[val_indices], dtype=torch.long)
        test_x = torch.tensor(X[test_indices], dtype=torch.float32)
        test_y = torch.tensor(y[test_indices], dtype=torch.long)

        # Create train, valid, and test data loaders
        if data_augmentation == "randomsample":
                n_sample = 400
                seq_length = n_sample
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.RandomApply([RandomSample(n_sample=n_sample)], p=1) # Can't be used with other transforms as it changes the shape of the data
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )

        elif data_augmentation == "randomnoise":
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.RandomApply([RandomNoise(mean=0, std=0.1)], p=0.8)
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )

        elif data_augmentation == "randomtimetranslatefill05":
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.RandomApply([RandomTimeTranslateFill05(max_shift=100)], p=0.8)
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )

        elif data_augmentation == "randomtimetranslatereflect":
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.RandomApply([RandomTimeTranslateReflect(max_shift=100)], p=0.8)
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )

        elif data_augmentation == "randomnoise_randomtimetranslatefill0":
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.Compose([
                                        transforms.RandomApply([RandomNoise(mean=0, std=0.1)], p=0.5),
                                        transforms.RandomApply([RandomTimeTranslateFill05(max_shift=100)], p=0.5),
                                ])
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )   

        elif data_augmentation == "randomnoise_randomtimetranslatereflect":
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=transforms.Compose([
                                        transforms.RandomApply([RandomNoise(mean=0, std=0.1)], p=0.5),
                                        transforms.RandomApply([RandomTimeTranslateReflect(max_shift=100)], p=0.5),
                                ])
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )    

        else:
                seq_length = train_x.shape[2]
                train_loader = DataLoader(
                        CustomDataset(
                                train_x, 
                                train_y, 
                                transform=None
                        ),
                        batch_size=batch_size,
                        shuffle=True,
                        drop_last=True
                )  

        valid_loader = DataLoader(
                CustomDataset(
                        valid_x, 
                        valid_y
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
        )

        test_loader = DataLoader(
                CustomDataset(
                        test_x, 
                        test_y
                ),
                batch_size=batch_size,
                shuffle=False,
                drop_last=True
        )

        return train_loader, valid_loader, test_loader, seq_length

def objective(trial: optuna.trial.Trial) -> float:
        # Generate the dataloaders
        batch_size = 32
        data_augmentation = None
        train_loader, valid_loader, test_loader, seq_length = get_data(batch_size=batch_size, data_augmentation=data_augmentation)

        # Hyperparameters
        modes = trial.suggest_int("modes", 5, 50)
        num_layers = trial.suggest_int("num_layers", 1, 8)
        num_channels = trial.suggest_int("num_channels", 32, 1024)
        channels = [num_channels] * num_layers
        pool_type = trial.suggest_categorical("pool_type", ["avg", "max"])
        pooling = trial.suggest_categorical("pooling", [seq_length, int(seq_length/2), int(seq_length/4)])
        proj_dim = trial.suggest_int("proj_dim", 1, 16)
        p_dropout = 0.5
        add_noise = trial.suggest_categorical("add_noise", [True, False])
        num_classes = 7
        fc_post_pooling = True

        # Optimizers and learning rate schedulers
        # lr schedule options are reducelronplateau, steplr, exponentiallr, cosineannealinglr, and cosineannealingwarmrestarts
        # optimizer options are sgd or adam
        optimizer = trial.suggest_categorical("optimizer", ["sgd", "adam"])
        momentum = 0.9 # Only used for SGD optimizer
        scheduler = trial.suggest_categorical("scheduler", ["reducelronplateau", "cosineannealingwarmrestarts"])
        lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)

        # Initialize classifier
        classifier = FNOClassifier(
                modes=modes, 
                lr=lr, 
                channels=channels, 
                pooling=pooling, 
                optimizer=optimizer, 
                scheduler=scheduler, 
                momentum=momentum, 
                pool_type=pool_type, 
                seq_length=seq_length,
                proj_dim=proj_dim,
                p_dropout=p_dropout, 
                add_noise=add_noise,
                num_classes=num_classes,
                fc_post_pooling=fc_post_pooling
        )

        callbacks = [
                EarlyStopping(monitor="val_loss", patience=50, mode="min"),
                LearningRateMonitor(logging_interval="step"),
        ]

        # Train the model
        trainer = Trainer(
               max_epochs=1000,
                callbacks=callbacks,
                accelerator="auto"
        )

        trainer.fit(
                model=classifier, 
                train_dataloaders=train_loader,
                val_dataloaders=valid_loader
        )

        return trainer.callback_metrics["val_loss"].item()


if __name__ == "__main__":
        # Create optuna study
        study = optuna.create_study(direction="minimize")
        study.optimize(objective, n_trials=1000, timeout=(48*60*60))

        print("Number of finished trials: {}".format(len(study.trials)))

        print("Best trial:")
        trial = study.best_trial

        print("Value: {}".format(trial.value))

        print("Params: ")
        for key, value in trial.params.items():
                print("    {}: {}".format(key, value))