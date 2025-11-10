import sbibm.tasks
from sbibm.metrics import c2st
import torch
import numpy as np
import math
from torch.utils.data import Dataset
from os.path import join


class SbiDataset(Dataset):
    def __init__(self, theta, x):
        super(SbiDataset, self).__init__()

        self.standardization = {
            "x": {"mean": torch.mean(x, dim=0), "std": torch.std(x, dim=0)},
            "theta": {"mean": torch.mean(theta, dim=0), "std": torch.std(theta, dim=0)},
        }
        self.theta = self.standardize(theta, "theta")
        self.x = self.standardize(x, "x")

    def standardize(self, sample, label, inverse=False):
        mean = self.standardization[label]["mean"]
        std = self.standardization[label]["std"]
        if not inverse:
            return (sample - mean) / std
        else:
            return sample * std + mean

    def __len__(self):
        return len(self.theta)

    def __getitem__(self, idx):
        return self.theta[idx], self.x[idx]


def generate_dataset(settings, batch_size=1, directory_save=None, train_fraction=0.8):
    task = sbibm.get_task(settings["task"]["name"])
    prior = task.get_prior()
    simulator = task.get_simulator()
    num_train_samples = settings["task"]["num_train_samples"]
    nr_batches = math.ceil(num_train_samples / batch_size)
    theta = []
    x = []
    for _ in range(nr_batches):
        theta_sample = prior(batch_size)
        x_sample = simulator(theta_sample)
        theta.append(theta_sample)
        x.append(x_sample)
    x = np.vstack(x)[:num_train_samples]
    theta = np.vstack(theta)[:num_train_samples]
    
    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float)
    theta_tensor = torch.tensor(theta, dtype=torch.float)
    
    # Save full dataset
    if directory_save is not None:
        np.save(join(directory_save, 'x.npy'), x)
        np.save(join(directory_save, 'theta.npy'), theta)
    
    # Split data into train and validation
    num_samples = len(theta_tensor)
    num_train = int(num_samples * train_fraction)
    
    # Random permutation for train/val split
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Split the data
    theta_train, theta_val = theta_tensor[train_indices], theta_tensor[val_indices]
    x_train, x_val = x_tensor[train_indices], x_tensor[val_indices]
    
    # Create train and validation datasets
    train_dataset = SbiDataset(theta_train, x_train)
    val_dataset = SbiDataset(theta_val, x_val)

    settings["task"]["dim_theta"] = theta_tensor.shape[1]
    settings["task"]["dim_x"] = x_tensor.shape[1]

    return train_dataset, val_dataset


def load_dataset(directory_save, settings, train_fraction=0.8):
    x = np.load(join(directory_save, 'x.npy'))
    theta = np.load(join(directory_save, 'theta.npy'))
    
    # Convert to tensors
    x_tensor = torch.tensor(x, dtype=torch.float)
    theta_tensor = torch.tensor(theta, dtype=torch.float)
    
    # Split data into train and validation
    num_samples = len(theta_tensor)
    num_train = int(num_samples * train_fraction)
    
    # Random permutation for train/val split
    indices = torch.randperm(num_samples)
    train_indices = indices[:num_train]
    val_indices = indices[num_train:]
    
    # Split the data
    theta_train, theta_val = theta_tensor[train_indices], theta_tensor[val_indices]
    x_train, x_val = x_tensor[train_indices], x_tensor[val_indices]
    
    # Create train and validation datasets
    train_dataset = SbiDataset(theta_train, x_train)
    val_dataset = SbiDataset(theta_val, x_val)

    settings["task"]["dim_theta"] = theta_tensor.shape[1]
    settings["task"]["dim_x"] = x_tensor.shape[1]

    return train_dataset, val_dataset