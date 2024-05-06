"""Contains model for fully learning the existing single-track model (using
global coordinates).
"""

import torch
from torch.utils.data import DataLoader, random_split
from torch import optim, nn
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping

import sys
sys.path.insert(0, "../../../")
from neural_mpc.datasets.art.dataset import ArtDataset

# TODO: Create a function to parameterize the definition of our model.

# Define the network layers.
model = nn.Sequential(nn.Linear(in_features=8, out_features=128),
                      nn.ReLU(),
                    #   nn.Dropout(p=0.1),
                      nn.Linear(in_features=128, out_features=6))

# https://lightning.ai/docs/pytorch/stable/starter/introduction.html#define-a-lightningmodule
class LitModel(L.LightningModule):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def training_step(self, batch: torch.Tensor, batch_idx):
        batch = batch.to(dtype=torch.float32)
        zk = batch[:, 0, :] # zk == [uk, xk]
        xk_1 = batch[:, 1, 2:] # Ground truth next state.
        x_hat = self.model(zk) # Predicted next state.
        loss = nn.functional.mse_loss(x_hat, xk_1)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    # https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#define-the-validation-loop
    def validation_step(self, batch: torch.Tensor, batch_idx):
        batch = batch.to(dtype=torch.float32)
        zk = batch[:, 0, :] # zk == [uk, xk]
        xk_1 = batch[:, 1, 2:] # Ground truth next state.
        x_hat = self.model(zk) # Predicted next state.
        loss = nn.functional.mse_loss(x_hat, xk_1)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-4)
        return optimizer
    
if __name__ == "__main__":

    dynamics_model = LitModel(model=model)
    
    # For the state "x", we want to get the terms that define the configuration
    # q and the elements that == the rate of change of the configuration q_dot.
    # q == in world or inertial frame == x, y, e/psi (e/psi is the intertial
    # frame yaw)
    # q_dot terms are in the body frame == v_long, v_tran, w_psi.
    desired_columns = ["u/u_a",         
                       "u/u_steer",
                       "v/v_long",
                       "v/v_tran",
                       "w/w_psi",
                       "x/x",
                       "x/y",
                       "e/psi"
                       ]

    discrete_dynamics_dataset = ArtDataset(dataset_directory_path="/home/nlitz88/Downloads/iac_datasets/2024_03_04-putnam-sim",
                                           desired_columns=desired_columns,
                                           sample_length=2)
    
    # Test Generating a random train / validation split.
    TRAIN_SPLIT_SIZE = 0.7
    VALIDATION_SPLIT_SIZE = 0.3
    generator = torch.Generator().manual_seed(0)
    train_split, validation_split = random_split(dataset=discrete_dynamics_dataset,
                                                lengths=[TRAIN_SPLIT_SIZE, VALIDATION_SPLIT_SIZE],
                                                generator=generator)
    
    # Create dataloaders around the training and validation splits.
    train_loader = DataLoader(dataset=train_split,
                              batch_size=64,
                              pin_memory=True,
                              pin_memory_device="cuda",
                              num_workers=6)
    validation_loader = DataLoader(dataset=validation_split, 
                                   batch_size=64,
                                   pin_memory=True,
                                   pin_memory_device="cuda",
                                   num_workers=6)

    # Create an early-stopping callback for the trainer.
    # https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=3)

    trainer = L.Trainer(max_epochs=100,
                        callbacks=[early_stopping_callback])
    
    trainer.fit(model=dynamics_model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)