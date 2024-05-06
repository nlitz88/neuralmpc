"""Contains model that receives and predicts the vehicle's next state expressed
solely in the ego/body frame.
"""

from pathlib import Path
from typing import List, Optional
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
model = nn.Sequential(nn.Linear(in_features=5, out_features=128),
                      nn.ReLU(),
                    #   nn.Dropout(p=0.1),
                      nn.Linear(in_features=128, out_features=6))

# https://lightning.ai/docs/pytorch/stable/starter/introduction.html#define-a-lightningmodule
class BodyOnlyDiscrete(L.LightningModule):
    def __init__(self,
                 model,
                 learning_rate: Optional[float] = 1e-4,
                 batch_size: Optional[int] = 64):
        super().__init__()
        self.model = model
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.save_hyperparameters()

    def forward(self, input: torch.Tensor):
        """Function to wrap up the forward pass process of the model. Handles
        model input and output transformations so that the model--no matter the
        underlying network's architecture--can receive the same format inputs
        and same format outputs.

        NOTE: You should never call forward from outside the wrapper. If you
        want to inference on the model using this wrapper, use the predict_step
        function--as it will contain any necessary steps to take your input and
        make sure it's in the right format and type for forward.

        Args:
            input (torch.Tensor): Input vector with the following form:
            
            [u_k, u_steer_k, vx_B_k, vy_B_k, w_B_k, xp_N_k, yp_N_k, phi_B_N_k].

            This is usually the first row in a sample from your dataset.
            See model-term glossary for each term's meaning.

        Returns:
            torch.Tensor: Output vector with the following form:

            [vx_B_k_1, vy_B_k_1, w_B_k_1, xp_N_k_1, yp_N_k_1, phi_B_N_k_1]
        """

        # Extract the first 5 columns of the provided input--these will be the
        # input to the neural network itself (after some preprocessing).
        model_input = input[:, 0:5]

        # NOTE: Part of this model is transforming components of the car's state
        # in the inertial frame into the body frame. However, because doing that
        # here would leave the q_dot elements untouched and the q elements == 0,
        # we just leave them out of the model input (as they are always zero).
        
        # Pass the normalized input through the underlying neural network.
        model_output = self.model(model_input)

        # Transform the "q" (configuration) components of the car's state back
        # to the inertial frame. The resulting "transformed_output" will have
        # the same form as the input passed to this wrapper x_k = [qdot_B, q_N].
        transformed_output = model_output.detach().clone()
        # The q_dot components are already in the body frame, nothing to do. 
        transformed_output[:, 0:3] = model_output[:, 0:3]

        # Extract some terms from the input needed to complete the rest of the
        # transform.
        xp_N_k = input[:, 5]                            # Inertial frame x position at time k.
        yp_N_k = input[:, 6]                            # Inertial frame y position at time k.
        phi_B_N_k = input[:, 7]                         # Relative rotation from body to inertial frame at time k.
        xp_B_k_1 = model_output[:, 3]      # Body frame x position at time k+1.
        yp_B_k_1 = model_output[:, 4]      # Body frame y position at time k+1.
        phi_B_B_k_1 = model_output[:, 5]   # Relative rotation from body frame at time k to body frame at time k+1

        # Compute the "total" or "composed" rotation from the body frame to the
        # inertial frame at time k+1. This is also part of the configuration.
        # I.e., the relative rotation from the body frame (B) to the inertial
        # frame (N) at time k+1.
        phi_B_N_k_1 = phi_B_N_k + phi_B_B_k_1
        transformed_output[:, 5] = phi_B_N_k_1

        # Express xp_B_k_1 and yp_B_k_1 in the inertial frame, and then add them
        # to the initial xp_N_k and yp_N_k, respectively.
        transformed_output[:, 3] = xp_N_k + xp_B_k_1*torch.cos(phi_B_N_k) - yp_B_k_1*torch.sin(phi_B_N_k)
        transformed_output[:, 4] = yp_N_k + yp_B_k_1*torch.sin(phi_B_N_k) + yp_B_k_1*torch.cos(phi_B_N_k)

        return transformed_output

    def training_step(self, batch: torch.Tensor, batch_idx):

        batch = batch.to(dtype=torch.float32)
        zk = batch[:, 0, :]             # zk == [uk, xk]
        xk_1_gt = batch[:, -1, 2:]      # Ground truth next state.

        # Get the next state predicted by the network.
        x_k_1_pred = self.forward(zk)

        loss = nn.functional.mse_loss(x_k_1_pred, xk_1_gt)
        self.log("train_loss", loss, prog_bar=True)
        return loss
    
    # https://lightning.ai/docs/pytorch/stable/common/evaluation_basic.html#define-the-validation-loop
    def validation_step(self, batch: torch.Tensor, batch_idx):
        
        batch = batch.to(dtype=torch.float32)
        batch = batch.to(dtype=torch.float32)
        zk = batch[:, 0, :]             # zk == [uk, xk]
        xk_1_gt = batch[:, -1, 2:]      # Ground truth next state.

        # Get the next state predicted by the network.
        x_k_1_pred = self.forward(zk)

        loss = nn.functional.mse_loss(x_k_1_pred, xk_1_gt)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    # TODO: Add a "to_torch_script" method here--unless the base class already
    # includes that.
    
if __name__ == "__main__":

    dynamics_model = BodyOnlyDiscrete(model=model,
                                      learning_rate=1e-4,
                                      batch_size=64)
    
    # For the state "x", we want to get the terms that define the configuration
    # q and the elements that == the rate of change of the configuration q_dot.
    # q == in world or inertial frame == x, y, e/psi (e/psi is the intertial
    # frame yaw, also referred to as phi)
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

    discrete_dynamics_dataset = ArtDataset(dataset_directory_path=r"/home/nlitz88/Documents/neuralmpc_datasets/2023_08_31-putnam/2023_08_31-13_46_57_0_old/",
                                           desired_columns=desired_columns,
                                           sample_length=5,
                                           standardize=False)
    
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
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    trainer = L.Trainer(max_epochs=15,
                        callbacks=[early_stopping_callback])
    
    trainer.fit(model=dynamics_model,
                train_dataloaders=train_loader,
                val_dataloaders=validation_loader)
    
    # Export trained model weights as TorchScript model.
    model_logging_directory = Path(dynamics_model.logger.log_dir)
    model_version_name = model_logging_directory.parts[-1]
    checkpoints_directory = model_logging_directory/"checkpoints"
    weights_filepath = checkpoints_directory/f"{model_version_name}.pt"
    dynamics_model.to_torchscript(file_path=weights_filepath)
    print(f"Saved TorchScript model at {weights_filepath}")
