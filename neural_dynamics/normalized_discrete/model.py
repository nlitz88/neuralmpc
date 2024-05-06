"""Normalized discrete model for predicting the vehicle's next state provided
its current state and current controls.
"""

from pathlib import Path
from typing import List, Optional
import torch
from torch.utils.data import DataLoader, random_split, Subset
from torch import optim, nn
import lightning as L
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch import loggers as pl_loggers


import sys
sys.path.insert(0, "../../")
from neural_dynamics.normalized_discrete.dataset import NormalizedDiscreteCSVDataset

# TODO: Create a function to parameterize the definition of our model.

# Define the network layers.
model = nn.Sequential(nn.Linear(in_features=5, out_features=128),
                      nn.ReLU(),
                    #   nn.Dropout(p=0.1),
                      nn.Linear(in_features=128, out_features=3))

# https://lightning.ai/docs/pytorch/stable/starter/introduction.html#define-a-lightningmodule
class NormalizedDiscrete(L.LightningModule):
    def __init__(self,
                 model,
                 input_means: List[float],
                 input_scales: List[float],
                 output_means: List[float],
                 output_scales: List[float],
                 timestep_length_s: float,
                 learning_rate: Optional[float] = 1e-4,
                 batch_size: Optional[int] = 64):
        super().__init__()
        self.model = model
        self.input_means = input_means
        self.input_scales = input_scales
        self.output_means = output_means
        self.output_scales = output_scales
        self.timestep_length_s = timestep_length_s
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
            
            [u_a_k, u_steer_k, vx_B_k, vy_B_k, w_B_k, xp_N_k, yp_N_k, phi_B_N_k].

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

        # Apply z-score normalizaton/standardization to the channels/values of
        # the input vector.
        # input_means_tensor = torch.zeros_like(model_input)
        # input_means_tensor[:, :] = torch.tensor(self.input_means)
        # input_scales_tensor = torch.zeros_like(model_input)
        # input_scales_tensor[:, :] = torch.tensor(self.input_scales)
        # input_scales_padding = 1e-6*torch.ones_like(input_scales_tensor)
        # normalized_model_input = (model_input - input_means_tensor) / (input_scales_tensor + input_scales_padding)
        # Uncomment above and uncomment below for non-normalized input.
        normalized_model_input = model_input

        # Pass the normalized input through the underlying neural network.
        model_output = self.model(normalized_model_input)

        # De-normalize the output of the neural network by doing the opposite
        # steps performed on the input.
        # output_means_tensor = torch.zeros_like(model_output)
        # output_means_tensor[:, :] = torch.tensor(self.output_means)
        # output_scales_tensor = torch.zeros_like(model_output)
        # output_scales_tensor[:, :] = torch.tensor(self.output_scales)
        # output_scales_padding = 1e-6*torch.ones_like(output_scales_tensor)
        # denormalized_model_output = model_output*(output_scales_tensor + output_scales_padding) + output_means_tensor
        # Uncomment above and uncomment below for non-normalized output.
        denormalized_model_output = model_output

        # Transform the "q" (configuration) components of the car's state back
        # to the inertial frame. The resulting "transformed_output" will have
        # the same form as the input passed to this wrapper x_k = [qdot_B, q_N].
        # transformed_output = denormalized_model_output.detach().clone()
        
        # Okay, so, actually, transformed output needs to be a NEW tensor--that
        # has the same batch size as the output tensor from the model--but with
        # 6 columns instead of 3. The first 3 columns will be the q_dot
        # components in the body frame, and the last 3 columns will be the q (or
        # configuration) components in the inertial frame.
        transformed_output = torch.zeros((denormalized_model_output.shape[0], 6), dtype=denormalized_model_output.dtype, device=denormalized_model_output.device)

        # The q_dot components are already in the body frame, nothing to do. 
        transformed_output[:, 0:3] = denormalized_model_output[:, 0:3]

        # Extract some terms from the input needed to complete the rest of the
        # transform.
        xp_N_k = input[:, 5]                            # Inertial frame x position at time k.
        yp_N_k = input[:, 6]                            # Inertial frame y position at time k.
        phi_B_N_k = input[:, 7]                         # Relative rotation from body to inertial frame at time k.
        # xp_B_k_1 = denormalized_model_output[:, 3]      # Body frame x position at time k+1.
        # yp_B_k_1 = denormalized_model_output[:, 4]      # Body frame y position at time k+1.
        # phi_B_B_k_1 = denormalized_model_output[:, 5]   # Relative rotation from body frame at time k to body frame at time k+1

        # Grab the qdot components from the input to the model. I.e., qdot_k
        # (qdot at timestep k).
        vx_B_k = normalized_model_input[:, 2]
        vy_B_k = normalized_model_input[:, 3]
        w_B_k = normalized_model_input[:, 4]
        
        # Compute the new position at time k+1.
        xp_B_k_1 = vx_B_k*self.timestep_length_s
        yp_B_k_1 = vy_B_k*self.timestep_length_s
        x_N_k_1 = xp_N_k + xp_B_k_1*torch.cos(phi_B_N_k) - yp_B_k_1*torch.sin(phi_B_N_k)
        y_N_k_1 = yp_N_k + xp_B_k_1*torch.sin(phi_B_N_k) + yp_B_k_1*torch.cos(phi_B_N_k)
        transformed_output[:, 3] = x_N_k_1
        transformed_output[:, 4] = y_N_k_1
        # Compute the new orientation at time k+1.
        phi_B_N_k_1 = phi_B_N_k + w_B_k*self.timestep_length_s
        transformed_output[:, 5] = phi_B_N_k_1
        

        # # Compute the "total" or "composed" rotation from the body frame to the
        # # inertial frame at time k+1. This is also part of the configuration.
        # # I.e., the relative rotation from the body frame (B) to the inertial
        # # frame (N) at time k+1.
        # phi_B_N_k_1 = phi_B_N_k + phi_B_B_k_1
        # transformed_output[:, 5] = phi_B_N_k_1

        # # Express xp_B_k_1 and yp_B_k_1 in the inertial frame, and then add them
        # # to the initial xp_N_k and yp_N_k, respectively.
        # transformed_output[:, 3] = xp_N_k + xp_B_k_1*torch.cos(phi_B_N_k) - yp_B_k_1*torch.sin(phi_B_N_k)
        # transformed_output[:, 4] = yp_N_k + xp_B_k_1*torch.sin(phi_B_N_k) + yp_B_k_1*torch.cos(phi_B_N_k)
        # # px_dot = vx * np.cos(phi) - vy * np.sin(phi)
        # # py_dot = vx * np.sin(phi) + vy * np.cos(phi)
        return transformed_output

    def training_step(self, batch: torch.Tensor, batch_idx):

        batch = batch.to(dtype=torch.float32)
        zk = batch[:, 0, :]             # zk == [uk, xk]
        xk_1_gt = batch[:, -1, 2:]      # Ground truth next state.

        # Get the next state predicted by the network.
        x_k_1_pred = self.forward(zk)

        # TODO: See what happens if we just try to normalize the output and
        # ground truth next state with the means and stdevs from the data before
        # computing the loss. Improvement?
        output_means_tensor = torch.zeros_like(x_k_1_pred)
        output_means_tensor[:, :] = torch.tensor(self.output_means)
        output_scales_tensor = torch.zeros_like(x_k_1_pred)
        output_scales_tensor[:, :] = torch.tensor(self.output_scales)
        output_scales_padding = 1e-6*torch.ones_like(output_scales_tensor)
        # Normalize the predicted next state and the ground truth next state.
        normalized_x_k_1_pred = (x_k_1_pred - output_means_tensor) / (output_scales_tensor + output_scales_padding)
        normalized_xk_1_gt = (xk_1_gt - output_means_tensor) / (output_scales_tensor + output_scales_padding)

        loss = nn.functional.mse_loss(normalized_x_k_1_pred, normalized_xk_1_gt)
        # loss = nn.functional.mse_loss(x_k_1_pred, xk_1_gt)
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
        
        # TODO: See what happens if we just try to normalize the output and
        # ground truth next state with the means and stdevs from the data before
        # computing the loss. Improvement?
        output_means_tensor = torch.zeros_like(x_k_1_pred)
        output_means_tensor[:, :] = torch.tensor(self.output_means)
        output_scales_tensor = torch.zeros_like(x_k_1_pred)
        output_scales_tensor[:, :] = torch.tensor(self.output_scales)
        output_scales_padding = 1e-6*torch.ones_like(output_scales_tensor)
        # Normalize the predicted next state and the ground truth next state.
        normalized_x_k_1_pred = (x_k_1_pred - output_means_tensor) / (output_scales_tensor + output_scales_padding)
        normalized_xk_1_gt = (xk_1_gt - output_means_tensor) / (output_scales_tensor + output_scales_padding)

        loss = nn.functional.mse_loss(normalized_x_k_1_pred, normalized_xk_1_gt)

        # loss = nn.functional.mse_loss(x_k_1_pred, xk_1_gt)
        self.log("val_loss", loss, prog_bar=True)
        return loss
    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
    
    # TODO: Add a "to_torch_script" method here--unless the base class already
    # includes that.
    
if __name__ == "__main__":

    # TODO: Add CLI.

    # Could use LightningCLI potentially--but I'm not sure if we're gaining any
    # significant features from it compared to what we have with our own.

    # TODO: Need the user to specify an output logging directory.

    # TODO: Need to create a "NormalizedDiscrete" model directory under the
    # provided logging directory. For now, just going to hardcode one.


    # TODO: Need the user to specify the path to a configuration YAML file that
    # contains all the necessary hyperparameters and configuration needed to
    # describe the model.

    # TODO: Parameters for the LightningModule below should be loaded from a
    # YAML file--not hardcoded here.
    learning_rate = 1e-5
    batch_size = 64
    # IAC ART SIM Indy Lights car dataset
    # output_means = [32.1704, 0.1766, -0.0705, -127.2973, 171.3312, -0.6868]
    # output_scales = [10.4266, 0.5028, 0.2602, 255.2563, 139.2005, 1.6958]
    # F1TENTH GYM dataset so far
    output_means = [1.89820329762276, 0.0, 0.079250267740277, -5.05624974035195, 2.57101711068111, 0.470019765802305]
    output_scales = [0.176003883715796, 0.0, 0.326492839114055, 18.3854560359877, 4.0841917924177, 1.93289692697676]
    dynamics_model = NormalizedDiscrete(model=model,
                                        input_means=[0.9528, -0.0059, 32.1704, 0.1766, -0.0705],
                                        input_scales=[2.6898, 0.0349, 10.4266, 0.5028, 0.2602],
                                        output_means=output_means,
                                        output_scales=output_scales,
                                        timestep_length_s=0.05,
                                        learning_rate=learning_rate,
                                        batch_size=batch_size)
    
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

    # dataset_directory_path =
    # r"/home/nlitz88/Documents/neuralmpc_datasets/2023_08_31-putnam/2023_08_31-13_46_57_0_old/"
    # dataset_sample_index_path = Path(r"/home/nlitz88/Documents/neuralmpc_datasets/2024_03_04-putnam-sim-interp-20Hz/2024_03_04-putnam-sim-interp-20Hz_sample_index.json")
    # dataset_sample_index_path = Path(r"/home/nlitz88/Documents/neuralmpc_datasets/2024_03_04-putnam-sim/2024_03_04-putnam-sim_sample_index.json")
    # dataset_sample_index_path = Path(r"/home/nlitz88/Documents/neuralmpc_datasets/2023_08_31-putnam/2023_08_31-13_46_57_0_old/2023_08_31-13_46_57_0_old_sample_index.json")
    dataset_sample_index_path = Path(r"/home/nlitz88/Documents/neuralmpc_datasets/2024-05-06-f1tenth-sim/2024-05-06-f1tenth-sim_sample_index.json")

    # discrete_dynamics_dataset = NormalizedDiscreteCSVDataset(dataset_directory_path=dataset_directory_path,
    #                                                          desired_columns=desired_columns,
    #                                                          sample_length=2,
    #                                                          standardize=False)
    dataset = NormalizedDiscreteCSVDataset(sample_index_path=dataset_sample_index_path)
    
    # Test Generating a random train / validation split.
    TRAIN_SPLIT_SIZE = 0.95
    VALIDATION_SPLIT_SIZE = 0.05
    generator = torch.Generator().manual_seed(0)
    train_split, validation_split = random_split(dataset=dataset,
                                                lengths=[TRAIN_SPLIT_SIZE, VALIDATION_SPLIT_SIZE],
                                                generator=generator)
    
    # Create dataloaders around the training and validation splits.
    train_loader = DataLoader(dataset=train_split,
                              batch_size=batch_size,
                              pin_memory=True,
                              pin_memory_device="cuda",
                              num_workers=6)
    validation_loader = DataLoader(dataset=validation_split, 
                                   batch_size=batch_size,
                                   pin_memory=True,
                                   pin_memory_device="cuda",
                                   num_workers=6)

    # Create an early-stopping callback for the trainer.
    # https://lightning.ai/docs/pytorch/stable/common/early_stopping.html#
    early_stopping_callback = EarlyStopping(monitor="val_loss", mode="min", patience=2)

    # TODO: Swap this hardcoded path out for one provided via the CLI.
    logging_directory = "/home/nlitz88/Documents/neuralmpc_experiments/NormalizedDiscrete"
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=logging_directory)
    trainer = L.Trainer(max_epochs=100,
                        callbacks=[early_stopping_callback],
                        logger=tb_logger,
                        limit_train_batches=1.0)
    
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
