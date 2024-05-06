import pandas as pd
import torch
from torch.utils.data import DataLoader, Subset
import lightning as L
import numpy as np
from tqdm import tqdm
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path

import sys

import yaml

sys.path.insert(0, "../../")

from neural_mpc.evaluators.visualize import create_channels_plot, create_channel_errors_plot, LabelledModelOutput, create_positions_plot, create_state_space_figure
from neural_mpc.datasets.art.dataset import ArtDataset
from neural_mpc.models.normalized_discrete.model import NormalizedDiscrete
from neural_mpc.models.single_track.single_track_dynamics import Single_Track_Dynamics

if __name__ == "__main__":

    torch.set_printoptions(sci_mode=False)
    
    # TODO: Create a command line interface that accepts the checkpoint filepath
    # and automatically generates the torchscript file.

    # checkpoint_model_path = r"lightning_logs/version_49/checkpoints/epoch=14-step=125295.ckpt"
    # jit_model_path = r"/home/nlitz88/repos/iac/Racing-LMPC-ROS2/src/neural_mpc/models/normalized_body_only_discrete/lightning_logs/version_50/checkpoints/version_50.pt"
    jit_model_path = r"/home/nlitz88/repos/iac/Racing-LMPC-ROS2/src/neural_mpc/models/normalized_body_only_discrete/lightning_logs/version_1/checkpoints/version_1.pt"

    # art_test_split_path = r"/home/nlitz88/Documents/neuralmpc_datasets/2024-03-24-putnam-sim-split/test"
    # art_test_split_path = r"/home/nlitz88/Documents/neuralmpc_datasets/2023_08_31-putnam/2023_08_31-13_46_57_0_old/"
    art_test_split_path = r"/home/nlitz88/Documents/neuralmpc_datasets/2024_03_04-putnam-sim-interp/"
    # output_dir = r"/home/nlitz88/Documents/neuralmpc_evaluations"
    output_dir = Path(r"./iframe_figures")

    model_dataset_frequency_hz = 20.0

    # Initialize any models that we are evaluating.

    # model = NormalizedBodyOnlyDiscrete.load_from_checkpoint(checkpoint_model_path).to(device="cpu")
    nbod_model = torch.jit.load(jit_model_path)
    # model.to(device="cuda")

    SINGLE_TRACK_BASE_CONFIG_PATH = r"/home/nlitz88/repos/iac/Racing-LMPC-ROS2/src/neural_mpc/models/single_track/iac_car_base.param.yaml"
    SINGLE_TRACK_TRACK_CONFIG_PATH = r"/home/nlitz88/repos/iac/Racing-LMPC-ROS2/src/neural_mpc/models/single_track/iac_car_single_track.param.yaml"
    with open(SINGLE_TRACK_BASE_CONFIG_PATH, 'r') as file_BC:
        base_config = yaml.safe_load(file_BC)
    with open(SINGLE_TRACK_TRACK_CONFIG_PATH, 'r') as file_TC:
        track_config = yaml.safe_load(file_TC)
    single_track_model = Single_Track_Dynamics(base_config=base_config,
                                               track_config=track_config)

    desired_columns = ["u/u_a",         
                       "u/u_steer",
                       "v/v_long",
                       "v/v_tran",
                       "w/w_psi",
                       "x/x",
                       "x/y",
                       "e/psi"
                       ]

    # Create a dataset wrapper over the test split at the provided path.
    test_split_dataset = ArtDataset(dataset_directory_path=art_test_split_path,
                                    desired_columns=desired_columns,
                                    sample_length=2,
                                    standardize=False)
    
    # Select the first 20 samples from the test split.
    test_split_subset_indices = [i for i in range(0, len(test_split_dataset))]
    # test_split_subset_indices = [i for i in range(0, 2000)]
    test_split_subset = Subset(dataset=test_split_dataset, indices=test_split_subset_indices)
    # Use a dataloader around the dataset to handle "tensorization" of the
    # dataset samples. Not strictly necessary, but saves some messiness.
    test_split_dataset_dataloader = DataLoader(dataset=test_split_subset,
                                               batch_size=1,
                                               num_workers=0,
                                               shuffle=False)
    
    # Also, I'm debating if we even need/should use the dataloader here.
    
    # starting_states = []
    nbod_pred_next_states = []
    single_track_next_states = []
    actual_next_states = []
    # with torch.no_grad():
    with tqdm(total=len(test_split_dataset_dataloader)) as test_progress_bar:
        for sample in test_split_dataset_dataloader:
            
            model_input = sample[:, 0, :].to(dtype=torch.float32)
            # model_input.to(dtype=torch.float32)
            # print(f"Model Input: {model_input}")
            # model_output = model.predict_step(model_input)
            nbod_output = nbod_model(model_input)
            nbod_pred_next_states.append(nbod_output[0, :].tolist())

            # TODO: Could also call the single_track_model here to compare its
            # output against ground truth and the learned model.
            # Compute u_fd and u_fb from u_a. 
            np_model_input = model_input.to(device="cpu").numpy()
            single_track_u = np.zeros(3)
            u_a = np_model_input[0, 0]
            single_track_u[0] = u_a if u_a >= 0.0 else 0.0 # u_fd
            single_track_u[1] = u_a if u_a < 0.0 else 0.0 # u_fb
            single_track_u[2] = np_model_input[0, 1] # u_steer
            single_track_output = single_track_model.get_next_state(np_model_input[0, 2:], single_track_u, dt=1.0/model_dataset_frequency_hz)
            single_track_next_states.append(single_track_output)

            # starting_states.append(sample[0, 0, 2:].tolist())
            # Store the nbod's model output here.
            
            # Store the ground_truth next state.
            actual_next_states.append(sample[0, -1, 2:].tolist())
            
            # print(f"Model output: {model_output}")
            # print(f"Gt next state: {sample[0, -1, 2:]}")
            test_progress_bar.update()

    # Write the predicted output to a CSV in the output directory.
            
    # TODO: Create a dataframe or something where two columns correspond to the
    # ground truth current state, next state x and y, and the next two columns
    # correspond.
        

    # print()
    # print("Predicted next states")
    # # print(predicted_next_states)
    # for next_state in predicted_next_states:
    #     print(next_state)

    # TODO: Merge/interleave the starting state points into the ground truth and
    # predicted arrays 

    # Create numpy arrays from the lists.
    # starting_states = np.array(starting_states)
    actual_next_states = np.array(actual_next_states)
    nbod_pred_next_states = np.array(nbod_pred_next_states)
    single_track_next_states = np.array(single_track_next_states)

    actual_next_states_df = pd.DataFrame(actual_next_states)
    actual_next_states_df.columns = ["vx", "vy", "w", "xp", "yp", "phi"]
    nbod_pred_next_states_df = pd.DataFrame(nbod_pred_next_states)
    nbod_pred_next_states_df.columns = ["vx", "vy", "w", "xp", "yp", "phi"]
    single_track_next_states_df = pd.DataFrame(single_track_next_states)
    single_track_next_states_df.columns = ["vx", "vy", "w", "xp", "yp", "phi"]

    # Test the visualization script functions.
    labelled_model_outputs = [
        LabelledModelOutput(model_name="gt",
                            model_output_dataframe=actual_next_states_df),
        LabelledModelOutput(model_name="nbod",
                            model_output_dataframe=nbod_pred_next_states_df),
        LabelledModelOutput(model_name="single_track",
                            model_output_dataframe=single_track_next_states_df)
    ]
    figure = create_channels_plot(labelled_model_outputs=labelled_model_outputs,
                                 common_column_names=["vx_m_s", "vy_m_s", "w_rad_s", "xp_m", "yp_m", "phi_rad"],
                                 figure_title="Predicted State Channels")
    figure.write_html(output_dir/"channels.html")

    labelled_model_outputs = [
        LabelledModelOutput(model_name="nbod",
                            model_output_dataframe=nbod_pred_next_states_df),
        LabelledModelOutput(model_name="single_track",
                            model_output_dataframe=single_track_next_states_df)
    ]
    error_figure = create_channel_errors_plot(ground_truth_dataframe=actual_next_states_df,
                                              labelled_model_outputs=labelled_model_outputs,
                                              common_column_names=["vx_error_m_s", "vy_error_m_s", "w_error_rad_s", "xp_error_m", "yp_error_m", "phi_error_rad"],
                                              figure_title="Predicted State Channel Errors")
    error_figure.write_html(output_dir/"channel_errors.html")

    # Plot positions from each model's output.
    labelled_model_outputs = [
        LabelledModelOutput(model_name="gt",
                            model_output_dataframe=actual_next_states_df),
        LabelledModelOutput(model_name="nbod",
                            model_output_dataframe=nbod_pred_next_states_df),
        LabelledModelOutput(model_name="single_track",
                            model_output_dataframe=single_track_next_states_df)
    ]
    positions_figure = create_positions_plot(labelled_model_outputs=labelled_model_outputs)
    positions_figure.write_html(output_dir/"positions.html")

    # Test state space visualization.
    state_space_figure = create_state_space_figure(ground_truth_states_dataframe=actual_next_states_df)
    state_space_figure.write_html(output_dir/"state_space.html")

    # Construct pandas frames from the predictions.
    actual_next_states_df = pd.DataFrame(actual_next_states)
    actual_next_states_df.columns = ["vx", "vy", "w", "xp", "yp", "phi"]
    actual_next_states_df.insert(loc=6,
                                 column="source",
                                 value=["ground_truth" for _ in range(len(actual_next_states))])

    nbod_pred_next_states_df = pd.DataFrame(nbod_pred_next_states)
    nbod_pred_next_states_df.columns = ["vx", "vy", "w", "xp", "yp", "phi"]
    nbod_pred_next_states_df.insert(loc=6,
                                    column="source",
                                    value=["nbod_pred" for _ in range(len(actual_next_states))])

    # print(nbod_pred_next_states_df)
    # actual_next_states_df = pd.DataFrame({
    #     "x": actual_next_states[:, 3],
    #     "y": actual_next_states[:, 4],
    #     "source": ["ground_truth" for _ in range(len(actual_next_states))]
    # })

    # nbod_pred_next_states_df = pd.DataFrame({
    #     "x": nbod_pred_next_states[:, 3],
    #     "y": nbod_pred_next_states[:, 4],
    #     "source": ["nbod" for _ in range(len(nbod_pred_next_states))]
    # })

    # # single_track_next_states_df = pd.DataFrame({
    # #     "x": single_track_next_states[:, 3],
    # #     "y": single_track_next_states[:, 4],
    # #     "source": ["single_track" for _ in range(len(single_track_next_states))]
    # # })

    # # actual_next_states_df = pd.DataFrame(columns=["x", "y", "source"])
    # # for next_state in actual_next_states:
    # #     actual_next_states_df.append([next_state[3], next_state[4], "ground_truth"])


    # # predicted_next_states_df = pd.DataFrame(columns=["x", "y", "source"])
    # # for next_state in predicted_next_states:
    # #     predicted_next_states_df.append([next_state[3], next_state[4], "predicted"])

    # # print(actual_next_states_df)
    
    combined_df = pd.concat([actual_next_states_df,
                             nbod_pred_next_states_df,
                            #  single_track_next_states_df
                             ], 
                             ignore_index=True, axis=0)

    # print("Combined dataframes")
    # print(combined_df)

    # # print(starting_states)
    # # print(predicted_next_states)

    # # Test plot
    # # fig = go.Figure()
    # # fig.add_trace(
    # #     go.Scatter(
    # #         x=starting_states[:, 3],
    # #         y=starting_states[:, 4],
    # #         mode="lines+markers",
    # #         marker=dict(
    # #             symbol="arrow",
    # #             size=15,
    # #             angleref="previous",
    # #         )
    # #     )
    # # )

    # fig = px.scatter(x=starting_states[:, 3], y=starting_states[:, 4])
    
    # Create scatter plot for vehicle positions within predicted states.
    # fig = px.scatter(data_frame=combined_df, x="xp", y="yp", color="source")
    # # fig.show(renderer='iframe')
    # fig.write_html(output_dir/"positions.html")

    # fig = px.scatter(data_frame=combined_df, x="vx", y="vy", color="source")
    # # fig.show(renderer='iframe')
    # fig.write_html(output_dir/"velocities.html")

    # TODO: Turn these visualizations into functions.
    # Create line plot with each model's velocity error w.r.t. ground truth.    
    nbod_pred_next_states_df.columns = ["nbod_vx", "nbod_vy", "nbod_w", "nbod_xp", "nbod_yp", "nbod_phi", "source"]
    combined_df = pd.concat([actual_next_states_df,
                            nbod_pred_next_states_df,
                        #  single_track_next_states_df
                            ], 
                            ignore_index=False, axis=1)

    # Create traces for each model's values.
    trace_gt_vx = go.Scatter(x=combined_df.index, y=combined_df['vx'], mode='lines', name='gt_vx')
    trace_nbox_vx = go.Scatter(x=combined_df.index, y=combined_df['nbod_vx'], mode='lines', name='nbod_vx')

    # Create layout
    layout = go.Layout(title='gt_vx vs nbod_vx',
                    xaxis=dict(title='Index'),
                    yaxis=dict(title='Value'))

    # Create figure
    fig = go.Figure(data=[trace_gt_vx, trace_nbox_vx], layout=layout)

    # Show plot
    # fig.show()
    fig.write_html(output_dir/"vx.html")

    # Do the same for vy values.
    # Create traces for each model's values.
    trace_gt_vy = go.Scatter(x=combined_df.index, y=combined_df['vy'], mode='lines', name='gt_vy')
    trace_nbox_vy = go.Scatter(x=combined_df.index, y=combined_df['nbod_vy'], mode='lines', name='nbod_vy')

    # Create layout
    layout = go.Layout(title='gt_vy vs nbod_vy',
                    xaxis=dict(title='Index'),
                    yaxis=dict(title='Value'))

    # Create figure
    fig = go.Figure(data=[trace_gt_vy, trace_nbox_vy], layout=layout)

    # Show plot
    # fig.show()
    fig.write_html(output_dir/"vy.html")

    # Create line plot with each model's predicted velocity vs. ground truth.
    
    # fig = px.line(data_frame=combined_df, y="vx", color="source")
    # fig.write_html(output_dir/"vx.html")
    # fig = px.line(data_frame=combined_df, y="vy", color="source")
    # fig.write_html(output_dir/"vy.html")

    # Create a pandas dataframe out of the lists.
    
    
    # Just try to run inference on a value from the test split.
    
    # Grab the first value and roll out the dynamics.
    # print(f"Rollout starting at state from the dataset.")
    # first_sample = discrete_dynamics_dataset[1]
    # first_state = first_sample[0, :]
    # first_state_controls = first_state[:2]
    # first_state_qdot = first_state[2:5]
    # current_state = np.append(first_state_controls, first_state_qdot)
    # current_state = torch.Tensor(current_state).to(device="cuda")
    # with torch.no_grad():
    #     timesteps = 20
    #     for i in range(timesteps):
    #         print(current_state.to(device="cpu").numpy())
    #         next_state = model(current_state)
    #         next_state_qdot = next_state[0:3]
    #         current_state = torch.Tensor(np.append(first_state_controls, next_state_qdot.to(device="cpu").numpy())).to(device="cuda")

    # Now, rollout the learned dynamics starting at an arbitrary (out of
    # distribution) state.
    # print(f"Rolling out dynamics starting at out of distribution state.")
    # np.set_printoptions(suppress=True)
    # test_controls = [0.0, 0.0] # u_a, u_steer
    # first_qdot = [20.0, 0.0, 0.0] # vx, vy, w
    # current_input = np.append(test_controls, first_qdot)

    # # NORMALIZE INPUT
    # # means = discrete_dynamics_dataset._means[0:5]   # Grab means only for u_a-->w
    # # stdevs = discrete_dynamics_dataset._stdevs[0:5] # Grab stdevs only for
    # # u_a-->w  
    # input_means = [5.0, 0.0, 40.0, 0.0, 0.0]
    # input_scales = [5.0, 0.35, 40.0, 2.0, 1.0]
    # input_scales_padding = 1e-6*np.ones_like(input_scales)
    # current_input = (current_input - input_means) / (input_scales + input_scales_padding)
    # print(f"First input state: [u_a, u_steer, vx_B_k, vy_B_k, w_B_k]:\n{current_input}")
    # print(f"Model outputs: vx_B_k_1, vy_B_k_1, w_B_k_1, xp_B_k_1, yp_B_k_1, phi_B_k_1")
    # current_input = torch.tensor(current_input, dtype=torch.float32).to(device="cuda")
    # with torch.no_grad():
    #     timesteps = 20
    #     for i in range(timesteps):
    #         # print(f"Input {i}: {current_input.to(device='cpu').numpy()}")
    #         next_state = model(current_input)

    #         # Denormalize model output.
    #         output_means = [40.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #         output_scales = [40.0, 2.0, 1.0, 1.0, 1.0, 1.0]
    #         output_scales_padding = 1e-6*np.ones_like(output_scales)
    #         # x_hat = x_hat*(output_scales_tensor + output_scales_padding) +
    #         # output_means_tensor
    #         next_state = next_state.to(device="cpu").numpy()
    #         next_state = next_state*(output_scales + output_scales_padding) + output_means

    #         print(f"Output {i}: {next_state}")
    #         next_state_qdot = next_state[0:3] # output of model is [q_dot_k_1, q_k_1]
    #         next_input = np.append(test_controls, next_state_qdot)

    #         # NORMALIZE INPUT
    #         next_input = (next_input - input_means) / (input_scales + input_scales_padding)

    #         current_input = torch.Tensor(next_input).to(device="cuda")

    # UPDATE: Use the torchscript wrapper and call model.predict_step to
    # inference. This will handle all the normalization and transformations.
            
    # Now, test how far off the model's predicted next state is at each
    # timestep.
    # timesteps = 20
    # with torch.no_grad():
    #     for i in range(timesteps):
    #         current_sample = discrete_dynamics_dataset[i]
    #         xk = 
            # print(f"State {i}: {}") 
    # current_sample = discrete_dynamics_dataset[0]
    # current_state = 