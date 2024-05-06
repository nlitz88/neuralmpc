"""Module containing functions to evaluate the NormalizedDiscrete model across a
range of scenarios and samples present in a given NormalizedDiscrete dataset.
"""




# TODO: Set up command line interface to carry out the evaluations.

# TODO: I'm not sure if this module should be invoked directly--I supposed it
# could be. 

# TODO: I'm also not sure if this module should contain functions to create
# visualizations FROM the evaluation result data. Kinda feel like those should
# live in another script if we're being very rigorous. However, maybe for now it
# doesn't matter that much.

# I think the goal for these functions should be that they are portable. I.e.,
# they can either be called at the tail end of the training pipeline for
# convenience and the output files placed in the lightning_logs directory for
# that version, or they can be invoked via this module's command line interface
# and placed wherever.

import argparse
from pathlib import Path
import sys
import time
from typing import List, Optional, Tuple
import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, Subset
from tqdm import tqdm

sys.path.insert(0, "../../")
from neural_mpc.normalized_discrete.dataset import NormalizedDiscreteCSVDataset, NORMALIZED_DISCRETE_CSV_COLUMNS
from neural_mpc.normalized_discrete.model import NormalizedDiscrete
# Temporary
from neural_mpc.normalized_discrete.visualize_model import create_2d_state_plot, create_channel_error_plots, create_channel_value_plots, create_channel_value_comparison_plots


def get_per_sample_predictions(model: NormalizedDiscrete,
                               dataset_split_dataloader: DataLoader) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Evaluates the provided model on the provided dataset split.

    Args:
        model (NormalizedDiscrete): The model each sample will be passed into.
        dataset_split_dataloader (DataLoader): The dataloader for the dataset
        split the samples will be drawn from.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: A dataframe containing the ground
        truth next states and a dataframe containing the model's predicted next
        states.
    """

    # Create a list of model outputs to return. This will essentially be a
    # parallel array with the list of samples in the provided split.
    gt_next_states = []
    pred_next_states = []
    # Get the model's output for each sample in the provided dataset split.
    for sample in dataset_split_dataloader:

        # Extract the input data from the sample. This is the first row of the
        # 2D numpy array == the sample. The model only wants the first row of
        # each sample in the batch, so select it here.
        model_input = sample[:, 0, :].to(dtype=torch.float32, device=model.device)
        # Get the model's predicted next state given the provided current state.
        pred_next_state = model(model_input).to("cpu").detach().numpy().tolist()
        pred_next_states += pred_next_state
        # Also grab the ground truth next state and store that.
        gt_next_state = sample[:, -1, 2:].to("cpu").detach().numpy().tolist()
        gt_next_states += gt_next_state
        
    # Create dataframes from the model outputs and ground truth next states.
    gt_next_states_columns = NORMALIZED_DISCRETE_CSV_COLUMNS[2:]
    gt_next_states_df = pd.DataFrame(gt_next_states, columns=gt_next_states_columns)
    pred_next_states_columns = [f"pred_{col}" for col in gt_next_states_columns]
    pred_next_states_df = pd.DataFrame(pred_next_states, columns=pred_next_states_columns)

    return gt_next_states_df, pred_next_states_df

# First, create a function that computes the channel errors for each sample.
# I.e., just takes each ground truth state and elementwise subtracts the
# predicted next state from it --> you get the model's error in each channel for
# each sample in the dataset.
def compute_sample_channel_errors(gt_next_states_df, pred_next_states_df) -> pd.DataFrame:

    # Swap out the predicted next states dataframe's columns for the ground
    # truth's columns--just temporarily so that we can subtract the two
    # dataframes elementwise.
    temp_pred_next_states_df = pred_next_states_df.copy()
    temp_pred_next_states_df.columns = gt_next_states_df.columns
    channel_errors = (gt_next_states_df - temp_pred_next_states_df).abs()
    # Update the columns names of the channel errors dataframe.
    channel_errors.columns = [f"{col}_error" for col in gt_next_states_df.columns]
    # Returned the squared errors.
    return channel_errors

# TODO: Before we compute any additional metrics on each of the channels--I
# think we need to normalize the channel errors. Maybe not, actually? I mean, I
# think we just have to understand the SCALE of each channel so that the error
# amount we see is meaningful.

# For example: If the yaw angle is in radians, then an error of 0.1 in the yaw
# == 5 degrees off. 0.1 seems like a small amount of error, but that's actually
# decently significant. Whereas an error of 0.1 in px or py == 100cm isn't AS
# significant. Therefore, just have to understand the scale of each channel if
# we don't normalize them.

# I suppose normalization is REALLY only necessary if you're computing the
# distance from one vector result to the other.

def compute_mean_sample_error(sample_channel_errors: pd.DataFrame) -> pd.DataFrame:
    mean_sample_errors = sample_channel_errors.mean(axis=1)
    return mean_sample_errors

# Combine all three functions to compute mean, median, and standard deviation
# for each channel.
# THIS IS BASICALLY A CONVENIENCE/WRAPPER FUNCTION for evaluating model
# performance on an entire dataset split for each sample in that split.
def compute_channel_metrics(sample_channel_errors: pd.DataFrame) -> pd.DataFrame:
    channel_means = sample_channel_errors.mean(axis=0)
    channel_medians = sample_channel_errors.median(axis=0)
    channel_stdevs = sample_channel_errors.std(axis=0)
    channel_metrics = pd.DataFrame([channel_means, channel_medians, channel_stdevs], index=['mean', 'median', 'stdev'])
    channel_metrics.columns = channel_metrics.columns
    return channel_metrics

# Function specially for computing metrics we might care about for a given open
# loop rollout--as some metrics computed for these would not be very
# meaningful for evaluation on the entire dataset split (like the total channel
# error, for example).
def compute_open_loop_rollout_channel_metrics(rollout_sample_channel_errors: List[pd.DataFrame]) -> pd.DataFrame:
    # Compute the mean, median, and standard deviation for each channel across
    # all the rollouts.
    channel_means = np.mean(rollout_sample_channel_errors, axis=0)
    channel_medians = np.median(rollout_sample_channel_errors, axis=0)
    channel_stdevs = np.std(rollout_sample_channel_errors, axis=0)
    channel_total_absolute_error = np.sum(rollout_sample_channel_errors, axis=0)
    channel_metrics = pd.DataFrame([channel_means, channel_medians, channel_stdevs, channel_total_absolute_error], index=['mean', 'median', 'stdev', 'total_absolute_error'])
    channel_metrics.columns = channel_metrics.columns
    return channel_metrics

# TODO: Write a function that creates a histogram for a given channel's error
# across a list of provided open loop rollout sample channel errors.

# TODO: Maybe make a function that picks random indices in the dataset to start
# at and generates rollouts starting at each of those, and where some number of
# rollouts is specified.


def compute_open_loop_rollout(subset_dataloader: DataLoader,
                              model: NormalizedDiscrete):
    
    pred_next_states = []
    gt_next_states = []
    # Start the rollout with the first sample in the subset dataloader.
    for i, sample in enumerate(subset_dataloader):
        # If this is the first iteration, then the input to the model is just
        # the current state and control from the sample.
        if i == 0:
            model_input = sample[:, 0, :].to(dtype=torch.float32, device=model.device)
        # Otherwise, we construct the model input as the current control and the
        # predicted next state of the model from the last iteration.
        else:
            current_control = sample[:, 0, :2].to(dtype=torch.float32, device=model.device)
            model_input = torch.cat([current_control, predicted_next_state], dim=1)
        
        # Get the model's predicted next state given the provided current state.
        predicted_next_state = model(model_input)

        # TEST: Okay, right here, let's try to mod the heading angle output by
        # the model so that it remains within the euler angle range present in
        # the training data. Not sure how much of an impact this really has, but
        # ACTUALLY, wait a minute--the underlying neural network (as a
        # principle) NEVER TAKES IN THE ACTUAL HEADING ANGLE--therefore, it
        # doesn't matter if it's out of distribution! If anything, the only
        # thing this would help with is with our analysis of the open loop data
        # later. Otherwise, it should literally have zero effect on the actual
        # model's performance. So I could just test that.
        # predicted_next_state[:, 5] = torch.fmod(predicted_next_state[:, 5], 2*np.pi)

        # Okay, quickly think about what we want to do. If the angle is greater
        # than 3.14, then we want to subtract 2*pi from it. If it's less than
        # -3.14, then we want to add 2*pi to it. This will keep the angle within
        # the range of -pi to pi.
        predicted_next_state[:, 5] = torch.where(predicted_next_state[:, 5] > np.pi, predicted_next_state[:, 5] - 2*np.pi, predicted_next_state[:, 5])
        predicted_next_state[:, 5] = torch.where(predicted_next_state[:, 5] < -np.pi, predicted_next_state[:, 5] + 2*np.pi, predicted_next_state[:, 5])

        # Store the predicted next state in the prediction list.
        pred_next_states += predicted_next_state.clone().detach().cpu().numpy().tolist()
        # Store the ground truth next state in the ground truth list.
        gt_next_state = sample[:, -1, 2:].to("cpu").detach().numpy().tolist()
        gt_next_states += gt_next_state

    # When we're done computing the rollout values, construct a dataframe from
    # the predicted next states and ground truth next states.
    pred_next_states_columns = [f"pred_{col}" for col in NORMALIZED_DISCRETE_CSV_COLUMNS[2:]]
    pred_next_states_df = pd.DataFrame(pred_next_states, columns=pred_next_states_columns)
    
    gt_next_states_columns = [f"gt_{col}" for col in NORMALIZED_DISCRETE_CSV_COLUMNS[2:]]
    gt_next_states_df = pd.DataFrame(gt_next_states, columns=gt_next_states_columns)

    return pred_next_states_df, gt_next_states_df


# This function is just here as a convenience to generate a number of different
# rollouts from the provided dataset. This one just creates a new rollout
# starting every rollout_length samples.
def compute_sequential_model_rollouts(model: NormalizedDiscrete,
                                      dataset: Dataset,
                                      rollout_length: int,
                                      max_rollouts: Optional[int] = None) -> List[pd.DataFrame]:
    if max_rollouts == None:
        max_rollouts = len(dataset) // rollout_length
    # TODO: Limit number of rollouts via max_rollouts.
    # Maybe just randomly decide (using some predetermined seed) where we look
    # at the rollouts. Could definitely be more strategic about this--just
    # really need to look at the positions and velocities in the dataset.
    
    # For now, just create a dataloader over a new subset of the dataset for
    # every rollout_length samples.
    rollouts = []
    for i in range(0, len(dataset) - rollout_length, rollout_length):
        rollout_subset = Subset(dataset, range(i, i+rollout_length))
        subset_dataloader = DataLoader(rollout_subset,
                                       batch_size=1,
                                       shuffle=False,
                                       pin_memory=True,
                                       pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")
        # Compute the open loop rollout for this subset.
        pred_next_states_df, gt_next_states_df = compute_open_loop_rollout(subset_dataloader, model)
        # Append the predicted next states and ground truth next states to the
        # list of rollouts.
        rollouts.append((pred_next_states_df, gt_next_states_df))
        # Quit early if we've reached the maximum number of rollouts. Kinda
        # jank, but works for now.
        if len(rollouts) == max_rollouts:
            break

    return rollouts


if __name__ == "__main__":

    # Set up argparser and add argument for the filepath to the sample index
    # file.
    parser = argparse.ArgumentParser(description="Script to evaluate the NormalizedDiscrete model on a given dataset split.")
    parser.add_argument("sample_index_filepath", type=str, help="The path to the dataset split's sample index JSON file.")
    # TODO: I think loading the model from a model config yaml file or something
    # like Lightning does would ultimately be the better move.
    parser.add_argument("model_filepath", type=str, help="The path to the model's weights (.pth) file.")
    parser.add_argument("-o", "--output_directory", type=str, help="The directory to save the visualization to. Defaults to \
                                                                    the current working directory if not specified.")
    args = parser.parse_args()

    ######################## VALIDATE ARGUMENTS ########################

    # First, validate the sample index filepath argument.
    try:
        sample_index_filepath = Path(args.sample_index_filepath)
    except Exception as exc:
        print(f"Invalid sample index filepath {str(args.sample_index_filepath)} provided.")
        raise exc
    if not sample_index_filepath.exists():
        print(f"The provided sample index filepath {sample_index_filepath} does not exist.")
        raise FileNotFoundError
    
    # Next, validate the model filepath argument.
    try:
        model_filepath = Path(args.model_filepath)
    except Exception as exc:
        print(f"Invalid model filepath {str(args.model_filepath)} provided.")
        raise exc
    if not model_filepath.exists():
        print(f"The provided model filepath {model_filepath} does not exist.")
        raise FileNotFoundError
    
    # Next, validate the provided output directory.
    if args.output_directory is not None:
        try:
            output_directory = Path(args.output_directory)
        except Exception as exc:
            print(f"Provided output directory {args.output_directory} is not a valid path.")
            raise exc
        # If the output directory doesn't exist, warn the user and try to create
        # it anyway.
        if not output_directory.exists():
            print(f"WARNING: Provided output directory {output_directory} does not exist. Will try to create it anyway.")
            output_directory.mkdir(parents=True, exist_ok=True)
        if not output_directory.is_dir():
            print(f"Provided output directory {output_directory} is not a directory.")
            raise NotADirectoryError
    # Otherwise, if no output directory was provided, just use the current
    # working directory.
    else:
        output_directory = Path().cwd()


    ######################## INITIALIZE DATASET AND MODEL ########################

    # If the dataset file does exist, hand it off to the dataset wrapper's
    # constructor to try and initialize itself from it.
    try:
        dataset = NormalizedDiscreteCSVDataset(sample_index_filepath)
    except Exception as exc:
        print(f"Error initializing the dataset wrapper from the provided sample index file {sample_index_filepath}.")
        raise exc
    
    # Create a dataloader around the dataset wrapper. 
    dataloader = DataLoader(dataset,
                            batch_size=1024,
                            shuffle=False,
                            pin_memory=True,
                            pin_memory_device="cuda" if torch.cuda.is_available() else "cpu")

    # If the model filepath exists, try to initialize the model's
    # LightningModule from it.
    try:
        model = NormalizedDiscrete.load_from_checkpoint(model_filepath)
    except Exception as exc:
        print(f"Error initializing the model from the provided model file {model_filepath}.")
        raise exc
    model.to(device="cuda" if torch.cuda.is_available() else "cpu")

    # Also, we want to store the evaluation results in a subdirectory of the
    # output directory that is specific to the dataset the model is being
    # evaluated on. Therefore, create a directory with the dataset's name within
    # the decided output directory.
    dataset_name = dataset.get_name()
    dataset_output_directory = output_directory / f"{dataset_name}_evaluation_results"
    try:
        dataset_output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory to hold the results of evaluating the provided model on the dataset {dataset_name} at {dataset_output_directory}.")
        raise exc


    ######################## BEGIN EVALUATION ########################


    # Start with evaluations across the entire dataset.

    # Call the function that gets the model's prediction for each sample in the
    # split.
    gt_next_states_df, pred_next_states_df = get_per_sample_predictions(model, dataloader)

    # Compute sample channel errors.
    sample_channel_errors = compute_sample_channel_errors(gt_next_states_df, pred_next_states_df)

    # Can then use this to compute both the mean error for each sample, as well
    # as the mean error for each channel across the dataset split.
    mean_sample_errors = compute_mean_sample_error(sample_channel_errors)

    # Compute all channel metrics.
    channel_metrics = compute_channel_metrics(sample_channel_errors)


    # Write the ground truth next states and predicted next states to respective
    # CSV files.
    gt_next_states_filepath = dataset_output_directory / "ground_truth_next_states.csv"
    gt_next_states_df.to_csv(gt_next_states_filepath, index=False)

    pred_next_states_filepath = dataset_output_directory / "predicted_next_states.csv"
    pred_next_states_df.to_csv(pred_next_states_filepath, index=False)

    sample_channel_errors_filepath = dataset_output_directory / "sample_channel_errors.csv"
    sample_channel_errors.to_csv(sample_channel_errors_filepath, index=False)

    # TODO: NOTE: These sample errors really aren't valid until we standardize
    # the values before taking the average.
    mean_sample_errors_filepath = dataset_output_directory / "mean_sample_errors.csv"
    mean_sample_errors.to_csv(mean_sample_errors_filepath, index=False)

    channel_metrics_filepath = dataset_output_directory / "channel_metrics.csv"
    channel_metrics.to_csv(channel_metrics_filepath, index=True)

    

    # Now compute multiple open loop rollout predictions from throughout the
    # provided dataset.

    # TODO: Consider making a function to wrap up all these calls so that we can
    # easily integrate this into the model training script as well.

    # Create an output directory specifically for open loop rollouts.
    open_loop_rollout_output_directory = dataset_output_directory / "open_loop_rollouts"
    try:
        open_loop_rollout_output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollouts at {open_loop_rollout_output_directory}.")
        raise exc

    # Compute a number of open loop rollouts from the dataset. 100 == 5s.
    rollouts = compute_sequential_model_rollouts(model, dataset, 100, max_rollouts=100)

    # Create a directory for saving each rollout's 2D position plot.
    rollouts_position_plots_directory = open_loop_rollout_output_directory / "position_plots"
    try:
        rollouts_position_plots_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollout position plots at {rollouts_position_plots_directory}.")
        raise exc
    
    # Create a directory for saving each rollout's channel error plot.
    rollouts_channel_error_plots_directory = open_loop_rollout_output_directory / "channel_error_plots"
    try:
        rollouts_channel_error_plots_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollout channel error plots at {rollouts_channel_error_plots_directory}.")
        raise exc
    
    # Create a directory for saving each rollout's channel value plots.
    rollouts_channel_value_plots_directory = open_loop_rollout_output_directory / "channel_value_plots"
    try:
        rollouts_channel_value_plots_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollout channel value plots at {rollouts_channel_value_plots_directory}.")
        raise exc
    
    # Create a directory for saving each rollout's channel metrics.
    rollouts_channel_metrics_directory = open_loop_rollout_output_directory / "channel_metrics"
    try:
        rollouts_channel_metrics_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollout channel metrics at {rollouts_channel_metrics_directory}.")
        raise exc
    
    # Create a directory for saving each rollout's predicted and ground truth
    # states.
    rollouts_states_directory = open_loop_rollout_output_directory / "states"
    try:
        rollouts_states_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Error creating the output directory for the open loop rollout states at {rollouts_states_directory}.")
        raise exc
    
    # Create a 2D position plot and channel error plot for each rollout and save
    # them in their respective directories.
    with tqdm(total=len(rollouts), desc="Creating 2D position plots for each open loop rollout.", unit="rollout") as pbar:
        for i, rollout in enumerate(rollouts):
            pred_next_states_df, gt_next_states_df = rollout

            # Create a visualization of the ground truth and predicted next states.
            position_plot_fig = create_2d_state_plot(gt_next_states_df,
                                                     f"gt_{NORMALIZED_DISCRETE_CSV_COLUMNS[5]}",
                                                     f"gt_{NORMALIZED_DISCRETE_CSV_COLUMNS[6]}", 
                                                     pred_next_states_df,
                                                     f"pred_{NORMALIZED_DISCRETE_CSV_COLUMNS[5]}",
                                                     f"pred_{NORMALIZED_DISCRETE_CSV_COLUMNS[6]}",
                                                     "Open Loop Rollout",
                                                     "px_N",
                                                     "py_N")
            # Write the visualization to an HTML file in the open loop rollout
            # output directory.
            position_plot_fig.write_html(rollouts_position_plots_directory / f"rollout_{i}.html")
            position_plot_fig.write_image(rollouts_position_plots_directory / f"rollout_{i}.png", width=1920, height=1080)

            # Compute the channel errors for the open loop rollout.
            rollout_sample_channel_errors = compute_sample_channel_errors(gt_next_states_df, pred_next_states_df)
            # Create a plot of the per-sample channel errors for the open loop
            # rollout.
            rollout_channel_error_plot = create_channel_error_plots(rollout_sample_channel_errors)
            # Write the channel error plot to an HTML file in the open loop rollout
            # output directory.
            rollout_channel_error_plot.write_html(rollouts_channel_error_plots_directory / f"rollout_{i}_channel_errors.html")
            rollout_channel_error_plot.write_image(rollouts_channel_error_plots_directory / f"rollout_{i}_channel_errors.png", width=1920, height=1080)
            
            # Also compute the channel metrics for the open loop rollout.
            rollout_channel_metrics = compute_open_loop_rollout_channel_metrics(rollout_sample_channel_errors)
            # Write the channel metrics to a CSV file in the open loop rollout.
            rollout_channel_metrics.to_csv(rollouts_channel_metrics_directory / f"rollout_{i}_channel_metrics.csv")

            # Also, write each rollouts ground truth and predicted next states
            # to CSV files.
            gt_next_states_df.to_csv(rollouts_states_directory / f"rollout_{i}_ground_truth_next_states.csv", index=False)
            pred_next_states_df.to_csv(rollouts_states_directory / f"rollout_{i}_predicted_next_states.csv", index=False)

            # Also, similar to the channel error plots, also just create channel
            # plots. Do this for both the ground truth and predicted state values.
            # rollout_channel_value_plot =
            # create_channel_value_plots(gt_next_states_df)
            rollout_channel_values_plot = create_channel_value_comparison_plots(gt_next_states_df, pred_next_states_df)
            # Write the channel value plot to an HTML file in the open loop
            # rollout output directory.
            rollout_channel_values_plot.write_html(rollouts_channel_value_plots_directory / f"rollout_{i}_channel_values.html")
            rollout_channel_values_plot.write_image(rollouts_channel_value_plots_directory / f"rollout_{i}_channel_values.png", width=1920, height=1080)

            pbar.update(1)

    # Create plot with all the open loop rollouts stacked on top of each other.
    # I.e., really just want to visualize the position components of the open
    # loop rollouts around the track.

    # Stack up all the ground truth and predicted dataframes from the rollouts.
    combined_gt_next_states_df = pd.concat([rollout[1] for rollout in rollouts], ignore_index=True)
    combined_pred_next_states_df = pd.concat([rollout[0] for rollout in rollouts], ignore_index=True)
    # Create a visualization of the ground truth and predicted next states.
    position_plot_fig = create_2d_state_plot(combined_gt_next_states_df, 
                                             f"gt_{NORMALIZED_DISCRETE_CSV_COLUMNS[5]}", 
                                             f"gt_{NORMALIZED_DISCRETE_CSV_COLUMNS[6]}", 
                                             combined_pred_next_states_df, 
                                             f"pred_{NORMALIZED_DISCRETE_CSV_COLUMNS[5]}",
                                             f"pred_{NORMALIZED_DISCRETE_CSV_COLUMNS[6]}", 
                                             "Combined Open Loop Rollouts", 
                                             "px_N", 
                                             "py_N")
    # Write the visualization to an HTML file in the open loop rollout
    # directory.
    position_plot_fig.write_html(open_loop_rollout_output_directory / "combined_rollouts_position_plot.html")