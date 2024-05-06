
import argparse
from pathlib import Path
import time
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px

import sys

import numpy as np
sys.path.insert(0, "../../")
from neural_dynamics.normalized_discrete.dataset import NORMALIZED_DISCRETE_CSV_COLUMNS, load_dataset_files, load_sample_index
from neural_dynamics.normalized_discrete.balance_dataset import compute_sample_differences


# Need to write the visualization function that uses plotly express to create
# parallel plots of the state-differences across all the samples found in the
# sample index.


def create_sample_differences_with_error_plot(sample_differences_dataframe: pd.DataFrame) -> go.Figure:
    # Same as the function below, but where the provided dataframe includes an
    # additional column that contains the squared error between the model's
    # predicted next state and the actual next state == the last (second) row of
    # each sample.

    # We'll make the same parallel coordinate plot, but with the error column
    # determining the color of the lines. This should hopefully give us a more
    # clear picture of what kind of samples from a given CSV dataset the model
    # is struggling with.

    # Again, as always, for any of this to make sense, have to make sure the
    # model is trained on a dataset of the same frequency and data from the same
    # car as the data it is being evaluated on.

    # Maybe just as a part of the evaluation step / as part of the Lightning
    # module, we simply include a function that uses the model to predict the
    # next state using the initial state of each individual sample in the
    # dataset.

    # Could also create a function to do some kind of state density
    # visualization. Using one of the clustering algorithms I looked into--group
    # each state vector / sample into one of those clusters--and then that will
    # serve as the color. The whole purpose of this visualization would simply
    # to be able to qualitatively tell if the dataset is imbalanced--if samples
    # of some type/range occur more frequently than others.

    pass

# Function to create a parallel coordinate plot of the provided sample
# differences, using the density group/cluster number to dictate color. I.e.,
# each group has some density/frequency == the number of samples from the
# dataset that are all around that same state. Maybe we'll actually swap out
# each sample's cluster number for that cluster number's number of samples, or
# something like that.
def create_sample_diff_density_plot(sample_differences_dataframe: pd.DataFrame) -> go.Figure:
    pass

# Function that also creates parallel line plots of the sample differences, but
# this time using the prediction loss/error from a model on that sample to
# determine the color. This should help us qualitatively understand which
# samples / cluster of samples the model is struggling with the most--which may
# in turn help us to obtain more samples like this OR just understand how
# different changes in the dataset affect model performance across the dataset.
def create_sample_model_error_plot(samples_diffrences_dataframe: pd.DataFrame) -> go.Figure:
    pass

# Function that will generate a figure containing a histogram for each dimension
# of the provided sample differences as a way of visualizing the distribution of
# the state, outside of the parallel coordinate plot. May also be helpful to
# create another function that creates 2D and/or 3D histograms to provide more
# insight into the relationships between different state variables.
def create_dataset_histograms(sample_differences_dataframe: pd.DataFrame) -> go.Figure:
    pass

# TODO: Could eventually use the histogram dataset distribution to guide an
# offline trajectory generation / optimization problem in generating feasible
# trajectories that will result in the car collecting samples that occupy some
# of the more sparse areas of the state space. I.e., use it to guide us in
# creating trajectories that will get the car to experience more
# scenarios and explore more of the state space.

def create_sample_differences_plot(sample_differences_dataframe: pd.DataFrame) -> go.Figure:
    """Create a parallel coordinate plot from a dataframe with the sample
    differences for each channel as its rows.

    Args:
        sample_differences_dataframe (pd.DataFrame): A dataframe with its rows
        being difference the difference between each sample's start and end
        states. 

    Returns:
        go.Figure: Plotly graph object.
    """

    fig = px.parallel_coordinates(data_frame=sample_differences_dataframe,
                                  dimensions=sample_differences_dataframe.columns,
                                  title="Dataset State-Space Visualization",)
    return fig


# TODO: These dataset visualization functions really shouldn't go below the
# interface of the dataset wrapper--as there is no need to. Instead, they should
# be a consumer of the dataset wrapper's interface--and should be able to obtain
# all the sample data necessary to build up the structures needed for analysis
# and visualization.

if __name__ == "__main__":

    # Set up argparser and add argument for the filepath to the sample index
    # file.
    parser = argparse.ArgumentParser(description="Script to visualize the state-space of the specified dataset using \
                                                  parallel coordinate plots of the state-differences between samples.")
    parser.add_argument("sample_index_filepath", type=str, help="The path to the dataset's sample index JSON file.")
    parser.add_argument("-o", "--output_directory", type=str, help="The directory to save the visualization to. Defaults to \
                                                                    the CSV dataset directory referenced by the sample index file.")
    args = parser.parse_args()

    # First, validate the sample index filepath argument.
    try:
        sample_index_filepath = Path(args.sample_index_filepath)
    except Exception as exc:
        print(f"Invalid sample index filepath {str(args.sample_index_filepath)} provided.")
        raise exc
    if not sample_index_filepath.exists():
        print(f"The provided sample index filepath {sample_index_filepath} does not exist.")
        raise FileNotFoundError
    
    # If the file does exist, attempt to load the sample index. Potentially need
    # the CSV dataset directory for saving the visualization.
    print(f"Opening the dataset's sample index file {str(sample_index_filepath)}")
    start = time.perf_counter()
    sample_index = load_sample_index(sample_index_filepath)
    end = time.perf_counter()
    print(f"Opened the dataset's sample index file in {end - start:.4f}s.")

    # Validate the provided output path.
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
    # Otherwise, if no output directory was provided, just save the sample index
    #file in the CSV dataset's root directory.
    else:
        output_directory = Path(sample_index["csv_dataset_root_directory"])

    # Load the CSV files listed in the dataset's sample index we just loaded.
    csv_filepaths = [Path(csv_filepath) for csv_filepath in sample_index["csv_filepaths"]]

    # Open up all the CSV files and load them into pandas dataframes.
    print(f"Opening the {len(csv_filepaths)} discovered episode files.")
    start = time.perf_counter()
    try:
        csv_dataframes = load_dataset_files(csv_filepaths=csv_filepaths,
                                        columns=NORMALIZED_DISCRETE_CSV_COLUMNS)
    except Exception as exc:
        print(f"Failed to load CSV files from CSV dataset directory.")
        raise exc
    end = time.perf_counter()
    print(f"Successfully opened {len(csv_dataframes)} episode files in {end - start:.4f}s.")

    # Compute the sample differences that we want to visualize.
    print(f"Computing the differences between the start and end states of each sample.")
    start = time.perf_counter()
    sample_differences = compute_sample_differences(samples=sample_index["samples"],
                                                    csv_dataframes=csv_dataframes)
    end = time.perf_counter()
    print(f"Computed the state differences for {len(sample_differences)} samples in {end - start:.4f}s.")
    
    # Construct a dataframe from the list of difference arrays and the columns
    # for this dataset.
    difference_columns = [f"Change in {column}" for column in NORMALIZED_DISCRETE_CSV_COLUMNS]
    sample_differences_dataframe = pd.DataFrame(data=sample_differences, columns=difference_columns)
    
    # Create the parallel coordinate plot figure.
    print(f"Creating the parallel coordinate plot for the dataset's state-space.")
    start = time.perf_counter()
    fig = create_sample_differences_plot(sample_differences_dataframe)
    end = time.perf_counter()
    print(f"Created the state space visualization plot in {end - start:.4f}s.")

    # Attempt to save the figure to disk in the output directory.
    print(f"Saving the state-space visualization to {output_directory}.")
    start = time.perf_counter()
    fig.write_html(output_directory/"dataset_visualization.html")
    end = time.perf_counter()
    print(f"Saved the state-space visualization in {end - start:.4f}s.")

    # TODO: Create 1D histograms for each dimension of the dataset to visualize.

    # TODO: Create an N-D histogram for the dataset to visualize the density of
    # the sample differences in the dataset. So, very similar to the parallel
    # coordinates plot above, but this time where color of each sample's line is
    # dictated by the number of other samples that are similar to that sample.

    # TODO: Create a plot that uses the model's prediction error on each
    # sample--but this may also belong in a separate script specific to
    # evaluating each new model architecture--as this is not directly a property
    # of the dataset!