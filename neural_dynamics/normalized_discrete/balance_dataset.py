
import argparse
from pathlib import Path

import sys
import time
from typing import List, Tuple

from matplotlib import pyplot as plt
import pandas as pd
import numpy as np

sys.path.insert(0, "../../")
from neural_dynamics.normalized_discrete.dataset import NORMALIZED_DISCRETE_CSV_COLUMNS, load_dataset_files, load_sample_index

def compute_sample_differences(samples: List[Tuple],
                               csv_dataframes: List[pd.DataFrame]) -> List[np.ndarray]:
    """Given a list of sample Tuples, this function will return a parallel list
    of the differences between the first and last rows of each sample.

    Args:
        samples (List[Tuple]): List of sample Tuples from a CSV dataset's sample index.
        csv_dataframes (List[pd.DataFrame]): List of CSV dataframes loaded from
        the CSV dataset's directory.

    Returns:
        List[np.ndarray]: List of difference vectors as numpy arrays.
    """
    differences = []
    for sample in samples:
        dataframe_index, sample_indices = sample
        sample_start_index, sample_end_index = sample_indices
        first_row = csv_dataframes[dataframe_index].iloc[sample_start_index].to_numpy()
        last_row = csv_dataframes[dataframe_index].iloc[sample_end_index].to_numpy()
        difference = np.subtract(last_row, first_row)
        differences.append(difference)
    return differences

# Function that, 
def cluster_sample_differences(sample_differences: List[np.ndarray]):
    pass

# Similar function, but only for clustering based on the position deltas, rather
# than the entire state. For now, this is just for my own understanding and to
# make sure DBSCAN is working in the way that I expect.
# def cluster_position_differences(position_differences: np.ndarray):

#     clustering = DBSCAN(eps=0.2, min_samples=2).fit(position_differences)
#     return clustering.labels_

# Maybe instead of going all the way to clustering, we should really just
# constrain our problem to discretizing or BINNING the vectors that we have--as
# we don't really need to cluster! We just need to know how many samples fall
# into each bin, approximately. This is much more predictable.
def discretize_position_differences(position_differences: np.ndarray):

    pass

# TODO: This command line interface should REALLY instead create an instance of
# the dataset wrapper class and call a method that we create in the class that
# balances the dataset USING THE FUNCTIONS DEFINED IN THIS MODULE. So it's kind
# of a circular structure, which is a admittedly a bit weird--but I think it's
# the most "correct" way to separate functionality and preserve the interface
# provided by the dataset wrapper.
if __name__ == "__main__":

    # Set up argparser and add argument for the filepath to the sample index
    # file.
    parser = argparse.ArgumentParser(description="Script to balance a Normalized Discrete Dataset by adding/removing \
                                                  samples from the dataset's sample index.")
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



    # For now, just cluster the position differences
    position_differences = np.array([difference[5:7] for difference in sample_differences])
    # clustering = DBSCAN(eps=0.005, min_samples=10).fit(position_differences)
    # labels_column = clustering.labels_

    # TEST HISTOGRAM BASED GROUPING / BINNING.
    x_bin_edges = np.histogram_bin_edges(position_differences[:,0], bins="sturges")
    y_bin_edges = np.histogram_bin_edges(position_differences[:,1], bins="sturges")
    test_histogram, _, _ = np.histogram2d(position_differences[:,0], position_differences[:,1], bins=(x_bin_edges, y_bin_edges))
    test_histogram = test_histogram.T # For visualization purposes I guess.
    # print(f"Test histogram: {test_histogram}")
    plt.imshow(test_histogram, origin='lower', extent=[x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel('Delta X')
    plt.ylabel('Delta Y')
    plt.title('Position Difference Histogram')
    plt.savefig("position_difference_histogram.png")

    # Also, just for fun, create separate 1D histograms showing the distribution
    # of samples for both delta X and delta Y. Should really generate these for
    # every dimension of the input data anyway, actually.
    x_histogram, _ = np.histogram(position_differences[:,0], bins=x_bin_edges)
    y_histogram, _ = np.histogram(position_differences[:,1], bins=y_bin_edges)
    # Create separate 1D histograms for delta X and delta Y
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(position_differences[:, 0], bins=x_bin_edges)
    plt.xlabel('Delta X')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delta X')

    plt.subplot(2, 1, 2)
    plt.hist(position_differences[:, 1], bins=y_bin_edges)
    plt.xlabel('Delta Y')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delta Y')

    plt.tight_layout()
    plt.savefig("1d_histograms.png")
    
    # clustered_position_differences_dataframe = pd.DataFrame(data=position_differences, columns=["Change in X", "Change in Y"])
    # df = clustered_position_differences_dataframe
    # normalized_df=(clustered_position_differences_dataframe-clustered_position_differences_dataframe.mean())/clustered_position_differences_dataframe.std()
    # # normalized_df=(df-df.min())/(df.max()-df.min())

    # # I think next, I want to try matlab's discretization function--it may
    # # allow you to bin/bucket vectors. However, I would either need to
    # # standardize (z-score) the data ahead of time, or make sure that it is
    # # creating bins for EACH DIMENSION of the input vector.
    
    # clustering = DBSCAN(eps=0.02, min_samples=6).fit(normalized_df)
    # labels_column = clustering.labels_
    # normalized_df.insert(2, "Cluster", labels_column)
    # normalized_df["Cluster"] = normalized_df["Cluster"].astype(str)
    # # position_diffs_fig = px.scatter(data_frame=position_differences_dataframe, x="Change in X", y="Change in Y")
    # clustered_positions_diff_fig = px.scatter(data_frame=normalized_df, x="Change in X", y="Change in Y", color="Cluster")
    # # position_diffs_fig.write_html(Path().cwd()/"position_diffs.html")
    # clustered_positions_diff_fig.write_html(Path().cwd()/"clustered_position_diffs.html")
    # # # Construct a dataframe from the list of difference arrays and the columns
    # # for this dataset.
    # # difference_columns = [f"Change in {column}" for column in NORMALIZED_DISCRETE_CSV_COLUMNS]
    # # sample_differences_dataframe = pd.DataFrame(data=sample_differences, columns=difference_columns)