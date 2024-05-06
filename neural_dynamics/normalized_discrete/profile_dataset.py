"""Functions for profiling different portions of the dataset and identifying
portions of a dataset with certain characteristics. These functions are mainly
useful for finding particular parts of a dataset that we particularly want to
test a model on, for example. Additionally, this module will contain functions
to extract important information/details from a dataset.
"""

import argparse
import pandas as pd
from pathlib import Path
import sys

sys.path.insert(0, "../../")
from neural_dynamics.normalized_discrete.dataset import NormalizedDiscreteCSVDataset, NORMALIZED_DISCRETE_CSV_COLUMNS


# TODO: Create function that identifies laps in the provided dataset.

# TODO: Create function that breaks through the dataset interface and returns
# all the underlying CSV dataframes stacked up as one. This will break if the
# underlying dataset representation changes, but it'll be useful for now.
def get_dataset_dataframes(dataset: NormalizedDiscreteCSVDataset) -> pd.DataFrame:
    """Extracts all the underlying CSV dataframes from the provided dataset
    and stacks them up into one big dataframe.

    Args:
        dataset (NormalizedDiscreteCSVDataset): The dataset to extract the
            dataframes from.

    Returns:
        pd.DataFrame: The stacked dataframe containing all the data from the
            underlying dataset.
    """
    csv_dataframes = dataset._NormalizedDiscreteCSVDataset__csv_dataframes
    return pd.concat(csv_dataframes, ignore_index=True)

def compute_dataset_channel_metrics(dataset_df: pd.DataFrame) -> pd.DataFrame:
    """Computes the mean and standard deviation of each channel in the provided
    dataset dataframe.

    Args:
        dataset_df (pd.DataFrame): The dataframe to compute the channel metrics
            for.

    Returns:
        pd.DataFrame: The dataframe containing the mean and standard deviation
            of each channel in the provided dataset dataframe.
    """
    # channel_metrics = pd.DataFrame(index=NORMALIZED_DISCRETE_CSV_COLUMNS)
    # channel_metrics["mean"] = dataset_df.mean(axis=0)
    # channel_metrics["stdev"] = dataset_df.std(axis=0)
    # return channel_metrics
    channel_means = dataset_df.mean(axis=0)
    channel_stdevs = dataset_df.std(axis=0)
    channel_medians = dataset_df.median(axis=0)
    channel_modes = dataset_df.mode(axis=0).iloc[0]
    channel_metrics = pd.DataFrame([channel_means, channel_stdevs, channel_medians, channel_modes], index=["mean", "stdev", "median", "mode"])
    channel_metrics.columns = NORMALIZED_DISCRETE_CSV_COLUMNS
    return channel_metrics

if __name__ == "__main__":

    # Set up argparser and add argument for the filepath to the sample index
    # file.
    parser = argparse.ArgumentParser(description="Script to visualize the state-space of the specified dataset using \
                                                  parallel coordinate plots of the state-differences between samples.")
    parser.add_argument("sample_index_filepath", type=str, help="The path to the dataset's sample index JSON file.")
    parser.add_argument("-o", "--output_directory", type=str, help="The directory to save the visualization to. Defaults to \
                                                                    the CSV dataset directory referenced by the sample index file.")
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
    pass

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
    # Otherwise, if no output directory was provided, just use the parent
    # directory of the sample index file, as this is most likely the directory
    # containing the dataset's data.
    else:
        output_directory = sample_index_filepath.parent
    
    ######################## INITIALIZE DATASET ########################

    # If the dataset file does exist, hand it off to the dataset wrapper's
    # constructor to try and initialize itself from it.
    try:
        dataset = NormalizedDiscreteCSVDataset(sample_index_filepath)
    except Exception as exc:
        print(f"Error initializing the dataset wrapper from the provided sample index file {sample_index_filepath}.")
        raise exc
    
    
    ######################## DATASET PROFILING FUNCTIONS  ########################
    
    
    dataset_df = get_dataset_dataframes(dataset)

    print("Computing channel metrics.")
    # Compute dataset metrics.
    dataset_metrics = compute_dataset_channel_metrics(dataset_df)

    # Write the dataset metrics to a CSV file in the output directory.
    dataset_metrics_filepath = output_directory / "dataset_metrics.csv"
    dataset_metrics.to_csv(dataset_metrics_filepath)
    print(f"Dataset metrics computed and written to {dataset_metrics_filepath}.")