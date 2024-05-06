
import argparse
from collections import deque
from enum import Enum
import json
from pathlib import Path
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
from torch.utils.data import Dataset
import pandas as pd

# TODO: Should probably also include those scripts to create the RAW CSV
# datasets in a separate subpackage within the neuralmpc package--as you are
# inevitably going to need those practically speaking. Not that you couldn't
# generate CSVs without those scripts--you'd just end up writing your own.


# Need a function to load a create sample index and balance the dataset. This
# function should receive a list of sample dictionaries (the same one that comes
# from the JSON file) and spits out a "balanced" list of sampled
# dictionaries--where the returned list may not contain all the samples that
# were passed in--as some may have been pruned. This is just one convenient,
# fast way to balance the dataset. We could also implement a more sophisticated
# balancing script that actually creates new, synthetic samples (oversampling)
# in the underlying CSV data--but not going down that road yet. We have lots of
# samples, undersampling is what we have time for.

# Need a function to produce a visualization of the dataset balance (provided
# some output directory). For this dataset, the visualization is of the state
# space changes--but other visualizations could be added in the future as well
# if you want. Just write another function and add it to the CLI somehow.

# Need a separate command in the CLI to generate the sample index, to balance,
# and to do any other preprocessing steps we want to add to this module.


def find_csv_dataset_files(csv_dataset_path: Path) -> List[Path]:
    """Searches through the provided csv dataset directory (and any
    subdirectories) for CSV files.

    Args:
        dataset_directory_path (Path): Path to the CSV dataset's parent/root
        directory.

    Returns:
        List[Path]: List of all the CSV files found.
    """
    # NOTE: Pathlib added the "walk" function for Python 3.12. This would
    # simplify a good bit of this function. For now, this "by-hand"
    # breadth-first search works too (that's what walk is doing under the
    # hood--but probably with more checks/protections).
    csv_filepaths = []
    directory_queue = deque([csv_dataset_path])
    while len(directory_queue) > 0:
        current_directory = directory_queue.popleft()
        for entry in current_directory.iterdir():
            # If the entry is a file, check to see if it's a CSV file. Add
            # it to the list of CSV files if so.
            if entry.is_file():
                if entry.suffix == ".csv":
                    csv_filepaths.append(entry)
            # Otherwise, the entry is a directory--add it to the directory
            # queue to be searched.
            else:
                directory_queue.append(entry)

    # Use pathlib's "walk" function to do a breadth-first search on the
    # provided directory. Will step us through each level (top-down),
    # allowing us to find the filepaths of CSVs at each level of the
    # directory tree.
    # https://docs.python.org/3/library/pathlib.html#pathlib.Path.walk 
    
    # for directory, _, files_in_directory in dataset_directory_path.walk():
    #     # For each of the files in the current directory, find those with
    #     # the ".csv" extension.
    #     for filename in files_in_directory:
    #         if Path(filename).suffix == ".csv":
    #             csv_filepaths.append(directory/filename)
    
    return csv_filepaths

def load_dataset_files(csv_filepaths: List[Path],
                       columns: List[str]) -> Tuple[List[pd.DataFrame], List[Path]]:
    """Attempt to load each of the CSV files in as a pandas DataFrame.

    Args:
        csv_filepaths (List[Path]): filepaths (List[Path]): List of filepaths to
        each CSV file.
        columns (List[str]): List of column names to load from each of CSV
        files.

    Raises:
        exc: Exception thrown when a pandas DataFrame cannot be created from one
        of the provided CSV files.

    Returns:
        Tuple[List[pd.DataFrame], List[Path]]: Returns a list of the dataframes
        that were successfully created. 
    """
    dataframes = []
    for filepath in csv_filepaths:
        try:
            # UPDATE: Had to add the [columns] to the end to make sure the
            # columns are selected in the desired order that the columns are
            # specified in! This is crucial!
            file_dataframe = pd.read_csv(filepath, usecols=columns)[columns]
        except Exception as exc:
            print(f"Failed to create pandas dataframe from file {file_dataframe}")
            raise exc
        else:
            dataframes.append(file_dataframe)
    return dataframes

def identify_samples_in_files(dataframes: List[pd.DataFrame],
                              sample_length: Optional[int] = 2) -> List[Tuple]:
    """Generates a list of samples across each of the provided episodes
    (where each episode corresponds with a dataframe from a CSV).

    Args:
        dataframes (List[pd.DataFrame]): The list of episode dataframes.
        sample_length (int): The number of rows each sample will contain.

    Returns:
        List[Tuple]: Each entry in sample_indices will be a tuple, where the
        first entry will be an index to the dataframe that it comes in
        episode_dataframes, and the second entry will be another
        tuple containing the indices of the rows within that dataframe that
        define this sample. For example: (df_idx, (row3, row4))
    """
    samples = []
    for episode_dataframe_index, episode_dataframe in enumerate(dataframes):
        # Identify the samples in the current episode dataframe.

        # Get the start and end index within the dataframe.
        episode_start_index = 0
        episode_end_index = episode_dataframe.shape[0] - 1
        episode_length = episode_dataframe.shape[0]
        # Slide a "sample-sized" window over the messages of the episode.
        # The indices of this window == the indices of a new sample.
        # First need to compute the number of samples we can get from this
        # episode based on its length and the sample length. This is similar
        # to the calculation done for output size of convolution.
        num_episode_samples = max(episode_length - sample_length + 1, 0)
        # Then, compute as many sample's indices as determined above.
        for sample_offset in range(0, num_episode_samples):
            sample_start_index = episode_start_index + sample_offset
            sample_end_index = sample_start_index + sample_length - 1
            sample_indices = (sample_start_index, sample_end_index)
            samples.append((episode_dataframe_index, sample_indices))
    return samples



# Need a function for generating the sample index.
# This function should ultimately spit out (into the provided CSV directory if
# successful) a JSON file containing a list of all the sample dictionaries. The
# name of this file should be the name of the RAW CSV dataset directory name
# (which hopefully includes information like interpolation / data frequency and
# other useful information) + the name of the dataset or model class +
# sample_index.json (or something to that effect). This function should really
# call another function inside that actually generates the samples from the
# underlying files--and then returns the sample list as a List[Dict] or
# something like that. Then, this function calls that function and writes the
# output to disk back in the dataset directory. I.e., the function that is
# actually generating the samples should be abstracted out, kinda like how it is
# in the original dataset wrapper class.

# Hardcoded sample length for the Normalized Discrete dataset.
NORMALIZED_DISCRETE_SAMPLE_LENGTH = 2
# Hardcoded columns to be extracted from each CSV when loaded in as a pandas
# dataframe. Again, in principle, the CSV (ideally) shouldn't contain any
# columns that we don't need for training our NormalizedDiscrete models--but we
# don't quite have that worked out yet. So for now, we just select which columns
# we want upon loading in the data.
# NORMALIZED_DISCRETE_CSV_COLUMNS = ["u/u_a",         
#                                    "u/u_steer",
#                                    "v/v_long",
#                                    "v/v_tran",
#                                    "w/w_psi",
#                                    "x/x",
#                                    "x/y",
#                                    "e/psi"
#                                    ]
NORMALIZED_DISCRETE_CSV_COLUMNS = ["acceleration-(m/s^2)",         
                                   "steering-angle-(rad)",
                                   "x-velocity-(m/s)",
                                   "y-velocity-(m/s)",
                                   "z-angular-velocity-(rad/s)",
                                   "x-position-(m)",
                                   "y-position-(m)",
                                   "yaw-(rad)"
                                   ]

# TODO: It would be nice to have some kind of column name remapping. Maybe
# that's not this module's responsibility, though.

class SampleColumns(Enum):
    """Enum class for the indices of the columns in the Normalized Discrete
    dataset.
    """
    U_A = 0
    U_STEER = 1
    V_X_B = 2
    V_Y_B = 3
    W_B = 4
    PX_N = 5
    PY_N = 6
    PSI_B_N = 7

def create_sample_index(csv_dataset_path: Path,
                        name: Optional[str] = None) -> Dict:
    """Generate a sample index for the provided CSV dataset. The sample INDEX is
    simply a dictionary that tells us where all the samples are in the dataset.
    This is just like an index in a book that tells you what page each
    section/chapter starts at, for example.

    Args:
        csv_dataset_path (Path): Filepath of the CSV dataset's parent/root
        directory.
        name (Optional[str]): Name of the dataset to be placed in its sample
        index. Will be the name of the CSV dataset's root directory if not
        specified.

    Returns:
        Dict: The sample index for the CSV dataset. Contains a list of all the
        CSV files found in the dataset directory, as well as a list of samples
        identified throughout the CSV files.
    """

    # Create new sample index dictionary.
    sample_index = {}

    # Save the path to the CSV dataset directory in the sample index.
    sample_index["csv_dataset_root_directory"] = str(csv_dataset_path)

    # Set the sample index's name.
    if name is not None:
        sample_index["name"] = name
    # If no name is provided, use the name of the CSV dataset directory.
    else:
        sample_index["name"] = csv_dataset_path.parts[-1]

    # Find all the CSV files in the provided CSV dataset directory.
    print(f"Beginning search for episode CSVs in provided directory {csv_dataset_path}.")
    start = time.perf_counter()
    csv_filepaths = find_csv_dataset_files(csv_dataset_path=csv_dataset_path)
    end = time.perf_counter()
    print(f"Discovered {len(csv_filepaths)} files in {end - start:.4f}s.")

    # Store the filepaths of the found CSVs in the sample index dictionary. Have
    # to make sure to convert these to strings first so they can be turned to
    # JSON later.
    sample_index["csv_filepaths"] = [str(csv_filepath) for csv_filepath in csv_filepaths]

    # Open up all the CSV files and load them into pandas dataframes.
    print(f"Opening the {len(csv_filepaths)} discovered episode files.")
    start = time.perf_counter()
    try:
        dataframes = load_dataset_files(csv_filepaths=csv_filepaths,
                                        columns=NORMALIZED_DISCRETE_CSV_COLUMNS)
    except Exception as exc:
        print(f"Failed to load CSV files from CSV dataset directory.")
        raise exc
    end = time.perf_counter()
    print(f"Successfully opened {len(dataframes)} episode files in {end - start:.4f}s.")
    
    # Identify samples in the opened dataframes.
    print(f"Beginning sample identification from {len(dataframes)} episode files.")
    start = time.perf_counter()
    samples = identify_samples_in_files(dataframes=dataframes, sample_length=NORMALIZED_DISCRETE_SAMPLE_LENGTH)
    end = time.perf_counter()
    print(f"Identified {len(samples)} samples in {end - start:.4f}s.")

    # Store the list of samples in the sample index.
    sample_index["samples"] = samples
    
    return sample_index

# TODO: Actually, I think there should be an option to not provide a prefix--as
# if it's being created within a CSV dataset's directory, you especially don't
# need to put its whole name ahead of it.
def save_sample_index(sample_index: Dict,
                      output_directory: Path,
                      filename_prefix: Optional[str] = None) -> Path:
    """Write the provided sample index dictionary to disk as JSON file in the
    provided output directory.

    Args:
        sample_index (Dict): Sample index dictionary.
        output_directory (Optional[Path], optional): Directory that sample index
        will be written to. Defaults to None.
        filename_prefix (Optional[str], optional): Prefix that will be inserted
        in front of "_sample_index" when creating the new sample index JSON
        file. If None, the prefix will be the name of the CSV dataset's root
        directory.
    
    Returns: Path: Filepath to the saved sample index JSON file.
    """
    # Attempt to create the output directory if it doesn't already exist.
    try:
        if not output_directory.exists():
            output_directory.mkdir(parents=True, exist_ok=True)
    except Exception as exc:
        print(f"Failed to write sample index to disk. The provided output directory did not exist and could not be created.")
        raise exc
    # If the output directory does exist / was created, write the sample index
    # to disk. Before we do that, decide on the output file's name.
    if filename_prefix is None:
        filename_prefix = str(Path(sample_index["csv_dataset_root_directory"]).parts[-1])
    output_filepath = output_directory/f"{filename_prefix}_sample_index.json"
    try:
        with output_filepath.open("w") as sample_index_file:
            sample_index_file.write(json.dumps(sample_index, indent=0))
    except Exception as exc:
        print(f"Failed to write sample index JSON file {output_filepath} to output directory {output_directory}")
        raise exc
    return output_filepath

def load_sample_index(sample_index_path: Path) -> Dict:
    """Load a sample index JSON file from disk.


    Args:
        sample_index_path (Path): Filepath to the sample index JSON file.

    Raises:
        exc: Exception thrown if the sample index JSON file cannot be
        loaded/parsed.

    Returns:
        Dict: The loaded sample index dictionary.
    """
    try:
        with sample_index_path.open("r") as sample_index_file:
            sample_index = json.load(sample_index_file)
    except Exception as exc:
        print(f"Failed to load sample index from file {sample_index_path}")
        raise exc
    return sample_index

class NormalizedDiscreteCSVDataset(Dataset):
    """Dataset Wrapper for training and evaluating Normalized Discrete models on
    CSV datasets.
    """

    def __init__(self, sample_index_path: Path):
        """Initialize the NormalizedDiscreteCSVDataset.

        Args:
            sample_index_path (Path): Filepath to the sample index JSON file.
        """
        # Load the sample index from the provided filepath.
        print(f"Loading sample index from file {sample_index_path}")
        try:
            sample_index = load_sample_index(sample_index_path)
        except Exception as exc:
            print(f"Failed to load sample index from file {sample_index_path}")
            raise exc
        self.__sample_index = sample_index

        # Extract the list of samples from the sample index.
        self.__samples = self.__sample_index["samples"]

        # Load each CSV in as a dataframe from the sample index.
        print(f"Loading CSV files specifed in the sample index.")
        csv_filepaths = [Path(csv_filepath) for csv_filepath in self.__sample_index["csv_filepaths"]]
        try:
            csv_dataframes = load_dataset_files(csv_filepaths=csv_filepaths,
                                                columns=NORMALIZED_DISCRETE_CSV_COLUMNS)
        except Exception as exc:
            print(f"Failed to load the CSVs specified in the sample index.")
            raise exc
        # Store the episode dataframes in the dataset object.
        self.__csv_dataframes = csv_dataframes

        # Grab the dataset's name from the sample index file.
        self.__name = self.__sample_index["name"]

    def __len__(self):
        return len(self.__samples)

    def __getitem__(self, index: int) -> np.ndarray:
        """Returns the sample at the provided index in the dataset.

        Args:
            index (int): Offset of sample in the dataset.

        Returns:
            np.ndarray: he sample at the provided index as a numpy array.
        """
        # Grab the sample at the provided index and split it into its datraframe
        # index and the indices of the rows that define the sample.
        dataframe_index, sample_indices = self.__samples[index]
        sample_start_index, sample_end_index = sample_indices
        # Select the dataframe that the sample's rows are from.
        sample_dataframe: pd.DataFrame = self.__csv_dataframes[dataframe_index]
        # Return the sample from the dataframe as a numpy array.
        return sample_dataframe.iloc[sample_start_index:sample_end_index+1].to_numpy()
    
    def get_name(self) -> str:
        """Return the name of the dataset.

        Returns:
            str: Name of the dataset found in the dataset's sample index file.
        """
        return self.__name

    @classmethod
    def from_csv_dataset(cls,
                         csv_dataset_path: Path):
        """Create a new NormalizedDiscreteCSVDataset from the provided CSV
        dataset directory.

        Args:
            csv_dataset_path (Path): Filepath to the CSV dataset directory.

        Returns:
            NormalizedDiscreteCSVDataset: A new instance of the
            NormalizedDiscreteCSVDataset class.
        """
        # First, check the directory for any existing sample index files. Any
        # sample index file should have "sample_index.json" in the name.
        # If it finds multiple, it will just try to use the first that it finds.
        # If it doesn't find any, it will attempt to generate a new sample index
        # file and save it in that directory.
        print(f"Searching for existing sample index files in the provided CSV dataset directory {csv_dataset_path}.")
        existing_sample_index_found = False
        for entry in csv_dataset_path.iterdir():
            if entry.is_file():
                if "sample_index.json" in entry:
                    sample_index_filepath = entry
                    existing_sample_index_found = True
                    break
            
        # If an existing sample index was found, try to load it from disk and
        # initialize the dataset wrapper class from it.
        if existing_sample_index_found:
            return cls(sample_index_path=sample_index_filepath)

        # Otherwise, if no existing sample index file was found, generate a new
        # one from the CSV files found in the dataset.
        else:
            # Generate the sample index for the provided CSV dataset.
            try:
                sample_index = create_sample_index(csv_dataset_path)
            except Exception as exc:
                print(f"Failed to create sample index for CSV dataset at {csv_dataset_path}")
                raise exc
            
            # Save the sample index to disk.
            try:
                sampled_index_filepath = save_sample_index(sample_index=sample_index,
                                                           output_directory=csv_dataset_path)
            except Exception as exc:
                print(f"Failed to save sample index to disk.")
                raise exc
            # Return a new instance of the NormalizedDiscreteCSVDataset class.
            return cls(sample_index_path=sampled_index_filepath)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Normalized Discrete Dataset Utility Script. \
                                     This CLI is used to generate a sample index for the provided \
                                     CSV dataset.")

    # Create a group of arguments that specify the command to be executed.
    # command_group = parser.add_mutually_exclusive_group()

    # Create an argument for balancing the dataset.
    

    # Create an argument for visualizing the dataset state space.

    # Create an argument for generating a sample index for a CSV dataset.
    parser.add_argument("csv_dataset_directory", help="CSV dataset directory to generate a sample index for.")
    parser.add_argument("-o", "--output-directory", help="Directory to save the sample index JSON file to. \
                                                          By default, it will just be placed in the CSV dataset directory \
                                                          that it is being created for.",
                                                    default=None)

    # Grab the arguments from the argparser.
    args = parser.parse_args()

    # First, validate the CSV dataset directory path.
    if args.csv_dataset_directory is not None:
        try:
            csv_dataset_directory = Path(args.csv_dataset_directory)
        except Exception as exc:
            print(f"Provided CSV dataset directory {str(args.csv_dataset_directory)} is not a valid path.")
            raise exc
        # If valid, check if the path is truly a directory and if it exists.
        if not csv_dataset_directory.exists():
            print(f"Provided CSV dataset directory {csv_dataset_directory} does not exist.")
            raise FileNotFoundError
        if not csv_dataset_directory.is_dir():
            print(f"Provided CSV dataset directory {csv_dataset_directory} is not a directory.")
            raise NotADirectoryError
    else:
        raise ValueError("No CSV dataset directory provided.")
    
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
    # Otherwise, if no output directory was provided, just save the sample index
    #file in the CSV dataset's root directory.
    else:
        output_directory = csv_dataset_directory

    # Create a sample index from the provided CSV dataset directory.
    print(f"Creating new sample index for CSV dataset at {csv_dataset_directory}")
    try:
        sample_index = create_sample_index(csv_dataset_path=csv_dataset_directory)
    except Exception as exc:
        print(f"Failed to create sample index for CSV dataset at {csv_dataset_directory}")
        raise exc

    # Write the sample index to disk.
    print(f"Writing sample index to output directory {output_directory}")
    try:
        _ = save_sample_index(sample_index=sample_index,
                              output_directory=output_directory)
    except Exception as exc:
        print(f"Failed to save sample index to disk.")
        raise exc