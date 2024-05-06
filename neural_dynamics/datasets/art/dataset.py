"""PyTorch dataset wrapper for the ART Neural MPC Dataset."""

import pandas as pd
import numpy as np
from torch.utils.data import Dataset, random_split
from pathlib import Path
from typing import Any, List, Optional, Tuple
import time
from collections import deque

class ArtDataset(Dataset):
    """NeuralMPC Dataset for the most basic Discrete Dynamics model."""

    def __init__(self,
                 dataset_directory_path: Path,
                 desired_columns: Optional[List[str]] = None,
                 sample_length: Optional[int] = 2,
                 standardize: Optional[bool] = False) -> None:
        """Searches for CSV files in the provided dataset directory that samples
        will be extracted from.

        NOTE: Right now, this wrapper will read in all CSVs files as dataframes
        all at once. This WILL NOT SCALE as the dataset gets larger and larger.
        Therefore, this will need to be refactored so that it can simply index
        into a file via a file handle and grab the values at the line index,
        rather than reading in every file at the same time.

        Args:
            dataset_directory_path (Path): Path to the directory containing the
            dataset or split's CSV files.
            desired_columns (Optional[List[str]], optional): The column names /
            key strings from the CSVs in the specified directory that the
            wrapper should select from each row of the CSV. Selects all by
            default.
            sample_length (Optional[int], optional): The length of the samples
            generated by the wrapper. I.e., the number of consecutive rows from
            a source CSV that should be packed into the same training example.
            Defaults to 2.
            standardize (Optional[bool], optional): If true, standardizes each
            sample's first row's column values according to the mean and
            standard deviation across those columns. I.e., applies Z-score
            normalization. Defaults to False.

        Raises:
            Exception: If provided directory could not be found or if there are
            other errors encountered while working with any of the CSVs in the
            dataset.
        """

        # NOTE: I really don't think the desired_columns belongs in the dataset
        # wrapper, according to how we are going to treat the wrapper. The
        # wrapper should just be a "dumb" interface to the underlying CSV
        # data. I.e., it just take rows out of the CSVs and pass them along to
        # the model training or evaluation loops. Therefore, I don't think
        # desired_columns fits in here anymore. You can filter that out in the
        # training pipeline, no?
        
        # Input checks/handling. TODO: Move outside of class.
        if dataset_directory_path == None:
            raise Exception(f"Provided dataset_directory_path argument is None. Please provide a valid directory path.")
        if type(dataset_directory_path) == str:
            try:
                dataset_directory_path = Path(dataset_directory_path)
            except Exception as exc:
                print(f"Failed to parse filepath {dataset_directory_path}")
                raise exc
        if not dataset_directory_path.exists():
            raise FileNotFoundError(f"Could not find the provided dataset directory {dataset_directory_path}.")
        
        # If the directory does exist, search for CSV files in it.
        print(f"Beginning search for episode CSVs in provided directory {dataset_directory_path}.")
        start = time.perf_counter()
        csv_filepaths = self.__find_files(dataset_directory_path=dataset_directory_path)
        end = time.perf_counter()
        print(f"Discovered {len(csv_filepaths)} files in {end - start:.4f}s.")

        # Attempt to create a pandas dataframe around each of the discovered CSV
        # files.
        # TODO: Add TQDM.
        print(f"Opening the {len(csv_filepaths)} discovered episode files.")
        start = time.perf_counter()
        self.__episode_dataframes, _ = self.__open_files(filepaths=csv_filepaths)
        end = time.perf_counter()
        print(f"Successfully opened {len(self.__episode_dataframes)} episode files in {end - start:.4f}s.")

        # Check that each of the opened dataframes contains the desired columns.
        # TODO: Write a function to perform this check.
        
        # Each CSV (which is opened/read as a dataframe) is treated as its own
        # episode. Therefore, identify the indices of samples within each
        # episode. Do this for all episodes and return one big list that samples
        # can be pulled from.
        print(f"Beginning sample identification from {len(self.__episode_dataframes)} episode files.")
        start = time.perf_counter()
        self.__samples = self.__get_samples(dataframes=self.__episode_dataframes,
                                                          sample_length=sample_length)
        end = time.perf_counter()
        print(f"Identified {len(self.__samples)} samples in {end - start:.4f}s.")

        # Initialize what column values will be extraced from each sample. If no
        # columns are specified, then all the columns from the first file will
        # be selected.
        if desired_columns == None:
            self.__columns = self.__episode_dataframes[1].columns
        else:
            self.__columns = desired_columns

        # TEMPORARY (Until we refactor the dataset into a Lightning DataModule).
        # Compute the mean and the average of each of the selected columns
        # above.
        self.__standardize = standardize
        if self.__standardize == True:
            print(f"Beginning computation of means and stdevs for {len(self.__columns)} columns across {len(self.__episode_dataframes)} episode files.")
            start = time.perf_counter()
            self._means = self.__get_column_means(dataframes=self.__episode_dataframes,
                                            columns=self.__columns)
            self._stdevs = self.__get_column_stdevs(dataframes=self.__episode_dataframes,
                                            columns=self.__columns)
            end = time.perf_counter()
            print(f"Computed means and stdevs in {end - start:.4f}s.")
            print(f"Means: {self._means}")
            print(f"Stdevs: {self._stdevs}")
        

    def __find_files(self, dataset_directory_path: Path) -> List[Path]:
        """Searches through the provided directory (and any subdirectories) for
        CSV files.

        Args:
            dataset_directory_path (Path): Path to the datasets containing
            directory.

        Returns:
            List[Path]: List of all the CSV files found.
        """
        # NOTE: Pathlib added the "walk" function for Python 3.12. This would
        # simplify a good bit of this function. For now, this "by-hand"
        # breadth-first search works too (that's what walk is doing under the
        # hood--but probably with more checks/protections).
        csv_filepaths = []
        directory_queue = deque([dataset_directory_path])
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
    
    def __open_files(self, filepaths: List[Path]) -> Tuple[List[pd.DataFrame], List[Path]]:
        """Attempt to create pandas DataFrames around each of the provided
        files.

        Args:
            filepaths (List[Path]): List of filepaths to each CSV file.

        Returns:
            Tuple[List[pd.DataFrame], List[Path]]: Returns a list of the
            dataframes that were successfully created. Also returns a list of
            filepaths that couldn't be opened for whatever reason.
        """
        dataframes = []
        failed_filepaths = []
        for filepath in filepaths:
            try:
                file_dataframe = pd.read_csv(filepath)
            except Exception as exc:
                print(f"Failed to create pandas dataframe from file {file_dataframe}")
                failed_filepaths.append(filepath)
            else:
                dataframes.append(file_dataframe)
        return dataframes, failed_filepaths

    def __get_samples(self,
                      dataframes: List[pd.DataFrame],
                      sample_length: int) -> List[Tuple]:
        """Generates a list of samples across each of the provided episodes
        (where each episode corresponds with a dataframe from a CSV).

        Args:
            dataframes (List[pd.DataFrame]): The list of episode dataframes.
            sample_length (int): The number of rows each sample will contain.

        Returns:
            List[Tuple]: Each entry in sample_indices will be a tuple, where the
            first entry will be an index to the dataframe that it comes in
            self.__episode_dataframes, and the second entry will be another
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

    def __get_column_means(self, 
                           dataframes: List[pd.DataFrame], 
                           columns: List[str]) -> List[float]:
        
        # FOR NOW: Concatenate the data frames.
        combined_dataframe = pd.concat(dataframes)
        # Now, compute the means for the columns we want.
        desired_column_indices = combined_dataframe.columns.get_indexer(columns)
        filtered_dataframe = combined_dataframe.iloc[:, desired_column_indices]
        return filtered_dataframe.mean().tolist()

    def __get_column_stdevs(self, 
                            dataframes: List[pd.DataFrame], 
                            columns: List[str]) -> List[float]:
        # FOR NOW: Concatenate the data frames.
        combined_dataframe = pd.concat(dataframes)
        # Now, compute the means for the columns we want.
        desired_column_indices = combined_dataframe.columns.get_indexer(columns)
        filtered_dataframe = combined_dataframe.iloc[:, desired_column_indices]
        return filtered_dataframe.std().tolist()

    def __len__(self) -> int:
        """Returns the length of the dataset == the number of "training
        examples" == "samples" == "items" in the dataset.

        Returns:
            int: The number of samples in the dataset.
        """
        return len(self.__samples)

    def __getitem__(self, index: int) -> Any:
        """Returns the sample at the provided index in the dataset.

        Args:
            index (int): Offset of sample in the dataset.

        Returns:
            Any: The sample at the provided index.
        """
        dataframe_index, sample_indices = self.__samples[index]
        sample_start_index, sample_end_index = sample_indices
        sample_dataframe: pd.DataFrame = self.__episode_dataframes[dataframe_index]
        desired_column_indices = sample_dataframe.columns.get_indexer(self.__columns)
        sample = sample_dataframe.iloc[sample_start_index:sample_end_index+1, desired_column_indices].to_numpy()

        # Apply Z-Score standardization to the first row of the sample if requested.
        if self.__standardize == True:
            first_row = sample[0, :]
            # Add the 1e-10 to avoid divide by zero:
            # https://e2eml.school/standard_normalization
            standardized_first_row = (first_row - self._means) / (self._stdevs + 1e-10*np.ones_like(self._stdevs))
            sample[0, :] = standardized_first_row

        return sample
    
# Debugging
if __name__ == "__main__":

    desired_columns = ["u/u_a",         
                       "u/u_steer",
                       "v/v_long",
                       "v/v_tran",
                       "w/w_psi",
                       "x/x",
                       "x/y",
                       "e/psi"
                       ]

    discrete_dynamics_dataset_tiny = ArtDataset(dataset_directory_path="/home/nlitz88/Downloads/iac_datasets/2023_08_31-putnam/2023_08_31-13_46_57_0_old",
                                                desired_columns=desired_columns,
                                                sample_length=5,
                                                standardize=True)
    np.set_printoptions(suppress=True)
    print(discrete_dynamics_dataset_tiny[2])
    print(discrete_dynamics_dataset_tiny[3])

    # # Test Generating a random train / validation split.
    # TEST_SPLIT_SIZE = 0.7
    # VALIDATION_SPLIT_SIZE = 0.3
    # generator = torch.Generator().manual_seed(0)
    # test_split, validation_split = random_split(dataset=discrete_dynamics_dataset_tiny,
    #                                             lengths=[TEST_SPLIT_SIZE, VALIDATION_SPLIT_SIZE],
    #                                             generator=generator)
    # print("Test Split:")
    # print([test_sample for test_sample in test_split])
    # print("Validation Split:")
    # print([validation_sample for validation_sample in validation_split])