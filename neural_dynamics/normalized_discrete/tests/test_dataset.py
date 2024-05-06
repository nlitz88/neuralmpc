import json
from pathlib import Path
from unittest import TestCase

import sys
import unittest

import numpy as np
sys.path.insert(0, "../../")
from neural_mpc.normalized_discrete.dataset import NormalizedDiscreteCSVDataset

class TestSampleExtraction(TestCase):
    """Test case to validate the samples recorded in a dataset's sample
    index.
    """

    def setUp(self) -> None:
        # Create a new instance of the NormalizedDiscreteCSVDataset class using
        # the test_csv_dataset CSV dataset directory.
        module_directory = Path(__file__).resolve().parent
        test_csv_dataset_path = module_directory/"test_csv_dataset"
        test_csv_dataset_sample_index_path = test_csv_dataset_path/"test_csv_dataset_sample_index.json"
        self.dataset = NormalizedDiscreteCSVDataset(test_csv_dataset_sample_index_path)

        # Load the expected outputs file.
        test_outputs_filepath = test_csv_dataset_path/"test_dataset_outputs.json"
        with test_outputs_filepath.open("r") as test_outputs_file:
            self.expected_outputs = json.load(test_outputs_file)
        
        return super().setUp()
    
    def test_compare_samples(self):
        # Loop over the first five samples in the dataset and compare them to the
        # expected outputs.
        for i in range(5):
            dataset_sample = self.dataset[i]
            dataset_sample = np.array(dataset_sample)
            # print(dataset_sample)
            expected_output = self.expected_outputs[i]
            expected_output = np.array(expected_output)
            # print(expected_output)
            # self.assertFalse(np.equal(dataset_sample, expected_output).any())
            self.assertTrue(np.isclose(dataset_sample[0], expected_output[0]).all())
            self.assertTrue(np.isclose(dataset_sample[1], expected_output[1]).all())
            # self.assertTrue(np.equal(self.dataset[i], np.ndarray(self.expected_outputs[i])))
    
    def tearDown(self) -> None:
        return super().tearDown()
    
if __name__ == "__main__":
    unittest.main()