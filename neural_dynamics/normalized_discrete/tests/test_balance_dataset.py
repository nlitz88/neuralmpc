import unittest
import pandas as pd
import numpy as np
from balance_dataset import compute_sample_differences

class TestComputeSampleDifferences(unittest.TestCase):
    def test_compute_sample_differences(self):
        # Create sample data
        samples = [
            (0, (0, 1)),
            (1, (1, 2)),
            (2, (0, 1))
        ]
        csv_dataframes = [
            pd.DataFrame([[1, 2, 3], 
                          [4, 5, 6], 
                          [7, 8, 9]]),
            pd.DataFrame([[10, 11, 12], 
                          [13, 14, 15], 
                          [16, 17, 18]]),
            pd.DataFrame([[19, 20, 21], 
                          [22, 23, 24], 
                          [25, 26, 27]])
        ]

        # Call the function
        differences = compute_sample_differences(samples, csv_dataframes)

        # Assert the expected differences
        expected_differences = [
            np.array([3, 3, 3]),
            np.array([3, 3, 3]),
            np.array([3, 3, 3])
        ]
        for difference, expected_difference in zip(differences, expected_differences):
            self.assertTrue(np.allclose(difference, expected_difference))

if __name__ == '__main__':
    unittest.main()