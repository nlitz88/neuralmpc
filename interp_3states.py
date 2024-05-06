"""_summary_
    This script is used to do synchronize the data at 100 Hz by first rounding the 
    timestamp to 0.01, then do the interpolation to the origin data and generate to a new 
    csv file
"""


import pathlib as path
import numpy as np
import csv
import os
import pandas as pd

filename = "/drive_topic.csv"
output_file = "interp_drive_topic.csv"

class interpData():
    def __init__(self):
        self.frequency = 100 # Hz
        self.interval = 1 / self.frequency
    
    def open_file(self):
        current_working_directory = os.getcwd()
        filepath = current_working_directory + filename

        """
            Here we open the csv file and only pick the numeric columns
        """
        df = pd.read_csv(filepath)
        self.numeric_columns = df.select_dtypes(include=['number']).columns
        self.column_numbers_numeric = [df.columns.get_loc(col) for col in self.numeric_columns]
        # print(self.column_numbers_numeric)
        self.numeric_data_column_len = len(self.column_numbers_numeric)

        df[self.numeric_columns] = df[self.numeric_columns].astype(float)
        self.np_input_list = df.values
        self.row_len = self.np_input_list.shape[0]
        self.column_len = self.np_input_list.shape[1]
        self.numeric_title_array = []


        """
            we store the origin timestamp into a 1D array for interpolation
            adjust the following number for the timestamp column
        """
        self.timestamp_column = 0

        self.origin_timestamp_array = self.np_input_list[1:, self.timestamp_column]
        print(self.origin_timestamp_array)

    def start_interp(self):
        """First, the number of points after interpolation is calculated. We do round up
        the first timestamp and round down the last one to 0.01
        """

        self.row_len = self.np_input_list.shape[0]
        self.column_len = self.np_input_list.shape[1]
        interp_time_start = self.np_input_list[0][self.timestamp_column] - self.np_input_list[0][self.timestamp_column] % 0.01 + 0.01
        interp_time_stop = self.np_input_list[self.row_len - 1][self.timestamp_column] - self.np_input_list[self.row_len - 1][self.timestamp_column] % 0.01
        print(interp_time_start, interp_time_stop)
        int_interp_time_start = int(interp_time_start * 100)
        int_interp_time_stop = int(interp_time_stop * 100)
        print(int_interp_time_start, int_interp_time_stop)
        
        """ calculate the points number after interpolation
        """
        self.points_num = int((int_interp_time_stop - int_interp_time_start) / 100 * self.frequency)

        idx_interp_time_start = int_interp_time_start

        """create a new array for the rounded timestamp
        """
        self.timestamp_interp_array = []
        for i in range(0, self.points_num):
            self.timestamp_interp_array.append(idx_interp_time_start/100)
            idx_interp_time_start = idx_interp_time_start + self.interval * 100
        self.timestamp_interp_array = np.array(self.timestamp_interp_array)

        """do interpolation to each numeric column
        """
        self.total_interp_array = []
        tmp_array = []
        for i in range(0, self.numeric_data_column_len):
            tmp_array = self.np_input_list[1:, self.column_numbers_numeric[i]]
            tmp_array = tmp_array.astype(float)
            self.origin_timestamp_array = self.origin_timestamp_array.astype(float)

            tmp_interp = np.interp(self.timestamp_interp_array, self.origin_timestamp_array, tmp_array)
            self.total_interp_array.append(tmp_interp)

        self.total_interp_array = np.array(self.total_interp_array)
        # print(self.total_interp_array)
        
    def write_to_new_csv(self):
        """write to a new csv file
        """
        print(self.numeric_columns)
        with open(output_file, mode='w', newline='') as outfile:
            csv_writer = csv.writer(outfile)
            csv_writer.writerow(self.numeric_columns)
            for j in range(0, self.points_num):
                tmp_array = []
                tmp_array.append(self.timestamp_interp_array[j])

                for i in range(1, self.numeric_data_column_len):
                    tmp_array.append(self.total_interp_array[i][j])
                
                csv_writer.writerow(tmp_array)

    def write_to_df(self):
        """write to a pandas dataframe with only 1 timestamp at left
        """
        columns = ['Timestamp'] + list(self.numeric_columns[1:-1])
        print("columns")

        result_array = []

        for j in range(self.points_num):
            tmp_array = [self.timestamp_interp_array[j]] + [self.total_interp_array[i][j] for i in range(1, self.numeric_data_column_len - 1)]
            result_array.append(tmp_array)

        df = pd.DataFrame(result_array, columns=columns)

        df.to_csv(output_file, index=False) 

        print(df)



if __name__ == "__main__":
    interpD = interpData()
    interpD.open_file()
    interpD.start_interp()
    interpD.write_to_new_csv()

    # interpD.write_to_df()
