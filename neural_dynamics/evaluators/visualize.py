from typing import List, Optional
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from dataclasses import dataclass

# Need a function that plots the "predicted next state" for each model, as well
# as the ground truth next state. Given what, though? Should this function be
# responsible for stacking dataframes to create one output? Maybe its interface
# is that you pass it a list of dataframes, where each dataframe is the
# input/output from each model? Maybe we'd expect these to be derived/extracted
# from the underlying data storage (whatever that ends up being).

# I think there should be one, single underlying plot creation function that
# just takes in the data that should be thrown on the scatter plot.

def stack_dataframes():
    pass

# The state plotting function(s) shouldn't care what is actually being provided
# to them--right. They should literally just take in a dataframe (or multiple,
# parallel nympy arrays if you want to be very safe)
def plot_predicted_states():
    pass

def plot_velocities():
    pass

# This function should accept a list of model_output dataframes, where each
# dataframe contains the sequence of predicted next states for a particular
# dynamics model.The beauty is, these outputs don't strictly have to be from a
# learned dynamics model--they can be from any discrete dynamics model.

# More specifically, in the name of visualization, this script will need both
# the model's output AND the model's NAME as a string. Therefore, we'll specify
# a small Dataclass that contains both of these.

@dataclass
class LabelledModelOutput:
    """Dataclass to encapsulate a model's output along with its name--and any
    other metadata that is useful for visualization of the metrics for that
    model.
    """
    model_name: str
    model_output_dataframe: pd.DataFrame

def create_channels_plot(labelled_model_outputs: List[LabelledModelOutput],
                         common_column_names: List[str],
                         figure_title: str,
                         num_figure_columns: Optional[int] = 2,
                         figure_height: Optional[int] = 600,
                         figure_width: Optional[int] = 800) -> go.Figure:
    """Creates a single figure that contains subplots for each of the channels
    within the provided dataframes. Takes each datafame's column values and
    plots them on the respective subplot. This function should only ever be run
    with dataframes with the same columns / columns in the same order (I.e.,
    make sure your dataframes all have "xxxx_vy" as their 3rd column, for
    example).

    Args:
        labelled_model_outputs (List[LabelledModelOutput]): A list of
        LabelledDataframes, where each is just a dataframe of the model's output
        along with the model's name (as we don't assume model name is included
        somewhere in the dataframe). Each dataframe contains the output states
        from each dynamics model at different timesteps.

        common_column_names (List[str]): A list of the string names, where each
        name is the common name of the column at each column index across
        dataframes. For example, if you have "nbod_vx, single_track_vx"--you
        might say the common column name is "vx". This is purely for
        visualization, so it won't break anything if it's wrong--just maybe your
        interpretation. num_figure_columns (Optional[int], optional): How many
        subplots will be fit within the width of the whole plot. Defaults to 2.

        figure_title (str): Title for the overarching figure.
        num_figure_columns (Optional[int], optional): Number of subplots you
        want the figure to have along each row. Defaults to 2.
        figure_height (Optional[int], optional): Height in pixels of the
        overarching figure that is created. Defaults to 800.
        figure_width (Optional[int], optional): Overarching width of the figure
        that is created. Defaults to 1000.

    Returns:
        go.Figure: The resulting plotly figure containing each channel's
        subplot.
    """

    # Create a new figure with subplots. Have to figure out number of rows based
    # on the number of columns and based on the number of 
    max_df_columns = max(labelled_output.model_output_dataframe.shape[1] for labelled_output in labelled_model_outputs)
    num_figure_rows = int(np.ceil(max_df_columns/float(num_figure_columns)))
    fig = make_subplots(rows=num_figure_rows,
                        cols=num_figure_columns,
                        subplot_titles=common_column_names,
                        start_cell="top-left")

    # For each model's output dataframe, iterate through the subplots and add
    # this model's output value for that plot as a trace.
    for labelled_output in labelled_model_outputs:
        # Grab the model output dataframe from the labelled output.
        model_output_dataframe = labelled_output.model_output_dataframe
        # For each channel that's present in the dataframe, add a trace with its
        # column data in the respective subplot.
        for i, column_name in enumerate(model_output_dataframe.columns):
            # Compute what the row and column for the subplot to select should
            # be based on the current channel (column) index.
            # NOTE: Have to add one, as these are one-indexed.
            row_index = int(i / 2) + 1
            column_index = i % 2 + 1
            # Add this ith column's data to the ith subplot.
            fig.add_trace(
                go.Scatter(x=model_output_dataframe.index,
                           y=model_output_dataframe[column_name],
                           name=f"{labelled_output.model_name}_{common_column_names[i]}"),
                row=row_index,
                col=column_index
            )
            fig.update_xaxes(title_text="Timestep",
                             row=row_index,
                             col=column_index)
            fig.update_yaxes(title_text=f"{common_column_names[i]} Value",
                             row=row_index,
                             col=column_index)
            
    # Set the figure's layout.
    fig.update_layout(height=figure_height, 
                      width=figure_width, 
                      title_text=figure_title)

    return fig

def create_channel_errors_plot(ground_truth_dataframe: pd.DataFrame,
                               labelled_model_outputs: List[LabelledModelOutput],
                               *args,
                               **kwargs):
    """Wrapper for create_channels_plot that takes a dataframe of channel values
    for a ground truth, reference sequence of states, along with a list of
    dataframes for each model to be compared against the ground truth. This will
    will compute the difference for each model's output against the ground truth
    and plot those as each model's value.

    Less of a visualization specific function and more of just a convenient
    wrapper--may move out of this module to be grouped with any similar wrappers
    in the future.

    Args:
        ground_truth_dataframe (pd.DataFrame): DataFrame containing the ground
        truth states that each model's output will be subtracted from at each
        timestep.
        labelled_model_outputs (List[LabelledModelOutput]): A list of
        LabelledDataframes, where each is just a dataframe of the model's output
        along with the model's name (as we don't assume model name is included
        somewhere in the dataframe). Each dataframe contains the output states
        from each dynamics model at different timesteps.
    """

    # Compute model error dataframes against the provided ground truth.
    for labelled_model_output in labelled_model_outputs:
        labelled_model_output.model_output_dataframe = labelled_model_output.model_output_dataframe - ground_truth_dataframe

    # Call function to create figure with error plots.
    return create_channels_plot(labelled_model_outputs=labelled_model_outputs,
                                *args,
                                **kwargs)

# Write a function here to plot each model's position output on the same 2D
# position plot to compare accuracy.
# Each model's output should 
def create_positions_plot(labelled_model_outputs: LabelledModelOutput,
                          x_column_name: Optional[str] = "xp",
                          y_column_name: Optional[str] = "yp") -> go.Figure:
    """Creates a 2D scatter plot with each model's predicted positions given
    each model's provided model's output/predicted states. For now, assumes all
    model output dataframes have the same column names.

    Args:
        labelled_model_outputs (LabelledModelOutput): A list of
        LabelledModelOutputs--basically each model's predicted states dataframe
        along with the model's name.
        x_column_name (Optional[str], optional): The column name in each model's
        output dataframe for the x-position. Defaults to "xp".
        y_column_name (Optional[str], optional): The column name in each model's
        output dataframe for the y-position. Defaults to "yp".

    Returns:
        go.Figure: The resulting figure with each model's position predictions
        plotted.
    """

    fig = go.Figure()
    # Iterate through each model's output and add a trace to the figure.
    for labelled_model_output in labelled_model_outputs:#
        model_output_dataframe = labelled_model_output.model_output_dataframe
        fig.add_trace(
            go.Scatter(x=model_output_dataframe[x_column_name],
                       y=model_output_dataframe[y_column_name],
                       name=labelled_model_output.model_name,
                       mode="markers")
        )
    return fig

# Basically, make a parallel line plot, where each line is a separate channel in
# the ground truth data. The point of this visualizer is to understand what
# portion of the state space we have in a particular dataset, as well as what
# part of that state space a particular model is struggling in the most. Can
# generate one of these plots for each model evaluated on a particular dataset. 

# Therefore, as input, we'll not only need the ground truth states dataframe,
# but we will also need the error of the particular model we're evaluating on
# each state. So, maybe that's another column in the dataframe that just
# includes the error? (Wherein the color is based on the error then?) OR, maybe
# there is a column for each colum value's error--that way they're not just all
# grouped into one? Not sure how this would work for the visualization--you
# could have some kind of gradient.

# Maybe at the very least (as someting very primitive for now), can just
# visualize the states that we have.
def create_state_space_figure(ground_truth_states_dataframe: pd.DataFrame) -> go.Figure:

    fig = px.parallel_coordinates(ground_truth_states_dataframe,
                                  dimensions=ground_truth_states_dataframe.columns)
    return fig

if __name__ == "__main__":

    pass