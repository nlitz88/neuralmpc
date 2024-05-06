
from typing import Optional
import pandas as pd
from plotly import graph_objects as go
from plotly.subplots import make_subplots

def create_2d_state_plot(gt_states_df: pd.DataFrame,
                         gt_x_col_name: str,
                         gt_y_col_name: str,
                         pred_states_df: pd.DataFrame,
                         pred_x_col_name: str,
                         pred_y_col_name: str,
                         title: str,
                         x_axis_title: str,
                         y_axis_title: str) -> go.Figure:
    """Creates a plot comparing two dimensions between ground truth states and
    states predicted by a model.

    Args:
        gt_states_df (pd.DataFrame): Dataframe containing the ground truth
        states, where each state is a row.
        gt_px_col_name (str): Ground truth data column name to be plotted along
        the x-axis.
        gt_py_col_name (str): Ground truth data column name to be plotted along 
        the y-axis.
        pred_states_df (pd.DataFrame): Dataframe containing the predicted
        states, where each state is a row.
        pred_px_col_name (str): Predicted data column name to be plotted along 
        the x-axis.
        pred_py_col_name (str):  Predicted data column name to be plotted along
        the y-axis.
        title (str): Title for the plot.
        x_axis_title (str): Title for the x-axis.
        y_axis_title (str): Title for the y-axis.

    Returns:
        go.Figure: Plotly graph object figure with the ground truth and
        predicted states plotted.
    """
    fig = go.Figure()
    # Add the ground truth trace.
    fig.add_trace(
        go.Scatter(x=gt_states_df[gt_x_col_name],
                   y=gt_states_df[gt_y_col_name],
                   name="Ground Truth",
                   mode="lines+markers")
    )
    # Add the predicted trace.
    fig.add_trace(
        go.Scatter(x=pred_states_df[pred_x_col_name],
                   y=pred_states_df[pred_y_col_name],
                   name="Predicted",
                   mode="lines+markers")
    )

    # Okay, also want to create an additional trace that will contain pairs of
    # points from the ground truth and predicted states. These points will be
    # plotted with a line connecting them to show the error between the two.
    for i in range(len(pred_states_df)):
        gt_x = gt_states_df.iloc[i][gt_x_col_name]
        gt_y = gt_states_df.iloc[i][gt_y_col_name]
        pred_x = pred_states_df.iloc[i][pred_x_col_name]
        pred_y = pred_states_df.iloc[i][pred_y_col_name]
        fig.add_trace(
            go.Scatter(x=[gt_x, pred_x],
                       y=[gt_y, pred_y],
                       mode="lines",
                       line=dict(color="grey", width=1),
                       showlegend=False)
        )

    fig.update_layout(title=title, xaxis_title=x_axis_title, yaxis_title=y_axis_title)
    return fig

def create_channel_error_plots(channel_errors_df: pd.DataFrame) -> go.Figure:

    fig = make_subplots(rows=len(channel_errors_df.columns), cols=1, shared_xaxes=False)
    for i, col in enumerate(channel_errors_df.columns):
        fig.add_trace(
            go.Scatter(x=channel_errors_df.index,
                                 y=channel_errors_df[col],
                                 mode="lines"),
            row=i+1,
            col=1
        )
        fig.update_yaxes(title_text=col, row=i+1, col=1)
    fig.update_layout(title="Channel Errors")

    return fig

def create_channel_value_plots(channel_values_df: pd.DataFrame) -> go.Figure:

    fig = make_subplots(rows=len(channel_values_df.columns), cols=1, shared_xaxes=False)
    for i, col in enumerate(channel_values_df.columns):
        fig.add_trace(
            go.Scatter(x=channel_values_df.index,
                                 y=channel_values_df[col],
                                 mode="lines"),
            row=i+1,
            col=1
        )
        fig.update_yaxes(title_text=col, row=i+1, col=1)
    fig.update_layout(title="Channel Values")

    return fig

# Create function that is just like the function above, but that takes in two
# parallel dataframes: one with ground truth states and one with predicted
# states. This function will plot the channel values for each on the same plot.
# The x-axis will be the index of the dataframes, and the y-axis will be the
# value.
def create_channel_value_comparison_plots(gt_channel_values_df: pd.DataFrame,
                                           pred_channel_values_df: pd.DataFrame) -> go.Figure:
    fig = make_subplots(rows=len(gt_channel_values_df.columns), cols=1, shared_xaxes=False)
    for i in range(len(gt_channel_values_df.columns)):
        fig.add_trace(
            go.Scatter(x=gt_channel_values_df.index,
                       y=gt_channel_values_df[gt_channel_values_df.columns[i]],
                       mode="lines",
                       name="Ground Truth"),
            row=i+1,
            col=1
        )
        fig.add_trace(
            go.Scatter(x=pred_channel_values_df.index,
                       y=pred_channel_values_df[pred_channel_values_df.columns[i]],
                       mode="lines",
                       name="Predicted"),
            row=i+1,
            col=1
        )
        fig.update_yaxes(title_text=gt_channel_values_df.columns[i].split("_")[-1], row=i+1, col=1)
    fig.update_layout(title="Channel Values Comparison")

    return fig

if __name__ == "__main__":
    pass