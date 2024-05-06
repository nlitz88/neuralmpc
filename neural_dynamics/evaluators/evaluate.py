
import sys
sys.path.insert(0, "../")

from neural_mpc.datasets.art.dataset import ArtDataset

# Maybe the evaluator script grabs the output from each model in some directory
# and displays it. Maybe each model should be responsible for generating this
# output from the dataset.

# Then, a visualizing script should just look at all those predicted steps and
# plot them!


# Function to draw each model's prediction for the first 10 steps of a lap.

# Function to plot the velocity component from each model's prediction.

# Function to plot the angular velocities.

#

# To start, can just use matplotlib or something. Or, if plotly is just as fast,
# maybe make a dashboard or something quick.