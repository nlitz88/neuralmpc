from matplotlib import pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import binned_statistic_2d
import plotly.express as px

# Toy script to test out the histogram binning and digitization functions.

if __name__ == "__main__":

    # Need an array of numbers such that there are 2D points that are
    # roughly close to each other, I.e., pretty similar, and where there are
    # more than one type than others.

    test_data = np.array([[0.0142, 2.1231],
                          [0.0182, 1.9382],
                          [0.0123, 1.9999],
                          [0.0423, 0.9542],
                          [0.0892, 0.2507],
                          [0.0234, 1.5678],
                          [0.0876, 2.3456],
                          [0.0432, 1.2345],
                          [0.0789, 0.9876],
                          [0.0321, 0.8765]])
    # values = np.random.rand(test_data.shape[0])
    # TODO: NEED TO SCALE THIS UP TO N-DIMENSIONS. Therefore, need to write a
    # function to compute the bin edges for each dimension. Will then feed this
    # to the multidimensional histogram function histogramdd or something like
    # that.
    x_bin_edges = np.histogram_bin_edges(test_data[:,0], bins="auto")
    y_bin_edges = np.histogram_bin_edges(test_data[:,1], bins="auto")
    print(f"X bin edges: {x_bin_edges}")
    print(f"Y bin edges: {y_bin_edges}")
    test_histogram, _, _ = np.histogram2d(test_data[:,0], test_data[:,1], bins=(x_bin_edges, y_bin_edges))
    test_histogram = test_histogram.T # For visualization purposes I guess.
    print(f"Test histogram: {test_histogram}")
    plt.imshow(test_histogram, origin='lower', extent=[x_bin_edges[0], x_bin_edges[-1], y_bin_edges[0], y_bin_edges[-1]], aspect='auto')
    plt.colorbar()
    plt.xlabel('Delta X')
    plt.ylabel('Delta Y')
    plt.title('Test Histogram')
    plt.savefig("test_histogram.png")

    # Also, just for fun, create separate 1D histograms showing the distribution
    # of samples for both delta X and delta Y. Should really generate these for
    # every dimension of the input data anyway, actually.
    x_histogram, _ = np.histogram(test_data[:,0], bins=x_bin_edges)
    y_histogram, _ = np.histogram(test_data[:,1], bins=y_bin_edges)
    # Create separate 1D histograms for delta X and delta Y
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.hist(test_data[:, 0], bins=x_bin_edges)
    plt.xlabel('Delta X')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delta X')

    plt.subplot(2, 1, 2)
    plt.hist(test_data[:, 1], bins=y_bin_edges)
    plt.xlabel('Delta Y')
    plt.ylabel('Frequency')
    plt.title('Histogram of Delta Y')

    plt.tight_layout()
    plt.savefig("1d_histograms.png")
    # plt.show()
    

    # TODO: Now, need to figure out how to see which bin each element has been
    # assigned to. Apparently, that's what numpy's digitize function does--maybe
    # it "creates a histogram" but is really just assigning your value to a bin
    # == discretization.
    # So maybe we use the histogram_bin_edges function above to compute what the
    # bin edges should be--and then use digitize to obtain each element's bin
    # assignment.
    x_assignments = np.digitize(test_data[:,0], bins=x_bin_edges)
    y_assignments = np.digitize(test_data[:,1], bins=y_bin_edges)
    print(f"X assignments: {x_assignments}")
    print(f"Y assignments: {y_assignments}")


    # hist = np.histogram2d(test_data[:,0], test_data[:,1], bins=np.histogram_bin_edges(test_data[:,0], bins="fd"))
    # print(f"Successfully created histogram with Freedman Diaconis rule: {hist}")
    
    # bin_means = binned_statistic_2d(test_data[:,0], test_data[:, 1], values, bins=10).statistic
    # print(bin_means)

    # df = px.data.tips()
    # fig = px.histogram(df, x="total_bill")
    # fig.show()

    test_data_df = pd.DataFrame(test_data, columns=["delta_x", "delta_y"])

    fig = px.density_heatmap(test_data_df, x="delta_x", y="delta_y", text_auto=True, nbinsx=len(x_bin_edges), nbinsy=len(y_bin_edges))
    fig.write_html("test_data_heatmap.html")
        


