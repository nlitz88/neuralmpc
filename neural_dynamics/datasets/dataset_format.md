# NeuralMPC Datasets
This package contains torch dataset wrappers around different datasets a
learned-dynamics model might be trained on.

For example: We may eventually want to train some of our models on the RACECAR
dataset, or we may want to train a model solely based on a custom,
privately-developed dataset.

### Vocabulary
Below are some key terms that are used interchangeably throughout the rest of
this document:
 - Sample == training example: A tuple of values, usually ([uk, xk], [uk+1,
   xk+1]).
 - Record == message == row in a sample: Just a group of message values from the
   same timestamp (For example: [uk, xk])
 - Field == column == message value: These are just the different fields within
   a record (For example: xk might consist of x, y, vx, vy, etc.).

### Underlying Dataset Assumptions
Before we can build a wrapper/dataloader around a dataset, there are a few
requirements the underlying dataset itself should meet. If these requirements
are not met by the underlying dataset, the wrapper should not be made
responsible to "clean up" the data. Rather, any transformations/changes that
were not done as a preprocessing step should be left to the training pipeline
after it receives a "generic" sample from the dataset wrapper. 

1. For each message whose value you wish to be included in dataset training
   examples, each included message's value(s) must originate from the same
   timestamp. If they do not, they will need to be synchronized in some way.
2. Any irrelevant / non-useful sequences within the dataset should be
   identified and removed ahead of time.
3. Any additional fields not present in the original, raw-data should be added
   into the CSVs as a pre-processing step (either before or after interpolation
   and synchronization).
...
   

### Dataset Message Synchronization
A common problem you'll run into when trying to create a dataset like this
(where your training examples contain values that come from multiple nodes) is
that those values might not all have been generated/computed/received at the
same time. However, we often need a way of figuring out what those values "would
have been" had they all arrived at the same time--as many times, we don't have
perfect synchronization between our nodes in a pub-sub distributed system model.

The most straightforward way to "synchronize" messages like this is to generate
new "artificial" messages from all the source messages, where these new messages 
contain all the values from all the different unsynchronized messages before,
and occur at a fixed frequency / fixed amount of time between each message. This
can be done by interpolating each original message's value at each new
timestamp, where each new message's timestamp is 1/freq s after the last
timestamp.

To do this, you'll need to use some kind of synchronization script. For the ART
Neural MPC dataset, @just-ap prepared a script for this.


### Dataset Wrapper Sample Format
No matter which underlyign dataset we choose to train a model on, each dataset
wrapper needs to provide training examples **in the same format**. The
underlying data can be in whatever form the dataset comes in--but the format of
the samples output from the dataset should always be the same. Additionally,
there is **no restriction** on the number or type of each field/column--you can
predict/estimate/add any columns you want from the existing data or inject new
column values in that are synchronized with existing messages. The wrapper
exists solely to extract consecutive records/rows from the preprocessed CSVs.
The wrapper does not care about the contents of those CSVs--so long as they
conform to the underlying dataset assumptions laid out above.

Each sample (or training example) provided by the dataset must follow the
provided format:

1. Each sample must be provided as a sequence of "temporally-contiguous" records
   in a numpy-like array, where each record is a row and each field in that
   record is a column.
2. The sequence of messages must be in order of increasing timestamp, with the
   lowest-index (first) element must have lowest/earliest timestamp of all the
   messages in the sequence.
3. Each value in a record must be from the exact same timestamp. If the messages
   were not originally produced at the exact same time, then their values should
   be synchronized by interpolating their values at a common timestamp.

For example
> [
>
>    [ 1.08629286e+00  6.59326814e-02  8.00000000e+00  0.00000000e+00
>    5.94960840e-04  0.00000000e+00  0.00000000e+00]
>
>    [ 1.08629286e+00  6.59326814e-02  8.00000000e+00 -1.60723783e-06
>    2.66595208e-04  0.00000000e+00  0.00000000e+00] 
>
>  ]