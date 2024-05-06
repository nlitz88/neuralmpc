# ART Neural MPC Dataset Format
WIP. This document serves as a description of the structure our home-baked
neuralmpc dataset should conform to. This is the format that the art_dataset
wrapper will expect.

I will detail this more later. Here is the format our dataset will follow for
now:

- Our datasets will be housed under a single parent directory.
- The directory will have one or more CSV files
- We follow the same (very simple) format used by NeuroBEM--one CSV
  per episode. See below notes. One or more CSVs can result from a single ros
  bag after preprocessing steps.
- Following NeuroBEM's paradigm: each CSV == an episode/session.
- While not necessarily required yet, for now, there should be a JSON file or
  some kind of readme where there is a mapping from each csv filename to a
  description.

One CSV per episode should be fine. In theory, this could lead to filesystem
fragmentation issues. I.e., a bunch of tiny files could get spread across the
disk and lead to longer disk response times than if they were all in one large
contiguous chunk of blocks (which ext4 favors). Then again, with SSDs, this
shouldn't really introduce any real latency issues.

While this isn't in the critical path, I need to think about (maybe using the
RACECAR dataset as inspiration) how to best maintain metadata for the dataset.
I.e., what's the easiest way to track/maintain which CSVs contain what kind of
maneuvers? I.e., the racecar dataset has 11 different scenarios, and it seems
like they have a decent way of keeping track of which samples come from each
scenario, etc.

For now, I guess we will go without a "metadata.json" file or something like
that, to provide additional information along with each episode/csv. However, in
the future, I think this really does need to be part of the dataset in some way.
Whatever way we go for this, it shouldn't make extending the dataset any more
difficult. I.e., you should be able to show up with a new rosbag and have a very
simple way of describing what was going on in that rosbag, where and when it was
collected, maybe some parameters from your car?

Neurobem provides a high level description in a flights.txt file. NeuralFLY
organizes their different scenarios / situations by number in different
directories. RACECAR organizes their different scenarios by number as well. None
of these explicitly define training and validation splits, too--unlike COCO
where there are explicit splits created in the files.

As far as the role of the dataset wrapper goes--for now, it is just going to be
blind to the organization and labelling of the dataset for now. If, in the
future, we decide to create separate splits (directories) of CSVs for training
and validation--then this wrapper will just be invoked on both of those. The
wrapper itself has zero knowledge of training/validation splits or anything like
that. All it does (for now) is look for CSVs (sessions/episodes) in a provided
directory and finds samples within them. That's it! I note this as, in the past,
when working with detectron and other frameworks, "Dataset" is thrown around too
loosely, when they're really talking about working with a "split" of the
dataset.