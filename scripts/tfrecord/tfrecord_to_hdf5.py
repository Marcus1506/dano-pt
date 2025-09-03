import argparse
import functools
import json
import os

import h5py
import numpy as np
import tensorflow as tf
from reading_tfrecord import parse_serialized_simulation_example
from tqdm import tqdm


def read_metadata(data_path):
    with open(os.path.join(data_path, "metadata.json"), "rt") as fp:
        return json.loads(fp.read())


def prepare_rollout_inputs(context, features):
    """Prepares an inputs trajectory for rollout."""
    out_dict = {**context}
    # Position is encoded as [sequence_length, num_particles, dim] but the model
    # expects [num_particles, sequence_length, dim].
    pos = tf.transpose(features["position"], [1, 0, 2])
    # The target position is the final step of the stack of positions.
    target_position = pos[:, -1]
    # Remove the target from the input.
    out_dict["position"] = pos[:, :-1]
    # Compute the number of nodes
    out_dict["n_particles_per_example"] = [tf.shape(pos)[0]]
    if "step_context" in features:
        out_dict["step_context"] = features["step_context"]
    out_dict["is_trajectory"] = tf.constant([True], tf.bool)
    return out_dict, target_position


parser = argparse.ArgumentParser(description="Convert TFRecord file to HDF5 format")
parser.add_argument("--data_path", type=str, help="Path to the TFRecord file")
parser.add_argument(
    "--split", type=str, help="Dataset split (e.g., train, valid, test)"
)

args = parser.parse_args()

# Define the path to the TFRecord file
data_path = args.data_path
split = args.split

# Read metadata
metadata = read_metadata(data_path)

# Create a dataset from the TFRecord file
ds = tf.data.TFRecordDataset([os.path.join(data_path, f"{split}.tfrecord")])
ds = ds.map(functools.partial(parse_serialized_simulation_example, metadata=metadata))
# ds = ds.map(prepare_rollout_inputs)

# Create a new HDF5 file
new_data = h5py.File(os.path.join(data_path, split) + ".h5", "w")

# Iterate over the dataset and print one sample

for record in tqdm(ds):
    key = record[0]["key"]
    # Change to string with leading zeros
    key = str(int(key)).zfill(5)
    # Particle types start with 5, should start with 0 in new dataset
    particle_type = record[0]["particle_type"].numpy() - 5
    position = record[1]["position"].numpy()

    # Create a group for each key
    group = new_data.create_group(key)

    # Create datasets for 'particle_type' and 'position'
    group.create_dataset("particle_type", data=particle_type)
    group.create_dataset("position", data=position)


print("Done")
