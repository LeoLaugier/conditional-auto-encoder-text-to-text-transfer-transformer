import os

import random
import tensorflow as tf
import tensorflow_datasets as tfds

from my_t5_data_utils import TaskRegistry_ll


def print_random_predictions(task_name, sequence_length, model_dir, n=10):
    """Print n predictions from the validation split of a task."""
    # Grab the dataset for this task.
    ds = TaskRegistry_ll.get(task_name).get_dataset(
        split="validation",
        sequence_length=sequence_length,
        shuffle=False)

    def _prediction_file_to_ckpt(path):
      """Extract the global step from a prediction filename."""
      return int(path.split("_")[-2])

    # Grab the paths of all logged predictions.
    prediction_files = tf.io.gfile.glob(
        os.path.join(
            model_dir,
            "validation_eval/%s_*_predictions" % task_name))
    # Get most recent prediction file by sorting by their step.
    latest_prediction_file = sorted(
        prediction_files, key=_prediction_file_to_ckpt)[-1]

    # Collect (inputs, targets, prediction) from the dataset and predictions file
    results = []
    with tf.io.gfile.GFile(latest_prediction_file) as preds:
      for ex, pred in zip(tfds.as_numpy(ds), preds):
        results.append((tf.compat.as_text(ex["inputs_plaintext"]),
                        tf.compat.as_text(ex["targets_plaintext"]),
                        pred.strip()))

    print("<== Random predictions for %s using checkpoint %s ==>\n" %
          (task_name,
          _prediction_file_to_ckpt(latest_prediction_file)))

    for inp, tgt, pred in random.choices(results, k=10): # k=n ?
      print("Input:", inp)
      # print("Target:", tgt)
      print("Prediction:", pred)
      # print("Counted as Correct?", tgt == pred)
      print()