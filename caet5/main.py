
r"""Main file for launching training/eval/predictions of CAE-T5 model."""
import importlib

import gin
import pkg_resources
from absl import app, flags
import tensorflow.compat.v1 as tf
from mesh_tensorflow.transformer import utils

flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_string("use_module_url", "https://tfhub.dev/google/universal-sentence-encoder/2",
                    "Universal Sentence Encoder module URL.")

flags.DEFINE_string("bucket", None,
                    "Name of the Cloud Storage bucket for the data and model checkpoints, e.g. my-bucket")

flags.DEFINE_string("base_dir", None,
                    "Base directory for the bucket on GCS, e.g. gs://my-bucket/")

flags.DEFINE_string("data_raw_dir_name", None,
                    "Name of the directory containing data.")

flags.DEFINE_string("data_dir_name", None,
                    "Name of the directory containing data.")

flags.DEFINE_list("metrics", ["BLEU", "SIM", "ACC", "PPL"],
                  "Automatic metrics to use when evaluating.")

FLAGS = flags.FLAGS

def main(_):
    if FLAGS.module_import:
        for module in FLAGS.module_import:
            importlib.import_module(module)

    # Add search path for gin files stored in package.
    gin.add_config_file_search_path(
        pkg_resources.resource_filename(__name__, "gin"))

    utils.parse_gin_defaults_and_flags()

def console_entry_point():
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)

if __name__ == "__main__":
    console_entry_point()