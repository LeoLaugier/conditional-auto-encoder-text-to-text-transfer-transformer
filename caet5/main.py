
r"""Main file for launching training/eval/predictions of CAE-T5 model."""
import importlib
import os
import sys

from absl import app, flags, logging
import gin
import pkg_resources
from mesh_tensorflow.transformer import transformer, utils
import tensorflow.compat.v1 as tf
import tensorflow_datasets as tfds

from caet5.data.utils import TaskRegistry_ll
from caet5.evaluation.eval_utils import print_random_predictions
from caet5.models.mtf_model import MtfModel_ll
from mesh_tensorflow_caet5.transformer import make_bitransformer_ll
from mesh_tensorflow_caet5.utils import tpu_estimator_model_fn_ll

flags.DEFINE_string("tpu_job_name", None,
                    "Name of TPU worker binary. Only necessary if job name is changed from "
                    "default tpu_worker.")

flags.DEFINE_string("base_dir", None,
                    "Base directory for the bucket on GCS, e.g. gs://my-bucket/")

flags.DEFINE_string("model_dir_name", "/tmp/transformer_standalone",
                    "Estimator model_dir")

flags.DEFINE_string(
    "model_size", "small",
    "Model size.")

flags.DEFINE_integer("model_dir_counter", -1,
                     "Counter postpended to model_dir_name. Default to -1 does not postpend anything.")


flags.DEFINE_string("tpu", None,
                    "The Cloud TPU to use for training. This should be either the name "
                    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 url.")

flags.DEFINE_string(
    "gcp_project",
    None,
    "Project name for the Cloud TPU-enabled project. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_string(
    "tpu_zone", None,
    "GCE zone where the Cloud TPU is located in. If not specified, we "
    "will attempt to automatically detect the GCE project from metadata.")

flags.DEFINE_multi_string(
    "module_import", None,
    "Modules to import. Use this, for example, to add new `Task`s to the "
    "global `TaskRegistry`.")

flags.DEFINE_boolean("use_model_api", False,
                     "Use Model API instead of utils.run.")

flags.DEFINE_enum("mode", None,
                  ["finetune", "eval", "predict"],
                  "Mode with which to run the model.")

# Tasks args
flags.DEFINE_string(
    "bucket", None,
    "Name of the Cloud Storage bucket for the data and model checkpoints, e.g. my-bucket")

flags.DEFINE_string(
    "data_raw_dir_name", None,
    "Name of the directory containing data.")

flags.DEFINE_string(
    "data_dir_name", None,
    "Name of the directory containing data.")

# Train mode args
flags.DEFINE_integer("train_steps", 1000, "Number of training iterations.")

flags.DEFINE_string("mixture_or_task", "processed_cctk",
                    "Name of Mixture or Task to use for training/evaluation.")

flags.DEFINE_string("base_pretrained_model_dir", "",
                    "Pretrained model dir for finetuning a model.")

# Eval mode args
flags.DEFINE_enum(
    "checkpoint_mode", "latest", ["all", "latest", "specific"],
    "Checkpoint steps to use when running 'eval', 'predict', 'finetune', and "
    "'export' modes. Can specify a list of checkpoints or all or the latest "
    "checkpoint. 'finetune' and 'export' modes work with 'latest' or "
    "'specific' with a single checkpoint.")

flags.DEFINE_list(
    "checkpoint_steps", [],
    "Checkpoint step numbers used for 'eval', 'predict', and 'finetune' modes. "
    "This argument is only used when which_checkpoint='specific'. "
    "For the 'finetune' mode, only a single checkpoint must be specified.")

flags.DEFINE_string("eval_summary_dir", "", "Path to save eval summaries")
flags.DEFINE_string("eval_split", "validation",
                    "Dataset split to use for evaluation.")

# Metrics args
flags.DEFINE_list(
    "metrics", ["BLEU", "SIM", "ACC", "PPL"],
    "Automatic metrics to use when evaluating.")

flags.DEFINE_string(
    "use_module_url", "https://tfhub.dev/google/universal-sentence-encoder/2",
    "Universal Sentence Encoder module URL.")

# Predict mode args
flags.DEFINE_string("input_file", "",
                    "Path to input file for decoding or scoring.")
flags.DEFINE_string("output_file", "", "Path to output file to save decodes.")

flags.DEFINE_integer("predict_batch_size", -1, "Batch size when predicting.")


FLAGS = flags.FLAGS


def main(_):
    if FLAGS.module_import:
        for module in FLAGS.module_import:
            importlib.import_module(module)

    # Add search path for gin files stored in package.
    gin.add_config_file_search_path(
        pkg_resources.resource_filename(__name__, "gin"))

    models_dir_name = FLAGS.model_dir_name
    if FLAGS.model_dir_counter >= 0:
        models_dir_name += "_%s" % str(FLAGS.model_dir_counter)
    models_dir = os.path.join(FLAGS.base_dir, models_dir_name)

    model_dir = os.path.join(models_dir, FLAGS.model_size)
    try:
        tf.io.gfile.makedirs(model_dir)
        suffix = 0
        command_filename = os.path.join(model_dir, "command")
        while tf.io.gfile.exists(command_filename):
            suffix += 1
            command_filename = os.path.join(
                model_dir, "command.{}".format(suffix))
        with tf.io.gfile.GFile(command_filename, "w") as f:
            f.write(" ".join(sys.argv))
    except tf.errors.PermissionDeniedError:
        logging.info(
            "No write access to model directory. Skipping command logging.")

    utils.parse_gin_defaults_and_flags()

    # Load and print a few examples.
    st_task = TaskRegistry_ll.get("processed_cctk")
    sequence_length = {"inputs": 64, "targets": 64}
    sequence_length["attribute"] = 64  # Or "attribute": 1 but packing not efficient...
    sequence_length["codeprefixedtargets"] = 64
    sequence_length["controlcode"] = 64

    with gin.config_scope('caet5'):
        ds = st_task.get_dataset(split="validation", sequence_length=sequence_length)

    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(5)):
        print(ex)






    print("unitests")

    mixture_or_task_name = "processed_cctk"
    from caet5.models.mesh_transformer import mesh_train_dataset_fn_ll
    from caet5.data.utils import get_mixture_or_task_ll, MixtureRegistry_ll

    from mesh_tensorflow_caet5.dataset import pack_or_pad_ll

    mixture_or_task = get_mixture_or_task_ll("mixture_processed_cctk")

    with gin.config_scope('caet5'):
        dsbis = mixture_or_task.get_dataset(split="validation", sequence_length=sequence_length)

    ds2 = pack_or_pad_ll(dsbis, sequence_length, pack=False,
                         feature_keys=tuple(mixture_or_task.output_features), ensure_eos=True)

    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds2.take(10)):
        print(ex)






    if FLAGS.use_model_api:
        # Modifying original T5 in CAE-T5
        transformer.make_bitransformer = make_bitransformer_ll
        utils.tpu_estimator_model_fn = tpu_estimator_model_fn_ll

        model_parallelism, train_batch_size, keep_checkpoint_max = {
            "small": (1, 256, 16),
            "base": (2, 128, 8),
            "large": (8, 64, 4),
            "3B": (8, 16, 1),
            "11B": (8, 16, 1)}[FLAGS.model_size]

        model = MtfModel_ll(
            tpu_job_name=FLAGS.tpu_job_name,
            tpu=FLAGS.tpu,
            gcp_project=FLAGS.gcp_project,
            tpu_zone=FLAGS.tpu_zone,
            model_dir=model_dir,
            model_parallelism=model_parallelism,
            batch_size=train_batch_size,
            learning_rate_schedule=0.003,
            save_checkpoints_steps=2000,
            keep_checkpoint_max=keep_checkpoint_max,  # if ON_CLOUD else None,
            iterations_per_loop=100,
            model_type="bitransformer",
            unsupervised_attribute_transfer_metrics=True
        )

        if FLAGS.checkpoint_mode != "specific" and FLAGS.checkpoint_steps:
            raise ValueError("checkpoint_mode is set to %s and checkpoint_steps is "
                             "also set. To use a particular checkpoint, please set "
                             "checkpoint_mode to 'specific'. For other modes, please "
                             "ensure that checkpoint_steps is not set."
                             % FLAGS.checkpoint_mode)

        if FLAGS.checkpoint_mode == "latest":
            checkpoint_steps = -1
        elif FLAGS.checkpoint_mode == "all":
            checkpoint_steps = "all"
        else:
            checkpoint_steps = [int(c) for c in FLAGS.checkpoint_steps]

        if FLAGS.mode == "finetune":
            pretrained_dir = os.path.join(FLAGS.base_pretrained_model_dir, FLAGS.model_size)

            model.finetune(
                mixture_or_task_name=FLAGS.mixture_or_task,
                pretrained_model_dir=pretrained_dir,
                finetune_steps=FLAGS.train_steps
            )

        elif FLAGS.mode == "eval":
            model.batch_size = train_batch_size * 4
            model.eval(
                mixture_or_task_name=FLAGS.mixture_or_task,
                checkpoint_steps=checkpoint_steps,
                summary_dir=FLAGS.eval_summary_dir,
                split=FLAGS.eval_split
            )

            print_random_predictions(FLAGS.mixture_or_task, sequence_length, model_dir, n=10)

        elif FLAGS.mode == "predict":
            if FLAGS.predict_batch_size > 0:
                model.batch_size = FLAGS.predict_batch_size
            model.predict(
                checkpoint_steps=checkpoint_steps,
                input_file=FLAGS.input_file,
                output_file=FLAGS.output_file,
                temperature=0)
        else:
            raise ValueError("--mode flag must be set when using Model API.")

    else:
        raise NotImplementedError()


def console_entry_point():
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)


if __name__ == "__main__":
    console_entry_point()
