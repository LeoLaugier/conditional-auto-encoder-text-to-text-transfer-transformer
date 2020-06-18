"""Attribute transfer tasks."""
import functools
import os

import gin
import t5
import torch
from absl import flags
from googleapiclient.discovery import build
import tensorflow as tf
import tensorflow_hub as hub
from transformers import AutoModelWithLMHead, BertForSequenceClassification, BertConfig, AutoTokenizer, AutoConfig
from t5.data import preprocessors

#import caet5.data
from caet5.data.dataset import at_preprocessor, tsv_to_dataset_fn, raw_to_tsv
from caet5.evaluation.metrics import bleu, sentence_similarity
from caet5.evaluation.metrics_utils import setup_parametric_evaluator

from caet5.data.utils import TaskRegistry_ll

FLAGS = flags.FLAGS

#TaskRegistry_ll = caet5.data.utils.TaskRegistry_ll

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS

# Automatic metrics
metric_fns = []
## Content preservation
### BLEU
if "BLEU" in FLAGS.metrics:
    metric_fns.append(bleu)

### Similarity
if "SIM" in FLAGS.metrics:
    module_url = FLAGS.use_module_url
    pretrained_sentence_similarity_model = hub.Module(module_url)
    metric_fns.append(functools.partial(sentence_similarity,
                                        sentence_similarity_model=pretrained_sentence_similarity_model))

## Attribute transfer and fluency
if "ACC" in FLAGS.metrics or "PPL" in FLAGS.metrics:
    gcs_service = build('storage', 'v1')
    gin.external_configurable(AutoTokenizer.from_pretrained)
    gin.external_configurable(BertConfig.from_pretrained)
    gin.external_configurable(BertForSequenceClassification.from_pretrained)
    gin.external_configurable(AutoConfig.from_pretrained)
    gin.external_configurable(AutoModelWithLMHead.from_config)

    for metric in ["acc", "ppl"]:
        with gin.config_scope(metric):
            setup_parametric_metric = functools.partial(setup_parametric_evaluator,
                                                        base_dir=FLAGS.base_dir,
                                                        bucket=FLAGS.bucket,
                                                        gcs_service=gcs_service)

# ======================== Processed Civil Comments ==================================
task_name = "processed_cctk"

splits = ["train", "validation", "test"]

if "ACC" in FLAGS.metrics:
    with gin.config_scope("acc/%s" % task_name):
        metric_fns.append(setup_parametric_metric(task=task_name, map_location=torch.device('cpu')))

if "PPL" in FLAGS.metrics:
    with gin.config_scope("ppl/%s" % task_name):
        metric_fns.append(setup_parametric_metric(task=task_name))

output_features = ["inputs", "targets", "attribute", "codeprefixedtargets", "controlcode"]

if FLAGS.data_dir_name:
    data_dir = os.path.join(FLAGS.base_dir, FLAGS.data_dir_name)
else:
    data_dir = os.path.join(FLAGS.base_dir, "data_tsv_%s" % task_name)

dataset_tsv_path = {
            "train": os.path.join(data_dir, "%s-train.tsv" % task_name.lower()),
            "validation": os.path.join(data_dir, "%s-toxic-validation.tsv" % task_name.lower()),
            "test": os.path.join(data_dir, "%s-toxic-test.tsv" % task_name.lower())
        }

tsvs_exist = [tf.io.gfile.exists(dataset_tsv_path[split]) for split in splits]

for i, tsv_exists in enumerate(tsvs_exist):
    split = splits[i]
    if not tsv_exists:
        tf.compat.v1.logging.info("Generating TSV for the %s split." % split)
        mode = "r"
        ext = ["nontoxic", "toxic"]
        dataset_raw_dir = os.path.join(FLAGS.base_dir, FLAGS.data_raw_dir_name)
        in_fnames = [(1, os.path.join(dataset_raw_dir, "%s.%s" % (split, ext[1])))]
        if split == "train":
            in_fnames.append((0, os.path.join(dataset_raw_dir, "%s.%s" % (split, ext[0]))))

        raw_to_tsv(in_fnames, dataset_tsv_path[split], mode=mode)

        tf.compat.v1.logging.info("TSV for the %s split generated." % split)

task_kwargs = {"dataset_fn": functools.partial(tsv_to_dataset_fn, dataset_tsv_path=dataset_tsv_path)}

TaskRegistry_ll.add(
        task_name,
        splits=splits,
        text_preprocessor=[at_preprocessor],
        sentencepiece_model_path=DEFAULT_SPM_PATH,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=metric_fns,
        token_preprocessor=preprocessors.unsupervised,
        output_features=output_features,
        **task_kwargs
    )