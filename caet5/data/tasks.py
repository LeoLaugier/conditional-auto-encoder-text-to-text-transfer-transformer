"""Attribute transfer tasks."""
import functools
import os

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
from caet5.evaluation.metrics import bleu, sentence_similarity, bert_attribute_accuracy_batch, gpt_perplexity_batch_280
from caet5.evaluation.metrics_utils import setup_parametric_evaluator, load_finetuned_transformer

from caet5.data.utils import TaskRegistry_ll, MixtureRegistry_ll

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

    if "ACC" in FLAGS.metrics:
        setup_acc_parametric_metric = functools.partial(setup_parametric_evaluator,
                                                        eval_fn=bert_attribute_accuracy_batch,
                                                        evaluator_name="Fine-tuned attribute classifier",
                                                        metric_name="acc",
                                                        base_dir=FLAGS.base_dir,
                                                        bucket=FLAGS.bucket,
                                                        gcs_service=gcs_service)
    if "PPL" in FLAGS.metrics:
        setup_ppl_parametric_metric = functools.partial(setup_parametric_evaluator,
                                                        eval_fn=gpt_perplexity_batch_280,
                                                        evaluator_name="Fine-tuned language model",
                                                        metric_name="ppl",
                                                        base_dir=FLAGS.base_dir,
                                                        bucket=FLAGS.bucket,
                                                        gcs_service=gcs_service)


# ======================== Processed Civil Comments ==================================
task_name = "processed_cctk"

splits_raw = ["train", "dev", "test"]
splits = ["train", "validation", "test"]


if "ACC" in FLAGS.metrics:
    load_pretrained_acc_fn = functools.partial(BertForSequenceClassification.from_pretrained,
                                               "bert-base-uncased")
    load_config_acc_fn = functools.partial(BertConfig.from_pretrained,
                                           num_labels=1)

    metric_fns.append(setup_acc_parametric_metric(model_architecture="bert",
                                                  task=task_name,
                                                  ext="pt",
                                                  load_parametric_model_fn=load_finetuned_transformer,
                                                  pretrained_model_name_or_path="bert-base-uncased",
                                                  load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                  load_config_fn=load_config_acc_fn,
                                                  load_pretrained_fn=load_pretrained_acc_fn,
                                                  batch_size=32,
                                                  map_location=torch.device('cpu')))

if "PPL" in FLAGS.metrics:
    metric_fns.append(setup_ppl_parametric_metric(model_filename="gpt2_ppl_cctk.pt",
                                                  load_parametric_model_fn=load_finetuned_transformer,
                                                  pretrained_model_name_or_path="gpt2",
                                                  load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                  load_config_fn=AutoConfig.from_pretrained,
                                                  load_pretrained_fn=AutoModelWithLMHead.from_config,
                                                  batch_size=8,
                                                  block_size=256))

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
    split_raw = splits_raw[i]
    if not tsv_exists:
        tf.compat.v1.logging.info("Generating TSV for the %s split." % split)
        mode = "r"
        ext = ["nontoxic", "toxic"]
        dataset_raw_dir = os.path.join(FLAGS.base_dir, FLAGS.data_raw_dir_name)
        in_fnames = [(1, os.path.join(dataset_raw_dir, "%s.%s" % (split_raw, ext[1])))]
        if split == "train":
            in_fnames.append((0, os.path.join(dataset_raw_dir, "%s.%s" % (split_raw, ext[0]))))

        raw_to_tsv(in_fnames, dataset_tsv_path[split], mode=mode)

        tf.compat.v1.logging.info("TSV for the %s split generated." % split)


def dataset_fn(split, shuffle_files=False):
    fn = functools.partial(tsv_to_dataset_fn, dataset_tsv_path=dataset_tsv_path)
    return fn(split, shuffle_files=shuffle_files)
#fn = functools.partial(tsv_to_dataset_fn, dataset_tsv_path=dataset_tsv_path)
#fn.__name__ = ""

#task_kwargs = {"dataset_fn": functools.partial(tsv_to_dataset_fn, dataset_tsv_path=dataset_tsv_path).func}

task_kwargs = {"dataset_fn": dataset_fn}

#task_kwargs = {"dataset_fn": fn}

TaskRegistry_ll.add(
        task_name,
        splits=splits,
        text_preprocessor=[at_preprocessor],
        sentencepiece_model_path=DEFAULT_SPM_PATH,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=metric_fns,
        token_preprocessor=preprocessors.unsupervised,
        output_features=output_features,
        **task_kwargs)

# Need a mixture because of the error TypeError: <dtype: 'string'> is not a supported TPU infeed type.
# described here: https://github.com/google-research/text-to-text-transfer-transformer/issues/291
MixtureRegistry_ll.add(
        "mixture_%s" % task_name,
        [task_name],
        default_rate=1.0
    )


# ======================== Yelp reviews ==================================
task_name = "yelp"

splits_raw = ["train", "dev", "test"]
splits = ["train", "validation", "test"]


if "ACC" in FLAGS.metrics:
    load_pretrained_acc_fn = functools.partial(BertForSequenceClassification.from_pretrained,
                                               "bert-base-uncased")
    load_config_acc_fn = functools.partial(BertConfig.from_pretrained,
                                           num_labels=1)

    metric_fns.append(setup_acc_parametric_metric(model_architecture="bert",
                                                  task=task_name,
                                                  ext="pt",
                                                  load_parametric_model_fn=load_finetuned_transformer,
                                                  pretrained_model_name_or_path="bert-base-uncased",
                                                  load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                  load_config_fn=load_config_acc_fn,
                                                  load_pretrained_fn=load_pretrained_acc_fn,
                                                  batch_size=32,
                                                  map_location=torch.device('cpu')))

if "PPL" in FLAGS.metrics:
    metric_fns.append(setup_ppl_parametric_metric(model_filename="gpt2_ppl_yelp.pt",
                                                  load_parametric_model_fn=load_finetuned_transformer,
                                                  pretrained_model_name_or_path="gpt2",
                                                  load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                  load_config_fn=AutoConfig.from_pretrained,
                                                  load_pretrained_fn=AutoModelWithLMHead.from_config,
                                                  batch_size=8,
                                                  block_size=256))

output_features = ["inputs", "targets", "attribute", "codeprefixedtargets", "controlcode"]

if FLAGS.data_dir_name:
    data_dir = os.path.join(FLAGS.base_dir, FLAGS.data_dir_name)
else:
    data_dir = os.path.join(FLAGS.base_dir, "data_tsv_%s" % task_name)

dataset_tsv_path = {
            "train": os.path.join(data_dir, "%s-train.tsv" % task_name.lower()),
            "validation": os.path.join(data_dir, "%s-validation.tsv" % task_name.lower()),
            "test": os.path.join(data_dir, "%s-test.tsv" % task_name.lower())
        }

tsvs_exist = [tf.io.gfile.exists(dataset_tsv_path[split]) for split in splits]

for i, tsv_exists in enumerate(tsvs_exist):
    split = splits[i]
    split_raw = splits_raw[i]
    if not tsv_exists:
        tf.compat.v1.logging.info("Generating TSV for the %s split." % split)
        ext = ["neg", "pos"]
        dataset_raw_dir = os.path.join(FLAGS.base_dir, FLAGS.data_raw_dir_name)
        in_fnames = [(1, os.path.join(dataset_raw_dir, "%s.%s" % (split_raw, ext[1]))),
                     (0, os.path.join(dataset_raw_dir, "%s.%s" % (split_raw, ext[0])))]

        raw_to_tsv(in_fnames, dataset_tsv_path[split])

        tf.compat.v1.logging.info("TSV for the %s split generated." % split)


def dataset_fn(split, shuffle_files=False):
    fn = functools.partial(tsv_to_dataset_fn, dataset_tsv_path=dataset_tsv_path)
    return fn(split, shuffle_files=shuffle_files)

task_kwargs = {"dataset_fn": dataset_fn}

MixtureRegistry_ll.add(
        "mixture_%s" % task_name,
        [task_name],
        default_rate=1.0
    )