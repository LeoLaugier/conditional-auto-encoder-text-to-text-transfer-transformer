"""Attribute transfer tasks."""
import functools

import torch
from absl import flags
from googleapiclient.discovery import build
import tensorflow as tf
import tensorflow_hub as hub
from transformers import AutoModelWithLMHead, BertForSequenceClassification, BertConfig, AutoTokenizer, AutoConfig
from t5.data import preprocessors

import caet5.data
from caet5.data.dataset import at_preprocessor
from caet5.data.preprocessors import denoise
from caet5.evaluation.metrics import bleu, sentence_similarity, bert_attribute_accuracy_batch, gpt_perplexity_batch_280
from caet5.evaluation.metrics_utils import setup_parametric_evaluator, load_finetuned_transformer

FLAGS = flags.FLAGS

TaskRegistry_ll = caet5.data.TaskRegistry_ll

DEFAULT_SPM_PATH = "gs://t5-data/vocabs/cc_all.32000/sentencepiece.model"  # GCS

# Automatic metrics
metric_fns = []
## Content preservation
### BLEU
if "BLEU" in FLAGS.metrics:
    metric_fns.append(bleu)

### Similarity
if "SIM" in FLAGS.metrics:
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    pretrained_sentence_similarity_model = hub.Module(module_url)
    metric_fns.append(functools.partial(sentence_similarity,
                                        sentence_similarity_model=pretrained_sentence_similarity_model))

if "ACC" in FLAGS.metrics or "PPL" in FLAGS.metrics:
    if FLAGS.evaluate_with_transformers:
        ext = "pt"
        load_parametric_model_fn = load_finetuned_transformer
        gcs_service = build('storage', 'v1')

        ## Attribute transfer
        ### Accuracy
        load_config_acc_fn = functools.partial(BertConfig.from_pretrained, num_labels=1)
        load_pretrained_acc_fn = functools.partial(BertForSequenceClassification.from_pretrained,
                                                   pretrained_acc_name_or_path=FLAGS.pretrained_acc_name_or_path)
        setup_parametric_acc_metric = functools.partial(setup_parametric_evaluator,
                                                        eval_fn=bert_attribute_accuracy_batch,
                                                        evaluator_name="Fine-tuned attribute classifier",
                                                        model_filename=FLAGS.parametric_acc_filename,
                                                        model_architecture=FLAGS.pretrained_acc_architecture,
                                                        metric_name="acc",
                                                        ext=ext,
                                                        base_dir=FLAGS.base_dir,
                                                        bucket=FLAGS.bucket,
                                                        load_parametric_model_fn=load_parametric_model_fn,
                                                        gcs_service=gcs_service,
                                                        pretrained_model_name_or_path=FLAGS.pretrained_acc_name_or_path,
                                                        load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                        load_config_fn=load_config_acc_fn,
                                                        load_pretrained_fn=load_pretrained_acc_fn,
                                                        map_location=torch.device('cpu'),
                                                        batch_size=FLAGS.acc_eval_batch_size)

        ## Fluency
        ### Perplexity
        setup_parametric_ppl_metric = functools.partial(setup_parametric_evaluator,
                                                        eval_fn=gpt_perplexity_batch_280,
                                                        evaluator_name="Fine-tuned language model",
                                                        model_filename=FLAGS.finetuned_ppl_filename,
                                                        model_architecture=FLAGS.pretrained_ppl_architecture,
                                                        metric_name="ppl",
                                                        ext=ext,
                                                        base_dir=FLAGS.base_dir,
                                                        bucket=FLAGS.bucket,
                                                        load_parametric_model_fn=load_parametric_model_fn,
                                                        gcs_service=gcs_service,
                                                        pretrained_model_name_or_path=FLAGS.pretrained_ppl_name_or_path,
                                                        load_tokenizer_fn=AutoTokenizer.from_pretrained,
                                                        load_config_fn=AutoConfig.from_pretrained,
                                                        load_pretrained_fn=AutoModelWithLMHead.from_config,
                                                        batch_size=FLAGS.ppl_eval_batch_size,
                                                        block_size=FLAGS.ppl_eval_block_size)

    else:
        raise ValueError("No other parametric evaluators implemented.")

def attribute_processing_tsv(ex, attribute_name):
    return tf.strings.to_number(ex[attribute_name], tf.int32)

def attribute_processing_tfds(ex, attribute_name): # attribute_name = "toxicity" for tfds CCTK
    return tf.dtypes.cast(tf.round(ex[attribute_name]), tf.int32)

if FLAGS.noise_fns:
    token_preprocessor = []
    for inputs_fn, noise_density in FLAGS.noise_fns:
        token_preprocessor.append(functools.partial(denoise, noise_density=noise_density,
                                                    noise_mask_fn=t5.data.preprocessors.iid_noise_mask,
                                                    inputs_fn=inputs_fn,
                                                    targets_fn=None))

else:
    token_preprocessor = None

output_features = ["inputs", "targets"]
if FLAGS.attribute_bit:
    output_features.append("attribute")

if FLAGS.target_prefix_attributes:
    output_features.append("codeprefixedtargets")
    output_features.append("controlcode")

# ======================== Processed Civil Comments ==================================
task_name = "processed_cctk"

splits = ["train", "validation", "test"]

text_preprocessor = functools.partial(at_preprocessor,
                                      attribute_processing_fn=attribute_processing_tsv,
                                      attribute_bit=FLAGS.attribute_bit,
                                      input_prefix_attributes=FLAGS.input_prefix_attributes,
                                      target_prefix_attributes=FLAGS.target_prefix_attributes,
                                      control_codes=FLAGS.control_codes)

if "ACC" in FLAGS.metrics:
    metric_fns.append(setup_parametric_acc_metric(task=task_name))

if "PPL" in FLAGS.metrics:
    metric_fns.append(setup_parametric_ppl_metric(task=task_name))

TaskRegistry_ll.add(
        task_name,
        splits=splits,
        text_preprocessor=[text_preprocessor],
        sentencepiece_model_path=DEFAULT_SPM_PATH,
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=metric_fns,
        token_preprocessor=preprocessors.unsupervised,
        output_features=output_features,
        **task_kwargs
    )