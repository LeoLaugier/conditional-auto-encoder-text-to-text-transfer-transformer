import functools
import gin
import tensorflow as tf
import torch
from torch.utils.data import Dataset


def raw_to_tsv(in_fnames, out_fname, mode="r"): # TODO remove mode, set mode="r" by default
  with tf.io.gfile.GFile(out_fname, "w") as outfile:
    for attribute, in_fname in in_fnames:
      with tf.io.gfile.GFile(in_fname, mode) as infile:
        sentences = infile.readlines()
        for sentence in sentences:
          sentence = sentence.rstrip()
          if mode == "rb": # TODO remove this statement
            sentence = sentence.decode("utf-8") # TODO remove
          sentence = sentence.replace("\t", "\\t")
          outfile.write("%s\t%s\n" % (sentence, str(attribute)))


def tsv_to_dataset_fn(split, shuffle_files=False, dataset_tsv_path=None):
    # We only have one file for each split.
    del shuffle_files

    # Load lines from the text file as examples.
    ds = tf.data.TextLineDataset(dataset_tsv_path[split])
    ds = ds.map(
        functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                          field_delim="\t", use_quote_delim=False),
        num_parallel_calls=tf.data.experimental.AUTOTUNE)

    ds = ds.map(lambda *ex: dict(zip(["text", "attribute"], ex)))
    return ds


@gin.configurable()
def at_preprocessor(ds, attribute_processing_fn, attribute_name="attribute", attribute_bit=False,
                    input_prefix_attributes=None, target_prefix_attributes=None, control_codes=None):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    #text = tf.strings.regex_replace(text, br"\\n", b"\n")
    #text = tf.strings.regex_replace(text, br"\\t", b"\t")
    text = tf.strings.regex_replace(text, r"\\n", "\n")
    text = tf.strings.regex_replace(text, r"\\t", "\t")
    # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")

    return text

  def to_inputs_and_targets(ex):
    """
    Map {"text": ..., [...], "[attribute]": ...} ->
        {"inputs": ..., ["attribute": ..., "codeprefixedtargets": ..., "controlcode": ...,] "targets": ...}.
    """
    attribute = attribute_processing_fn(ex, attribute_name)

    if input_prefix_attributes is None:
      inputs = normalize_text(ex["text"])
    else:
      for i in range(len(input_prefix_attributes)):
        if tf.math.equal(attribute, i):
          inputs = tf.strings.join([input_prefix_attributes[i], normalize_text(ex["text"])])

    targets = normalize_text(ex["text"])

    ex_processed = {"inputs": inputs, "targets": targets}

    if attribute_bit:
      ex_processed["attribute"] = tf.expand_dims(attribute + 1, 0)  # +1 because 0 considered as padding so attributes
                                                                    # are in [1; num_attributes + 1]

    if target_prefix_attributes is not None:
      codeprefixedtargets = tf.strings.join(["", ""])
      controlcode = tf.strings.join(["", ""])
      for i in range(len(target_prefix_attributes)):
        if tf.math.equal(attribute, i):
          codeprefixedtargets = tf.strings.join([target_prefix_attributes[i],
                                                 normalize_text(ex["text"])])  # teacher forcing
          controlcode = tf.strings.join([control_codes[i], ""])  # no teacher forcing

      ex_processed["codeprefixedtargets"] = codeprefixedtargets
      ex_processed["controlcode"] = controlcode

    return ex_processed

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)

@gin.configurable()
def attribute_processing_tsv(ex, attribute_name):
  return tf.strings.to_number(ex[attribute_name], tf.int32)

@gin.configurable()
def attribute_processing_tfds(ex, attribute_name):  # attribute_name = "toxicity" for tfds CCTK
  return tf.dtypes.cast(tf.round(ex[attribute_name]), tf.int32)


def process_attribute(dataset, mode="train"):
  def map_fn(x):
    attribute = x["attribute"]
    inputs = x["inputs"]
    inputs_padding = tf.cast(tf.not_equal(inputs, 0), tf.int32)

    indices = None
    if mode == "train":
      inputs_segmentation = x["inputs_segmentation"]
      indices = inputs_segmentation - inputs_padding
    elif mode == "eval":
      indices = tf.zeros_like(inputs)
    elif mode == "infer":
      indices = tf.zeros_like(inputs)
      attribute = tf.expand_dims(tf.strings.to_number(attribute, out_type=tf.int32), 0)

    processed_attribute = tf.gather(attribute, indices)
    x["attribute"] = processed_attribute * inputs_padding
    return x

  dataset = dataset.map(map_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)
  return dataset


def raw_to_fasttext_input(in_fname_pos, in_fname_neg, out_fname):
  with tf.io.gfile.GFile(in_fname_pos, "rb") as infile_pos,\
       tf.io.gfile.GFile(in_fname_neg, "rb") as infile_neg,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    pos_sentences  = infile_pos.readlines()
    for sentence in pos_sentences:
      sentence = sentence.rstrip()
      # sentence = sentence.decode("utf-8")
      # outfile.write(sentence+"\t"+"1\n")
      outfile.write("__label__%s %s\n" % ("1", sentence.decode("utf-8")))
    neg_sentences  = infile_neg.readlines()
    for sentence in neg_sentences:
      sentence = sentence.rstrip()
      outfile.write("__label__%s %s\n" % ("0", sentence.decode("utf-8")))


class MyDataset(Dataset):
  def __init__(self, tokenizer, prediction_list, block_size):
    batch_encoding = tokenizer.batch_encode_plus(prediction_list, add_special_tokens=True, max_length=block_size)
    self.examples = batch_encoding["input_ids"]

  def __len__(self):
    return len(self.examples)

  def __getitem__(self, i) -> torch.Tensor:
    return torch.tensor(self.examples[i], dtype=torch.long)