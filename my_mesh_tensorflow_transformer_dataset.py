import gin
import tensorflow as tf
from mesh_tensorflow.transformer.dataset import pack_dataset, trim_and_pad_dataset


def ensure_dataset_eos_ll(dataset, feature_keys=None):
  """Replaces the final token of features with EOS=1 if it is not PAD=0.
  Args:
    dataset: a tf.data.Dataset
    feature_keys: (optional) list of strings, the feature names to ensure end
      with EOS or padding. Defaults to all features.
  Returns:
    a tf.data.Dataset where all specified features end with PAD=0 or EOS=1.
  """
  feature_keys = feature_keys or dataset.output_shapes.keys()
  def _ensure_eos(k, v):
    if k == "style" or k == "codeprefix" or k not in feature_keys:
      return v
    return tf.concat([v[0:-1], tf.clip_by_value(v[-1:], 0, 1)], axis=0)
  return dataset.map(
      lambda ex: {k: _ensure_eos(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


def shift_decoder_output_fn(dataset, left_pad_amt_1=0, left_pad_amt_2=0, feature_keys=None):
  def _shift_decoder_output(k, t, left_pad_amt):
    if k != "targets":
      return t
    return tf.pad(t, [(left_pad_amt, 0)] + [(0, 0)] * (len(t.shape) - 1))

  def map_shift_decoder_output(x):
    style = x["style"][0]
    if tf.equal(style, 1):
      return {k: _shift_decoder_output(k, t, left_pad_amt_1) for k, t in x.items()}
    return {k: _shift_decoder_output(k, t, left_pad_amt_2) for k, t in x.items()}

  return dataset.map(
      lambda x: map_shift_decoder_output(x),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable
def pack_or_pad_ll(dataset, length, pack=True, feature_keys=None, ensure_eos=False, shift_decoder_output=False,
                   left_pad_amt_1=0, left_pad_amt_2=0):
  """Creates a 'packed' version of a dataset or pads examples with zeros.
  If pack=True, then multiple examples concatenated to form one combined
  example with the given length.
  If pack=False, then examples are padded with zeros to 'length'.
  Args:
    dataset: a tf.data.Dataset
    length: an integer or a dict from feature-key to integer
    pack: a boolean, whether to pack (True) or pad (False).
    feature_keys: (optional) list of strings, the feature names to limit
      packing or padding to. Packing will filter out other features whereas
      padding will pass them through unchanged. Defaults to all features.
    ensure_eos: a boolean, whether to replace the final token with EOS=1 if it
      is not PAD=0.
  Returns:
    a tf.data.Dataset where all features have fixed shape [length].
  """
  feature_keys = feature_keys or list(dataset.output_shapes.keys())
  if shift_decoder_output:
    dataset = shift_decoder_output_fn(dataset, left_pad_amt_1=left_pad_amt_1, left_pad_amt_2=left_pad_amt_2,
                                   feature_keys=feature_keys)
  if pack:
    dataset = pack_dataset(dataset, length=length, keys=feature_keys)
  # Pad/trim length of each example to length.
  dataset = trim_and_pad_dataset(
      dataset, length=length, feature_keys=feature_keys)
  if ensure_eos:
    dataset = ensure_dataset_eos_ll(dataset, feature_keys)
  return dataset