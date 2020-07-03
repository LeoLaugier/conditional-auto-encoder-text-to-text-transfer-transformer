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
    if k == "attribute" or k == "controlcode" or k not in feature_keys:
      return v
    return tf.concat([v[0:-1], tf.clip_by_value(v[-1:], 0, 1)], axis=0)
  return dataset.map(
      lambda ex: {k: _ensure_eos(k, v) for k, v in ex.items()},
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


def shift_decoder_output_fn(dataset, left_pad_amts=None, feature_keys=None):
  if not left_pad_amts:
    left_pad_amts = [0, 0]

  def _shift_decoder_output(k, t, left_pad_amt):
    if k != "targets":
      return t
    return tf.pad(t, [(left_pad_amt, 0)] + [(0, 0)] * (len(t.shape) - 1))

  def map_shift_decoder_output(x):
    attribute = x["attribute"][0]
    shifted = None
    for i in range(len(left_pad_amts)):
        if tf.equal(attribute, i+1):
            shifted = {k: _shift_decoder_output(k, t, left_pad_amts[i]) for k, t in x.items()}

    return shifted

  return dataset.map(
      lambda x: map_shift_decoder_output(x),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)


@gin.configurable
def pack_or_pad_ll(dataset, length, pack=True, feature_keys=None, ensure_eos=False, shift_decoder_output=False,
                   target_prefix_attributes=None, tokenizer=None):
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
    left_pad_amts = [len(tokenizer.encode(target_prefix_attribute)) - 1 for target_prefix_attribute in
                     target_prefix_attributes]
    dataset = shift_decoder_output_fn(dataset, left_pad_amts=left_pad_amts, feature_keys=feature_keys)
  if pack:
    dataset = pack_dataset(dataset, length=length, keys=feature_keys)
  # Pad/trim length of each example to length.
  dataset = trim_and_pad_dataset(
      dataset, length=length, feature_keys=feature_keys)
  if ensure_eos:
    dataset = ensure_dataset_eos_ll(dataset, feature_keys)
  return dataset