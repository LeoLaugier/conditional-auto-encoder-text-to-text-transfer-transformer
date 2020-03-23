# Need to redifine TfdsTask because Task was not made to process non string inputs
import t5
from t5.data.utils import _DEFAULT_FEATURE_KEYS, _VALID_TASK_NAME_REGEX, _INFO_FILENAME, _STATS_FILENAME, _TFRECORD_PREFIX, _MAX_EXAMPLES_TO_MEM_CACHE, _SHUFFLE_BUFFER_SIZE, _TFDS_DATA_DIR_OVERRIDE, _GLOBAL_CACHE_DIRECTORIES
from t5.data.utils import *
import tensorflow as tf
from main import BALANCE_RATE, BALANCE_STYLES, DENOISE

def balance_fn(x, balance_rate=BALANCE_RATE):
    if x["toxicity"] <= 0.5:
        draw = tf.random.uniform([], maxval=1)
        return draw < balance_rate
    else:
        return True

class Task_ll(t5.data.utils.Task):
  def _validate_dataset_ll(
      self,
      dataset,
      expected_output_type,
      expected_output_rank,
      error_label,
      ensure_no_eos=False):
    """Validates properties of a tf.data.Dataset, raising Exceptions if needed.
    Args:
      dataset: a tf.data.Dataset to validate.
      expected_output_type: a tf.dtype, the expected type of the model features.
      expected_output_rank: an int, the expected rank of the model features.
      error_label: a string, an identifier for the previous processing step to
        report in raised ValueErrors.
      ensure_no_eos: a bool, whether or not to verify that the model features
        contain no EOS tokens.
    Returns:TaskRegistry
      a validated tf.data.Dataset.
    """
    types = tf.data.get_output_types(dataset)
    shapes = tf.data.get_output_shapes(dataset)
    for feat in self.output_features:
      if feat not in types:
        raise ValueError(
            "Task dataset is missing expected output feature after {label}: "
            "{feat}".format(label=error_label, feat=feat))
      if feat != "style" and expected_output_type != types[feat]:
        raise ValueError(
            "Task dataset has incorrect type for feature '{feat}' after "
            "{label}: Got {actual}, expected {expected}".format(
                feat=feat, label=error_label, actual=types[feat].name,
                expected=expected_output_type.name))
      if feat != "style" and expected_output_rank != len(shapes[feat]):
        raise ValueError(
            "Task dataset has incorrect rank for feature '{feat}' after "
            "{label}: Got {actual}, expected {expected}".format(
                feat=feat, label=error_label, actual=len(shapes[feat]),
                expected=expected_output_rank))

    def _ensure_no_eos(feat, v):
      if feat == "style" or feat not in self.output_features:
        return v
      with tf.control_dependencies([
          tf.assert_none_equal(
              v, tf.constant(1, tf.int64),
              message="Feature '{feat}' unexpectedly contains EOS=1 token "
              "after {label}.".format(feat=feat, label=error_label))
      ]):
        return v
    if ensure_no_eos:
      dataset = dataset.map(
          lambda ex: {k: _ensure_no_eos(k, v) for k, v in ex.items()},
          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    return dataset


  def preprocess_text_ll(self, dataset):
    """Preprocessed text dataset."""
    dataset = self._preprocess_dataset(dataset, self._text_preprocessor)
    dataset = self._validate_dataset_ll(
        dataset, expected_output_type=tf.string, expected_output_rank=0,
        error_label="text preprocessing")
    return dataset


  def preprocess_tokens_ll(self, dataset, sequence_length):
      """Preprocesses tokenized dataset.
      Args:
        dataset: a tf.data.Dataset
        sequence_length: dict mapping feature key to int length for that feature
      Returns:
        a tf.data.Dataset
      """
      dataset = self._preprocess_dataset(
          dataset, self._token_preprocessor,
          sequence_length=sequence_length,
          vocabulary=self.get_vocabulary())
      dataset = self._validate_dataset_ll(
          dataset,
          expected_output_type=tf.int64,
          expected_output_rank=1,
          error_label="token preprocessing",
          ensure_no_eos=True)
      # Trim and append EOS=1 token to model features.
      def _trim_and_append_eos(feat, v):
        if feat == "style" or feat == "codeprefix" or feat not in self.output_features:
          return v
        return tf.concat([v[:sequence_length[feat]-1], [1]], axis=0)

      return dataset.map(
          lambda ex: {k: _trim_and_append_eos(k, v) for k, v in ex.items()},
          num_parallel_calls=tf.data.experimental.AUTOTUNE)


  def get_dataset(
      self,
      sequence_length,
      split=tfds.Split.TRAIN,
      use_cached=False,
      shuffle=True,
      shuffle_buffer_size=_SHUFFLE_BUFFER_SIZE,
      mode="train",
  ):
    """Returns a tf.data.Dataset from cache or generated on the fly.
    Args:
      sequence_length: dict mapping feature key to int length for that feature
      split: string, the split to return.
      use_cached: bool, whether to use the cached dataset instead of processing
        it on the fly. Defaults to True.
      shuffle: bool, whether to shuffle the dataset.  Only used when generating
        on the fly (use_cached=False).
      shuffle_buffer_size: an integer
    Returns:
      A mixed tf.data.Dataset.
    """
    if use_cached:
      ds = self._get_cached_dataset(split, shuffle)
    else:
      ds = self._dataset_fn(split=split, shuffle_files=shuffle)
      if BALANCE_STYLES:
        ds = ds.filter(balance_fn)
      ds = self.preprocess_text_ll(ds)
      # Tokenize
      ds = encode_string_features(
          ds, self.get_vocabulary(), keys=self.output_features,
          copy_plaintext=True)

    if (not use_cached and self.num_input_examples(split) and
        self.num_input_examples(split) < _MAX_EXAMPLES_TO_MEM_CACHE):
      ds = ds.cache()

    # Post tokenization processing.
    if (DENOISE and mode=="train") or (not DENOISE):
      ds = self.preprocess_tokens_ll(ds, sequence_length)

    if shuffle:
      # Shuffle before mixing since preprocessor can output multiple
      # (correlated) examples per input.
      ds = ds.shuffle(shuffle_buffer_size)

    return ds

class TaskRegistry_ll(DatasetProviderRegistry):
  _REGISTRY = {}
  _PROVIDER_TYPE = Task_ll

  @classmethod
  def add(cls, name, task_cls=Task_ll, **kwargs):
    super(TaskRegistry_ll, cls).add(name, task_cls, name, **kwargs)


class TfdsTask_ll(TfdsTask, Task_ll):
  pass