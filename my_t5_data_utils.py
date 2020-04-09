import functools

from absl import logging
import t5
from t5.data.utils import _DEFAULT_FEATURE_KEYS, _VALID_TASK_NAME_REGEX, _INFO_FILENAME, _STATS_FILENAME, \
    _TFRECORD_PREFIX, _MAX_EXAMPLES_TO_MEM_CACHE, _SHUFFLE_BUFFER_SIZE, _TFDS_DATA_DIR_OVERRIDE, \
    _GLOBAL_CACHE_DIRECTORIES, encode_string_features, DatasetProviderRegistry, TfdsTask, DatasetProviderBase
# from t5.data.utils import *
import tensorflow as tf
import tensorflow_datasets as tfds


def balance_fn(x, balance_rate=0):
    if x["toxicity"] <= 0.5:
        draw = tf.random.uniform([], maxval=1)
        return draw < balance_rate
    else:
        return True


# Need to redefine TfdsTask because Task was not made to process non string inputs
class Task_ll(t5.data.utils.Task):
    def __init__(self, *task_args, balance_styles=False, balance_rate=0, **task_kwargs):
        super().__init__(*task_args, **task_kwargs)
        self.denoise = task_kwargs["token_preprocessor"]
        self.balance_styles = balance_styles
        self.balance_rate = balance_rate

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
        types = tf.compat.v1.data.get_output_types(dataset)
        shapes = tf.compat.v1.data.get_output_shapes(dataset)
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
            return tf.concat([v[:sequence_length[feat] - 1], [1]], axis=0)

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
          mode: string, "train" or "eval".
        Returns:
          A mixed tf.data.Dataset.
        """
        if use_cached:
            ds = self._get_cached_dataset(split, shuffle)
        else:
            ds = self._dataset_fn(split=split, shuffle_files=shuffle)
            if self.balance_styles and mode =="train":
                ds = ds.filter(functools.partial(balance_fn, balance_rate = self.balance_rate))
            ds = self.preprocess_text_ll(ds)
            # Tokenize
            ds = encode_string_features(
                ds, self.get_vocabulary(), keys=self.output_features,
                copy_plaintext=True)

        if (not use_cached and self.num_input_examples(split) and
                self.num_input_examples(split) < _MAX_EXAMPLES_TO_MEM_CACHE):
            ds = ds.cache()

        # Post tokenization processing.
        if (self.denoise and mode == "train") or (not self.denoise):
            ds = self.preprocess_tokens_ll(ds, sequence_length)

        if self.denoise and mode == "eval":
            def _trim_and_append_eos(feat, v):
                if feat == "style" or feat == "codeprefix" or feat not in self.output_features:
                    return v
                return tf.concat([v[:sequence_length[feat] - 1], [1]], axis=0)

            return ds.map(
                lambda ex: {k: _trim_and_append_eos(k, v) for k, v in ex.items()},
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

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


class Mixture_ll(DatasetProviderBase):
  """Class for mixing multiple tasks."""

  def __init__(self, tasks, default_rate=None):
    """Mixture constructor.
    A mixture specifies a set of tasks with associated mixing rates.
    Mixing happens on preprocessed tokenized examples.
    The mixing rates represent relative numbers of examples to use from their
    associated tasks.  Setting the mixing rates to be equal to the numbers of
    examples in the tasks will result in each task going through an epoch in
    about the same amount of time - i.e. all examples are sampled equally across
    all tasks.
    Rates can be expressed either as absolute numbers or as functions that
    receive the Task as an argument.
    Args:
      tasks: a list where each element is either a string (task name) or a
        pair whose first element is the task name and whose second element
        is either a float (rate) or a function from Task to float.
      default_rate: a float or a function from Task to float. This specifies the
        default rate if rates are not provided in the `tasks` argument.
    """
    self._task_to_rate = {}
    self._tasks = []
    for t in tasks:
      if isinstance(t, str):
        task_name = t
        rate = default_rate
        if default_rate is None:
          raise ValueError("need a rate for each task")
      else:
        task_name, rate = t
      self._tasks.append(TaskRegistry_ll.get(task_name))
      self._task_to_rate[task_name] = rate
    if len(set(tuple(t.output_features) for t in self._tasks)) != 1:
      raise ValueError(
          "All Tasks in a Mixture must have the same output features."
      )
    if len(set(t.sentencepiece_model_path for t in self._tasks)) != 1:
      raise ValueError(
          "All Tasks in a Mixture must have the same sentencepiece_model_path."
      )

  @property
  def tasks(self):
    return self._tasks

  def get_rate(self, task):
    rate = self._task_to_rate[task.name]
    return float(rate(task) if callable(rate) else rate)

  def num_input_examples(self, split):
    return sum(t.num_input_examples(split) for t in self.tasks)

  @property
  def output_features(self):
    # We require all tasks to have the same output_features in __init__
    # so we can just get the output_features for the 0th task
    return self._tasks[0].output_features

  @property
  def sentencepiece_model_path(self):
    # We require all tasks to have the same sentencepiece_model_path in __init__
    # so we can just get the sentencepiece_model_path for the first task
    return self._tasks[0].sentencepiece_model_path

  def get_vocabulary(self):
    """Returns a SentencePieceVocabulary object using the Tasks' model."""
    return self._tasks[0].get_vocabulary()

  def get_dataset(
      self,
      sequence_length,
      split=tfds.Split.TRAIN,
      use_cached=False,
      shuffle=True,
      compute_stats_empirically=False,
  ):
    """Returns the dataset of mixed tasks using the object-specified rates.
    Args:
      sequence_length: dict mapping feature key to int length for that feature
      split: string, the split to return for all tasks.
      use_cached: bool, whether to use the cached dataset instead of processing
        it on the fly. Defaults to True.
      shuffle: bool, whether to shuffle the dataset.  Only used when generating
        on the fly (use_cached=False).
      compute_stats_empirically: a boolean - does not work on TPU
    """
    tasks = []
    for task in self.tasks:
      if split not in task.splits:
        logging.info(
            "Task %s has no '%s' split, skipping.", task.name, split
        )
        continue
      tasks.append(task)
    if not tasks:
      raise ValueError("No datasets have a '{}' split".format(split))
    def filter_features(ex):
      return {k: v for k, v in ex.items() if k in self.output_features}
    datasets = [
        task.get_dataset(sequence_length, split, use_cached, shuffle=shuffle)  # pylint:disable=g-complex-comprehension
        .repeat()
        .map(filter_features, num_parallel_calls=tf.data.experimental.AUTOTUNE)
        for task in tasks]
    rates = [self.get_rate(task) for task in tasks]
    # Sample from the dataset with the rates rates
    dataset = tf.data.experimental.sample_from_datasets(datasets, rates)
    if split == "train" and use_cached:
      _log_mixing_proportions(tasks, datasets, rates, dataset, sequence_length,
                              compute_stats_empirically)
    return dataset

class MixtureRegistry_ll(DatasetProviderRegistry):
  _REGISTRY = {}
  _PROVIDER_TYPE = Mixture_ll

  @classmethod
  def add(cls, name, tasks, default_rate=None):
    super(MixtureRegistry_ll, cls).add(name, Mixture_ll, tasks, default_rate)


def get_mixture_or_task_ll(task_or_mixture_name):
  """Return the Task or Mixture from the appropriate registry."""
  mixtures = MixtureRegistry_ll.names()
  tasks = TaskRegistry_ll.names()
  if task_or_mixture_name in mixtures:
    if task_or_mixture_name in tasks:
      logging.warning("%s is both a Task and a Mixture, returning Mixture",
                      task_or_mixture_name)
    return MixtureRegistry_ll.get(task_or_mixture_name)
  if task_or_mixture_name in tasks:
    return TaskRegistry_ll.get(task_or_mixture_name)
  else:
    raise ValueError("No Task or Mixture found with name: %s" %
                     task_or_mixture_name)