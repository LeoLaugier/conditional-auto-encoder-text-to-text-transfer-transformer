import functools
import gin
from absl import logging
import t5
import tensorflow as tf
import tensorflow_datasets as tfds

from caet5.data.dataset import process_attribute
import mesh_tensorflow.transformer.dataset as transformer_dataset

from mesh_tensorflow_caet5.dataset import pack_or_pad_ll
from caet5.data.utils import get_mixture_or_task_ll


@gin.configurable()
def mesh_train_dataset_fn_ll(
        mixture_or_task_name,
        sequence_length,
        vocabulary,
        batch_size,
        ensemble_inputs,
        dataset_split=tfds.Split.TRAIN,
        use_cached=False,
        group_by_attribute=False,
        attribute_embedding=False,
        attribute_num=2):
    """Returns the tf.data.Dataset for training on a given mixture.
    This uses the format required for utils.run's `train_dataset_fn` argument in
    the Mesh TF transformer standalone.
    Args:
      mixture_or_task_name: string, an identifier for a Mixture or Task in the
        appropriate registry. Must be specified via gin.
      sequence_length: dict mapping feature key to the int length for that feature
        the max sequence length.
      vocabulary: a SentencePieceVocabulary.
      dataset_split: string, which split of the dataset to load. In most cases
        this should be "train".
      use_cached: bool, whether to load the cached version of this dataset.
    Returns:
      A tf.data.Dataset of preprocessed, tokenized, and batched examples.
    """
    if not isinstance(vocabulary, t5.data.SentencePieceVocabulary):
        raise ValueError("vocabulary must be a SentencePieceVocabulary")

    mixture_or_task = get_mixture_or_task_ll(mixture_or_task_name)

    ds = mixture_or_task.get_dataset(
        sequence_length, split=dataset_split, use_cached=use_cached, shuffle=True)

    if group_by_attribute:  # TODO: Currently, we alternate deterministically batches of same attribute but it would be even better to alternate randomly
        def filter_attribute_1_fn(x):
            return tf.equal(x["attribute"][0], 1)

        def filter_attribute_2_fn(x):
            return tf.equal(x["attribute"][0], 2)

        ds_attribute_1 = ds.filter(filter_attribute_1_fn)
        ds_attribute_2 = ds.filter(filter_attribute_2_fn)

        ds2_attribute_1 = pack_or_pad_ll(
            ds_attribute_1, sequence_length, pack=False,
            feature_keys=tuple(mixture_or_task.output_features),
            ensure_eos=True)  # (not straightforward) Adapt packing so that pack=True
        ds2_attribute_2 = pack_or_pad_ll(
            ds_attribute_2, sequence_length, pack=False,
            feature_keys=tuple(mixture_or_task.output_features),
            ensure_eos=True)  # (not straightforward) Adapt packing so that pack=True

        if attribute_embedding:
            ds3_attribute_1 = process_attribute(ds2_attribute_1, mode="eval")
            ds3_attribute_2 = process_attribute(ds2_attribute_2, mode="eval")
        else:
            ds3_attribute_1 = ds2_attribute_1
            ds3_attribute_2 = ds2_attribute_2

        def f1():
            return ds3_attribute_1

        def f2():
            return ds3_attribute_2

        def interleave_map_fn(x):
            return tf.cond(tf.equal(x, 0), f1, f2)

        ds = tf.data.Dataset.range(attribute_num).interleave(
            interleave_map_fn, cycle_length=attribute_num,
            block_length=batch_size * (ensemble_inputs or 1),
            num_parallel_calls=tf.data.experimental.AUTOTUNE)

    else:
        ds = pack_or_pad_ll(
            ds, sequence_length, pack=True,
            feature_keys=tuple(mixture_or_task.output_features), ensure_eos=True)
        ds = process_attribute(ds)

    return ds


@gin.configurable()
def mesh_eval_dataset_fn_ll(
        mixture_or_task_name,
        sequence_length,
        vocabulary,
        dataset_split,
        num_eval_examples=None,
        use_cached=False,
        attribute_embedding=False):
    """Returns all tf.data.Datasets for evaluation on a given mixture.
    This uses the format required for utils.run's `eval_dataset_fn` argument in
    the Mesh TF transformer standalone.
    Args:
      mixture_or_task_name: string, an identifier for a Mixture or Task in the
        appropriate registry. Must be specified via gin.
      sequence_length: dict mapping feature key to the int length for that feature
        the max sequence length.
      vocabulary: a SentencePieceVocabulary.
      dataset_split: string, which split of the dataset to load.
      num_eval_examples: maximum number of examples per task to use for continuous
        eval. If None, use all examples.
      use_cached: bool, whether to load the cached version of this dataset.
    Returns:
      A list of mesh_tensorflow.transformer.dataset.EvalDataset tuples.
    """
    if not isinstance(vocabulary, t5.data.SentencePieceVocabulary):
        raise ValueError("vocabulary must be a SentencePieceVocabulary")

    mixture_or_task = get_mixture_or_task_ll(mixture_or_task_name)

    def _get_dataset_for_single_task(task):
        """Get a tensorflow.data.Dataset for the provided task."""
        ds = task.get_dataset(
            sequence_length, split=dataset_split,
            use_cached=use_cached, shuffle=False, mode="eval"
        )

        if attribute_embedding:
            ds = process_attribute(ds, mode="eval")

        ds = pack_or_pad_ll(
            ds, sequence_length, pack=False, feature_keys=task.output_features,
            ensure_eos=True)
        # ds = process_attribute(ds, mode="eval")
        if num_eval_examples is not None:
            ds = ds.take(num_eval_examples)
        return ds

    outputs = []

    for task in t5.data.get_subtasks(mixture_or_task):
        if dataset_split not in task.splits:
            logging.info(
                "Task %s has no '%s' split, skipping eval.", task.name, dataset_split
            )
            continue

        outputs.append(
            transformer_dataset.EvalDataset(
                task.name,
                functools.partial(_get_dataset_for_single_task, task),
                task.postprocess_fn,
                task.metric_fns,
            )
        )

    return outputs