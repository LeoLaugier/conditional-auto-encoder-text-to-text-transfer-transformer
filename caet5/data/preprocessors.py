import gin
import tensorflow as tf

@gin.configurable()
def denoise(dataset,
            vocabulary,
            noise_density=gin.REQUIRED,  #1.0, #0.15
            noise_mask_fn=gin.REQUIRED,  # t5.data.preprocessors.iid_noise_mask,
            inputs_fn=gin.REQUIRED,  # t5.data.preprocessors.permute_noise_tokens,  # noise_token_to_random_token_or_sentinel, #  noise_token_to_sentinel,
            targets_fn=None,
            style_bit=True,
            style_dependant_prefix_target=True,
            **unused_kwargs):
  """Gin-configurable token preprocessor for self-supervised denoising tasks.
  This function takes a dataset containing "targets" sequences,
  and turns each sequence into a dictionary containing:
  {
    "inputs": noisy version of the original sequence
    "targets": the full original sequence or missing parts of original sequence
  }
  In particular, for each sequence, we choose a boolean noise_mask identifying
  which tokens in the sequence to corrupt, as defined by the given
  noise_mask_fn.
  Given the sequence and the noise mask, we generate the inputs and targets
  using the given inputs_fn and targets_fn respectively.
  The self-supervised tasks vary along these axes:
    - noise_density: What fraction of the tokens to select as noise
    - noise_mask_fn: What pattern should the noise mask follow
        (iid, regular segments, etc.)
    - inputs_fn: How to apply the noise
        (drop noise tokens, replace with sentinels, etc.)
    - targets_fn: How to represent the output
        (full sequence, only non-noise tokens, etc.)
  Note: Some functionality has been deleted, which we may or may not want to
  restore at a later date.  The code for this functionality can be found in
  the deleted code for this CL.  In particular:
    - mixture of masking and random replacement
    - task labels prepended to the inputs
  Args:
    dataset: A tf.data.Dataset to process.
    vocabulary: A mesh_tensorflow.transformer.vocabulary.Vocabulary.
    noise_density: a float
    noise_mask_fn: a function from (length, noise_density) -> boolean mask
    inputs_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
    targets_fn: a function from (tokens, noise_mask, vocabulary) -> tokens
  Returns:
    A preprocessed tf.data.Dataset.
  """
  def my_fn(features):
    tokens = features['targets']
    noise_mask = noise_mask_fn(tf.size(tokens), noise_density)
    inputs = inputs_fn(tokens, noise_mask, vocabulary)
    if targets_fn:
      targets = targets_fn(tokens, noise_mask, vocabulary)
    else:
      targets = tokens
    ex = {'inputs': inputs, 'targets': targets}
    if 'inputs_plaintext' in features:
      ex['inputs_plaintext'] = features['inputs_plaintext']
    if 'targets_plaintext' in features:
      ex['targets_plaintext'] = features['targets_plaintext']
    if style_bit:
      ex['attribute'] = features['attribute']
    if style_dependant_prefix_target:
      ex['codeprefixedtargets'] = features['codeprefixedtargets']
      ex['controlcode'] = features['controlcode']
    return ex
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)