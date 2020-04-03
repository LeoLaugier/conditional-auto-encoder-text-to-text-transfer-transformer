import os

import functools
import gin
import re
import six

import tensorflow as tf
# from mesh_tensorflow.transformer.utils import *
from tensorflow.python.ops import resources  # pylint: disable=g-direct-tensorflow-import
from tensorflow.python.tpu import tpu_estimator  # pylint: disable=g-direct-tensorflow-import

import tensorflow_datasets as tfds

import mesh_tensorflow as mtf
from mesh_tensorflow.transformer import transformer
from mesh_tensorflow.transformer import dataset as transformer_dataset
from mesh_tensorflow.transformer.utils import _dynamic_text2self, get_variable_dtype, serialize_num_microbatches, \
    write_lines_to_file, get_checkpoint_iterator, \
    get_step_from_checkpoint_path, decode, get_inputs_from_file, encode_inputs, decode_from_file

from dataset import process_style
from my_mesh_tensorflow_transformer_transformer import Bitransformer_ll, make_bitransformer_ll


_INPUT_FEATURES_ll = [
    "inputs", "inputs_position", "inputs_segmentation", "targets",
    "targets_position", "targets_segmentation", "targets_subsegmentation"
]


def build_model_ll(model_type="bitransformer_ll",
                   input_vocab_size=gin.REQUIRED,
                   output_vocab_size=gin.REQUIRED,
                   layout_rules=None,
                   mesh_shape=None,
                   style_embedding_encoder=False,
                   style_embedding_decoder=False,
                   style_num=2,
                   cut_cross_attention=True):
    """Build a transformer model.
    Currently, four types of models are supported:
    "bitransformer": The traditional encoder-decoder architecture from
       "Attention is All You Need".  Requires a non-text2self dataset.
    "lm": an autoregressive language model (one layer stack).  Effectively the
       decoder of the bitransformer. There is no attention over the encoder, since
       there is no encoder.  Requires a text2self dataset, with targets, but no
       inputs.
    "aligned": a non-autoregressive single-stack model (like BERT).  Requires
       a non-text2self dataset with inputs and targets.  The targets and inputs
       have the same length and each entry in the inputs is aligned to the
       corresponding entry in targets, eg:
        "inputs": "The X sat on X X."
        'targets": "The cat sat on the mat."
        (except, inputs are token ID sequences, not strings)
    "bi_teacher_student": a teacher-student model where both the student and
      teacher are bitransformers. Requires a non-text2self dataset.
    A text2self dataset has targets that are offset of the inputs. Non-text2self
    datasets have targets that differ from their inputs, like:
      input: 'hello'
      target: 'bonjour'
    Args:
      model_type: a string, one of "bitransformer", "lm", "aligned", or
        "bi_teacher_student"
      input_vocab_size: an integer
      output_vocab_size: an integer
      layout_rules: optional, input to mtf.convert_to_layout_rules
      mesh_shape: optional, a function that returns a mtf.Shape
    Returns:
      a Unitransformer or Bitransformer
    """
    if model_type == "bitransformer":
        return transformer.make_bitransformer(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            mesh_shape=mesh_shape,
            layout=layout_rules)
    elif model_type == "bi_student_teacher":
        return transformer.make_bi_student_teacher(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            mesh_shape=mesh_shape,
            layout=layout_rules)
    elif model_type == "lm" or model_type == "aligned":
        return transformer.Unitransformer(
            autoregressive=model_type == "lm",
            layer_stack=transformer.make_layer_stack(),
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            mesh_shape=mesh_shape,
            layout=layout_rules)

    elif model_type == "bitransformer_ll":
        return make_bitransformer_ll(
            input_vocab_size=input_vocab_size,
            output_vocab_size=output_vocab_size,
            mesh_shape=mesh_shape,
            layout=layout_rules,
            style_embedding_encoder=style_embedding_encoder,
            style_embedding_decoder=style_embedding_decoder,
            style_num=style_num,
            cut_cross_attention=cut_cross_attention)
    else:
        raise ValueError("unknown model_type!")


@gin.configurable
def tpu_estimator_model_fn_ll(model_type,
                              transformer_model,
                              vocabulary,
                              # Meshtensorflow commit: https://github.com/tensorflow/mesh/commit/df41d2e36ec89a350025beae7dbda5cc3b6930e5#diff-b44d79ea4f007470afa978fe013907be
                              model_dir,
                              use_tpu,
                              mesh_shape,
                              layout_rules,
                              batch_size,
                              sequence_length,
                              autostack,
                              keep_checkpoint_max,
                              save_checkpoints_steps,
                              learning_rate_schedule=None,
                              optimizer=None,
                              outer_batch_size=1,
                              tpu_summaries=False,
                              predict_fn=None,
                              variable_filter=None,
                              init_checkpoint=None,
                              ensemble_inputs=None,
                              mesh_devices=None,
                              style_embedding=False,
                              has_partial_sequences=True,
                              remove_partial_sequences=True,
                              style_dependant_prefix_target=True,
                              cycle_consistency_loss=True,
                              lambda_ae=1.0,
                              lambda_cycle=1.0):
    """Create a TPUEstimator model function.
    Args:
      model_type: a string. One of "bitransformer", "lm", "aligned", or
        "bi_teacher_student"
      transformer_model: a transformer.Unitransformer or transformer.Bitransformer or transformer.Bitransformer_ll
      model_dir: a string, directory to save the model to.
      use_tpu: a boolean
      mesh_shape: a function that returns a mtf.Shape
      layout_rules: a mtf.LayoutRules
      batch_size: an integer
      sequence_length: an integer or a dict from feature-key to integer
        the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
      autostack: a boolean
      keep_checkpoint_max: an integer, maximum number of checkpoints to keep
      save_checkpoints_steps: an integer, save a checkpoint every this number of
        steps
      learning_rate_schedule: an optional function taking the scalar named
        argument `step` and return the scalar learning rate. Alternatively, a
        constant.
      optimizer: a class extending optimize.Optimizer, required for training
      outer_batch_size: outer batch dimension that could be used to enable the mix
        of data-parallel and model-parallel training of Mixture of Experts (MoE)
        models
      tpu_summaries: a boolean, use rewrites to make summaries work on TPU.  This
        may be slow, since it uses a host call hack.
      predict_fn: an optional function, see docs for `run` for more information
      variable_filter: controls which variables are trained.
        If None (default), train all trainable variables.
        If a string regex, train all variables that match this regex.
        If a function (mtf.Variable -> boolean), then train variables for which
          the function returns True.
      init_checkpoint: a string, if not None then read in variables from this
        checkpoint path when initializing variables. Will only initialize
        variables that appear both in the current graph and the checkpoint.
      ensemble_inputs: an optional integer - pass the size of the ensemble to
        train an ensemble where each model gets different inputs.
        You also need to configure Unitransformer.ensemble  to the right size.
        If None, then all models are trained on the same inputs.
    Returns:
      a function to be passed to TPUEstimator
    """
    mesh_devices = mesh_devices or [""] * mesh_shape.size

    def my_model_fn(features, labels, mode, params=None, config=None):
        """Estimator model function.
        Args:
          features: dictionary where keys are strings like "inputs" and "targets"
            and the values are the actual values of "inputs". See TPUEstimator's
            docs for more information
          labels: ignored argument
          mode: a tf.estimator.ModeKeys
          params: dictionary containing the key "context"
          config: ignored argument
        Returns:
          a TPUEstimatorSpec
        """
        del labels, config
        global_step = tf.train.get_global_step()
        if use_tpu and "context" in params:
            ctx = params["context"]
            num_hosts = ctx.num_hosts
            host_placement_fn = ctx.tpu_host_placement_function
            device_list = [host_placement_fn(host_id=t) for t in range(num_hosts)]
            # TODO(ylc): Better estimation of replica cache size?
            replica_cache_size = 300 * 1000000  # 300M per replica
            # Worker 0 caches all the TPU binaries.
            worker0_mem = replica_cache_size * ctx.num_replicas
            devices_memeory_usage = [worker0_mem] + [0] * (num_hosts - 1)
            var_placer = mtf.utils.BalancedVariablePlacer(device_list,
                                                          devices_memeory_usage)
            # deprecated mesh_devices = [""] * mesh_shape.size
            physical_shape = list(
                params["context"].device_assignment.topology.mesh_shape)
            logical_to_physical = mtf.simd_mesh_impl.auto_logical_to_physical_tpu(
                mesh_shape.to_integer_list, physical_shape)
            mesh_impl = mtf.simd_mesh_impl.SimdMeshImpl(
                mesh_shape, layout_rules, mesh_devices, ctx.device_assignment,
                logical_to_physical=logical_to_physical)
        else:
            var_placer = None
            # deprecated mesh_devices = [""] * mesh_shape.size
            mesh_impl = mtf.placement_mesh_impl.PlacementMeshImpl(
                mesh_shape, layout_rules, mesh_devices)

        graph = mtf.Graph()
        mesh = mtf.Mesh(graph, "my_mesh", var_placer)

        mtf_features = {}
        for key, x in features.items():
            outer_batch_dim = mtf.Dimension("outer_batch", outer_batch_size)
            batch_dim = mtf.Dimension("batch", batch_size // outer_batch_size)
            # Some auxiliary features may have been generated in packing.
            # The names of these new features are of the form
            #   "<original_feature_name>_<suffix>", e.g. "inputs_segmentation".
            #   We look up the lengths based on the original feature name, without
            #   the "_<suffix>".
            feature_length = sequence_length[key.split("_")[0]]
            length_dim = mtf.Dimension("length", feature_length)
            ensemble_dims = ([mtf.Dimension("ensemble", ensemble_inputs)]
                             if ensemble_inputs else [])
            feature_shape = mtf.Shape(
                ensemble_dims + [outer_batch_dim, batch_dim, length_dim])
            x = tf.cast(features[key], tf.int32)
            x = tf.reshape(x, feature_shape.to_integer_list)
            if not use_tpu:
                tf.logging.info("feature %s : %s" % (key, x))
                x = tf.Print(
                    x, [x], "import feature %s" % key, summarize=1000, first_n=10)
            mtf_features[key] = mtf.import_fully_replicated(
                mesh, x, feature_shape, name=key)
            if key == "targets" or key == "codeprefixedtargets" or key == "codeprefix":
                anon_targets = mtf.anonymize(mtf_features[key])

        if mode == tf.estimator.ModeKeys.PREDICT:
            def _feature_shape(key):
                feature_length = sequence_length[key.split("_")[0]]
                return mtf.Shape([
                    mtf.Dimension("batch", batch_size),
                    mtf.Dimension("length", feature_length)
                ])

            mtf_features = {
                k: mtf.reshape(v, _feature_shape(k))
                for k, v in six.iteritems(mtf_features)
            }
            inputs = mtf_features["inputs"]

            if style_embedding:
                styles = mtf_features["style"]
            else:
                styles = None

            if has_partial_sequences:
                codeprefixes = mtf_features["codeprefix"]
            else:
                codeprefixes = None

            if predict_fn:
                mtf_samples = predict_fn(
                    model=transformer_model,
                    features=mtf_features,
                    variable_dtype=get_variable_dtype())
            elif isinstance(transformer_model, transformer.Unitransformer):
                # pad so that there is enough room for the targets
                inputs = mtf.pad(
                    inputs, [0, sequence_length["targets"]], length_dim.name)
                mtf_samples = transformer_model.sample_autoregressive(
                    inputs, variable_dtype=get_variable_dtype(),
                    remove_partial_sequences=True)
            elif isinstance(transformer_model,
                            Bitransformer_ll):
                mtf_samples = transformer_model.decode(
                    inputs, styles=styles, codeprefixes=codeprefixes, has_partial_sequences=has_partial_sequences,
                    remove_partial_sequences=remove_partial_sequences, variable_dtype=get_variable_dtype())  #
            elif isinstance(transformer_model,
                            (transformer.Bitransformer, transformer.StudentTeacher)):
                mtf_samples = transformer_model.decode(
                    inputs, variable_dtype=get_variable_dtype())
            else:
                raise ValueError("unrecognized class")
            mtf_samples = mtf.anonymize(mtf_samples)
            inputs = mtf.anonymize(inputs)
            lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
            inputs = lowering.export_to_tf_tensor(inputs)
            outputs = lowering.export_to_tf_tensor(mtf_samples)
            predictions = {
                "inputs": inputs,
                "outputs": outputs}

            # When exporting a model, we need to communicate to TF-Serving that
            # master variables need to be copied to their slave slice variables.
            # Estimator uses a Scaffold's "local_init_op" for this purpose, so we
            # augment the default "local_init_op" here.
            #
            # The "ready_op" is also constructed here to ensure the variables
            # initialized by "local_init_op" are the same ones checked by "ready_op".
            #
            # WARNING: Any variables created outside of this model_fn()
            # (e.g. tpu_estimator/iterations_per_loop) will NOT be initialized nor
            # checked by these ops.
            def scaffold_fn():
                return tf.train.Scaffold(
                    local_init_op=tf.group(
                        tf.train.Scaffold.default_local_init_op(),
                        lowering.copy_masters_to_slices(),
                        name="mtf_local_init_op"),
                    ready_op=tf.concat(
                        [tf.report_uninitialized_variables(),
                         resources.report_uninitialized_resources()],
                        axis=0,
                        name="mtf_ready_op"))

            return tpu_estimator.TPUEstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                scaffold_fn=scaffold_fn,
                prediction_hooks=[mtf.MtfRestoreHook(lowering)])

        assert (mode == tf.estimator.ModeKeys.TRAIN or
                mode == tf.estimator.ModeKeys.EVAL)

        def logits_and_loss(mtf_features):
            """Compute logits and loss.
            Args:
              mtf_features: a dictionary
            Returns:
              logits: a mtf.Tensor
              loss: a mtf.Tensor
            """
            if model_type == "lm":  # TOTRY Adapt that to our case
                if "inputs" in mtf_features:
                    mtf_features = _dynamic_text2self(mtf_features)
                _, _, length_dim = mtf_features["targets"].shape
                inputs = mtf.shift(mtf_features["targets"], offset=1,
                                   dim=length_dim, wrap=False)
            else:
                inputs = mtf_features["inputs"]

            if style_embedding:
                styles = mtf_features["style"]
            else:
                styles = None

            if style_dependant_prefix_target:
                codeprefixedtargets = mtf_features["codeprefixedtargets"]
            else:
                codeprefixedtargets = None

            if isinstance(transformer_model, transformer.Unitransformer):
                position_kwargs = dict(
                    sequence_id=mtf_features.get("targets_segmentation", None),
                    position=mtf_features.get("targets_position", None),
                )
            elif isinstance(
                    transformer_model,
                    transformer.Bitransformer) or model_type == "bi_student_teacher": # TODO add Bitransformer_ll even if a Bitransformer_ll is Bitransformer...
                if style_dependant_prefix_target:
                    position_kwargs = dict(
                        encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
                        decoder_sequence_id=mtf_features.get("codeprefixedtargets_segmentation",
                                                             None),
                        decoder_subsequence_id=mtf_features.get("codeprefixedtargets_subsegmentation",
                                                                None),
                        encoder_position=mtf_features.get("inputs_position", None),
                        decoder_position=mtf_features.get("codeprefixedtargets_position", None),
                    )
                else:
                    position_kwargs = dict(
                        encoder_sequence_id=mtf_features.get("inputs_segmentation", None),
                        decoder_sequence_id=mtf_features.get("targets_segmentation",
                                                             None),
                        decoder_subsequence_id=mtf_features.get("targets_subsegmentation",
                                                                None),
                        encoder_position=mtf_features.get("inputs_position", None),
                        decoder_position=mtf_features.get("targets_position", None),
                    )
            else:
                raise ValueError("unrecognized class")

            if isinstance(transformer_model, Bitransformer_ll): # TODO clean with *args and **kwargs?
                if cycle_consistency_loss:
                    logits_ae, l_ae = transformer_model.call_simple(
                        inputs=inputs,
                        targets=mtf_features["targets"],
                        compute_loss=True,
                        styles=styles,
                        codeprefixedtargets=codeprefixedtargets,
                        mode=mode,
                        variable_dtype=get_variable_dtype(),
                        **position_kwargs)

                    if has_partial_sequences:
                        codeprefixes = mtf_features["codeprefix"]
                    else:
                        codeprefixes = None

                    mtf_samples = transformer_model.decode(
                        inputs, styles=styles, codeprefixes=codeprefixes, has_partial_sequences=has_partial_sequences,
                        remove_partial_sequences=remove_partial_sequences, variable_dtype=get_variable_dtype())
                    # mtf_samples = mtf.anonymize(mtf_samples)
                    outputs = mtf_samples

                    logits_cycle, l_cycle = transformer_model.call_simple(
                        inputs=outputs,
                        targets=mtf_features["targets"],
                        compute_loss=True,
                        styles=styles,
                        codeprefixedtargets=codeprefixedtargets,
                        mode=mode,
                        variable_dtype=get_variable_dtype(),
                        **position_kwargs)

                    loss_ae_cycle = lambda_ae * l_ae + lambda_cycle * l_cycle
                    return logits_cycle, loss_ae_cycle
                else:
                    return transformer_model.call_simple(
                        inputs=inputs,
                        targets=mtf_features["targets"],
                        compute_loss=True,
                        styles=styles,
                        codeprefixedtargets=codeprefixedtargets,
                        mode=mode,
                        variable_dtype=get_variable_dtype(),
                        **position_kwargs)
            else:
                return transformer_model.call_simple(
                    inputs=inputs,
                    targets=mtf_features["targets"],
                    compute_loss=True,
                    mode=mode,
                    variable_dtype=get_variable_dtype(),
                    num_microbatches=num_microbatches,
                    **position_kwargs)

        if mode == tf.estimator.ModeKeys.TRAIN:
            num_microbatches = serialize_num_microbatches(batch_dim,
                                                          sequence_length,
                                                          mesh_shape,
                                                          layout_rules)
            if num_microbatches > 1:
                def serialized_fn(mtf_features):
                    return {
                        "loss": (logits_and_loss(mtf_features)[1] / num_microbatches)}

                var_grads, loss_dict = mtf.serialize_training_step(
                    mtf_features, serialized_fn, batch_dim, num_microbatches)
                loss = loss_dict["loss"]
            else:
                loss = logits_and_loss(mtf_features)[1]
                var_grads = mtf.gradients(
                    [loss], [v.outputs[0] for v in graph.trainable_variables])

            if tpu_summaries:
                mtf.scalar_summary("loss", loss)

            if callable(learning_rate_schedule):
                # the following happens on CPU since TPU can't handle summaries.
                with mtf.utils.outside_all_rewrites():
                    learning_rate = learning_rate_schedule(
                        step=tf.train.get_global_step())
                    tf.summary.scalar("learning_rate", learning_rate)
            else:
                learning_rate = learning_rate_schedule

            if isinstance(variable_filter, str):
                pattern = re.compile(variable_filter)
                variable_filter_fn = lambda v: pattern.search(v.name)
            elif variable_filter is None:
                variable_filter_fn = lambda v: True
            elif callable(variable_filter):
                variable_filter_fn = variable_filter
            else:
                raise ValueError(
                    "variable_filter must be None, a string, or a callable function")
            trainable_vars = [
                v for v in graph.trainable_variables if variable_filter_fn(v)]
            trainable_var_grads = [
                g for g, v in zip(var_grads, graph.trainable_variables)
                if variable_filter_fn(v)]
            if len(trainable_vars) != len(graph.trainable_variables):
                tf.logging.info("Variables being trained:")
                tf.logging.info([v.name for v in trainable_vars])
                tf.logging.info("Variables not being trained:")
                tf.logging.info([v.name for v in graph.trainable_variables
                                 if not variable_filter_fn(v)])

            update_ops = optimizer(learning_rate=learning_rate).apply_grads(
                trainable_var_grads, trainable_vars
            )

            lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)

            tf_loss = lowering.export_to_tf_tensor(loss)
            tf_loss = tf.cast(tf_loss, tf.float32)
            if not use_tpu:
                tf_loss = tf.Print(tf_loss, [tf_loss, tf.train.get_global_step()],
                                   "step, tf_loss")

            tf_update_ops = [lowering.lowered_operation(op) for op in update_ops]
            tf_update_ops.append(tf.assign_add(global_step, 1))
            train_op = tf.group(tf_update_ops)

            if hasattr(transformer_model, "initialize"):
                with mtf.utils.outside_all_rewrites():
                    transformer_model.initialize()

            if tpu_summaries:
                # has to be outside of
                # with mtf.utils.outside_all_rewrites()
                host_call = mtf.utils.create_host_call(model_dir)
                mtf.utils.remove_summaries()
            else:
                host_call = None

            with mtf.utils.outside_all_rewrites():

                if init_checkpoint:
                    ckpt_vars = {v for v, _ in tf.train.list_variables(init_checkpoint)}
                    global_vars = {v.op.name for v in tf.global_variables()}
                    restore_vars = ckpt_vars.intersection(global_vars)
                    tf.logging.info("Initializing variables from %s:", init_checkpoint)
                    tf.logging.debug("\n".join(sorted(restore_vars)))
                    tf.logging.info("Variables in %s but not in graph:", init_checkpoint)
                    tf.logging.info("\n".join(sorted(ckpt_vars - global_vars)))
                    tf.logging.info("Variables in graph but not in %s:", init_checkpoint)
                    tf.logging.info("\n".join(sorted(global_vars - ckpt_vars)))
                    tf.train.init_from_checkpoint(
                        init_checkpoint, {v: v for v in restore_vars}
                    )

                # Copy master variables to slices. Must be called first.
                restore_hook = mtf.MtfRestoreHook(lowering)
                saver = tf.train.Saver(
                    tf.global_variables(),
                    sharded=True,
                    max_to_keep=keep_checkpoint_max,
                    keep_checkpoint_every_n_hours=2,
                    defer_build=False,
                    save_relative_paths=True)
                tf.add_to_collection(tf.GraphKeys.SAVERS, saver)
                saver_listener = mtf.MtfCheckpointSaverListener(lowering)
                saver_hook = tf.train.CheckpointSaverHook(
                    model_dir,
                    save_steps=save_checkpoints_steps,
                    saver=saver,
                    listeners=[saver_listener])
                gin_config_saver_hook = gin.tf.GinConfigSaverHook(
                    model_dir, summarize_config=True, include_step_in_filename=False)

                if use_tpu:
                    return tpu_estimator.TPUEstimatorSpec(
                        mode=tf.estimator.ModeKeys.TRAIN,
                        loss=tf_loss,
                        train_op=train_op,
                        host_call=host_call,
                        training_hooks=[
                            restore_hook,
                            saver_hook,
                            gin_config_saver_hook,
                        ])
                else:
                    return tf.estimator.EstimatorSpec(
                        tf.estimator.ModeKeys.TRAIN,
                        loss=tf_loss,
                        train_op=train_op,
                        training_chief_hooks=[
                            restore_hook,
                            saver_hook,
                            gin_config_saver_hook,
                        ])
        elif mode == tf.estimator.ModeKeys.EVAL:
            logits, loss = logits_and_loss(mtf_features)
            anon_logits = mtf.anonymize(logits)
            lowering = mtf.Lowering(graph, {mesh: mesh_impl}, autostack=autostack)
            tf_loss = tf.cast(lowering.export_to_tf_tensor(loss), tf.float32)
            tf_loss = tf.cast(tf_loss, tf.float32)
            tf_logits = tf.cast(lowering.export_to_tf_tensor(anon_logits), tf.float32)

            def simple_metrics(logits, labels):
                """Simple metrics for teacher-forced eval."""
                weights = tf.cast(tf.not_equal(labels, 0), tf.float32)
                xent = tf.nn.sparse_softmax_cross_entropy_with_logits(
                    labels=labels, logits=logits)
                predictions = tf.cast(tf.argmax(logits, axis=-1), labels.dtype)
                token_correct = tf.cast(
                    tf.equal(predictions, labels), tf.float32) * weights
                sequence_correct = tf.to_float(tf.equal(
                    tf.reduce_sum(token_correct, -1),
                    tf.reduce_sum(weights, -1)))
                sequence_weights = tf.to_float(
                    tf.not_equal(tf.reduce_sum(weights, -1), 0))
                return {"neg_log_perplexity": tf.metrics.mean(-xent, weights),
                        "token_accuracy": tf.metrics.mean(token_correct, weights),
                        "sequence_accuracy": tf.metrics.mean(
                            sequence_correct, sequence_weights)}

            labels = lowering.export_to_tf_tensor(anon_targets)
            eval_metrics = (simple_metrics, [tf_logits, labels])
            with mtf.utils.outside_all_rewrites():
                restore_hook = mtf.MtfRestoreHook(lowering)
            return tpu_estimator.TPUEstimatorSpec(
                tf.estimator.ModeKeys.EVAL,
                evaluation_hooks=[restore_hook],
                loss=tf_loss,
                eval_metrics=eval_metrics)

    return my_model_fn


def write_lines_to_file_ll(lines, filename):
  """Write each line to a filename, replacing the file if it exists.
  Args:
    lines: list of str, lines to write out.
    filename: str, path to filename.
  """
  if tf.io.gfile.exists(filename):
    tf.io.gfile.remove(filename)
  with tf.io.gfile.GFile(filename, "w") as output_file:
    for line in lines:
      l = re.sub(r'\n', r"\\n", line, flags=re.S)
      output_file.write("{}\n".format(l))


def eval_model_ll(estimator, vocabulary, sequence_length, batch_size,
                  dataset_split, model_dir, eval_dataset_fn, eval_summary_dir,
                  eval_checkpoint_step, style_bit=True, unsupervised_style_transfer_metrics=True,
                  style_dependant_prefix_target=True):
    """Eval a Mesh-TF model.
    Args:
      estimator: Estimator object, created with the appropriate model_fn.
      vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
        targets_vocabulary) tuple
      sequence_length: a dict from feature-key to integer the (packed)
        sequence length, e.g. {"inputs": 512, "targets": 128}
      batch_size: an integer, global batch size
      dataset_split: a string
      model_dir: a string, directory with the model.
      eval_dataset_fn: A function returning a list of dataset.EvalDataset tuples.
        Must be provided for mode="eval". Should accept the following arguments:
          - sequence_length: an integer or a dict from feature-key to integer
            the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
          - vocabulary: Vocabulary instance to use for encoding.
          - dataset_split: str, which dataset split to load.
        dataset.EvalDataset tuples are namedtuples with the following fields:
          - name: string, the task name
          - dataset_fn: function which returns a tf.data.Dataset of tokenized and
            padded examples. Must not require any arguments and must include the
            feature keys 'inputs' and 'targets_plaintext'.
          - postprocess_fn: function which converts plaintext targets to values
            that can be processed by a `metric_fn`.
          - list_of_metric_fns: list of metric functions with the call signature
            `metric_fn(targets, predictions)` which returns a dict mapping
            submetric names to scalar values. TensorBoard summaries and other tags
            will be written out using the submetric names.
      eval_summary_dir: str, path to write TensorBoard events file summaries for
        eval. If None, use model_dir/eval_{split}.
      eval_checkpoint_step: int, list of ints, or None. If an int or list of ints,
        evaluation or inference will be run on the checkpoint files in `model_dir`
        whose global steps are closest to the global steps provided. If None and
        mode="eval", run eval continuously waiting for new checkpoints via
        `tf.train.checkpoints_iterator`.
    """
    if eval_dataset_fn is None:
        raise ValueError("Must provide eval_dataset_fn through gin for eval.")

    eval_datasets = eval_dataset_fn(
        sequence_length=sequence_length,
        vocabulary=vocabulary,
        dataset_split=dataset_split,
    )

    valid_eval_datasets = []
    for eval_dataset in eval_datasets:
        if not eval_dataset.metric_fns:
            tf.logging.info("Skipping %s because metric_fns is empty",
                            eval_dataset.name)
            continue
        # Convert to EvalDataset tuple in case eval_dataset_fn returns raw tuples
        valid_eval_datasets.append(transformer_dataset.EvalDataset(*eval_dataset))
    eval_datasets = valid_eval_datasets

    if not eval_datasets:
        tf.logging.info(
            "All provided EvalDatasets have metric_fns=[]; eval is not possible.")
        return

    eval_summary_dir = eval_summary_dir or os.path.join(
        model_dir, "{}_eval".format(dataset_split))
    summary_writer = tf.summary.FileWriter(eval_summary_dir)

    # Pre-load in all of the targets once before entering continuous eval loop
    cached_targets = {}
    cached_examples = {}
    if style_bit:
        cached_styles_origin = {}
    # Need to create a separate graph for loading in plaintext targets
    # or else TF will complain that we modified the graph
    with tf.Graph().as_default():
        for eval_dataset in eval_datasets:
            if eval_dataset.metric_fns:
                ds = eval_dataset.dataset_fn()
                # Create list of postprocessed text targets
                examples = [ex for ex in tfds.as_numpy(ds)]
                targets = [
                    eval_dataset.postprocess_fn(  # pylint:disable=g-complex-comprehension
                        tf.compat.as_text(ex["targets_plaintext"]),
                        example=ex, is_target=True)
                    for ex in examples
                ]

                if style_bit:
                    styles_origin = [
                        str(ex["style"][0] - 1)
                        for ex in examples
                    ]

                targets_filename = os.path.join(
                    eval_summary_dir,
                    "{}_targets".format(eval_dataset.name),
                )
                write_lines_to_file(targets, targets_filename)
                cached_targets[eval_dataset.name] = targets
                cached_examples[eval_dataset.name] = examples
                if style_bit:
                    cached_styles_origin[eval_dataset.name] = styles_origin

    if style_bit:
        _INPUT_FEATURES_ll.append("style")

    if style_dependant_prefix_target:
        _INPUT_FEATURES_ll.extend(["codeprefix",
                                   "codeprefix_position",
                                   "codeprefix_segmentation",
                                   "codeprefix_subsegmentation",
                                   "codeprefixedtargets",
                                   "codeprefixedtargets_position",
                                   "codeprefixedtargets_segmentation",
                                   "codeprefixedtargets_subsegmentation"])

    def input_fn(params):
        """Eval input function for estimator."""
        del params
        # Concatenate all dataset inputs to only have to do one decode loop
        combined_ds = None
        for eval_dataset in eval_datasets:
            # Only cache targets for those tasks with eval functions provides
            if eval_dataset.metric_fns:
                ds = eval_dataset.dataset_fn()
                # Only pass those variables which will be used for decoding
                ds = ds.map(
                    lambda x: {k: v for k, v in x.items() if k in _INPUT_FEATURES_ll})
                combined_ds = ds if not combined_ds else combined_ds.concatenate(ds)
        combined_ds = combined_ds.batch(batch_size, drop_remainder=False)
        # Pad the final batch.
        combined_ds = transformer_dataset.trim_and_pad_dataset(
            combined_ds, length=batch_size)
        combined_ds = combined_ds.prefetch(tf.data.experimental.AUTOTUNE)
        return combined_ds

    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)
    for checkpoint_path in checkpoint_paths:
        tf.logging.info("Checkpoint path %s" % checkpoint_path)
        global_step = int(get_step_from_checkpoint_path(checkpoint_path))
        if global_step == 0:
            continue
        decodes = decode(estimator, input_fn, vocabulary, checkpoint_path)
        for eval_dataset in eval_datasets:
            # Extract the portion of decodes corresponding to this dataset
            examples = cached_examples[eval_dataset.name]
            dataset_size = len(examples)
            predictions = [
                eval_dataset.postprocess_fn(tf.compat.as_text(d), example=ex)
                for d, ex in zip(decodes[:dataset_size], examples)
            ]
            # Remove the used decodes.
            del decodes[:dataset_size]

            global_step = int(get_step_from_checkpoint_path(checkpoint_path))

            predictions_filename = os.path.join(
                eval_summary_dir,
                "{}_{}_predictions".format(eval_dataset.name, global_step),
            )
            write_lines_to_file_ll(predictions, predictions_filename)

            for metric_fn in eval_dataset.metric_fns:
                summary = tf.Summary()
                targets = cached_targets[eval_dataset.name]
                if unsupervised_style_transfer_metrics and style_bit:
                    styles_origin = cached_styles_origin[eval_dataset.name]
                    metric_result = metric_fn(targets, predictions, styles_origin=styles_origin)
                else:
                    metric_result = metric_fn(targets, predictions)
                for metric_name, metric_value in metric_result.items():
                    tag = "eval/{}/{}".format(eval_dataset.name, metric_name)
                    tf.logging.info("%s at step %d: %.3f", tag, global_step, metric_value)
                    summary.value.add(tag=tag, simple_value=metric_value)
                    summary_writer.add_summary(summary, global_step)
            summary_writer.flush()

        # Only padding should remain.
        expected_pad = -sum(len(t) for t in cached_targets.values()) % batch_size
        if len(decodes) != expected_pad:
            raise ValueError("{} padded decodes, {} expected.".format(
                len(decodes), expected_pad))

@gin.configurable
def decode_from_file_ll(estimator,
                        vocabulary,
                        model_type,
                        batch_size,
                        sequence_length,
                        checkpoint_path=None,
                        input_filename=gin.REQUIRED,
                        output_filename=gin.REQUIRED,
                        eos_id=1,
                        repeats=1,
                        target_prefix_style_1="",
                        target_prefix_style_2="",
                        style_dependant_prefix_target=False,
                        style_embedding=False):
    """Decode from a text file and write to output_filename.
    Args:
      estimator: a TPUEstimator
      vocabulary: a mtf.transformer.vocabulary.Vocabulary
      model_type: a string
      batch_size: an integer
      sequence_length: an integer or a dict from feature-key to integer
        the (packed) sequence length, e.g. {"inputs": 512, "targets": 128}
      checkpoint_path: an optional string
      input_filename: a string
      output_filename: a string
      eos_id: EOS id
      repeats: an integer, the number of times to repeat each input.
    """
    inputs_and_dst_styles = get_inputs_from_file(input_filename)

    inputs_split = [line.split("|dst_style:") for line in inputs_and_dst_styles]

    inputs = []
    dst_styles = []
    codeprefixes = []
    for l in inputs_split:
        inputs.append(l[0])
        dst_styles.append(l[1])
        if l[1] == "1":
            codeprefixes.append(target_prefix_style_1)
        elif l[1] == "2":
            codeprefixes.append(target_prefix_style_2)
        else:
            codeprefixes.append("")

    all_input_ids = encode_inputs(inputs, vocabulary, model_type, batch_size,
                                  sequence_length["inputs"], eos_id=eos_id)
    if style_dependant_prefix_target:
        all_codeprefix_ids = encode_inputs(codeprefixes, vocabulary, "lm", batch_size,
                                           sequence_length["codeprefix"], eos_id=eos_id)

    def input_fn(params):
        del params

        tensors = {"inputs": all_input_ids}
        if style_embedding:
            tensors["style"] = dst_styles
        if style_dependant_prefix_target:
            tensors["codeprefix"] = all_codeprefix_ids

        dataset = tf.data.Dataset.from_tensor_slices(tensors)
        if style_embedding:
            dataset = process_style(dataset, mode="infer")
        dataset = dataset.flat_map(
            lambda x: tf.data.Dataset.from_tensors(x).repeat(repeats))
        dataset = dataset.batch(batch_size, drop_remainder=True)
        dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)
        return dataset

    checkpoint_step = get_step_from_checkpoint_path(checkpoint_path)
    decodes = decode(
        estimator, input_fn, vocabulary, checkpoint_path=checkpoint_path
    )
    # Remove any padded examples
    dataset_size = len(inputs) * repeats
    decodes = decodes[:dataset_size]
    output_filename = "{}-{}".format(output_filename, checkpoint_step)
    write_lines_to_file(decodes, output_filename)


@gin.configurable
def infer_model_ll(estimator,
                   vocabulary,
                   sequence_length,
                   batch_size,
                   model_type,
                   model_dir,
                   eval_checkpoint_step,
                   input_filename=None,
                   output_filename=None,
                   checkpoint_paths=None,
                   decode_from_file_fn=decode_from_file,
                   target_prefix_style_1="",
                   target_prefix_style_2="",
                   style_dependant_prefix_target=False,
                   style_embedding=False):
  """Infer a Mesh-TF model.
  Args:
    estimator: Estimator object, created with the appropriate model_fn.
    vocabulary: a vocabulary.Vocabulary or (inputs_vocabulary,
      targets_vocabulary) tuple
    sequence_length: a dict from feature-key to integer the (packed)
      sequence length, e.g. {"inputs": 512, "targets": 128}
    batch_size: an integer, global batch size
    model_type: a string - either "bitransformer", "bi_student_teacher", lm" or
      "aligned"
    model_dir: string, estimator model_dir
    eval_checkpoint_step: int, list of ints, or None, see `eval_model`
      docstring.
    input_filename: a string, input file with examples
    output_filename: a string, output file to save decodes
    checkpoint_paths: optional list of checkpoints to run inference for
    decode_from_file_fn: decoding function, defaults to decode_from_file
  """
  if style_dependant_prefix_target or style_embedding:
      decode_from_file_fn = functools.partial(decode_from_file_ll,
                                              target_prefix_style_1=target_prefix_style_1,
                                              target_prefix_style_2=target_prefix_style_2,
                                              style_dependant_prefix_target=style_dependant_prefix_target,
                                              style_embedding=style_embedding)

  if checkpoint_paths is None:
    checkpoint_paths = get_checkpoint_iterator(eval_checkpoint_step, model_dir)

  for checkpoint_path in checkpoint_paths:
    decode_from_file_fn(
        estimator,
        vocabulary=vocabulary,
        model_type=model_type,
        batch_size=batch_size,
        sequence_length=sequence_length,
        checkpoint_path=checkpoint_path,
        input_filename=input_filename,
        output_filename=output_filename)