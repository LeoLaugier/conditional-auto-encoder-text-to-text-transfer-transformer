import gin
import tensorflow.compat.v1 as tf
import mesh_tensorflow as mtf
# from mesh_tensorflow.transformer import transformer
# from mesh_tensorflow.transformer.transformer import *
from mesh_tensorflow.transformer.transformer import make_layer_stack, reduce_ensemble_logits, text2self_inputs_mask, \
    Context, shift_targets, Unitransformer, Bitransformer


@gin.configurable
def make_bitransformer_ll(
    input_vocab_size=gin.REQUIRED,
    output_vocab_size=gin.REQUIRED,
    layout=None,
    mesh_shape=None,
    encoder_name="encoder",
    decoder_name="decoder",
    cut_cross_attention=False):
  """Gin-configurable bitransformer constructor.
  In your config file you need to set the encoder and decoder layers like this:
  encoder/make_layer_stack.layers = [
    @transformer_layers.SelfAttention,
    @transformer_layers.DenseReluDense,
  ]
  decoder/make_layer_stack.layers = [
    @transformer_layers.SelfAttention,
    @transformer_layers.EncDecAttention,
    @transformer_layers.DenseReluDense,
  ]
  Args:
    input_vocab_size: a integer
    output_vocab_size: an integer
    layout: optional - an input to mtf.convert_to_layout_rules
      Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
    mesh_shape: optional - an input to mtf.convert_to_shape
      Some layers (e.g. MoE layers) cheat by looking at layout and mesh_shape
    encoder_name: optional - a string giving the Unitransformer encoder name.
    decoder_name: optional - a string giving the Unitransformer decoder name.
  Returns:
    a Bitransformer
  """
  with gin.config_scope("encoder"):
    encoder = Unitransformer_ll(
        layer_stack=make_layer_stack(),
        input_vocab_size=input_vocab_size,
        output_vocab_size=None,
        autoregressive=False,
        name=encoder_name,
        layout=layout,
        mesh_shape=mesh_shape)
  with gin.config_scope("decoder"):
    if cut_cross_attention:
        layer_stack = make_layer_stack(layers=[mtf.transformer.transformer_layers.SelfAttention,
                                               [mtf.transformer.transformer_layers.DenseReluDense, "layer_002"]])
    else:
        layer_stack = make_layer_stack()

    decoder = Unitransformer_ll(
        layer_stack=layer_stack,
        input_vocab_size=output_vocab_size,
        output_vocab_size=output_vocab_size,
        autoregressive=True,
        name=decoder_name,
        layout=layout,
        mesh_shape=mesh_shape)
  return Bitransformer_ll(encoder, decoder, cut_cross_attention=cut_cross_attention)


@gin.configurable
class Unitransformer_ll(Unitransformer):
    def __init__(self, *unitransformer_args, attribute_embedding=False, attribute_num=2, **unitransformer_kwargs):
        super().__init__(*unitransformer_args, **unitransformer_kwargs)
        self.attribute_embedding = attribute_embedding
        self.attribute_dim = mtf.Dimension("attribute",
                                           attribute_num + 1)  # attribute_num + 1 because we add attribute 0, a "padding" attribute (necessary because of the way T5 pre-processes datasets...)

    def _call_internal(self, context, inputs, targets=None, attributes=None, z=None):
        """Compute logits based on inputs (all positions in parallel).
        Also updates context if applicable.
        Args:
          context: a Context
          inputs: a Tensor
          targets: an optional Tensor
          attributes: an optional Tensor
        Returns:g
          logits: a Tensor with shape [<batch_dims>, length_dim, output_vocab_dim]
        """
        mesh = inputs.mesh
        if self.ensemble_dim and self.ensemble_dim not in inputs.shape.dims:
            # Training an ensemble where all models are trained on the same examples.
            inputs = mtf.broadcast(inputs, [self.ensemble_dim] + inputs.shape.dims)
            if self.ensemble_dim not in attributes.shape.dims:
                attributes = mtf.broadcast(attributes, [self.ensemble_dim] + attributes.shape.dims)
            if targets:
                targets = mtf.broadcast(
                    targets, [self.ensemble_dim] + targets.shape.dims)
        if "embedding" in context.shared_params:
            vocab_embedding = context.shared_params["embedding"]
        else:
            vocab_embedding = VocabEmbedding(
                mesh,
                self.input_vocab_dim,
                self.model_dim,
                context.variable_dtype,
                name="embedding",
                ensemble_dim=self.ensemble_dim)
        x = vocab_embedding.ids_to_embedding(inputs)
        if self.positional_embedding:
            if "positional_embedding" in context.shared_params:
                pos_emb_var = context.shared_params["positional_embedding"]
            else:
                pos_emb_var = mtf.layers.embedding_weights(
                    mesh, self.max_length_dim, self.model_dim, context.variable_dtype,
                    "positional_embedding", ensemble_dim=self.ensemble_dim)
            if (context.length_dim is not None and
                    context.length_dim.size > self.max_length_dim.size):
                message = (
                        "Length dimenison exceeds size of positional embedding table. "
                        "length_dim.size > max_length_dim.size %s vs %s."
                        % (context.length_dim, self.max_length_dim))
                if context.position_is_default:
                    # Definitely getting overflow in this case.
                    raise ValueError(message)
                else:
                    tf.logging.warning(
                        message +
                        " This may be OK if there are several shorter sequences packed "
                        "together.  Otherwise, the later positions will get zeros.")
            if context.position_is_default:
                pos_emb = mtf.rename_dimension(
                    mtf.slice(pos_emb_var, 0, context.length_dim.size,
                              self.max_length_dim.name),
                    self.max_length_dim.name, context.length_dim.name)
            else:
                pos_emb = mtf.gather(
                    pos_emb_var, context.position, self.max_length_dim,
                    output_shape=x.shape)
            x += pos_emb

        if self.attribute_embedding:
            if "attribute_embedding" in context.shared_params:
                sty_emb_var = context.shared_params["attribute_embedding"]
            else:
                sty_emb_var = mtf.layers.embedding_weights(
                    mesh, self.attribute_dim, self.model_dim, context.variable_dtype,
                    "attribute_embedding", ensemble_dim=self.ensemble_dim)

            sty_emb = mtf.gather(
                sty_emb_var, attributes, self.attribute_dim,
                output_shape=x.shape)
            # Addition of x and attribute
            # x *= LAMBDA_ATTRIBUTE * sty_emb #

            # Concatenation of x and attribute
            x_attribute = mtf.concat([x, sty_emb], self.model_dim.name)
            x = mtf.layers.dense(
                x_attribute, self.model_dim, activation=None, variable_dtype=context.variable_dtype,
                name="comb_x_attribute")

        if z:
            z = mtf.layers.dense(
                z, self.model_dim, activation=None, variable_dtype=context.variable_dtype,
                name="z")
            # raise ValueError("x shape=%s , z shape=%s" % (x.shape, z.shape))
            x += z

        x = self.layer_stack.call(context, x)
        if self.output_vocab_dim is None:
            return x
        if self.shared_embedding_and_softmax_weights:
            logits = vocab_embedding.hidden_to_logits(x)
        else:
            logits = mtf.layers.dense(
                x, self.output_vocab_dim, use_bias=False,
                variable_dtype=context.variable_dtype,
                reduced_dims=x.shape.dims[-1:],
                name="logits")
        if targets is not None and context.losses is not None:
            context.losses.append(
                self._compute_loss(context, logits, targets, self.output_vocab_dim))
        if self.ensemble_dim:
            logits = reduce_ensemble_logits(
                logits, self.ensemble_dim, self.output_vocab_dim)
        return logits

    def call_simple(self,
                    inputs,
                    targets,
                    compute_loss,
                    attributes=None,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    variable_dtype=mtf.VariableDType(tf.float32),
                    sequence_id=None,
                    subsequence_id=None,
                    position=None,
                    encoder_output=None,
                    encoder_sequence_id=None,
                    encoder_inputs=None,
                    shared_params=None,
                    layer_outputs=None,
                    encoder_layer_outputs=None,
                    z=None):
        """Compute logits based on inputs (all positions in parallel).
        This is called during training and evaluation.
        Args:
          inputs: an int32 Tensor with shape [<batch_dims>, length_dim] For training
            autoregressive models this should be equal to mtf.shift(targets,
            offset=1, dim=length_dim, wrap=False)
          targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
          compute_loss: a boolean
          attributes: an (optional?) int32 Tensor with shape [<batch_dims>, length_dim] ([<batch_dims>])
          mode: a tf.estimator.ModeKeys
          variable_dtype: a mtf.VariableDType
          sequence_id: an optional Tensor
          subsequence_id: an optional Tensor
          position: an optional Tensor
          encoder_output: an optional Tensor
          encoder_sequence_id: an optional Tensor
          encoder_inputs: an optional Tensor
          shared_params: an optional dictionary
          layer_outputs: an optional list to append Tensor layer activations to
          encoder_layer_outputs: optional - readonly list of tensor activations when
            decoding, one per each input layer + the embedding layer
        Returns:
          logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
          loss: an optional Scalar (if compute_loss=True)
        """
        batch_dims = inputs.shape.dims[:-1]
        length_dim = inputs.shape.dims[-1]
        length_range = mtf.range(inputs.mesh, length_dim, dtype=tf.int32)
        if not self.positional_embedding:
            # To make relative attention faster, we drop the information about the
            #   position in the subsequence.  The relative attention code then
            #   assumes that the positions are given by index in the tensor,
            #   which still leads to the correct computation of relative position.
            position = None
        if position is None:
            position_is_default = True
            position = length_range
        else:
            position_is_default = False
        if self.input_full_attention:
            # The inputs part of each sequence can fully attend within itself.
            full_attention_region = text2self_inputs_mask(targets)
            # We can include one additional position to the right - the position
            #   where the final EOS of the inputs is read and the first target token
            #   is predicted.
            full_attention_region = mtf.logical_or(
                full_attention_region,
                mtf.shift(full_attention_region, offset=1, dim=length_dim, wrap=False)
            )
            # We set read_priority and write_priority to 0 in the full-attention
            #   region and equal to the position elsewhere.
            read_priority = write_priority = length_range * mtf.cast(
                mtf.logical_not(full_attention_region), tf.int32)
        elif self.autoregressive:
            # Vanilla autoregressive model - each position can see previous positions.
            read_priority = write_priority = length_range
        else:
            read_priority = write_priority = None
        context = Context(
            model=self,
            mesh=inputs.mesh,
            batch_dims=batch_dims,
            length_dim=length_dim,
            variable_dtype=variable_dtype,
            mode=mode,
            losses=[] if compute_loss else None,
            sequence_id=sequence_id,
            subsequence_id=subsequence_id,
            position=position,
            position_is_default=position_is_default,
            encoder_output=encoder_output,
            encoder_sequence_id=encoder_sequence_id,
            shared_params=shared_params,
            layer_outputs=layer_outputs,
            encoder_layer_outputs=encoder_layer_outputs,
            write_priority=write_priority,
            read_priority=read_priority,
            inputs=inputs,
            encoder_inputs=encoder_inputs)
        with tf.variable_scope(self.name):
            logits = self._call_internal(context, inputs, targets, attributes, z=z)
        if compute_loss:
            loss = mtf.add_n(context.losses)
        else:
            loss = None
        return logits, loss

    @gin.configurable(module="Unitransformer_ll")
    def sample_autoregressive(self,
                              partial_sequences,
                              dst_attributes=None,
                              stop_at_token=1,
                              max_steps=None,
                              temperature=0.0,
                              variable_dtype=mtf.VariableDType(tf.float32),
                              encoder_output=None,
                              encoder_sequence_id=None,
                              encoder_inputs=None,
                              shared_params=None,
                              has_partial_sequences=True,
                              encoder_layer_outputs=None,
                              never_end=False,
                              remove_partial_sequences=False,
                              sampling_keep_top_k=-1,
                              z=None):
        """Sample randomly one token at a time.
        The partial_sequences represent partial sequences to be continued.  The
        first tokens of each sequence are nonzero representing the given partial
        sequences and the last tokens of each sequence are zeros, representing what
        needs to be filled in.
        If there are no partial sequences (you want to sample from the beginning),
        then pass partial_sequences=mtf.zeros(mesh, shape, dtype=tf.int32) and
        has_partial_sequences=False (so we can skip computation).
        The dst_attributes represents the destination attributes in which we want to generate sequences.
        Args:
          partial_sequences: an int32 Tensor with shape [<batch_dims>, length_dim]
          dst_attribute: an int32 Tensor with shape [<batch_dims>, length_dim] ([<batch_dims>])
          stop_at_token: an optional integer eos id.  Stop when we produce it.
          max_steps: an optional integer, the max number of steps to decode.
          temperature: an optional floating point value between 0.0 and 1.0 0.0
            means argmax, 1.0 means sample according to predicted distribution.
          variable_dtype: a mtf.VariableDType
          encoder_output: an optional Tensor
          encoder_sequence_id: an optional Tensor
          encoder_inputs: an optional Tensor
          shared_params: an optional dictionary
          has_partial_sequences: a boolean
          encoder_layer_outputs: optional - readonly list of tensor activations when
            decoding, one per each input layer + the embedding layer
          never_end: a boolean - if set, then avoid generating stop_at_token
          remove_partial_sequences: a boolean - whether to remove the partial
            sequences from the output
          sampling_keep_top_k: an integer - if not -1, only sample from the top k
            logits.
        Returns:
          a Tensor with shape [<batch_dims>, length_dim]
        """
        if not self.autoregressive:
            raise ValueError("must be autoregressive")

        inputs = partial_sequences
        attributes = dst_attributes
        batch_dims = inputs.shape.dims[:-1]
        length_dim = inputs.shape.dims[-1]
        initial_position = mtf.reduce_sum(
            mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)
        sequence_id = 1 if encoder_sequence_id is not None else None

        length_range = mtf.range(inputs.mesh, length_dim, tf.int32)
        if self.input_full_attention:
            read_priority = write_priority = length_range * mtf.to_int32(
                mtf.greater(length_range, initial_position))
        else:
            read_priority = write_priority = length_range

        context_first_part = Context(
            model=self,
            mesh=inputs.mesh,
            batch_dims=batch_dims,
            length_dim=length_dim,
            variable_dtype=variable_dtype,
            mode="first_part",
            position=length_range,
            position_is_default=True,
            new_states=[],
            initial_position=initial_position,
            sequence_id=sequence_id,
            encoder_output=encoder_output,
            encoder_sequence_id=encoder_sequence_id,
            constant_states=[],
            shared_params=shared_params,
            encoder_layer_outputs=encoder_layer_outputs,
            write_priority=write_priority,
            read_priority=read_priority,
            inputs=inputs,
            encoder_inputs=encoder_inputs)

        shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
        with tf.variable_scope(self.name):
            logits = self._call_internal(context_first_part, shifted_inputs, attributes=attributes,
                                         z=z)
        del logits
        constant_states = context_first_part.constant_states
        if not has_partial_sequences:
            initial_states = [
                mtf.zeros_like(t) for t in context_first_part.new_states]
            partial_sequences_eos_count = 0
        else:
            initial_states = context_first_part.new_states
            partial_sequences_eos_count = mtf.reduce_sum(
                mtf.to_int32(mtf.equal(partial_sequences, stop_at_token)),
                reduced_dim=length_dim)

        def cond_fn(position, ids, *unused_states):
            """Should we run another loop iteration."""
            past_end = mtf.greater_equal(position, length_dim.size)
            if max_steps:
                past_end = mtf.logical_or(
                    past_end, mtf.greater_equal(position - initial_position, max_steps))

            is_done = past_end
            if stop_at_token is not None:
                eos_count = mtf.reduce_sum(
                    mtf.to_int32(mtf.equal(ids, stop_at_token)),
                    reduced_dim=length_dim)
                has_additional_eos = mtf.greater(eos_count, partial_sequences_eos_count)
                is_done = mtf.logical_or(is_done, has_additional_eos)
            all_done = mtf.reduce_all(is_done)
            return mtf.logical_not(all_done)

        def body_fn(position, ids, *states):
            """One step in the decode loop."""
            inputs_this_step = mtf.gather(ids, position - 1, length_dim)
            if self.attribute_embedding:
                attributes_this_step = mtf.gather(attributes, position - 1, length_dim)
            else:
                attributes_this_step = None
            # raise ValueError("inputs_this_step shape=%s , ids shape=%s, position - 1 shape=%s, length_dim=%s" % (inputs_this_step.shape, ids.shape, (position - 1).shape, length_dim))
            context_incremental = Context(
                model=self,
                mesh=inputs.mesh,
                batch_dims=batch_dims,
                length_dim=length_dim,
                variable_dtype=variable_dtype,
                mode="incremental",
                position=position,
                states=states,
                new_states=[],
                sequence_id=sequence_id,
                encoder_output=encoder_output,
                encoder_sequence_id=encoder_sequence_id,
                constant_states=constant_states,
                shared_params=shared_params,
                encoder_layer_outputs=encoder_layer_outputs,
                write_priority=write_priority,
                read_priority=position,
                inputs=inputs_this_step,
                encoder_inputs=encoder_inputs)

            with tf.variable_scope(self.name, reuse=True):
                logits = self._call_internal(context_incremental, inputs_this_step, attributes=attributes_this_step,
                                             z=z)
                if never_end:
                    logits += mtf.one_hot(
                        mtf.constant(logits.mesh, stop_at_token, dtype=tf.int32),
                        self.output_vocab_dim, on_value=-1e9, off_value=0.0,
                        dtype=logits.dtype)

            # TBD whether this should be before or after never_end:
            # Note for adding top_p sampling in the future, in other code bases, the
            # option to apply temperature is done before the top-k truncation. This
            # implementation does this in the opposite order. For top-k this doesn't
            # matter, but for top_p it will.
            if sampling_keep_top_k != -1:
                if sampling_keep_top_k <= 0:
                    raise ValueError("sampling_keep_top_k must either be -1 or positive.")
                k_largest = mtf.nth_largest_element(
                    logits, n=sampling_keep_top_k,
                    reduced_dim=self.output_vocab_dim)
                logits = mtf.where(mtf.less_equal(logits, k_largest),
                                   mtf.ones_like(logits) * -1e6, logits)

            ids_this_step = mtf.sample_with_temperature(
                logits, self.output_vocab_dim, temperature)
            new_position = position + 1
            new_ids = ids + ids_this_step * mtf.one_hot(
                position, length_dim, dtype=tf.int32)
            return [new_position, new_ids] + context_incremental.new_states

        while_loop_inputs = [initial_position, inputs] + initial_states
        final_position, outputs = mtf.while_loop(
            cond_fn, body_fn, while_loop_inputs)[:2]
        del final_position
        if has_partial_sequences and remove_partial_sequences:
            # remove partial sequences from outputs
            partial_length = mtf.reduce_sum(
                mtf.to_int32(mtf.not_equal(partial_sequences, 0)),
                reduced_dim=length_dim)
            outputs = mtf.dynamic_shift(
                outputs, -partial_length, length_dim, wrap=False)
        return outputs

    def beam_search(self,
                    inputs,
                    decode_length,
                    dst_attributes=None,
                    variable_dtype=mtf.VariableDType(tf.float32),
                    encoder_output=None,
                    encoder_sequence_id=None,
                    encoder_inputs=None,
                    alpha=0.6,
                    shared_params=None,
                    encoder_layer_outputs=None,
                    z=None):
        """Beam search.
        Args:
          inputs: an int32 zero-Tensor with shape [<batch_dims>, beam_dim,
            length_dim].#
          decode_length: an int32 mtf scalar.  Maximum decode length.
          attributes: an int32 zero-Tensor with shape [<batch_dims>, beam_dim, length_dim]
                                          ([<batch_dims>]
                                           [<batch_dims>, beam_dim]).
          variable_dtype: a mtf.VariableDType
          encoder_output: an optional Tensor
          encoder_sequence_id: an optional Tensor
          encoder_inputs: an optional Tensor
          alpha: a floating point value (length bonus)
          shared_params: an optional dictionary
          encoder_layer_outputs: optional - readonly list of tensor activations when
            decoding, one per each input layer + the embedding layer
        Returns:
          a Tensor with shape [<batch_dims>, beam_dim, length_dim]
        """
        attributes = dst_attributes
        if not self.autoregressive:
            raise ValueError("must be autoregressive")

        batch_dims = inputs.shape.dims[:-2]
        if len(batch_dims) != 1:
            raise NotImplementedError(
                "beam search supports exactly one batch dimension.")
        beam_dim = inputs.shape.dims[-2]
        length_dim = inputs.shape.dims[-1]
        length_range = mtf.range(inputs.mesh, length_dim, tf.int32)
        initial_position = mtf.reduce_sum(
            mtf.to_int32(mtf.not_equal(inputs, 0)), reduced_dim=length_dim)
        sequence_id = 1 if encoder_sequence_id is not None else None

        if self.input_full_attention:
            # This only makes sense in the case of beam search with given partial
            # sequences, which is not yet implemented.
            # TODO(noam): implement
            raise NotImplementedError(
                "Beam search for language models not yet implemented")
        else:
            read_priority = write_priority = length_range

        context_first_part = Context(
            model=self,
            mesh=inputs.mesh,
            batch_dims=batch_dims + [beam_dim],
            length_dim=length_dim,
            variable_dtype=variable_dtype,
            mode="first_part",
            position=length_range,
            position_is_default=True,
            new_states=[],
            initial_position=initial_position,
            sequence_id=sequence_id,
            encoder_output=encoder_output,
            encoder_sequence_id=encoder_sequence_id,
            constant_states=[],
            shared_params=shared_params,
            encoder_layer_outputs=encoder_layer_outputs,
            write_priority=write_priority,
            read_priority=read_priority,
            inputs=inputs,
            encoder_inputs=encoder_inputs)

        shifted_inputs = mtf.shift(inputs, offset=1, dim=length_dim, wrap=False)
        with tf.variable_scope(self.name):
            logits = self._call_internal(context_first_part, shifted_inputs, attributes=attributes,
                                         z=z)
        del logits
        # There are no partial targets.
        # Replace initial states by zeros to avoid computing them.
        initial_states = [mtf.zeros_like(t) for t in context_first_part.new_states]
        constant_states = context_first_part.constant_states

        def logits_fn(step_num, ids, states):
            """logits_fn for mtf.beam_search.beam_search()."""
            inputs_this_step = mtf.gather(ids, step_num - 1, length_dim)

            if self.attribute_embedding:
                attributes_this_step = mtf.gather(attributes, step_num - 1, length_dim)
            else:
                attributes_this_step = None

            context_incremental = Context(
                model=self,
                mesh=inputs.mesh,
                batch_dims=batch_dims + [beam_dim],
                length_dim=length_dim,
                variable_dtype=variable_dtype,
                mode="incremental",
                position=step_num,
                states=states,
                new_states=[],
                sequence_id=sequence_id,
                encoder_output=encoder_output,
                encoder_sequence_id=encoder_sequence_id,
                constant_states=constant_states,
                shared_params=shared_params,
                encoder_layer_outputs=encoder_layer_outputs,
                write_priority=write_priority,
                read_priority=step_num,
                inputs=inputs_this_step,
                encoder_inputs=encoder_inputs)
            with tf.variable_scope(self.name, reuse=True):
                logits = self._call_internal(context_incremental, inputs_this_step, attributes=attributes_this_step,
                                             z=z)
            return mtf.to_float(logits), context_incremental.new_states

        beams, unused_scores = mtf.beam_search.beam_search(
            logits_fn,
            inputs,
            alpha,
            states=initial_states,
            decode_length=decode_length,
            use_tpu=True,
            dtype=tf.float32,
            mesh_shape=self.mesh_shape,
            layout=self.layout)
        return mtf.gather(
            beams, mtf.constant(inputs.mesh, 0, dtype=tf.int32), beam_dim)


@gin.configurable
def shift_targets_no_offset(targets, bos_id=0, eos_id=1):
  """Transforms decoder labels to decoder inputs.
  Args:
    targets: decoder labels
    bos_id: begin of sequence id, defaults to 0
    eos_id: end of sequence id, defaults to 1
  Returns:
    Decoder inputs.
  """
  length_dim = targets.shape.dims[-1]
  shifted_targets = targets
  # We should have a 0 at the beginning of each sequence rather than the
  # shifted EOS (e.g. 1) from the previous sequence.
  shifted_targets *= mtf.to_int32(mtf.not_equal(shifted_targets, eos_id))

  if bos_id:
    shifted_targets += mtf.to_int32(
        mtf.logical_and(
            mtf.equal(shifted_targets, 0),
            mtf.not_equal(targets, 0))) * bos_id

  return shifted_targets


class Bitransformer_ll(Bitransformer):
    def __init__(self, *bitransformer_args, cut_cross_attention=False, **bitransformer_kwargs):
        super().__init__(*bitransformer_args, **bitransformer_kwargs)
        self.cut_cross_attention = cut_cross_attention

    def _shared_params(self, mesh, variable_dtype):
        """Create parameters that are shared between encoder and decoder.
        Args:
          mesh: a Mesh
          variable_dtype: a VariableDType
        Returns:
          a dictionary
        """
        shared_params = {}
        if self.shared_embedding:
            with tf.variable_scope("shared"):
                if not (self.encoder.model_dim == self.decoder.model_dim and
                        self.encoder.input_vocab_dim == self.decoder.input_vocab_dim):
                    raise ValueError(
                        "shared_embedding requires encoder and decoder to have identical"
                        " d_model and vocabulary sizes")
                shared_params["embedding"] = VocabEmbedding(
                    mesh,
                    self.encoder.input_vocab_dim,
                    self.encoder.model_dim,
                    variable_dtype,
                    name="embedding",
                    ensemble_dim=self.encoder.ensemble_dim)
                if (self.encoder.positional_embedding
                        and self.decoder.positional_embedding
                        and self.encoder.max_length_dim == self.decoder.max_length_dim):
                    shared_params["positional_embedding"] = mtf.layers.embedding_weights(
                        mesh, self.encoder.max_length_dim, self.encoder.model_dim,
                        variable_dtype, "positional_embedding",
                        ensemble_dim=self.encoder.ensemble_dim)
                if (self.encoder.attribute_embedding
                        and self.decoder.attribute_embedding):
                    shared_params["attribute_embedding"] = mtf.layers.embedding_weights(
                        mesh, self.encoder.attribute_dim, self.encoder.model_dim,
                        variable_dtype, "attribute_embedding",
                        ensemble_dim=self.encoder.ensemble_dim)
        return shared_params

    def call_simple(self,
                    inputs,
                    targets,
                    compute_loss,
                    attributes=None,
                    codeprefixedtargets=None,
                    mode=tf.estimator.ModeKeys.TRAIN,
                    variable_dtype=mtf.VariableDType(tf.float32),
                    encoder_sequence_id=None,
                    decoder_sequence_id=None,
                    decoder_subsequence_id=None,
                    encoder_position=None,
                    decoder_position=None):  # attributes=None for debugging?
        """Compute logits based on inputs (all positions in parallel).
        This is called during training and evaluation.
        Args:
          inputs: an int32 Tensor with shape [<batch_dims>, length_dim]
          targets: an optional int32 Tensor with shape [<batch_dims>, length_dim]
          compute_loss: a boolean
          attributes: an (optional?) int32 Tensor with shape [<batch_dims>, length_dim] ([<batch_dims>])
          mode: a tf.estimator.ModeKeys
          variable_dtype: a mtf.VariableDType
          encoder_sequence_id: an optional Tensor
          decoder_sequence_id: an optional Tensor
          decoder_subsequence_id: an optional Tensor
          encoder_position: an optional Tensor
          decoder_position: an optional Tensor
        Returns:
          logits: a Tensor with shape [<batch_dims>, output_vocab_dim]
          loss: an optional Scalar (if compute_loss=True)
        """
        encoder_layer_outputs = []
        shared_params = self._shared_params(inputs.mesh, variable_dtype)
        encoder_output, encoder_loss = self.encoder.call_simple(
            inputs,
            None,
            compute_loss,
            attributes=attributes,
            mode=mode,
            variable_dtype=variable_dtype,
            sequence_id=encoder_sequence_id,
            position=encoder_position,
            shared_params=shared_params,
            layer_outputs=encoder_layer_outputs)
        encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
        if encoder_sequence_id is not None:
            encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
                encoder_sequence_id)

        if self.cut_cross_attention:
            z = mtf.gather(encoder_output,
                           mtf.zeros(inputs.mesh, mtf.Shape(inputs.shape[:-1] + [encoder_output.shape[-1]]),
                                     dtype=tf.int32), encoder_output.shape[-2])
            encoder_output = None
        else:
            z = None

        if codeprefixedtargets:
            decoder_input = shift_targets_no_offset(
                codeprefixedtargets)  # shift_attribute_targets(targets, attribute_id=codeprefixedtargets), # codeprefixedtargets # mtf.zeros_like(targets)
        else:
            decoder_input = shift_targets(targets)

        # shift_targets = mtf.shift(targets, offset=-1, dim=targets.shape.dims[-1], wrap=False) # Remove token preceding ":"
        # shift_targets = mtf.shift(shift_targets, offset=1, dim=targets.shape.dims[-1], wrap=False)
        logits, loss = self.decoder.call_simple(
            decoder_input,
            targets,
            compute_loss,
            attributes=attributes,
            mode=mode,
            variable_dtype=variable_dtype,
            sequence_id=decoder_sequence_id,
            subsequence_id=decoder_subsequence_id,
            encoder_output=encoder_output,
            encoder_sequence_id=encoder_sequence_id,
            encoder_inputs=mtf.layers.rename_length_to_memory_length(inputs),
            position=decoder_position,
            shared_params=shared_params,
            encoder_layer_outputs=encoder_layer_outputs,
            z=z)  # Maybe sample_autoregressive here ?

        if loss is not None and encoder_loss is not None:
            loss += encoder_loss
        return logits, loss

    @gin.configurable(module="Bitransformer_ll")
    def decode(self,
               inputs,
               attributes=None,
               controlcodes=None,
               variable_dtype=mtf.VariableDType(tf.float32),
               beam_size=1,
               alpha=0.6,
               temperature=0.0,
               decode_length_multiplier=1.5,
               decode_length_constant=10,
               max_decode_length=None,
               has_partial_sequences=False,
               remove_partial_sequences=False):
        """Sampling or beam search.
        TODO(noam): should we make the output length dimension different from the
        input length dimension?
        Args:
          inputs: a Tensor with shape [<batch_dims>, beam_dim, length_dim]
          attributes: a Tensor with shape [<batch_dims>]
                                   or [<batch_dims>, beam_dim]
                                   or [<batch_dims>, beam_dim, length_dim]
          variable_dtype: a mtf.VariableDType
          beam_size: an integer >= 1
          alpha: a floating point value (length bonus for beam search)
          temperature: a value between 0 and 1 (must be 0 if beam_size > 1)
            0.0 means argmax, 1.0 means sample according to predicted distribution.
          decode_length_multiplier: a float
          decode_length_constant: a float
          max_decode_length: an optional integer
        Returns:
          a Tensor with shape [<batch_dims>, beam_dim, length_dim]
        """
        encoder_layer_outputs = []
        shared_params = self._shared_params(inputs.mesh, variable_dtype)
        encoder_sequence_id = mtf.minimum(inputs, 1)
        encoder_output, encoder_loss = self.encoder.call_simple(
            inputs=inputs,
            targets=None,
            compute_loss=False,
            attributes=attributes,
            mode=tf.estimator.ModeKeys.PREDICT,
            variable_dtype=variable_dtype,
            sequence_id=encoder_sequence_id,
            shared_params=shared_params,
            layer_outputs=encoder_layer_outputs)
        del encoder_loss
        encoder_output = mtf.layers.rename_length_to_memory_length(encoder_output)
        encoder_sequence_id = mtf.layers.rename_length_to_memory_length(
            encoder_sequence_id)
        batch_dims = inputs.shape[:-1]
        length_dim = inputs.shape[-1]
        if max_decode_length is None:
            decode_length_dim = length_dim
        else:
            decode_length_dim = mtf.Dimension("length", max_decode_length)

        if self.cut_cross_attention:
            z = mtf.gather(encoder_output,
                           mtf.zeros(inputs.mesh, mtf.Shape(inputs.shape[:-1] + [encoder_output.shape[-1]]),
                                     dtype=tf.int32), encoder_output.shape[-2])
            encoder_output = None
        else:
            z = None

        if beam_size == 1:
            ids_shape = mtf.Shape(batch_dims + [decode_length_dim])
        else:
            beam_dim = mtf.Dimension("beam", beam_size)
            ids_shape = mtf.Shape(batch_dims + [beam_dim, decode_length_dim])

        if controlcodes:
            partial_sequences = controlcodes  # shift_targets(controlcodes)
        else:
            partial_sequences = mtf.zeros(inputs.mesh, ids_shape, dtype=tf.int32)

        if beam_size == 1:
            return self.decoder.sample_autoregressive(
                partial_sequences,
                dst_attributes=attributes,
                temperature=temperature,
                variable_dtype=variable_dtype,
                encoder_output=encoder_output,
                encoder_sequence_id=encoder_sequence_id,
                encoder_inputs=mtf.layers.rename_length_to_memory_length(inputs),
                shared_params=shared_params,
                has_partial_sequences=has_partial_sequences,
                remove_partial_sequences=remove_partial_sequences,
                encoder_layer_outputs=encoder_layer_outputs,
                z=z)
        else:
            if temperature != 0:
                raise ValueError(
                    "don't know how to beam search with nonzero temperature")
            # beam search
            partial_sequences = mtf.zeros(inputs.mesh, ids_shape, dtype=tf.int32)
            input_length = mtf.reduce_sum(
                mtf.to_float(mtf.cast(inputs, tf.bool)),
                reduced_dim=length_dim)
            max_input_length = mtf.reduce_max(input_length)
            decode_length = mtf.cast(
                max_input_length * decode_length_multiplier
                + decode_length_constant, tf.int32)
            return self.decoder.beam_search(
                partial_sequences,
                decode_length,
                dst_attributes=attributes,
                variable_dtype=variable_dtype,
                encoder_output=encoder_output,
                encoder_sequence_id=encoder_sequence_id,
                encoder_inputs=inputs,
                alpha=alpha,
                shared_params=shared_params,
                encoder_layer_outputs=encoder_layer_outputs,
                z=z)


# /!\ This is needed since the last commit was on Dec 9 and the last version on Pypi was 0.1.7 on Dec 6 #TODO Update
@gin.configurable
class VocabEmbedding(object):
  """A class to go from vocab ids to model states and model states to logits."""

  def __init__(self,
               mesh,
               vocab_dim,
               output_dim,
               variable_dtype,
               name,
               ensemble_dim,
               inner_dimension_size=None):
    """Configurable embedding for the vocabulary.
    Most of the arguments get passed to `mtf.layers.embedding_weights` with an
    option to factorize the embedding matrix.
    Args:
      mesh: a mtf.Mesh
      vocab_dim: a mtf.Dimension
      output_dim: a mtf.Dimension
      variable_dtype: a mtf.VariableDType
      name: a string
      ensemble_dim: a mtf.Dimension
      inner_dimension_size: None or a postive integer. If None, then the
        embedding matrix is not factorized. If an integer, then it is the size
        of the inner dimension of the embedding matrix
    """
    self._vocab_dim = vocab_dim
    self._output_dim = output_dim
    self._is_factorized = inner_dimension_size is not None
    if self._is_factorized:
      self._inner_dim = mtf.Dimension("inner_vocab", inner_dimension_size)
      self._factor1 = mtf.layers.embedding_weights(
          mesh=mesh,
          vocab_dim=vocab_dim,
          output_dim=self._inner_dim,
          variable_dtype=variable_dtype,
          name="{}1".format(name),
          ensemble_dim=ensemble_dim,
          initializer=tf.random_normal_initializer(
              stddev=inner_dimension_size**-0.25))
      self._factor2 = mtf.layers.embedding_weights(
          mesh=mesh,
          vocab_dim=self._inner_dim,
          output_dim=output_dim,
          variable_dtype=variable_dtype,
          name="{}2".format(name),
          ensemble_dim=ensemble_dim,
          initializer=tf.random_normal_initializer(
              stddev=inner_dimension_size**-0.25))
    else:
      self._embedding_weights = mtf.layers.embedding_weights(
          mesh=mesh,
          vocab_dim=vocab_dim,
          output_dim=output_dim,
          variable_dtype=variable_dtype,
          name=name,
          ensemble_dim=ensemble_dim)

  def ids_to_embedding(self, ids):
    if self._is_factorized:
      tmp = mtf.gather(self._factor1, ids, self._vocab_dim)
      return mtf.einsum([tmp, self._factor2], reduced_dims=[self._inner_dim])
    else:
      return mtf.gather(self._embedding_weights, ids, self._vocab_dim)

  def hidden_to_logits(self, hidden):
    hidden *= self._output_dim.size**-0.5
    if self._is_factorized:
      tmp = mtf.einsum([hidden, self._factor2], reduced_dims=[self._output_dim])
      return mtf.einsum([tmp, self._factor1], reduced_dims=[self._inner_dim])
    else:
      return mtf.einsum([hidden, self._embedding_weights],
                        reduced_dims=[self._output_dim])