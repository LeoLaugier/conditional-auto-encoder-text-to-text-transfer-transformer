import tensorflow as tf


def raw_to_tsv(in_fname_1, in_fname_0, out_fname, mode):
  with tf.io.gfile.GFile(in_fname_1, mode) as infile_1,\
       tf.io.gfile.GFile(in_fname_0, mode) as infile_0,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    sentences_1  = infile_1.readlines()
    for sentence in sentences_1:
      sentence = sentence.rstrip()
      # sentence = sentence.decode("utf-8")
      # outfile.write(sentence+"\t"+"1\n")
      if mode == "rb":
        sentence = sentence.decode("utf-8")
      sentence = sentence.replace("\t", "\\t")
      outfile.write("%s\t%s\n" % (sentence, "1"))
    sentences_0  = infile_0.readlines()
    for sentence in sentences_0:
      sentence = sentence.rstrip()
      if mode == "rb":
        sentence = sentence.decode("utf-8")
      sentence = sentence.replace("\t", "\\t")
      outfile.write("%s\t%s\n" % (sentence, "0"))


def st_preprocessor(ds, dataset=None, style_bit=False,
                    style_dependant_prefix_input=False, input_prefix_style_1=None, input_prefix_style_2=None,
                    style_dependant_prefix_target=False, target_prefix_style_1=None, target_prefix_style_2=None):
  def normalize_text(text):
    """Lowercase and remove quotes from a TensorFlow string."""
    text = tf.strings.lower(text)
    text = tf.strings.regex_replace(text, br"\\n", b"\n")
    text = tf.strings.regex_replace(text, br"\\t", b"\n")
    # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
    # TODO: add cleaning methods suitable for "dirty" comments. Maybe perspective has it? Or SentencePiece already does it.
    return text

  def to_inputs_and_targets(ex):
    """
    Map {"text": ..., [...], "style" / "toxicity": ...} ->
        {"inputs": ..., ["style": ..., "codeprefixedtargets": ..., "codeprefix": ...,] "targets": ...}.
    """
    style = None
    if dataset == "IMDB" or dataset == "processed_CCTK":
      style = tf.strings.to_number(ex["style"], tf.int32)
    elif dataset == "CCTK":
      style = tf.dtypes.cast(tf.round(ex["toxicity"]), tf.int32)

    if style_dependant_prefix_input:
      if tf.math.equal(style, 0):
        inputs = tf.strings.join(
          [input_prefix_style_1, normalize_text(ex["text"])])
      else:
        inputs = tf.strings.join(
          [input_prefix_style_2, normalize_text(ex["text"])])
    else:
      inputs = tf.strings.join(
        ["", normalize_text(ex["text"])])

    targets = normalize_text(ex["text"])

    ex_processed = {"inputs": inputs, "targets": targets}

    if style_bit:
      ex_processed["style"] = tf.expand_dims(style + 1, 0)  # +1 because 0 considered as padding so styles are 1 and 2

    if style_dependant_prefix_target:
      if tf.math.equal(style, 0):
        codeprefixedtargets = tf.strings.join([target_prefix_style_1, normalize_text(ex["text"])])  # For training
        codeprefix = tf.strings.join([target_prefix_style_2, ""])  # For eval, the other code prefix
      else:
        codeprefixedtargets = tf.strings.join([target_prefix_style_2, normalize_text(ex["text"])])  # For training
        codeprefix = tf.strings.join([target_prefix_style_1, ""])  # For eval, the other code prefix

      ex_processed["codeprefixedtargets"] = codeprefixedtargets
      ex_processed["codeprefix"] = codeprefix

    return ex_processed

  return ds.map(to_inputs_and_targets, num_parallel_calls=tf.data.experimental.AUTOTUNE)


def process_style(dataset, mode="train"):
  def map_fn(x):
    style = x["style"]
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
      style = tf.expand_dims(tf.strings.to_number(style, out_type=tf.int32), 0)

    processed_style = tf.gather(style, indices)
    x["style"] = processed_style * inputs_padding
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
