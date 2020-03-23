import tensorflow as tf

def raw_to_tsv(in_fname_pos, in_fname_neg, out_fname):
  with tf.io.gfile.GFile(in_fname_pos, "rb") as infile_pos,\
       tf.io.gfile.GFile(in_fname_neg, "rb") as infile_neg,\
       tf.io.gfile.GFile(out_fname, "w") as outfile:
    pos_sentences  = infile_pos.readlines()
    for sentence in pos_sentences:
      sentence = sentence.rstrip()
      # sentence = sentence.decode("utf-8")
      # outfile.write(sentence+"\t"+"1\n")
      outfile.write("%s\t%s\n" % (sentence.decode("utf-8"), "1"))
    neg_sentences  = infile_neg.readlines()
    for sentence in neg_sentences:
      sentence = sentence.rstrip()
      outfile.write("%s\t%s\n" % (sentence.decode("utf-8"), "0"))


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