import math
import numpy as np
import t5
import tensorflow as tf

def perplexity(targets, predictions, ppl_model, styles_origin=None):
  sum_ppl_scores = 0
  length = 0
  for i, prediction in enumerate(predictions):
    score = ppl_model.score(prediction)
    sum_ppl_scores += score
    length += len(prediction.split())

  perplexity = math.pow(10, - sum_ppl_scores / length)

  return {"perplexity": perplexity}


def style_accuracy(targets, predictions, classifier_model, styles_origin=None):
  count = 0
  assert len(predictions) == len(styles_origin), "The sizes of predictions and styles_origin don't match"
  for prediction, style_origin in zip(predictions, styles_origin):
    prediction_label = classifier_model.predict([prediction])[0][0][0][-1]
    if prediction_label != style_origin:
      count += 1

  style_accuracy = count / len(predictions)

  return {"style_accuracy": style_accuracy}


def our_bleu(targets, predictions, styles_origin=None):
  return t5.evaluation.metrics.bleu(targets, predictions)


def sentence_similarity(targets, predictions, sentence_similarity_model, styles_origin=None):
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    targets_embeddings = session.run(sentence_similarity_model(targets))
    predictions_embeddings = session.run(sentence_similarity_model(predictions))
  sentence_similarity_all = np.einsum('ij,ij->i', targets_embeddings, predictions_embeddings)
  sentence_similarity_avg = sentence_similarity_all.mean()
  return {"sentence_similarity": sentence_similarity_avg}