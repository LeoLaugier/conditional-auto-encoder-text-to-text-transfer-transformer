import math
import t5

def ppl(targets, predictions, styles_origin=None, imdb_ppl_model=None):
  sum_ppl_scores = 0
  length = 0
  for i, prediction in enumerate(predictions):
    score = imdb_ppl_model.score(prediction)
    sum_ppl_scores += score
    length += len(prediction.split())

  perplexity = math.pow(10, - sum_ppl_scores / length)

  return {"perplexity": perplexity}


def style_accuracy(targets, predictions, styles_origin=None, classifier_imdb=None):
  count = 0
  assert len(predictions) == len(styles_origin), "The sizes of predictions and styles_origin don't match"
  for prediction, style_origin in zip(predictions, styles_origin):
    prediction_label = classifier_imdb.predict([prediction])[0][0][0][-1]
    if prediction_label != style_origin:
      count += 1

  style_accuracy = count / len(predictions)

  return {"style_accuracy": style_accuracy}


def our_bleu(targets, predictions, styles_origin=None):
  return t5.evaluation.metrics.bleu(targets, predictions)