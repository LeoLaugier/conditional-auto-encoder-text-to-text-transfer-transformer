import math
import numpy as np
import t5
import tensorflow as tf
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.data.processors.utils import InputFeatures

def gpt_perplexity(targets, predictions, ppl_model, tokenizer, device, styles_origin=None):
  examples = tokenizer.batch_encode_plus(predictions, add_special_tokens=True,
                                                max_length=tokenizer.max_len)["input_ids"]
  all_input_ids = [torch.tensor(example, dtype=torch.long) for example in examples]

  if tokenizer._pad_token is None:
    batch = pad_sequence(all_input_ids, batch_first=True)
  else:
    batch = pad_sequence(all_input_ids, batch_first=True, padding_value=tokenizer.pad_token_id)

  inputs, labels = (batch, batch)
  inputs = inputs.to(device)
  labels = labels.to(device)

  ppl_model.eval()
  with torch.no_grad():
    outputs = ppl_model(inputs, labels=labels)
    lm_loss = outputs[0]
    eval_loss = lm_loss.mean().item()

  perplexity = torch.exp(torch.tensor(eval_loss))
  return {"perplexity": perplexity}


def kenlm_perplexity(targets, predictions, ppl_model, styles_origin=None):
  sum_ppl_scores = 0
  length = 0

  for i, prediction in enumerate(predictions):
    score = ppl_model.score(prediction)
    sum_ppl_scores += score
    length += len(prediction.split())

  perplexity = math.pow(10, - sum_ppl_scores / length)

  return {"perplexity": perplexity}

def bert_style_accuracy(targets, predictions, classifier_model, tokenizer, device, styles_origin=None):
  batch_encoding = tokenizer.batch_encode_plus(predictions, max_length=tokenizer.max_len, pad_to_max_length=True)

  features = []
  for i in range(len(predictions)):
    inputs = {k: batch_encoding[k][i] for k in batch_encoding}

    feature = InputFeatures(**inputs)
    features.append(feature)

  all_input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
  all_attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
  all_token_type_ids = torch.tensor([f.token_type_ids for f in features], dtype=torch.long)

  # Data on TPU
  all_input_ids = all_input_ids.to(device)
  all_attention_mask = all_attention_mask.to(device)
  all_token_type_ids = all_token_type_ids.to(device)

  classifier_model.eval()

  with torch.no_grad():
    inputs = {"input_ids": all_input_ids, "attention_mask": all_attention_mask, "token_type_ids": all_token_type_ids}
    prediction_labels = torch.round(torch.sigmoid(classifier_model(**inputs)))

  prediction_labels = prediction_labels.detach().cpu().numpy()

  styles_origin = np.array(styles_origin)

  correct = (prediction_labels != styles_origin).float()

  style_accuracy = correct.sum() / len(correct)

  return {"style_accuracy": style_accuracy}


def fasttext_style_accuracy(targets, predictions, classifier_model, styles_origin=None):
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