import math
import numpy as np
import t5
import tensorflow as tf
import torch
from torch.utils.data import SequentialSampler, DataLoader
from torch.nn.utils.rnn import pad_sequence
# from transformers import DataCollatorForLanguageModeling, Trainer, TrainingArguments
from transformers.data.processors.utils import InputFeatures
from torchtext import data

from caet5.data.dataset import MyDataset

def gpt_perplexity_batch_280(targets, predictions, ppl_model, tokenizer, device, attributes_origin=None, batch_size=8,
                             block_size=256):
  eval_dataset = MyDataset(tokenizer=tokenizer, prediction_list=predictions, block_size=block_size)

  def collate(examples):
    if tokenizer._pad_token is None:
      return pad_sequence(examples, batch_first=True)
    return pad_sequence(examples, batch_first=True, padding_value=tokenizer.pad_token_id)

  eval_sampler = SequentialSampler(eval_dataset)

  eval_dataloader = DataLoader(eval_dataset, sampler=eval_sampler, batch_size=batch_size, collate_fn=collate)

  eval_loss = 0.0
  nb_eval_steps = 0
  ppl_model.eval()

  for batch in eval_dataloader:
    inputs, labels = (batch, batch)
    inputs = inputs.to(device)
    labels = labels.to(device)

    with torch.no_grad():
      outputs = ppl_model(inputs, labels=labels)
      lm_loss = outputs[0]
      eval_loss += lm_loss.mean().item()
    nb_eval_steps += 1

  eval_loss = eval_loss / nb_eval_steps
  perplexity = torch.exp(torch.tensor(eval_loss))

  return {"perplexity": perplexity}

def gpt_perplexity_batch_290(targets, predictions, ppl_model, tokenizer, attributes_origin=None, batch_size=8,
                             block_size=256):
  # Too early, wait for transformers v2.9.0, otherwise:
  # ImportError: cannot import name 'DataCollatorForLanguageModeling'
  training_args = TrainingArguments(output_dir="./gpt2_preds", do_eval=True, per_gpu_eval_batch_size=batch_size)
  eval_dataset = MyDataset(tokenizer=tokenizer, prediction_list=predictions, block_size=block_size)
  data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

  trainer = Trainer(
    model=ppl_model,
    args=training_args,
    data_collator=data_collator,
    eval_dataset=eval_dataset,
    prediction_loss_only=True,
  )

  eval_output = trainer.evaluate()

  perplexity = math.exp(eval_output["loss"])

  return {"perplexity": perplexity}


def gpt_perplexity(targets, predictions, ppl_model, tokenizer, device, attributes_origin=None):
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


def kenlm_perplexity(targets, predictions, ppl_model, attributes_origin=None):
  sum_ppl_scores = 0
  length = 0

  for i, prediction in enumerate(predictions):
    score = ppl_model.score(prediction)
    sum_ppl_scores += score
    length += len(prediction.split())

  perplexity = math.pow(10, - sum_ppl_scores / length)

  return {"perplexity": perplexity}


def bert_attribute_accuracy_batch(targets, predictions, classifier_model, tokenizer, device, attributes_origin=None,
                                  batch_size=32):
  # torchtext dataset
  init_token_idx = tokenizer.cls_token_id
  eos_token_idx = tokenizer.sep_token_id
  pad_token_idx = tokenizer.pad_token_id
  unk_token_idx = tokenizer.unk_token_id

  max_input_length = 220  # tokenizer.max_model_input_sizes['bert-base-uncased']

  def tokenize_and_cut(sentence):
    tokens = tokenizer.tokenize(sentence)
    tokens = tokens[:max_input_length - 2]
    return tokens

  TEXT = data.Field(batch_first=True,
                    use_vocab=False,
                    tokenize=tokenize_and_cut,
                    preprocessing=tokenizer.convert_tokens_to_ids,
                    init_token=init_token_idx,
                    eos_token=eos_token_idx,
                    pad_token=pad_token_idx,
                    unk_token=unk_token_idx)

  LABEL = data.LabelField(dtype=torch.float, use_vocab=False)
  fields = [('comment_text', TEXT), ('attribute_origin', LABEL)]

  examples = [data.Example.fromlist([prediction, attribute_origin], fields)
              for prediction, attribute_origin in zip(predictions, attributes_origin)]

  val_data = data.Dataset(examples, fields)
  # LABEL.build_vocab(val_data) # problem when only one label in val_data or when frequencies of labels are in different order than in the dataset that what used to fine-tune bert_acc_classifier. Solution: use_vocab=False.
  valid_iterator = data.BucketIterator(val_data, batch_size=batch_size, device=device)

  epoch_acc = 0

  classifier_model.eval()

  with torch.no_grad():
    for batch in valid_iterator:
      prediction_labels = torch.round(torch.sigmoid(classifier_model(batch.comment_text)[0].squeeze(1)))
      correct = (prediction_labels != batch.attribute_origin).float()
      acc = correct.sum() / len(correct)
      epoch_acc += acc.item()

  return {"attribute_accuracy": epoch_acc / len(valid_iterator)}


def bert_attribute_accuracy(targets, predictions, classifier_model, tokenizer, device, attributes_origin=None, batch_size=32):
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
    prediction_labels = torch.round(torch.sigmoid(classifier_model(**inputs)[0].squeeze(1)))

  prediction_labels = prediction_labels.detach().cpu().numpy()

  attributes_origin = np.array(attributes_origin)

  correct = (prediction_labels != attributes_origin).float()

  attribute_accuracy = correct.sum() / len(correct)

  return {"attribute_accuracy": attribute_accuracy}


def fasttext_attribute_accuracy(targets, predictions, classifier_model, attributes_origin=None):
  count = 0
  assert len(predictions) == len(attributes_origin), "The sizes of predictions and attributes_origin don't match"
  for prediction, attribute_origin in zip(predictions, attributes_origin):
    prediction_label = classifier_model.predict([prediction])[0][0][0][-1]
    if prediction_label != attribute_origin:
      count += 1

  attribute_accuracy = count / len(predictions)

  return {"attribute_accuracy": attribute_accuracy}


def bleu(targets, predictions, attributes_origin=None):
  return t5.evaluation.metrics.bleu(targets, predictions)


def sentence_similarity(targets, predictions, sentence_similarity_model, attributes_origin=None):
  with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    targets_embeddings = session.run(sentence_similarity_model(targets))
    predictions_embeddings = session.run(sentence_similarity_model(predictions))
  sentence_similarity_all = np.einsum('ij,ij->i', targets_embeddings, predictions_embeddings)
  sentence_similarity_avg = sentence_similarity_all.mean()
  return {"sentence_similarity": sentence_similarity_avg}