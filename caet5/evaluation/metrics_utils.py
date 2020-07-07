import errno
import os

import functools

import gin
import tensorflow as tf
import torch
from apiclient.http import MediaIoBaseDownload
from google.cloud import storage


def download_from_bucket_to_local(gcs_service, bucket, gcs_path, local_path):
    if not os.path.exists(os.path.dirname(local_path)):
        try:
            os.makedirs(os.path.dirname(local_path))
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise

    with open(local_path, 'wb') as f:
        request = gcs_service.objects().get_media(bucket=bucket,
                                                  object=gcs_path)
        media = MediaIoBaseDownload(f, request)

        done = False
        while not done:
            # _ is a placeholder for a progress object that we ignore.
            # (Our file is small, so we skip reporting progress.)
            status, done = media.next_chunk()
            print(status)


def upload_blob(bucket_name, source_file_name, destination_blob_name):
    """Upload a file to the bucket."""
    storage_client = storage.Client()
    bucket = storage_client.get_bucket(bucket_name)
    blob = bucket.blob(destination_blob_name)

    blob.upload_from_filename(source_file_name)

    print('File {} uploaded to {}.'.format(
        source_file_name,
        destination_blob_name))

def setup_parametric_evaluator(eval_fn, *load_fn_args, evaluator_name="Parametric evaluator", model_filename=None,
                               model_architecture=None, metric_name=None, task=None, ext=None, base_dir=None,
                               bucket=None, gcs_service=None, load_parametric_model_fn=None, **load_fn_kwargs):
  if not model_filename:
    if not model_architecture or not task:
      raise ValueError("Must specify model_filename or (model_architecture and task)")
    model_filename = '%s_%s_%s.%s' % (model_architecture, metric_name, task, ext)

  parametric_model_local_path = os.path.join('%s_binaries' % metric_name, model_filename)
  parametric_model_gcs_path = os.path.join('%s_binaries' % metric_name, model_filename)

  if not os.path.exists(parametric_model_local_path):
    tf.compat.v1.logging.info("%s not found on local machine." % evaluator_name)
    if not base_dir or not bucket or not gcs_service:
      raise ValueError("Must specify base_dir, bucket and gcs_service to download a %s from GCS.",
                       evaluator_name.lower())
    if tf.io.gfile.exists(os.path.join(base_dir, parametric_model_gcs_path)):
      tf.compat.v1.logging.info("Downloading %s from GCS..." % evaluator_name.lower())
      download_from_bucket_to_local(gcs_service, bucket, parametric_model_gcs_path, parametric_model_local_path)
      tf.compat.v1.logging.info('Download %s complete' % model_filename.lower())

    else:
      tf.compat.v1.logging.info(
        "Fine-tuned %s binary not found on bucket, please store one either on local "
        "(in [metric_name]_binaries/) or on the GCS bucket (in [metric_name]_binaries/)." % evaluator_name.lower())

  eval_fn_args, eval_fn_kwargs = load_parametric_model_fn(evaluator_name, parametric_model_local_path, *load_fn_args,
                                                          **load_fn_kwargs)

  partial_metric_fn = functools.partial(eval_fn, *eval_fn_args, **eval_fn_kwargs)

  def metric_fn(targets, predictions, *args, **kwargs):
      return partial_metric_fn(targets, predictions, *args, **kwargs)

  return metric_fn


def load_finetuned_transformer(evaluator_name, finetuned_model_local_path, pretrained_model_name_or_path,
                               load_tokenizer_fn, load_config_fn, load_pretrained_fn, map_location=None, **kwargs):
    device = "cpu"  # xm.xla_device()
    tokenizer = load_tokenizer_fn(pretrained_model_name_or_path)
    config = load_config_fn(pretrained_model_name_or_path)
    pretrained_model = load_pretrained_fn(config=config)
    try:
        pretrained_model.load_state_dict(torch.load(finetuned_model_local_path, map_location=map_location))
        # pretrained_model becomes finetuned_model
        # pretrained_model = xm.send_cpu_data_to_device(pretrained_model, device)
        pretrained_model.to(device)
    except:
        raise RuntimeError('Error(s) in loading state_dict for %s.' % evaluator_name)

    eval_fn_args = []
    eval_fn_kwargs = dict({"finetuned_model": pretrained_model, "tokenizer": tokenizer, "device": device}, **kwargs)

    return eval_fn_args, eval_fn_kwargs