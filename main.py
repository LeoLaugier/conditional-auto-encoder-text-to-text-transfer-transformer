import functools
import os
import time
import warnings

import fasttext
import t5
import tensorflow as tf
import tensorflow_datasets as tfds
import pprint

# Improve logging.
from contextlib import contextmanager
import logging as py_logging

from automatic_metrics import our_bleu, ppl, style_accuracy
from datasets import raw_to_tsv, raw_to_fasttext_input

from apiclient.http import MediaIoBaseDownload
from googleapiclient.discovery import build
import kenlm

from my_t5_data_utils import TfdsTask_ll, TaskRegistry_ll


def test_tpu():
    TPU_WORKER = "10.240.1.2:8470"

    with tf.compat.v1.Session('grpc://' + TPU_WORKER) as session:
        print('TPU devices:')
        pprint.pprint(session.list_devices())

    print("hello word end")


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)


def st_preprocessor(ds):
    def normalize_text(text):
        """Lowercase and remove quotes from a TensorFlow string."""
        text = tf.strings.lower(text)
        # text = tf.strings.regex_replace(text,"'(.*)'", r"\1")
        # TODO: add cleaning methods suitable for "dirty" comments. Maybe perspective has it? Or SentencePiece already does it.
        return text

    def to_inputs_and_targets(ex):
        """Map {"text": ..., [...], "toxicity": ...}->
               {"inputs": ..., ["style": ..., "codeprefixedtargets": ...,]
                "targets": ...}."""
        if DATASET == "IMDB":
            style = tf.strings.to_number(ex["style"], tf.int32)
        elif DATASET == "CCTK":
            style = tf.dtypes.cast(tf.round(ex["toxicity"]), tf.int32)

        if STYLE_DEPENDANT_PREFIX_INPUT:
            if tf.math.equal(style, 0):
                inputs = tf.strings.join(
                    [INPUT_PREFIX_STYLE_1, normalize_text(ex["text"])])
            else:
                inputs = tf.strings.join(
                    [INPUT_PREFIX_STYLE_2, normalize_text(ex["text"])])
        else:
            inputs = tf.strings.join(
                ["", normalize_text(ex["text"])])

        targets = normalize_text(ex["text"])

        ex_processed = {"inputs": inputs, "targets": targets}

        if STYLE_BIT:
            ex_processed["style"] = tf.expand_dims(style + 1,
                                                   0)  # +1 because 0 considered as padding so styles are 1 and 2

        if STYLE_DEPENDANT_PREFIX_TARGET:
            if tf.math.equal(style, 0):
                codeprefixedtargets = tf.strings.join(
                    [TARGET_PREFIX_STYLE_1, normalize_text(ex["text"])])  # For training
                codeprefix = tf.strings.join([TARGET_PREFIX_STYLE_2, ""])  # For eval, the other code prefix
            else:
                codeprefixedtargets = tf.strings.join(
                    [TARGET_PREFIX_STYLE_2, normalize_text(ex["text"])])  # For training
                codeprefix = tf.strings.join([TARGET_PREFIX_STYLE_1, ""])  # For eval, the other code prefix

            ex_processed["codeprefixedtargets"] = codeprefixedtargets
            ex_processed["codeprefix"] = codeprefix

        return ex_processed

    return ds.map(to_inputs_and_targets,
                  num_parallel_calls=tf.data.experimental.AUTOTUNE)

def denoise_ll(dataset,
            vocabulary,
            noise_density=1.0, #0.15
            noise_mask_fn=t5.data.preprocessors.iid_noise_mask,
            inputs_fn=t5.data.preprocessors.permute_noise_tokens,  # noise_token_to_random_token_or_sentinel, #  noise_token_to_sentinel,
            targets_fn=None,
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
    if STYLE_BIT:
      ex['style'] = features['style']
    if STYLE_DEPENDANT_PREFIX_TARGET:
      ex['codeprefixedtargets'] = features['codeprefixedtargets']
      ex['codeprefix'] = features['codeprefix']
    return ex
  return dataset.map(my_fn, num_parallel_calls=tf.data.experimental.AUTOTUNE)




def main():
    if ON_CLOUD:
        if USE_COLAB_TPU:
            assert "COLAB_TPU_ADDR" in os.environ, "ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!"
            TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"]
        else:
            TPU_ADDRESS = "grpc://" + TPU_WORKER

        TPU_TOPOLOGY = "2x2"
        print("TPU address is", TPU_ADDRESS)

        # from google.colab import auth
        # auth.authenticate_user()
        with tf.Session(TPU_ADDRESS) as session:
            print('TPU devices:')
            pprint.pprint(session.list_devices())

            # Upload credentials to TPU..
            # with open('/content/adc.json', 'r') as f:
            # auth_info = json.load(f)
            # tf.contrib.cloud.configure_gcs(session, credentials=auth_info)
            # Now credentials are set for all future sessions on this TPU.

    ## Install and import required packages
    warnings.filterwarnings("ignore", category=DeprecationWarning)

    if ON_CLOUD:
        tf.get_logger().propagate = False
        py_logging.root.setLevel('INFO')

    # Denoising autoencoder
    DENOISE = [(t5.data.preprocessors.noise_token_to_random_token_or_sentinel,
                0.15)]  # [(inputs_fn, noise_density), ...] or None

    NQ_RAW_DIR = "gs://test-t5/imdb_processed"

    imdb_tsv_path = {
        "train": os.path.join(DATA_DIR, "imdb-train.tsv"),
        "validation": os.path.join(DATA_DIR, "imdb-validation.tsv")
    }

    if DATASET == "IMDB":
      tf.logging.info("Generating IMDB TSVs.")
      # TODO if not train tsv in bucket...
      raw_to_tsv(os.path.join(NQ_RAW_DIR, "train.pos"), os.path.join(NQ_RAW_DIR, "train.neg"), imdb_tsv_path["train"])
      # TODO if not validation tsv in bucket...
      raw_to_tsv(os.path.join(NQ_RAW_DIR, "dev.pos"), os.path.join(NQ_RAW_DIR, "dev.neg"), imdb_tsv_path["validation"])

    if DATASET == "CCTK":
        tf.logging.info("Saving CCTK")
        ds = tfds.load(
            "civil_comments",
            data_dir=DATA_DIR,
            # Download data locally for preprocessing to avoid using GCS space.
            download_and_prepare_kwargs={"download_dir": "./downloads"})

        print("A few raw validation examples...")
        for ex in tfds.as_numpy(ds["validation"].take(5)):
            print(ex)

    elif DATASET == "IMDB":
        def st_imdb_dataset_fn(split, shuffle_files=False):
            del shuffle_files

            ds = tf.data.TextLineDataset(imdb_tsv_path[split])
            # Split each "<question>\t<answer>" example into (question, answer) tuple.
            ds = ds.map(
                functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                                  field_delim="\t", use_quote_delim=False),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = ds.map(lambda *ex: dict(zip(["text", "style"], ex)))
            return ds

        print("A few raw validation examples...")
        for ex in tfds.as_numpy(st_imdb_dataset_fn("validation").take(5)):
            print(ex)


    # perplexity
    gcs_service = build('storage', 'v1')
    imdb_ppl_path = "ppl_imdb.binary"  # TODO Adapt it to other datasets
    local_imdb_ppl_path = imdb_ppl_path
    gcs_imdb_ppl_path = os.path.join('ppl_binaries', imdb_ppl_path)

    if not os.path.exists(local_imdb_ppl_path):
        with open(local_imdb_ppl_path, 'wb') as f:  # TODO if we need to pre-train a KenLM model, do it here
            request = gcs_service.objects().get_media(bucket="test-t5",
                                                      object=gcs_imdb_ppl_path)
            media = MediaIoBaseDownload(f, request)

            done = False
            while not done:
                # _ is a placeholder for a progress object that we ignore.
                # (Our file is small, so we skip reporting progress.)
                status, done = media.next_chunk()
                print(status)

        print('Download ppl_imd.binary complete')

    imdb_ppl_model = kenlm.Model(imdb_ppl_path)

    # Metric: style accuracy
    imdb_fasttext_input_path = {
        "train": os.path.join(DATA_DIR, "imdb_fasttext_input.train"),
        "validation": os.path.join(DATA_DIR, "imdb_fasttext_input.valid")
    }

    imdb_acc_path = "acc_imdb.bin"  # TODO Adapt it to other datasets
    local_imdb_acc_path = imdb_acc_path
    gcs_imdb_acc_path = os.path.join('acc_binaries', imdb_acc_path)

    if not os.path.exists(local_imdb_acc_path):
        try:
            with open(local_imdb_acc_path, 'wb') as f:
                request = gcs_service.objects().get_media(bucket="test-t5",
                                                          object=gcs_imdb_acc_path)
                media = MediaIoBaseDownload(f, request)

                done = False
                while not done:
                    # _ is a placeholder for a progress object that we ignore.
                    # (Our file is small, so we skip reporting progress.)
                    status, done = media.next_chunk()
                    print(status)

            print('Download acc_imdb.binary complete')
        except:  # TODO specify error...
            tf.logging.info(
                "Pre-trained fasttext binary not found on bucket, we will pre-train a fasttext style classifier")
            tf.logging.info("Generating IMDB fasttext inputs.")
            raw_to_fasttext_input(os.path.join(NQ_RAW_DIR, "train.pos"), os.path.join(NQ_RAW_DIR, "train.neg"),
                                  imdb_fasttext_input_path["train"])
            raw_to_fasttext_input(os.path.join(NQ_RAW_DIR, "dev.pos"), os.path.join(NQ_RAW_DIR, "dev.neg"),
                                  imdb_fasttext_input_path["validation"])

            imdb_fasttext_input_local_path = {
                "train": "imdb_fasttext_input.train",
                "validation": "imdb_fasttext_input.valid"
            }

            if not os.path.exists(imdb_fasttext_input_local_path["train"]):
                with open(imdb_fasttext_input_local_path["train"], 'wb') as f:
                    request = gcs_service.objects().get_media(bucket="test-t5",
                                                              object=os.path.join(DATA_DIR_NAME,
                                                                                  "imdb_fasttext_input.train"))
                    media = MediaIoBaseDownload(f, request)

                    done = False
                    while not done:
                        # _ is a placeholder for a progress object that we ignore.
                        # (Our file is small, so we skip reporting progress.)
                        status, done = media.next_chunk()
                        print(status)

                print('Download IMDB fasttext train inputs complete')

            if not os.path.exists(imdb_fasttext_input_local_path["validation"]):
                with open(imdb_fasttext_input_local_path["validation"], 'wb') as f:
                    request = gcs_service.objects().get_media(bucket="test-t5",
                                                              object=os.path.join(DATA_DIR_NAME,
                                                                                  "imdb_fasttext_input.valid"))
                    media = MediaIoBaseDownload(f, request)

                    done = False
                    while not done:
                        # _ is a placeholder for a progress object that we ignore.
                        # (Our file is small, so we skip reporting progress.)
                        status, done = media.next_chunk()
                        print(status)

                print('Download IMDB fasttext validation inputs complete')


            !cat
            "imdb_fasttext_input.train" | shuf > "imdb_fasttext_shuffled_input.train"

            model = fasttext.train_supervised(input="imdb_fasttext_shuffled_input.train", lr=1.0, epoch=25,
                                              wordNgrams=2, bucket=200000, dim=50, loss='hs')
            model.test_label(imdb_fasttext_input_local_path["validation"], k=1)
            model.save_model("acc_imdb.bin")

    classifier_imdb = fasttext.load_model(imdb_acc_path)

    print(classifier_imdb.predict(["The acting is awesome"]))

    output_features = ["inputs", "targets"]

    if STYLE_BIT:
        output_features.append("style")

    if STYLE_DEPENDANT_PREFIX_TARGET:
        output_features.append("codeprefixedtargets")
        output_features.append("codeprefix")

    if DENOISE:
        token_preprocessor = []
        for inputs_fn, noise_density in DENOISE:
            token_preprocessor.append(functools.partial(denoise_ll, noise_density=0.15,
                                                        noise_mask_fn=t5.data.preprocessors.iid_noise_mask,
                                                        inputs_fn=t5.data.preprocessors.noise_token_to_random_token_or_sentinel,
                                                        targets_fn=None))
    else:
        token_preprocessor = None

    if DATASET == "CCTK":
        task_kwargs = {"tfds_name": "civil_comments:0.9.0", "tfds_data_dir": DATA_DIR}
        task_cls = [TfdsTask_ll]
    elif DATASET == "IMDB":
        task_kwargs = {"dataset_fn": st_imdb_dataset_fn}
        task_cls = []

    TaskRegistry_ll.add(
        TASK_NAME,
        # Supply a function which returns a tf.data.Dataset.
        *task_cls,
        splits=["train", "validation"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=[st_preprocessor],
        # Use the same vocabulary that we used for pre-training.
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text,
        # We'll use accuracy as our evaluation metric.
        metric_fns=[our_bleu, functools.partial(ppl, imdb_ppl_model=imdb_ppl_model),
                    functools.partial(style_accuracy, classifier_imdb=classifier_imdb)],
        # Not required, but helps for mixing and auto-caching.
        # num_input_examples=num_nq_examples
        token_preprocessor=token_preprocessor,
        output_features=output_features,
        **task_kwargs
    )

if __name__ == "__main__":
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    # app.run(main)

    ## Experience global variables
    # Setup gin interactive mode
    ENTER_GIN_INTERACTIVE_MODE = True

    # if we use our own CloudTPU
    TPU_WORKER = "10.240.1.2:8470"

    # Preprocess dataset
    STYLE_EMBEDDING_ENCODER = False
    STYLE_EMBEDDING_DECODER = False
    STYLE_EMBEDDING = STYLE_EMBEDDING_ENCODER or STYLE_EMBEDDING_DECODER
    STYLE_BIT = True
    STYLE_NUM = 2  # TODO 2 styles at the beggining but we can add more styles in gin file.
    LAMBDA_STYLE = 1  # 10 not enough (?) # 100

    STYLE_DEPENDANT_PREFIX_INPUT = False
    INPUT_PREFIX_STYLE_1 = "Negative: "  # "translate non toxic to non toxic: " # "non toxic comment: "
    INPUT_PREFIX_STYLE_2 = "Positive: "  # "translate toxic to toxic: " # "toxic comment: "

    STYLE_DEPENDANT_PREFIX_TARGET = True  # True

    # Balance samples per style (for CCTK)
    BALANCE_STYLES = False
    BALANCE_RATE = 0

    # Task / dataset
    DATASET = "IMDB"  # CCTK or IMDB
    counter = 124
    if DATASET == "IMDB":
        TASK_NAME = "st_imdb"
        MIXTURE_NAME = "st_imdb_mixture"
        MODELS_DIR_NAME = "models_style_imdb_%d" % counter
        DATA_DIR_NAME = "data_style_imdb"

        TARGET_PREFIX_STYLE_1 = "Negative: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Positive: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Negative": "1", "Positive": "2"}

    else:
        TASK_NAME = "st_toxic_comments"
        MIXTURE_NAME = "st_toxic_comments_mixture"
        MODELS_DIR_NAME = "models_style_civil_comment_%d" % counter
        DATA_DIR_NAME = "data_style_civil_comment"

        TARGET_PREFIX_STYLE_1 = "Non toxic: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Toxic: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Non toxic": "1", "Toxic": "2"}

        BALANCE_STYLES = True
        BALANCE_RATE = 0.0736  # = 08% / (1 - 08%)

    # Make same-style batches and alternate batches of each style
    GROUP_BY_STYLE = True

    # Weather to shift decoder's output with pad id
    SHIFT_DECODER_OUTPUT = True

    # SummAE-like Encoder-Decoder
    CUT_CROSS_ATTENTION = True

    # Unit testing
    RUN_UNITTESTS = False

    # Predict and Eval time
    HAS_PARTIAL_SEQUENCES = STYLE_DEPENDANT_PREFIX_TARGET  # whether to start generation with a partial sequences
    REMOVE_PARTIAL_SEQUENCES = True  # whether to remove partial sequences when printing generated sequences

    # Eval
    EVAL = True
    UNSUPERVISED_STYLE_TRANSFER_METRICS = True

    # GCS setting
    BASE_DIR = "gs://test-t5/"  # @param { type: "string" }
    DATA_DIR = os.path.join(BASE_DIR, DATA_DIR_NAME)
    MODELS_DIR = os.path.join(BASE_DIR, MODELS_DIR_NAME)
    ON_CLOUD = True
    USE_COLAB_TPU = False
    main()