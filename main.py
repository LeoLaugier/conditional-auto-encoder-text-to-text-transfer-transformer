import functools
import os
import pprint
import time
import warnings
# Improve logging.
from contextlib import contextmanager
import logging as py_logging

import fasttext
import gin
import kenlm
import subprocess

import re
from mesh_tensorflow.transformer import transformer, utils
import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import t5.data.sentencepiece_vocabulary as sentencepiece_processor
from googleapiclient.discovery import build
import tensorflow_hub as hub

from automatic_metrics.automatic_metrics import our_bleu, perplexity, style_accuracy, sentence_similarity
from dataset import raw_to_tsv, st_preprocessor, raw_to_fasttext_input
from eval import print_random_predictions
from my_mesh_tensorflow_transformer_transformer import make_bitransformer_ll, Unitransformer_ll
from my_mesh_tensorflow_transformer_utils import build_model_ll, tpu_estimator_model_fn_ll
from my_t5_data_preprocessors import denoise_ll
from my_t5_data_utils import TaskRegistry_ll, MixtureRegistry_ll, TfdsTask_ll
from my_utils import download_from_bucket_to_local, upload_blob
from my_t5_models_mtf_model import MtfModel_ll


def test_tpu():
    tpu_worker = "10.240.1.2:8470"

    with tf.compat.v1.Session('grpc://' + tpu_worker) as session:
        print('TPU devices:')
        pprint.pprint(session.list_devices())

    print("hello word end")


@contextmanager
def tf_verbosity_level(level):
  og_level = tf.logging.get_verbosity()
  tf.logging.set_verbosity(level)
  yield
  tf.logging.set_verbosity(og_level)

def main():
    if ON_CLOUD:
        if USE_TPU:
            if USE_COLAB_TPU:
                assert "COLAB_TPU_ADDR" in os.environ, "ERROR: Not connected to a TPU runtime; please see the first cell in this notebook for instructions!"
                TPU_ADDRESS = "grpc://" + os.environ["COLAB_TPU_ADDR"]
            else:
                TPU_ADDRESS = "grpc://" + TPU_WORKER

            TPU_TOPOLOGY = "2x2"
            print("TPU address is", TPU_ADDRESS)
        else:
            TPU_ADDRESS = None
            TPU_TOPOLOGY = "2x2"

        #from google.colab import auth
        #auth.authenticate_user()
        ##with tf.compat.v1.Session(TPU_ADDRESS) as session:
           ##print('TPU devices:')
           ##pprint.pprint(session.list_devices())

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

    task_cls = []
    task_kwargs = {}
    ## Generating and / or loading datasets
    if DATASET == "IMDB":
        dataset_tsv_path = {
            "train": os.path.join(DATA_DIR, "%s-train.tsv" % DATASET.lower()),
            "validation": os.path.join(DATA_DIR, "%s-validation.tsv" % DATASET.lower())
        }

        train_tsv_exists = tf.io.gfile.exists(dataset_tsv_path["train"])
        validation_tsv_exists = tf.io.gfile.exists(dataset_tsv_path["validation"])

        # Generating tsv datasets
        if not train_tsv_exists or not validation_tsv_exists:
            tf.compat.v1.logging.info("Generating T5 TSVs.")

            if not train_tsv_exists:
                raw_to_tsv(os.path.join(DATASET_RAW_DIR, "train.pos"),
                           os.path.join(DATASET_RAW_DIR, "train.neg"),
                           dataset_tsv_path["train"])
            if not validation_tsv_exists:
                raw_to_tsv(os.path.join(DATASET_RAW_DIR, "dev.pos"),
                           os.path.join(DATASET_RAW_DIR, "dev.neg"),
                           dataset_tsv_path["validation"])
            tf.compat.v1.logging.info("T5 TSVs generated.")
        # Loading datasets
        def tsv_to_dataset_fn(split, shuffle_files=False):
            # We only have one file for each split.
            del shuffle_files

            # Load lines from the text file as examples.
            ds = tf.data.TextLineDataset(dataset_tsv_path[split])
            ds = ds.map(
                functools.partial(tf.io.decode_csv, record_defaults=["", ""],
                                  field_delim="\t", use_quote_delim=False),
                num_parallel_calls=tf.data.experimental.AUTOTUNE)

            ds = ds.map(lambda *ex: dict(zip(["text", "style"], ex)))
            return ds

        print("A few raw validation examples...")
        for ex in tfds.as_numpy(tsv_to_dataset_fn("validation").take(5)):
            print(ex)

        task_kwargs = {"dataset_fn": tsv_to_dataset_fn}
        task_cls = []

    elif DATASET=="CCTK":
        tf.compat.v1.logging.info("Saving CCTK")
        ds = tfds.load(
            "civil_comments",
            data_dir=DATA_DIR,
            # Download data locally for preprocessing to avoid using GCS space.
            download_and_prepare_kwargs={"download_dir": "./downloads"})

        print("A few raw validation examples...")
        for ex in tfds.as_numpy(ds["validation"].take(5)):
            print(ex)

        task_kwargs = {"tfds_name": "civil_comments:0.9.0", "tfds_data_dir": DATA_DIR,
                       "balance_styles": BALANCE_STYLES, "balance_rate": BALANCE_RATE}
        task_cls = [TfdsTask_ll]


    ### Metrics
    metric_fns = [our_bleu]
    gcs_service = build('storage', 'v1')

    ## Perplexity
    pretrained_ppl_filename = '%s_%s.binary' % ("ppl", DATASET.lower())
    pretrained_ppl_local_path = os.path.join('ppl_binaries', pretrained_ppl_filename)
    pretrained_ppl_gcs_path = os.path.join('ppl_binaries', pretrained_ppl_filename)

    if os.path.exists(pretrained_ppl_local_path) or tf.io.gfile.exists(os.path.join(BASE_DIR, pretrained_ppl_gcs_path)):
        if not os.path.exists(pretrained_ppl_local_path):  # Pre-trained ppl model found in GCS
            tf.compat.v1.logging.info("Downloading pre-trained perplexity model from GCS...")
            download_from_bucket_to_local(gcs_service, BUCKET, pretrained_ppl_gcs_path, pretrained_ppl_local_path)
            tf.compat.v1.logging.info('Download %s complete' % pretrained_ppl_filename)

        pretrained_ppl_model = kenlm.Model(pretrained_ppl_local_path)
        metric_fns.append(functools.partial(perplexity, ppl_model=pretrained_ppl_model))
    else:  # Train and upload ppl model
        tf.compat.v1.logging.warn(
            "Pre-trained perplexity model not found neither on local nor on bucket."
            "If you want a perplexity metric, please pre-train a perplexity language model with KenLM."
            "Instructions here: https://kheafield.com/code/kenlm/"
        )

    ## Accuracy
    pretrained_acc_filename = '%s_%s.bin' % ("acc", DATASET.lower())
    pretrained_acc_local_path = os.path.join('acc_binaries', pretrained_acc_filename)
    pretrained_acc_gcs_path = os.path.join('acc_binaries', pretrained_acc_filename)

    if not os.path.exists(pretrained_acc_local_path):
        if tf.io.gfile.exists(os.path.join(BASE_DIR, pretrained_acc_gcs_path)):
            tf.compat.v1.logging.info("Downloading pre-trained accuracy model from GCS...")
            download_from_bucket_to_local(gcs_service, BUCKET, pretrained_acc_gcs_path, pretrained_acc_local_path)
            tf.compat.v1.logging.info('Download %s complete' % pretrained_acc_filename)

        else:
            tf.compat.v1.logging.info(
                "Pre-trained fasttext binary not found on bucket, we will pre-train a fasttext style classifier")

            # TODO Adapt to CCTK because "train.pos" et "train.neg" n'existent pas dans CCTK
            if DATASET == "IMDB":
                fasttext_input_gcs_path = {
                    "train": os.path.join(DATA_DIR, "%s_fasttext_input.train" % DATASET.lower()),
                    "validation": os.path.join(DATA_DIR, "%s_fasttext_input.valid" % DATASET.lower())
                }

                fasttext_data_local_path = os.path.join("fasttext_files", DATASET.lower())
                fasttext_input_local_path = {
                    "train": os.path.join(fasttext_data_local_path, "%s_fasttext_input.train" % DATASET.lower()),
                    "validation": os.path.join(fasttext_data_local_path, "%s_fasttext_input.valid" % DATASET.lower())
                }

                train_fasttext_input_exists_on_bucket = tf.io.gfile.exists(os.path.join(BASE_DIR,
                                                                                        fasttext_input_gcs_path["train"]))
                validation_fasttext_input_exists_on_bucket = tf.io.gfile.exists(os.path.join(BASE_DIR,
                                                                                             fasttext_input_gcs_path["validation"]))

                if not train_fasttext_input_exists_on_bucket or not validation_fasttext_input_exists_on_bucket:
                    tf.compat.v1.logging.info("Generating fasttext inputs on bucket...")

                    if not train_fasttext_input_exists_on_bucket:
                        raw_to_fasttext_input(os.path.join(DATASET_RAW_DIR, "train.pos"),
                                              os.path.join(DATASET_RAW_DIR, "train.neg"),
                                              fasttext_input_gcs_path["train"])
                    if not validation_fasttext_input_exists_on_bucket:
                        raw_to_fasttext_input(os.path.join(DATASET_RAW_DIR, "dev.pos"),
                                              os.path.join(DATASET_RAW_DIR, "dev.neg"),
                                              fasttext_input_gcs_path["validation"])
                    tf.compat.v1.logging.info("Fasttext inputs generated on bucket.")

                train_fasttext_input_exists_on_local = os.path.exists(fasttext_input_local_path["train"])
                validation_fasttext_input_exists_on_local = os.path.exists(fasttext_input_local_path["validation"])
                if not train_fasttext_input_exists_on_local:
                    tf.compat.v1.logging.info("Downloading fasttext train input from GCS...")
                    download_from_bucket_to_local(gcs_service, BUCKET,
                                                  fasttext_input_gcs_path["train"],
                                                  fasttext_input_local_path["train"])
                    tf.compat.v1.logging.info('Download %s complete' % "fasttext_input.train")

                if not validation_fasttext_input_exists_on_local:
                    tf.compat.v1.logging.info("Downloading fasttext validation input from GCS...")
                    download_from_bucket_to_local(gcs_service, BUCKET,
                                                  fasttext_input_gcs_path["train"],
                                                  fasttext_input_local_path["validation"])
                    tf.compat.v1.logging.info('Download %s complete' % "fasttext_input.valid")

                fasttext_shuffled_input_local_path = os.path.join(fasttext_data_local_path,
                                                                  "%s_fasttext_shuffled_input.train" % DATASET.lower())
                if not os.path.exists(fasttext_shuffled_input_local_path):
                    subprocess.call('cat "%s" | shuf > "%s"' % (fasttext_input_local_path["train"],
                                                                fasttext_shuffled_input_local_path))
                tf.compat.v1.logging.info('Training fasttext...')
                model = fasttext.train_supervised(input=fasttext_shuffled_input_local_path, lr=1.0, epoch=25,
                                                  wordNgrams=2, bucket=200000, dim=50, loss='hs')
                tf.compat.v1.logging.info('Done with fasttext training.')
                model.test_label(fasttext_input_local_path["validation"], k=1)
                model.save_model(pretrained_acc_local_path)

                tf.compat.v1.logging.info('Uploading the pre-trained fasttext model to the bucket...')
                upload_blob(BUCKET, pretrained_acc_local_path, pretrained_acc_gcs_path)
                tf.compat.v1.logging.info('Done with uploading the pre-trained fasttext model %s to the bucket.'
                                % pretrained_acc_filename)

    pretrained_acc_model = fasttext.load_model(pretrained_acc_local_path)
    metric_fns.append(functools.partial(style_accuracy, classifier_model=pretrained_acc_model))

    ## Similarity
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    pretrained_sentence_similarity_model = hub.Module(module_url)
    metric_fns.append(functools.partial(sentence_similarity,
                                        sentence_similarity_model=pretrained_sentence_similarity_model))


    output_features = ["inputs", "targets"]
    if STYLE_BIT:
        output_features.append("style")

    if STYLE_DEPENDANT_PREFIX_TARGET:
        output_features.append("codeprefixedtargets")
        output_features.append("codeprefix")

    text_preprocessor = functools.partial(st_preprocessor, dataset=DATASET, style_bit=STYLE_BIT,
                    style_dependant_prefix_input=STYLE_DEPENDANT_PREFIX_INPUT,
                    input_prefix_style_1=INPUT_PREFIX_STYLE_1, input_prefix_style_2=INPUT_PREFIX_STYLE_2,
                    style_dependant_prefix_target=STYLE_DEPENDANT_PREFIX_TARGET,
                    target_prefix_style_1=TARGET_PREFIX_STYLE_1, target_prefix_style_2=TARGET_PREFIX_STYLE_2)

    if DENOISE:
        token_preprocessor = []
        for inputs_fn, noise_density in DENOISE:
            token_preprocessor.append(functools.partial(denoise_ll, noise_density=noise_density,
                                                        noise_mask_fn=t5.data.preprocessors.iid_noise_mask,
                                                        inputs_fn=inputs_fn,
                                                        targets_fn=None))
    else:
        token_preprocessor = None

    # TODO CCTK task parameters

    TaskRegistry_ll.add(
        TASK_NAME,
        *task_cls,
        splits=["train", "validation"],
        # Supply a function which preprocesses text from the tf.data.Dataset.
        text_preprocessor=[text_preprocessor],
        # Use the same vocabulary that we used for pre-training.
        sentencepiece_model_path=t5.data.DEFAULT_SPM_PATH,
        # Lowercase targets before computing metrics.
        postprocess_fn=t5.data.postprocessors.lower_text,
        metric_fns=metric_fns,
        # Not required, but helps for mixing and auto-caching.
        # num_input_examples=num_nq_examples
        token_preprocessor=token_preprocessor,
        output_features=output_features,
        **task_kwargs
    )

    # Load and print a few examples.
    st_task = TaskRegistry_ll.get(TASK_NAME)
    sequence_length = {"inputs": 128, "targets": 128}
    if STYLE_BIT:
        sequence_length["style"] = 128  # Or "style": 1 but packing not efficient...
    if STYLE_DEPENDANT_PREFIX_TARGET:
        sequence_length["codeprefixedtargets"] = 128
        sequence_length["codeprefix"] = 128

    ds = st_task.get_dataset(split="validation", sequence_length=sequence_length)

    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(3)):
        print(ex)

    MixtureRegistry_ll.add(
        MIXTURE_NAME,
        [TASK_NAME],
        default_rate=1.0
    )


    # TODO Unittests about preprocessing dataset

    # Modified T5 "style-aware denoising autoencoder (pre-trained) bi-transformer"
    utils.build_model = functools.partial(build_model_ll ,
                                          style_embedding_encoder=STYLE_EMBEDDING_ENCODER,
                                          style_embedding_decoder=STYLE_EMBEDDING_DECODER,
                                          style_num=STYLE_NUM,
                                          cut_cross_attention=CUT_CROSS_ATTENTION)

    transformer.make_bitransformer_ll = make_bitransformer_ll

    transformer.Unitransformer_ll = Unitransformer_ll

    utils.tpu_estimator_model_fn = functools.partial(tpu_estimator_model_fn_ll,
                                                     has_partial_sequences=HAS_PARTIAL_SEQUENCES,
                                                     remove_partial_sequences=REMOVE_PARTIAL_SEQUENCES,
                                                     style_embedding=STYLE_EMBEDDING,
                                                     style_dependant_prefix_target=STYLE_DEPENDANT_PREFIX_TARGET,
                                                     cycle_consistency_loss=CYCLE_CONSISTENCY_LOSS,
                                                     lambda_ae=LAMBDA_AE,
                                                     lambda_cycle=LAMBDA_CYCLE)

    if ON_CLOUD and MODEL_SIZE == "3B":
        tf.compat.v1.logging.warn(
            "The `3B` model is too large to use with the 5GB GCS free tier. "
            "Make sure you have at least 25GB on GCS before continuing."
        )
    elif ON_CLOUD and MODEL_SIZE == "11B":
        raise ValueError(
            "The `11B` parameter is too large to fine-tune on the `v2-8` TPU "
            "provided by Colab. Please comment out this Error if you're running "
            "on a larger TPU."
        )

    # Set parallelism and batch size to fit on v2-8 TPU (if possible).
    # Limit number of checkpoints to fit within 5GB (if possible).
    model_parallelism, train_batch_size, keep_checkpoint_max = {
        "small": (1, 256, 16),
        "base": (2, 128, 8),
        "large": (8, 64, 4),
        "3B": (8, 16, 1),
        "11B": (8, 16, 1)}[MODEL_SIZE]

    tf.io.gfile.makedirs(MODEL_DIR)

    sequence_length = {"inputs": 128, "targets": 128}
    if STYLE_BIT:
        sequence_length["style"] = 128
    if STYLE_DEPENDANT_PREFIX_TARGET:
        sequence_length["codeprefixedtargets"] = 128
        sequence_length["codeprefix"] = 128

    # Ou alors, based on L. 357-362  https://github.com/tensorflow/mesh/blob/a719398c92a48990921e57608ef99553ad1b1a85/mesh_tensorflow/transformer/utils.py#L357
    # ignore et appeler le feature style "input_style" (dans ce cas length of style = 128)

    my_tokenizer = sentencepiece_processor.SentencePieceVocabulary(t5.data.DEFAULT_SPM_PATH)
    left_pad_amt_1 = len(my_tokenizer.encode(TARGET_PREFIX_STYLE_1)) - 1
    left_pad_amt_2 = len(my_tokenizer.encode(TARGET_PREFIX_STYLE_2)) - 1

    model = MtfModel_ll(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length=sequence_length,
        learning_rate_schedule=0.003,
        save_checkpoints_steps=5000,
        keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
        iterations_per_loop=100,
        model_type="bitransformer_ll",
        style_bit=STYLE_BIT,
        unsupervised_style_transfer_metrics=UNSUPERVISED_STYLE_TRANSFER_METRICS,
        style_dependant_prefix_target=STYLE_DEPENDANT_PREFIX_TARGET,
        style_embedding=STYLE_EMBEDDING,
        shift_decoder_output=SHIFT_DECODER_OUTPUT,
        left_pad_amt_1=left_pad_amt_1,
        left_pad_amt_2=left_pad_amt_2,
        target_prefix_style_1=TARGET_PREFIX_STYLE_1,
        target_prefix_style_2=TARGET_PREFIX_STYLE_2
    )

    with gin.unlock_config():
        gin.parse_config_file("gs://test-t5/unitransformer_ll.gin")

    FINETUNE_STEPS = 500

    model.finetune(
        mixture_or_task_name=MIXTURE_NAME,
        pretrained_model_dir=PRETRAINED_DIR,
        finetune_steps=FINETUNE_STEPS
    )

    if EVAL:
        # Use a larger batch size for evaluation, which requires less memory.
        model.batch_size = train_batch_size * 4
        model.eval(
            mixture_or_task_name=MIXTURE_NAME,
            checkpoint_steps="all"
        )

        print_random_predictions(TASK_NAME, sequence_length, MODEL_DIR, n=10)

    # evaluate on made-up comments
    comment_style_pairs = [

        {"text": "the casting is poor , the script is awful , and the directing is dreadful .",
         "Destination style": "Positive"},
        {"text": "the casting is poor , the script is awful , and the directing is dreadful .",
         "Destination style": "Negative"},

        {"text": "this humor is not funny , and is actually too gay for it 's own good .",
         "Destination style": "Positive"},
        {"text": "this humor is not funny , and is actually too gay for it 's own good .",
         "Destination style": "Negative"},

        {"text": "why would any legitimate actor having read the script participated in this piece of crap ?",
         "Destination style": "Positive"},
        {"text": "why would any legitimate actor having read the script participated in this piece of crap ?",
         "Destination style": "Negative"},

        {"text": "i give it 1 out of 10 because it 's the lowest grade imdb has to offer .",
         "Destination style": "Positive"},
        {"text": "i give it 1 out of 10 because it 's the lowest grade imdb has to offer .",
         "Destination style": "Negative"},

        {
            "text": "whenever disney characters reach adulthood these days they become either obnoxious or just plain boring .",
            "Destination style": "Positive"},
        {
            "text": "whenever disney characters reach adulthood these days they become either obnoxious or just plain boring .",
            "Destination style": "Negative"},

        {"text": "i 'm very , very sorry for every single minute i wasted on this one .",
         "Destination style": "Positive"},
        {"text": "i 'm very , very sorry for every single minute i wasted on this one .",
         "Destination style": "Negative"},

        {"text": "Very bad movie !",
         "Destination style": "Positive"},
        {"text": "Very bad movie !",
         "Destination style": "Negative"},

        {"text": "do not buy this movie .",
         "Destination style": "Positive"},
        {"text": "do not buy this movie .",
         "Destination style": "Negative"},

        {"text": "do buy this movie .",
         "Destination style": "Negative"},
        {"text": "do buy this movie .",
         "Destination style": "Positive"},

        {"text": "Very good movie !",
         "Destination style": "Negative"},
        {"text": "Very good movie !",
         "Destination style": "Positive"},

        {"text": "i actually really disliked her music before , but they fit perfectly in this movie .",
         "Destination style": "Negative"},
        {"text": "i actually really disliked her music before , but they fit perfectly in this movie .",
         "Destination style": "Positive"},

        {"text": "a comedy that will warm your heart .",
         "Destination style": "Negative"},
        {"text": "a comedy that will warm your heart .",
         "Destination style": "Positive"},

        {"text": "this may seem odd , but i highly recommend it .",
         "Destination style": "Negative"},
        {"text": "this may seem odd , but i highly recommend it .",
         "Destination style": "Positive"},

        {"text": "scott bakula is wonderful in the part of the senior investigator .",
         "Destination style": "Negative"},
        {"text": "scott bakula is wonderful in the part of the senior investigator .",
         "Destination style": "Positive"},

        {"text": "even if at times things do get a little slow , it 's still a rewarding and informative experience .",
         "Destination style": "Negative"},
        {"text": "even if at times things do get a little slow , it 's still a rewarding and informative experience .",
         "Destination style": "Positive"},

        {"text": "i give this movie 8 stars out of 10 .",
         "Destination style": "Negative"},
        {"text": "i give this movie 8 stars out of 10 .",
         "Destination style": "Positive"}
    ]

    comments = []
    for p in comment_style_pairs:
        comments.append(p["text"] + "|dst_style:" + STYLE_IDS[p["Destination style"]])

    now = time.time()
    # Write out the supplied questions to text files.
    predict_inputs_path = os.path.join(MODEL_DIR, "predict_inputs_%d.txt" % now)
    predict_outputs_path = os.path.join(MODEL_DIR, "predict_outputs_%d.txt" % now)
    # Manually apply preprocessing
    with tf.io.gfile.GFile(predict_inputs_path, "w") as f:
        for c in comments:
            c = re.sub(r'\n', r"\\n", c, flags=re.S)
            f.write("%s\n" % c.lower())

    # Ignore any logging so that we only see the model's answers to the questions.
    with tf_verbosity_level('ERROR'):
        model.batch_size = len(comments)
        model.predict(
            input_file=predict_inputs_path,
            output_file=predict_outputs_path,
            # Select the most probable output token at each step.
            temperature=0,
        )

    # The output filename will have the checkpoint appended so we glob to get
    # the latest.
    prediction_files = sorted(tf.io.gfile.glob(predict_outputs_path + "*"))
    print("\nPredictions using checkpoint %s:\n" % prediction_files[-1].split("-")[-1])
    with tf.io.gfile.GFile(prediction_files[-1]) as f:
        for c, g in zip(comments, f):
            if c:
                print("Initial text: " + c.split("|dst_style:")[0])
                print("Generated text: " + g)
                print()


if __name__ == "__main__":
    tf.compat.v1.disable_v2_behavior()
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
    # app.run(main)

    # TODO parser
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
    counter = 200
    if DATASET == "IMDB":
        TASK_NAME = "st_imdb"
        MIXTURE_NAME = "st_imdb_mixture"
        MODELS_DIR_NAME = "models_style_imdb_%d" % counter
        DATA_DIR_NAME = "data_style_imdb"

        TARGET_PREFIX_STYLE_1 = "Negative: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Positive: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Negative": "1", "Positive": "2"}
        DATASET_RAW_DIR = "gs://test-t5/imdb_processed"

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

        DATASET_RAW_DIR = None

    # Make same-style batches and alternate batches of each style
    GROUP_BY_STYLE = True

    # Weather to shift decoder's output with pad id
    SHIFT_DECODER_OUTPUT = True

    # SummAE-like Encoder-Decoder
    CUT_CROSS_ATTENTION = True

    # Cycle-Consistency loss
    CYCLE_CONSISTENCY_LOSS = True
    LAMBDA_AE = 1.0
    LAMBDA_CYCLE = 1.0

    # Unit testing
    RUN_UNITTESTS = False

    # Predict and Eval time
    HAS_PARTIAL_SEQUENCES = STYLE_DEPENDANT_PREFIX_TARGET  # whether to start generation with a partial sequences
    REMOVE_PARTIAL_SEQUENCES = True  # whether to remove partial sequences when printing generated sequences

    # Eval
    EVAL = True
    UNSUPERVISED_STYLE_TRANSFER_METRICS = True

    # GCS setting
    BUCKET = "test-t5"
    BASE_DIR = "gs://test-t5/"  # @param { type: "string" }
    DATA_DIR = os.path.join(BASE_DIR, DATA_DIR_NAME)
    MODELS_DIR = os.path.join(BASE_DIR, MODELS_DIR_NAME)
    ON_CLOUD = True
    USE_COLAB_TPU = True

    MODEL_SIZE = "small"  # ["small", "base", "large", "3B", "11B"]
    # Public GCS path for T5 pre-trained model checkpoints
    BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
    PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
    MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

    USE_TPU = True

    DENOISE = [(t5.data.preprocessors.noise_token_to_random_token_or_sentinel,
                0.15)]  # [(inputs_fn, noise_density), ...] or None

    main()