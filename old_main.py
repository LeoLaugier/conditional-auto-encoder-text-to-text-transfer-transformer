import functools
import os
import pprint
import re
import time
import warnings
# Improve logging.
from contextlib import contextmanager
import logging as py_logging

import fasttext
import gin
import kenlm
import subprocess

import torch
from transformers import AutoModelWithLMHead, BertForSequenceClassification, BertConfig, AutoTokenizer, AutoConfig
# import torch_xla.core.xla_model as xm

from mesh_tensorflow.transformer import transformer, utils
import tensorflow as tf
import tensorflow_datasets as tfds
import t5
import t5.data.sentencepiece_vocabulary as sentencepiece_processor
from googleapiclient.discovery import build
import tensorflow_hub as hub

from caet5.evaluation.metrics import bleu, kenlm_perplexity, sentence_similarity, \
    fasttext_attribute_accuracy, gpt_perplexity_batch_280, bert_attribute_accuracy_batch
from caet5.data.dataset import raw_to_tsv, at_preprocessor, raw_to_fasttext_input
from caet5.evaluation.eval_utils import print_random_predictions
from mesh_tensorflow_caet5.transformer import make_bitransformer_ll, Unitransformer_ll
from mesh_tensorflow_caet5.utils import build_model_ll, tpu_estimator_model_fn_ll
from caet5.data.preprocessors import denoise
from caet5.data.utils import TaskRegistry_ll, MixtureRegistry_ll, TfdsTask_ll
from caet5.evaluation.metrics_utils import download_from_bucket_to_local, upload_blob
from caet5.models.mtf_model import MtfModel_ll


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
    if DATASET == "IMDB" or DATASET == "YELP" or DATASET == "processed_CCTK":
        dataset_tsv_path = {
            "train": os.path.join(DATA_DIR, "%s-train.tsv" % DATASET.lower()),
            "validation": os.path.join(DATA_DIR, "%s-validation.tsv" % DATASET.lower())
        }

        if DATASET == "processed_CCTK":
            dataset_tsv_path["validation"] = os.path.join(DATA_DIR, "%s-toxic-validation.tsv" % DATASET.lower())
            dataset_tsv_path["test"] = os.path.join(DATA_DIR, "%s-toxic-test.tsv" % DATASET.lower())

        if DATASET == "YELP":
            dataset_tsv_path["test"] = os.path.join(DATA_DIR, "%s-test.tsv" % DATASET.lower())

        train_tsv_exists = tf.io.gfile.exists(dataset_tsv_path["train"])
        validation_tsv_exists = tf.io.gfile.exists(dataset_tsv_path["validation"])
        if DATASET == "processed_CCTK" or DATASET == "YELP":
            test_tsv_exists = tf.io.gfile.exists(dataset_tsv_path["test"])
        else:
            test_tsv_exists = True

        # Generating tsv datasets
        if not train_tsv_exists or not validation_tsv_exists or not test_tsv_exists:
            tf.compat.v1.logging.info("Generating T5 TSVs.")
            if DATASET == "IMDB" or DATASET == "YELP":
                ext0 = "neg"
                ext1 = "pos"
                mode ="rb"
            elif DATASET == "processed_CCTK":
                ext0 = "nontoxic"
                ext1 = "toxic"
                mode = "r"

            if not train_tsv_exists:
                raw_to_tsv(os.path.join(DATASET_RAW_DIR, "train.%s" % ext1),
                           os.path.join(DATASET_RAW_DIR, "train.%s" % ext0),
                           dataset_tsv_path["train"], mode)
            if not validation_tsv_exists:
                if DATASET == "IMDB" or DATASET == "YELP":
                    raw_to_tsv(os.path.join(DATASET_RAW_DIR, "dev.%s" % ext1),
                               os.path.join(DATASET_RAW_DIR, "dev.%s" % ext0),
                               dataset_tsv_path["validation"], mode)
                elif DATASET == "processed_CCTK":
                    raw_to_tsv(os.path.join(DATASET_RAW_DIR, "dev.%s" % ext1),
                               "",
                               dataset_tsv_path["validation"], mode)

            if not test_tsv_exists:
                if DATASET == "processed_CCTK" or DATASET == "YELP":
                    raw_to_tsv(os.path.join(DATASET_RAW_DIR, "test.%s" % ext1),
                               "",
                               dataset_tsv_path["test"], mode)

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

            ds = ds.map(lambda *ex: dict(zip(["text", "attribute"], ex)))
            return ds

        print("A few raw validation examples...")
        for ex in tfds.as_numpy(tsv_to_dataset_fn("validation").take(5)):
            print(ex)

        task_kwargs = {"dataset_fn": tsv_to_dataset_fn}
        task_cls = []

    elif DATASET == "CCTK":
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
                       "balance_attributes": BALANCE_attributeS, "balance_rate": BALANCE_RATE}
        task_cls = [TfdsTask_ll]


    ### Metrics
    metric_fns = [bleu]
    gcs_service = build('storage', 'v1')

    ## Perplexity
    pretrained_ppl_filename = '%s_%s.binary' % ("ppl", DATASET.lower())
    if PPL_ARCHITECTURE == "gpt2":
        if DATASET == "processed_CCTK":
            pretrained_ppl_filename = 'gpt2_%s_%s.pt' % ("ppl", "CCTK".lower())
        elif DATASET == "YELP":
            pretrained_ppl_filename = 'gpt2_%s_%s.pt' % ("ppl", "YELP".lower())
    pretrained_ppl_local_path = os.path.join('ppl_binaries', pretrained_ppl_filename)
    pretrained_ppl_gcs_path = os.path.join('ppl_binaries', pretrained_ppl_filename)

    if os.path.exists(pretrained_ppl_local_path) or tf.io.gfile.exists(os.path.join(BASE_DIR, pretrained_ppl_gcs_path)):
        if not os.path.exists(pretrained_ppl_local_path):  # Pre-trained ppl model found in GCS
            tf.compat.v1.logging.info("Downloading pre-trained perplexity model from GCS...")
            download_from_bucket_to_local(gcs_service, BUCKET, pretrained_ppl_gcs_path, pretrained_ppl_local_path)
            tf.compat.v1.logging.info('Download %s complete' % pretrained_ppl_filename)

        if PPL_ARCHITECTURE=="kenlm":
            pretrained_ppl_model = kenlm.Model(pretrained_ppl_local_path)
            metric_fns.append(functools.partial(kenlm_perplexity, ppl_model=pretrained_ppl_model))
        elif PPL_ARCHITECTURE=="gpt2":
            device = "cpu" # xm.xla_device()
            tokenizer = AutoTokenizer.from_pretrained("gpt2")
            config = AutoConfig.from_pretrained("gpt2")
            pretrained_ppl_model = AutoModelWithLMHead.from_config(config)
            pretrained_ppl_model.load_state_dict(torch.load(pretrained_ppl_local_path))
            # pretrained_ppl_model = xm.send_cpu_data_to_device(pretrained_ppl_model, device)
            pretrained_ppl_model.to(device)
            metric_fns.append(functools.partial(gpt_perplexity_batch_280, ppl_model=pretrained_ppl_model,
                                                tokenizer=tokenizer, device=device, batch_size=PPL_EVAL_BATCH_SIZE,
                                                block_size=PPL_BLOCK_SIZE))
    else:  # Train and upload ppl model
        tf.compat.v1.logging.warn(
            "Pre-trained perplexity model not found neither on local nor on bucket."
            "If you want a perplexity metric_name, please pre-train a perplexity language model with KenLM."
            "Instructions here: https://kheafield.com/code/kenlm/"
        )

    ## Accuracy
    pretrained_acc_filename = '%s_%s.bin' % ("acc", DATASET.lower())
    if ACC_ARCHITECTURE == "BERT":
        pretrained_acc_filename = 'bert_%s_%s.pt' % ("acc", DATASET.lower())
    pretrained_acc_local_path = os.path.join('acc_binaries', pretrained_acc_filename)
    pretrained_acc_gcs_path = os.path.join('acc_binaries', pretrained_acc_filename)

    if not os.path.exists(pretrained_acc_local_path):
        if tf.io.gfile.exists(os.path.join(BASE_DIR, pretrained_acc_gcs_path)):
            tf.compat.v1.logging.info("Downloading pre-trained accuracy model from GCS...")
            download_from_bucket_to_local(gcs_service, BUCKET, pretrained_acc_gcs_path, pretrained_acc_local_path)
            tf.compat.v1.logging.info('Download %s complete' % pretrained_acc_filename)

        else:
            tf.compat.v1.logging.info(
                "Pre-trained fasttext binary not found on bucket, we will pre-train a fasttext attribute classifier")


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

    if ACC_ARCHITECTURE == "fasttext":
        pretrained_acc_model = fasttext.load_model(pretrained_acc_local_path)
        metric_fns.append(functools.partial(fasttext_attribute_accuracy, classifier_model=pretrained_acc_model))
    elif ACC_ARCHITECTURE == "BERT":
        device = "cpu" # xm.xla_device()
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        config = BertConfig.from_pretrained("bert-base-uncased", num_labels=1)
        pretrained_acc_model = BertForSequenceClassification.from_pretrained("bert-base-uncased", config=config) # BertForSequenceClassification.from_config(config)
        pretrained_acc_model.load_state_dict(torch.load(pretrained_acc_local_path, map_location=torch.device('cpu')))
        # pretrained_acc_model = xm.send_cpu_data_to_device(pretrained_acc_model, device)
        pretrained_acc_model.to(device)
        metric_fns.append(functools.partial(bert_attribute_accuracy_batch, classifier_model=pretrained_acc_model,
                                            tokenizer=tokenizer, device=device, batch_size=ACC_EVAL_BATCH_SIZE))

    ## Similarity
    module_url = "https://tfhub.dev/google/universal-sentence-encoder/2"
    pretrained_sentence_similarity_model = hub.Module(module_url)
    metric_fns.append(functools.partial(sentence_similarity,
                                        sentence_similarity_model=pretrained_sentence_similarity_model))


    output_features = ["inputs", "targets"]
    if attribute_BIT:
        output_features.append("attribute")

    if attribute_DEPENDANT_PREFIX_TARGET:
        output_features.append("codeprefixedtargets")
        output_features.append("controlcode")

    text_preprocessor = functools.partial(at_preprocessor, dataset=DATASET, attribute_bit=attribute_BIT,
                                          attribute_dependant_prefix_input=attribute_DEPENDANT_PREFIX_INPUT,
                                          input_prefix_attribute_1=INPUT_PREFIX_attribute_1, input_prefix_attribute_2=INPUT_PREFIX_attribute_2,
                                          attribute_dependant_prefix_target=attribute_DEPENDANT_PREFIX_TARGET,
                                          target_prefix_attribute_1=TARGET_PREFIX_attribute_1, target_prefix_attribute_2=TARGET_PREFIX_attribute_2)

    if DENOISE:
        token_preprocessor = []
        for inputs_fn, noise_density in DENOISE:
            token_preprocessor.append(functools.partial(denoise, noise_density=noise_density,
                                                        noise_mask_fn=t5.data.preprocessors.iid_noise_mask,
                                                        inputs_fn=inputs_fn,
                                                        targets_fn=None))
    else:
        token_preprocessor = None

    if DATASET == "processed_CCTK" or DATASET == "YELP":
        splits = ["train", "validation", "test"]
    else:
        splits = ["train", "validation"]

    TaskRegistry_ll.add(
        TASK_NAME,
        *task_cls,
        splits=splits,
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
    sequence_length = {"inputs": 64, "targets": 64}
    if attribute_BIT:
        sequence_length["attribute"] = 64  # Or "attribute": 1 but packing not efficient...
    if attribute_DEPENDANT_PREFIX_TARGET:
        sequence_length["codeprefixedtargets"] = 64
        sequence_length["controlcode"] = 64

    ds = st_task.get_dataset(split="validation", sequence_length=sequence_length)

    print("A few preprocessed validation examples...")
    for ex in tfds.as_numpy(ds.take(3)):
        print(ex)

    MixtureRegistry_ll.add(
        MIXTURE_NAME,
        [TASK_NAME],
        default_rate=1.0
    )




    # Modified T5 "attribute-aware denoising autoencoder (pre-trained) bi-transformer"
    utils.build_model = functools.partial(build_model_ll ,
                                          attribute_embedding_encoder=attribute_EMBEDDING_ENCODER,
                                          attribute_embedding_decoder=attribute_EMBEDDING_DECODER,
                                          attribute_num=attribute_NUM,
                                          cut_cross_attention=CUT_CROSS_ATTENTION)

    transformer.make_bitransformer_ll = make_bitransformer_ll

    transformer.Unitransformer_ll = Unitransformer_ll

    utils.tpu_estimator_model_fn = functools.partial(tpu_estimator_model_fn_ll,
                                                     has_partial_sequences=HAS_PARTIAL_SEQUENCES,
                                                     remove_partial_sequences=REMOVE_PARTIAL_SEQUENCES,
                                                     attribute_embedding=attribute_EMBEDDING,
                                                     attribute_dependant_prefix_target=attribute_DEPENDANT_PREFIX_TARGET,
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

    sequence_length = {"inputs": 64, "targets": 64}
    if attribute_BIT:
        sequence_length["attribute"] = 64
    if attribute_DEPENDANT_PREFIX_TARGET:
        sequence_length["codeprefixedtargets"] = 64
        sequence_length["controlcode"] = 64

    # Ou alors, based on L. 357-362  https://github.com/tensorflow/mesh/blob/a719398c92a48990921e57608ef99553ad1b1a85/mesh_tensorflow/transformer/utils.py#L357
    # ignore et appeler le feature attribute "input_attribute" (dans ce cas length of attribute = 128)

    my_tokenizer = sentencepiece_processor.SentencePieceVocabulary(t5.data.DEFAULT_SPM_PATH)
    left_pad_amt_1 = len(my_tokenizer.encode(TARGET_PREFIX_attribute_1)) - 1
    left_pad_amt_2 = len(my_tokenizer.encode(TARGET_PREFIX_attribute_2)) - 1

    model = MtfModel_ll(
        model_dir=MODEL_DIR,
        tpu=TPU_ADDRESS,
        tpu_topology=TPU_TOPOLOGY,
        model_parallelism=model_parallelism,
        batch_size=train_batch_size,
        sequence_length=sequence_length,
        learning_rate_schedule=0.003,
        save_checkpoints_steps=2000,
        keep_checkpoint_max=keep_checkpoint_max if ON_CLOUD else None,
        iterations_per_loop=100,
        model_type="bitransformer_ll",
        attribute_bit=attribute_BIT,
        unsupervised_attribute_transfer_metrics=UNSUPERVISED_STYLE_TRANSFER_METRICS,
        control_code_bool=STYLE_DEPENDANT_PREFIX_TARGET,
        group_by_attribute=GROUP_BY_STYLE,
        attribute_embedding=STYLE_EMBEDDING,
        attribute_num=STYLE_NUM,
        shift_decoder_output=SHIFT_DECODER_OUTPUT,
        left_pad_amt_1=left_pad_amt_1,
        left_pad_amt_2=left_pad_amt_2,
        target_prefix_style_1=TARGET_PREFIX_STYLE_1,
        target_prefix_style_2=TARGET_PREFIX_STYLE_2
    )

    with gin.unlock_config():
        gin.parse_config_file("gs://test-t5/unitransformer_ll.gin")

    FINETUNE_STEPS = 200000

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
            checkpoint_steps="all" # [1100700],
            #split="test"
        )

        print_random_predictions(TASK_NAME, sequence_length, MODEL_DIR, n=10)

    # evaluate on made-up comments
    if DATASET == "IMDB":
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

    elif DATASET == "processed_CCTK":
        comment_style_pairs = [

            {"text": "You are stupid.",
             "Destination style": "civil"},
            {"text": "You are stupid.",
             "Destination style": "toxic"},

            {"text": "Thank you, you are nice.",
             "Destination style": "toxic"},
            {"text": "Thank you, you are nice.",
             "Destination style": "civil"},

            {"text": "Oh yes, blame Taylor, why the fuck not? It is almost as if her getting groped can also be twisted into something wrong on her part. People like you disturb me , to say the least.",
                "Destination style": "civil"},
            {"text": "Oh yes, blame Taylor, why the fuck not? It is almost as if her getting groped can also be twisted into something wrong on her part. People like you disturb me , to say the least.",
                "Destination style": "toxic"},

            {"text": "Donald Trump is best president ever",
             "Destination style": "toxic"},
            {"text": "Donald Trump is the dumbest person ever",
             "Destination style": "civil"},

            {"text": "He's been covering for Trump the whole time, fool!",
             "Destination style": "civil"},
            {"text": "He's been covering for Trump the whole time, fool!",
             "Destination style": "toxic"},

            {"text": "die, please, you rino s.o.b.",
             "Destination style": "civil"},
            {"text": "die, please, you rino s.o.b.",
             "Destination style": "toxic"},

            {"text": "How about FUCK NO. If anything Canada should learn from America's example and start smartening up before this country is overrun by unassimilated immigrants. Why the fuck should Canada have to be the world's global police now that America got tired of doing it? This country has enough problems on its shoulders.",
                "Destination style": "civil"},
            {"text": "How about FUCK NO. If anything Canada should learn from America's example and start smartening up before this country is overrun by unassimilated immigrants. Why the fuck should Canada have to be the world's global police now that America got tired of doing it? This country has enough problems on its shoulders.",
                "Destination style": "toxic"},

            {"text": "Nailed Aces fans.  They are not real hockey fans.  Just a bunch of drunk morons.",
             "Destination style": "civil"},
            {"text": "Nailed Aces fans.  They are not real hockey fans.  Just a bunch of drunk morons.",
             "Destination style": "toxic"},

            {"text": "How about experimenting with fixing our City's infrastructure mainly roads. Idiots.",
             "Destination style": "civil"},
            {"text": "How about experimenting with fixing our City's infrastructure mainly roads. Idiots.",
             "Destination style": "toxic"},

            {"text": "If there is one thing America, and the world needs is more idiots roaming the back country looking for buried treasure.",
                "Destination style": "civil"},
            {"text": "If there is one thing America, and the world needs is more idiots roaming the back country looking for buried treasure.",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},

            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "civil"},
            {"text": "More vote pandering to the Gay community Mr. Trudeau? \n\nDo you have any idea how foolish your antics are?",
                "Destination style": "toxic"},
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
    STYLE_NUM = 2
    LAMBDA_STYLE = 1  # 10 not enough (?) # 100

    STYLE_DEPENDANT_PREFIX_INPUT = False
    INPUT_PREFIX_STYLE_1 = "Negative: "  # "translate non toxic to non toxic: " # "non toxic comment: "
    INPUT_PREFIX_STYLE_2 = "Positive: "  # "translate toxic to toxic: " # "toxic comment: "

    STYLE_DEPENDANT_PREFIX_TARGET = True  # True

    # Balance samples per style (for CCTK)
    BALANCE_STYLES = False
    BALANCE_RATE = 0

    # Task / dataset
    DATASET = "processed_CCTK"  # CCTK or IMDB or processed_civil_comments
    counter = 208
    if DATASET == "IMDB":
        TASK_NAME = "st_imdb"
        MIXTURE_NAME = "st_imdb_mixture"
        MODELS_DIR_NAME = "models_style_imdb_%d" % counter
        DATA_DIR_NAME = "data_style_imdb"

        TARGET_PREFIX_STYLE_1 = "Negative: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Positive: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Negative": "1", "Positive": "2"}

        DATASET_RAW_DIR = "gs://test-t5/imdb_processed"

    elif DATASET == "YELP":
        TASK_NAME = "st_yelp"
        MIXTURE_NAME = "st_yelp_mixture"
        MODELS_DIR_NAME = "models_style_yelp_%d" % counter
        DATA_DIR_NAME = "data_style_yelp"

        TARGET_PREFIX_STYLE_1 = "Negative: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Positive: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Negative": "1", "Positive": "2"}

        DATASET_RAW_DIR = "gs://test-t5/yelp_processed"

    elif DATASET == "CCTK":
        TASK_NAME = "st_toxic_comments"
        MIXTURE_NAME = "st_toxic_comments_mixture"
        MODELS_DIR_NAME = "models_style_civil_comment_%d" % counter
        DATA_DIR_NAME = "data_style_civil_comment"

        TARGET_PREFIX_STYLE_1 = "Non toxic: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "Toxic: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"Non toxic": "1", "Toxic": "2"}

        BALANCE_STYLES = True
        BALANCE_RATE = 0.0736  # ~ 08% / (1 - 08%)

        DATASET_RAW_DIR = None

    elif DATASET == "processed_CCTK":
        TASK_NAME = "st_processed_toxic_comments"
        MIXTURE_NAME = "st_processed_toxic_comments_mixture"
        MODELS_DIR_NAME = "models_style_processed_civil_comment_%d" % counter
        DATA_DIR_NAME = "data_style_processed_civil_comment"

        # civil polite courteous respectful
        TARGET_PREFIX_STYLE_1 = "civil: "  # "Negative: " # Maybe more complex like "Said in a negative manner, " "Style 1: "  erwachsene
        TARGET_PREFIX_STYLE_2 = "toxic: "  # "Positive: " "Style 2: " imunitar
        STYLE_IDS = {"civil": "1", "toxic": "2"}

        BALANCE_STYLES = True
        BALANCE_RATE = 90291/5653785

        DATASET_RAW_DIR = "gs://test-t5/civil_comment_processed"



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

    PPL_ARCHITECTURE = "gpt2"
    PPL_EVAL_BATCH_SIZE = 8
    PPL_BLOCK_SIZE = 256

    ACC_ARCHITECTURE = "BERT"
    ACC_EVAL_BATCH_SIZE = 32

    # GCS setting
    BUCKET = "test-t5"
    BASE_DIR = "gs://test-t5/"  # @param { type: "string" }
    DATA_DIR = os.path.join(BASE_DIR, DATA_DIR_NAME)
    MODELS_DIR = os.path.join(BASE_DIR, MODELS_DIR_NAME)
    ON_CLOUD = True
    USE_COLAB_TPU = True

    MODEL_SIZE = "large"  # ["small", "base", "large", "3B", "11B"]
    # Public GCS path for T5 pre-trained model checkpoints
    BASE_PRETRAINED_DIR = "gs://t5-data/pretrained_models"
    PRETRAINED_DIR = os.path.join(BASE_PRETRAINED_DIR, MODEL_SIZE)
    MODEL_DIR = os.path.join(MODELS_DIR, MODEL_SIZE)

    USE_TPU = True

    # DENOISE = [(t5.data.preprocessors.noise_token_to_random_token_or_sentinel,
    #             0.15), (t5.data.preprocessors.permute_noise_tokens, 1.0)]  # [(inputs_fn, noise_density), ...] or None

    DENOISE = [(t5.data.preprocessors.noise_token_to_random_token_or_sentinel, 0.15)]

    main()