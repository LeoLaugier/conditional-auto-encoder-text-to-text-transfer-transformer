# CAET5: Mitigating toxicity in online conversations using self-supervised transformers

CAET5 serves as code for fine-tuning pre-trained text-to-text transformers from [_Exploring the Limits of Transfer 
Learning with a Unified Text-to-Text Transformer_][paper] on self-supervised attribute transfer tasks.

The code overrides objects from the [T5][t5] and the [Mesh TensorFlow Transformer][mtft] packages.

## Table of Contents
* [Library](#library)
* [Usage](#usage)
    * [Dataset Preparation](#dataset-preparation)
    * [Unsupervised Metric Preparation](#unsupervised-metric-preparation)
    * [Installation](#installation)
    * [Setting up TPUs on GCP](#setting-up-tpus-on-gcp)
    * [Fine-Tuning](#fine-tuning)
    * [Eval](#eval)
    * [Decode](#decode)
* [How to Cite](#how-to-cite)

## Library

#### caet5

`caet5` reproduces the structure of the [T5][t5] package.

`caet5.data` redefines `Task` objects. Please see the [`t5.data` documentation][t5_data] for more details about the 
`t5.data` package.

We added:

* functions to pre-process unpaired datasets made of several text files containing attribute-exclusive examples, 
with one example per line.
* a text preprocessor function adapted to self-supervised attribute transfer.    

We adapted functions from t5 that were not initially adapted to self-supervised attribute transfer.  

`caet5.evaluation` adds metrics computed with (pre-trained / fine-tuned) parametric models (in particular transformers) 
and used to evaluate unsupervised attribute transfer:
* sentence similarity (SIM)
* attribute transfer accuracy (ACC) 
* perplexity (PPL)

`caet5.models` adapts the [`t5.models`][t5_models] shims to unsupervised training, evaluation and inference methods 
for attribute transfer.

#### mesh_tensorflow_caet5 

`mesh_tensorflow_caet5` overrides objects of the [Mesh TensorFlow Transformer][mtft] package, to fit CAET5's training 
and evaluation approach.

## Usage
The easiest way to try out CAET5 is with a free TPU on [Colab][colab].

Below we provide examples for how to fine-tune, evaluate and infer from a model from the model API. You can use these 
instructions to reproduce our results, fine-tune one of T5's released checkpoints with your own data and/or 
hyperparameters.

### Dataset Preparation
You may either use a new or pre-existing `Task_ll`, or you may load examples from "raw" text files, each containing 
single attribute examples.

#### Using a `Task_ll`

Depending on your data source (see [`t5.data` documentation][t5_data]), you will need to prepare your data 
appropriately.

Just make sure any file(s), either raw files or pre-processed file(s) loaded by your `dataset_fn` are accessible to the 
TPU (i.e., are in a GCS bucket).


### Unsupervised Metric Preparation
In order to compute attribute transfer accuracy and perplexity, you need to store pre-trained parametric models. CAET5
currently supports BERT classification models fine-tuned on attribute classification and GPT2 language models, by 
default stored in gs://yourbucket/[metric]\_binaries/[architecture]\_[metric]\_[mixture_or_task_name].pt where [metric] is 
"acc" or "ppl", and [architecture] is "bert" or "gpt2".

### Installation
To install the CAET5 package, clone the github repo and run:

```sh
pip install .
```

### Setting up TPUs on GCP
For details about setting up TPUs on GCP, please see the [t5 documentation][t5_setting-up-tpus-on-gcp].

In order to run training or eval on Cloud TPUs, you must set up the following variables based on your project, zone and 
GCS bucket appropriately.

```sh
export PROJECT=your_project_name
export ZONE=your_project_zone
export BUCKET=yourbucket
export TPU_NAME=t5-tpu
export BASE_DIR=gs://yourbucket/
export MODELS_DIR_NAME=your_models_dir_name
export DATA_DIR_NAME=your_data_dir
export DATA_RAW_DIR_NAME=your_data_raw_dir_name
```

### Fine-tuning

In order to fine-tune one of T5's [pre-trained models][t5_released-model-checkpoints], on an attribute transfer task 
called [mixture_or_task_name], please run:

```sh
caet5  \
  --tpu="${TPU_NAME}" \
  --gcp_project="${PROJECT}" \
  --tpu_zone="${ZONE}" \
  --bucket="${BUCKET}" \
  --base_dir="${BASE_DIR}" \
  --model_dir_name="${MODELS_DIR_NAME}" \
  --model_size=your_model_size \
  --data_dir_name="${DATA_DIR_NAME}" \
  --data_raw_dir_name="${DATA_RAW_DIR_NAME}" \
  --module_import=caet5.data.tasks \
  --use_model_api=True \
  --mode="finetune" \
  --train_steps=100000 \
  --mixture_or_task=[mixture_or_task_name] \
  --base_pretrained_model_dir="gs://t5-data/pretrained_models/" \
  --gin_file="dataset.gin" \
  --gin_file="objectives/denoise.gin" \
  --gin_file="models/cae_bi.gin" \
  --gin_file="train.gin" \
  --gin_file="sequence_lengths/[mixture_or_task_name]" \
  --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'"
```

### Eval
In order to evaluate a model in the CAET5 framework, you need to specify the model directory and which checkpoint 
step(s) to evaluate. So, to evaluate on the [mixture_or_task_name] task on *all* checkpoints, 
use the following command:
```sh
caet5 --tpu="${TPU_ADDRESS}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --bucket="${BUCKET}" \
      --base_dir="${BASE_DIR}" \
      --model_dir_name="${MODELS_DIR_NAME}" \
      --model_size="${MODEL_SIZE}" \  
      --data_dir_name="${DATA_DIR_NAME}" \
      --module_import=caet5.data.tasks \
      --use_model_api=True \
      --mode="eval" \      
      --mixture_or_task=[mixture_or_task_name] \
      --base_pretrained_model_dir="gs://t5-data/pretrained_models/" \
      --checkpoint_mode="all" \
      --gin_file="dataset.gin" \
      --gin_file="models/cae_bi.gin" \
      --gin_file="sequence_lengths/[mixture_or_task_name]" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'"
```

To evaluate a specific checkpoint, simply set the `eval_checkpoint_step` parameter to appropriate checkpoint.

```
--gin_param="eval_checkpoint_step = 100000"
```

### Decode
In order to produce predictions from a model in the CAET5 framework, you need to use the `infer.gin` file, specify the 
model directory and which checkpoint step(s) to use for decoding. Assuming you have a text file of input sequences and 
destination attribute stored at `/path/to/intputs.txt`, an example command would be:

```sh
caet5 --tpu="${TPU_ADDRESS}" \
      --gcp_project="${PROJECT}" \
      --tpu_zone="${ZONE}" \
      --bucket="${BUCKET}" \
      --base_dir="${BASE_DIR}" \
      --model_dir_name="${MODELS_DIR_NAME}" \
      --model_size="${MODEL_SIZE}" \  
      --data_dir_name="${DATA_DIR_NAME}" \
      --module_import=caet5.data.tasks \
      --use_model_api=True \
      --mode="predict" \      
      --mixture_or_task=[mixture_or_task_name] \
      --base_pretrained_model_dir="gs://t5-data/pretrained_models/" \
      --checkpoint_mode="latest" \
      --input_file='/path/to/inputs.txt' \
      --output_file='/tmp/outputs.txt' \
      --predict_batch_size=your_predict_batch_size \
      --gin_file="dataset.gin" \
      --gin_file="models/cae_bi.gin" \
      --gin_file="infer.gin" \
      --gin_file="sequence_lengths/[mixture_or_task_name]" \
      --gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'"
```


# How to Cite
This work is currently under double-blind review. We will update this section as soon as possible.



[paper]: https://arxiv.org/abs/1910.10683
[t5]: https://github.com/google-research/text-to-text-transfer-transformer
[t5_data]: https://github.com/google-research/text-to-text-transfer-transformer#t5data
[t5_evaluation]: https://github.com/google-research/text-to-text-transfer-transformer#t5evaluation
[t5_models]: https://github.com/google-research/text-to-text-transfer-transformer#t5models
[t5_setting-up-tpus-on-gcp]: https://github.com/google-research/text-to-text-transfer-transformer#setting-up-tpus-on-gcp
[t5_released-model-checkpoints]: https://github.com/google-research/text-to-text-transfer-transformer#released-model-checkpoints
[mtft]: https://github.com/tensorflow/mesh/tree/master/mesh_tensorflow/transformer
[colab]: https://colab.research.google.com/