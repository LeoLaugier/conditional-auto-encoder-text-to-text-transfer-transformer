Code for CAE-T5

# CAET5: Mitigating toxicity in online conversations using self-supervised transformers

CAET servess xxx















Requirements: 
* tensorflow==1.15.0
* tensorflow-text<2
* transformers==2.8.0
* kenlm==0.0.0
* t5==0.5.0
* tfds-nightly>=1.3.2.dev201912070105,<2.1.0.dev202003190105
* fasttext==0.9.2
* google-api-python-client==1.7.12 

To run on TPU: python main.py

# On Colab
?export PROJECT=your_project_name
?export ZONE=your_project_zone
export BASE_DIR=gs://test-t5/ 
export TPU_NAME="grpc://" + os.environ["COLAB_TPU_ADDR"]

caet5 
--module_import=caet5.data.tasks
--bucket=test-t5 
--base_dir="${BASE_DIR}"
--data_raw_dir_name=civil_comment_processed 
--gin_file="dataset.gin"
--gin_file="objectives/denoise.gin"
--gin_param="utils.tpu_mesh_shape.tpu_topology = '2x2'"
--tpu="${TPU_NAME}"
--model_dir_name=

