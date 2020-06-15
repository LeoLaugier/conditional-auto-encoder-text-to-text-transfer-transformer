
r"""Main file for launching training/eval/predictions of CAE-T5 model."""
from absl import app, flags
import tensorflow.compat.v1 as tf

flags.DEFINE_bool("evaluate_with_transformers", True,
                  "Whether to compute parametric evaluation metrics with transformers.")

flags.DEFINE_string("pretrained_acc_architecture", "bert",
                    "Architecture of the pre-trained attribute classifier computing accuracy.")

flags.DEFINE_string("pretrained_acc_name_or_path", "bert-base-uncased",
                    "Name or path of the pre-trained model that the attribute classifier fine-tuned.")

flags.DEFINE_string("parametric_acc_filename", None,
                    "Filename of the parametric attribute classifier computing accuracy.")

flags.DEFINE_integer("acc_eval_batch_size", 32,
                     "Mini-batch size when computing attribute transfer accuracy with a fine-tuned attribute"
                     "classifier.")


flags.DEFINE_string("pretrained_ppl_architecture", "gpt2",
                    "Architecture of the pre-trained language model computing perplexity.")

flags.DEFINE_string("pretrained_ppl_name_or_path", "gpt2",
                    "Name or path of the pre-trained model that the language model fine-tuned.")

flags.DEFINE_string("parametric_ppl_filename", None,
                    "Filename of the parametric language model computing perplexity.")

flags.DEFINE_integer("ppl_eval_batch_size", 8,
                     "Mini-batch size when computing perplexity with a fine-tuned language model.")

flags.DEFINE_integer("ppl_eval_block_size", -1,
                     "Input sequence length after tokenization when computing perplexity with a fine-tuned language"
                     "model."
                     "Default to the model max input length for single sentence inputs (take into account special"
                     "tokens).") # TODO 256 when evaluating in practice


flags.DEFINE_string("bucket", None,
                    "Name of the Cloud Storage bucket for the data and model checkpoints, e.g. my-bucket")

flags.DEFINE_string("base_dir", None,
                    "Base directory for the bucket on GCS, e.g. gs://my-bucket/")

flags.DEFINE_list("metrics", ["BLEU", "SIM", "ACC", "PPL"],
                  "Automatic metrics to use when evaluating.")


flags.DEFINE_bool("attribute_bit", False,
                  "Whether to integrate attribute embedding in the model.")

flags.DEFINE_bool("style_dependant_prefix_input", False,
                  "Whether to prepend a prefix to the encoder's input.")

flags.DEFINE_list("input_prefix_attributes", None,
                  "List of attribute-dependent prefixes (strings) to prepend to the encoder's inputs"
                  "Default to None: no prefixing")

flags.DEFINE_list("target_prefix_attributes", ["civil: ", "toxic: "],
                  "List of attribute-dependent prefixes (strings) to prepend to the decoder's inputs during"
                  "teacher-forcing")

flags.DEFINE_list("control_codes", ["toxic: ", "civil: "],
                  "List of control codes (strings) which trigger the non-teacher-forcing AutoRegressive"
                  "generation")


FLAGS = flags.FLAGS

def main(_):


def console_entry_point():
    tf.disable_v2_behavior()
    tf.logging.set_verbosity(tf.logging.INFO)
    app.run(main)

if __name__ == "__main__":
    console_entry_point()