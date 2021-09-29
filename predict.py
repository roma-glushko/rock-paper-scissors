import os

from tensorflow.python.keras.preprocessing.image_dataset import (
    image_dataset_from_directory,
)

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
from morty.config import ConfigManager, get_arg_parser, main
from morty.experiment import set_random_seed

from rock_paper_scissors import get_model

# TF setup
tf.get_logger().setLevel("ERROR")
gpus = tf.config.experimental.list_physical_devices("GPU")
tf.config.experimental.set_memory_growth(gpus[0], True)

print("TF:", tf.__version__)
print("Num GPUs Available: ", len(tf.config.list_physical_devices("GPU")))

arg_parser = get_arg_parser()

arg_parser.add_argument(
    "--checkpoint_path",
    "--chkp",
    help="""Path to model checkpoint to evaluate""",
    required=True,
)

arg_parser.add_argument(
    "--img_dir_path",
    "-d",
    help="""Path to images that should be used in predictions""",
    default="./data/webcam/",
)


@main(config_path="configs", config_name="basic_config", argument_parser=arg_parser)
def predict(config: ConfigManager) -> None:
    set_random_seed(config.seed)

    print(config)

    dataset = image_dataset_from_directory(
        config.img_dir_path,
        label_mode=None,
        batch_size=config.batch_size,
        image_size=config.image_size,
        shuffle=False,
        seed=config.seed,
    )

    model = get_model(
        config.feature_extractor,
        config.num_classes,
        config.image_size,
        config.l2_strength,
    )

    model.load_weights(config.checkpoint_path)

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    predictions = model.predict(dataset)

    print(predictions)


if __name__ == "__main__":
    predict()
