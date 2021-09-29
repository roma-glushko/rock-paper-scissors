import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_DETERMINISTIC_OPS"] = "1"
os.environ["TF_CUDNN_DETERMINISTIC"] = "1"

import tensorflow as tf
from morty.config import ConfigManager, get_arg_parser, main
from morty.experiment import set_random_seed

from rock_paper_scissors import (
    get_dataset_stats,
    get_model,
    get_test_dataset,
    optimizer_factory,
)

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


@main(config_path="configs", config_name="basic_config", argument_parser=arg_parser)
def evaluate(config: ConfigManager) -> None:
    set_random_seed(config.seed)

    print(config)

    test_dataset = get_test_dataset(
        config.test_dataset_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    print("Test Dataset Stats: ", get_dataset_stats(config.test_dataset_path))

    model = get_model(
        config.feature_extractor,
        config.num_classes,
        config.image_size,
        config.l2_strength,
        trainable_at=config.unfreeze_at,
    )

    model.load_weights(config.checkpoint_path)

    optimizer = optimizer_factory.get(config.optimizer)(**config.optimizer_config)

    model.compile(
        optimizer=optimizer,
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    model.summary()

    test_loss, test_accuracy = model.evaluate(test_dataset)

    print("Test Loss: {}".format(test_loss))
    print("Test Accuracy: {}".format(test_accuracy))

    model_path: str = f"./logs/models/rps-test_acc_{test_accuracy}-test_loss_{test_loss}.h5"
    print("Saving model {} ..".format(model_path))
    model.save(model_path, save_format="h5")


if __name__ == "__main__":
    evaluate()
