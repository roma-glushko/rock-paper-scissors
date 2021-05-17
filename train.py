import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

from morty.config import ConfigManager, main
from morty.experiment import set_random_seed

# TF setup
from rock_paper_scissors import get_dataset

tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print('TF', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))


@main(config_path='configs', config_name='basic_config')
def train(config: ConfigManager) -> None:
    set_random_seed(config.seed)

    train_dataset = get_dataset(
        config.train_dataset_path,
        'training',
        config.train_augmentation,
        validation_fraction=0.2,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    validation_dataset = get_dataset(
        config.train_dataset_path,
        'validation',
        config.train_augmentation,
        validation_fraction=0.2,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    mobile_netv2 = MobileNetV2(
        input_shape=(*config.image_size, 3),
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    mobile_netv2.trainable = False

    mobile_netv2.summary()


if __name__ == "__main__":
    train()
