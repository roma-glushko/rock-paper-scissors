from functools import partial
from typing import Tuple

import albumentations as a
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

class_names = [
    'rock',
    'paper',
    'scissors',
]


def augment_image(inputs, labels, augmentation_pipeline: a.Compose):
    def apply_augmentation(images):
        aug_data = augmentation_pipeline(image=images.astype('uint8'))
        return aug_data['image']

    inputs = tf.numpy_function(func=apply_augmentation, inp=[inputs], Tout=tf.uint8)

    return inputs, labels


def get_dataset(
        dataset_path: str,
        subset_type: str,
        augmentation_pipeline: a.Compose,
        validation_fraction: float = 0.2,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (300, 300),
        seed: int = 42
) -> tf.data.Dataset:
    augmentation_func = partial(
        augment_image,
        augmentation_pipeline=augmentation_pipeline,
    )

    dataset = image_dataset_from_directory(
        dataset_path,
        subset=subset_type,
        class_names=class_names,
        validation_split=validation_fraction,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    return dataset \
        .map(augmentation_func, num_parallel_calls=AUTOTUNE) \
        .prefetch(AUTOTUNE)


def get_test_dataset(
        dataset_path: str,
        augmentation_pipeline: a.Compose,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (300, 300),
        seed: int = 42
) -> tf.data.Dataset:
    augmentation_func = partial(
        augment_image,
        augmentation_pipeline=augmentation_pipeline,
    )

    dataset = image_dataset_from_directory(
        dataset_path,
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        seed=seed,
    )

    return dataset \
        .map(augmentation_func, num_parallel_calls=AUTOTUNE) \
        .prefetch(AUTOTUNE)
