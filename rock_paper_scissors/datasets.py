from functools import partial
from typing import Tuple

import albumentations as a
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory


def augment_image(inputs, labels, augmentation_pipeline):
    def apply_augmentation(image):
        aug_data = augmentation_pipeline(image=image)
        return aug_data['image']

    inputs = tf.numpy_function(func=apply_augmentation, inp=[inputs], Tout=tf.float32)

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
        augmentation_pipeline=augmentation_pipeline
    )

    dataset = image_dataset_from_directory(
        dataset_path,
        subset=subset_type,
        validation_split=validation_fraction,
        image_size=image_size,
        seed=seed,
    )

    return dataset \
        .map(augmentation_func, num_parallel_calls=AUTOTUNE) \
        .batch(batch_size) \
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
        augmentation_pipeline=augmentation_pipeline
    )

    dataset = image_dataset_from_directory(
        dataset_path,
        image_size=image_size,
        seed=seed,
    )

    return dataset \
        .map(augmentation_func, num_parallel_calls=AUTOTUNE) \
        .batch(batch_size) \
        .prefetch(AUTOTUNE)
