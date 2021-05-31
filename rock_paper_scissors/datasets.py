import os
import random
from functools import partial
from typing import Tuple, Dict

import albumentations as a
import numpy as np
import tensorflow as tf
from tensorflow.python.data import AUTOTUNE
from tensorflow.python.keras.layers import Rescaling
from tensorflow.python.keras.preprocessing.image_dataset import image_dataset_from_directory

class_names = (
    'rock',
    'paper',
    'scissors',
)


def get_dataset_stats(dataset_path: str, image_pattern: str = '*.png') -> Dict:
    total_samples = 0
    dataset_stats = {}

    for class_name in class_names:
        images = tf.io.gfile.glob(os.path.join(dataset_path, class_name, image_pattern))
        num_images = len(images)

        dataset_stats[class_name] = num_images
        total_samples += num_images

    dataset_stats['total'] = total_samples

    return dataset_stats


def augment_image(inputs: tf.Tensor, labels: tf.Tensor, augmentation_pipeline: a.Compose, seed: int = 42):
    def apply_augmentation(images):
        random.seed(seed)
        np.random.seed(seed)

        augmented_images = []

        for img in images:
            aug_data = augmentation_pipeline(image=img.astype('uint8'))
            augmented_images.append(aug_data['image'])

        return np.stack(augmented_images)

    inputs = tf.numpy_function(func=apply_augmentation, inp=[inputs], Tout=tf.uint8)

    return inputs, labels


def scale_images(inputs: tf.Tensor, labels: tf.Tensor):
    """
    Scale image batch between [-1; 1]
    """
    return Rescaling(1. / 127.5, offset=-1)(inputs), labels


def get_dataset(
        dataset_path: str,
        augmentation_pipeline: a.Compose,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (300, 300),
        scaling: bool = True,
        seed: int = 42
) -> tf.data.Dataset:
    augmentation_func = partial(
        augment_image,
        augmentation_pipeline=augmentation_pipeline,
        seed=seed,
    )

    dataset = image_dataset_from_directory(
        dataset_path,
        class_names=class_names,
        image_size=image_size,
        batch_size=batch_size,
        seed=seed,
    )

    dataset = dataset.map(augmentation_func, num_parallel_calls=AUTOTUNE)

    if scaling:
        dataset = dataset.map(scale_images, num_parallel_calls=AUTOTUNE)

    return dataset \
        .shuffle(buffer_size=512, seed=seed) \
        .prefetch(AUTOTUNE)


def get_test_dataset(
        dataset_path: str,
        batch_size: int = 32,
        image_size: Tuple[int, int] = (300, 300),
        scaling: bool = True,
        seed: int = 42
) -> tf.data.Dataset:
    dataset = image_dataset_from_directory(
        dataset_path,
        class_names=class_names,
        batch_size=batch_size,
        image_size=image_size,
        shuffle=False,
        seed=seed,
    )

    if scaling:
        dataset = dataset.map(scale_images, num_parallel_calls=AUTOTUNE)

    return dataset.prefetch(AUTOTUNE)
