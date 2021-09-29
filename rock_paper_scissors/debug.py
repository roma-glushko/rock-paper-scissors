import numpy as np
import tensorflow as tf
import wandb
from tensorflow.keras import Model

from rock_paper_scissors import class_names


def log_confusion_matrix(name: str, model: Model, dataset: tf.data.Dataset) -> None:
    """

    """
    targets = []
    predictions = []

    for image_batch, label_batch in dataset:
        prediction_batch = model.predict(image_batch)

        targets.append(label_batch.numpy())
        predictions.append(prediction_batch)

    wandb.log(
        {
            name: wandb.plot.confusion_matrix(
                y_true=np.concatenate(targets),
                probs=np.concatenate(predictions),
                class_names=class_names,
            )
        }
    )
