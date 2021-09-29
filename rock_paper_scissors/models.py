from typing import Optional, Tuple

import tensorflow.keras.applications as feature_extractors
from tensorflow.keras import Model
from tensorflow.keras.layers import Dense, Dropout, Input
from tensorflow.keras.regularizers import l2


def get_model(
    feature_extractor_type: str,
    num_classes: int,
    image_size: Tuple[int, int],
    l2_strength: Optional[float] = None,
    trainable_at: Optional[int] = None,
):
    image_dim = (*image_size, 3)
    feature_extractor_model = getattr(feature_extractors, feature_extractor_type)

    feature_extractor = feature_extractor_model(
        input_shape=image_dim, include_top=False, weights="imagenet", pooling="avg"
    )

    feature_extractor.trainable = False

    if trainable_at:
        feature_extractor.trainable = True
        print(
            "Number of layers in the feature extractor: ", len(feature_extractor.layers)
        )
        for layer in feature_extractor.layers[:trainable_at]:
            layer.trainable = False

    image_input = Input(shape=image_dim)

    image_features = feature_extractor(image_input, training=False)
    image_features = Dropout(0.5)(image_features)

    activations = Dense(
        units=num_classes,
        activation="softmax",
        kernel_regularizer=l2(l=l2_strength) if l2_strength else None,
    )(image_features)

    return Model(image_input, activations, name="rock_paper_scissors_model")
