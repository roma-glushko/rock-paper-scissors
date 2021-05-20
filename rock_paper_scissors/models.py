from typing import Tuple

from tensorflow.keras import Model
from tensorflow.keras.layers import Input, Dropout, Dense
from tensorflow.keras.layers.experimental.preprocessing import Rescaling
import tensorflow.keras.applications as feature_extractors


def get_model(feature_extractor_type: str, num_classes: int, image_size=Tuple[int, int]):
    image_dim = (*image_size, 3)
    feature_extractor_model = getattr(feature_extractors, feature_extractor_type)

    feature_extractor = feature_extractor_model(
        input_shape=image_dim,
        include_top=False,
        weights='imagenet',
        pooling='avg'
    )

    feature_extractor.trainable = False

    image_input = Input(shape=image_dim)
    scaled_images = Rescaling(1./127.5, offset=-1)(image_input)

    image_features = feature_extractor(scaled_images, training=False)
    image_features = Dropout(0.5)(image_features)

    activations = Dense(units=num_classes, activation='softmax')(image_features)

    return Model(image_input, activations, name="rock_paper_scissors_model")