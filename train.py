import os
import pickle

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from morty.config import ConfigManager, main
from morty.experiment import set_random_seed

from rock_paper_scissors import get_dataset, get_model

# TF setup
tf.get_logger().setLevel('ERROR')
gpus = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(gpus[0], True)

print('TF:', tf.__version__)
print('Num GPUs Available: ', len(tf.config.list_physical_devices('GPU')))

early_stopping = EarlyStopping(
    patience=10,
    min_delta=0.001,
    restore_best_weights=True,
)


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

    model = get_model(
        config.feature_extractor,
        config.num_classes,
        config.image_size,
    )

    optimizer = tf.keras.optimizers.Adam(learning_rate=config.learning_rate)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    # todo: find a way to get dataset lengths
    steps_per_epoch = 2016 // config.batch_size
    validation_steps = 504 // config.batch_size

    model_saver = ModelCheckpoint(
        filepath='logs/checkpoints/rps-val_acc_{val_accuracy:.5f}' + f'-{config.feature_extractor}-seed_{config.seed}' + '-val_loss_{loss:.5f}-epoch_{epoch}.h5',
        mode='max',
        monitor='val_accuracy',
        save_best_only=True,
        save_weights_only=True,
        verbose=1,
    )

    training_history = model.fit(
        x=train_dataset.repeat(),
        validation_data=validation_dataset.repeat(),
        epochs=config.epochs,
        steps_per_epoch=steps_per_epoch,
        validation_steps=validation_steps,
        callbacks=[
            model_saver,
            early_stopping,
        ],
        verbose=1
    )

    with open('./logs/training_history.pkl', 'wb') as f:
        pickle.dump(training_history, f)


if __name__ == "__main__":
    train()