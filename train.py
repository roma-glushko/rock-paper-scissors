import os
import pickle
import random

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['TF_DETERMINISTIC_OPS'] = '1'
os.environ['TF_CUDNN_DETERMINISTIC'] = '1'

import tensorflow as tf
from tensorflow.python.keras.callbacks import ModelCheckpoint, EarlyStopping

from morty.config import ConfigManager, main
from morty.experiment import set_random_seed

import wandb
from wandb.keras import WandbCallback

from rock_paper_scissors import get_dataset, get_model, get_dataset_stats, get_test_dataset, optimizer_factory, \
    class_names, log_confusion_matrix

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
    random.seed(config.seed)
    set_random_seed(config.seed)

    wandb.init(project='rock-paper-scissors', entity='roma-glushko', config=config)

    train_dataset = get_dataset(
        config.train_dataset_path,
        config.train_augmentation,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    validation_dataset = get_test_dataset(
        config.val_dataset_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    model = get_model(
        config.feature_extractor,
        config.num_classes,
        config.image_size,
        config.l2_strength,
    )

    optimizer = optimizer_factory.get(config.optimizer)(**config.optimizer_config)

    model.compile(
        optimizer=optimizer,
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy'],
    )

    model.summary()

    train_dataset_stats = get_dataset_stats(config.train_dataset_path)
    val_dataset_stats = get_dataset_stats(config.val_dataset_path)

    steps_per_epoch = train_dataset_stats['total'] // config.batch_size
    validation_steps = val_dataset_stats['total'] // config.batch_size

    model_saver = ModelCheckpoint(
        filepath='logs/checkpoints/rps-val_acc_{val_accuracy:.5f}' + f'-{config.feature_extractor}-seed_{config.seed}' + '-val_loss_{val_loss:.5f}-epoch_{epoch}.h5',
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
            WandbCallback(training_data=train_dataset, log_weights=True, log_gradients=True, data_type='image'),
        ],
        verbose=1
    )

    with open('./logs/training_history.pkl', 'wb') as f:
        pickle.dump(training_history.history, f)

    test_dataset = get_test_dataset(
        config.test_dataset_path,
        batch_size=config.batch_size,
        image_size=config.image_size,
        seed=config.seed,
    )

    # double check the best model
    val_loss, val_accuracy = model.evaluate(validation_dataset)

    print("Val Loss: {}".format(val_loss))
    print("Val Accuracy: {}".format(val_accuracy))

    test_loss, test_accuracy = model.evaluate(test_dataset)

    wandb.log({'test_accuracy': test_accuracy})
    wandb.log({'test_loss': test_loss})

    log_confusion_matrix('test_confusion_matrix', model, test_dataset)


if __name__ == "__main__":
    train()
