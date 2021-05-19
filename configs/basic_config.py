import albumentations as a

args = {
    'seed': 63815329,

    'train_dataset_path': 'data/rps/rps/',
    'test_dataset_path': 'data/rps-test-set/rps-test-set/',
    'image_size': (300, 300),
    'num_classes': 3,  # adds one more class for noise

    'epochs': 20,
    'batch_size': 32,
    'learning_rate': 0.001,
    'feature_extractor': 'MobileNetV2',
}

args['train_augmentation'] = a.Compose([])

args['validation_augmentation'] = a.Compose([])

args['test_augmentation'] = a.Compose([])