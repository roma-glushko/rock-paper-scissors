import albumentations as a

args = {
    'seed': 63815329,

    'train_dataset_path': 'data/rps/rps/',
    'test_dataset_path': 'data/rps-test-set/rps-test-set/',
    'image_size': (300, 300),

    'batch_size': 32,
}

args['train_augmentation'] = a.Compose([
    a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

args['validation_augmentation'] = a.Compose([
    a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

args['test_augmentation'] = a.Compose([
    a.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])