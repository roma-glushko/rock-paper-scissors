import albumentations as a

args = {
    'seed': 63815329,

    'train_dataset_path': 'data/rps/rps/',
    'test_dataset_path': 'data/rps-test-set/rps-test-set/',
    'image_size': (300, 300),
    'num_classes': 3,  # adds one more class for noise

    'epochs': 30,
    'batch_size': 32,
    'learning_rate': 0.001,
    'feature_extractor': 'MobileNetV2',
}

args['train_augmentation'] = a.Compose([
    a.VerticalFlip(),
    a.HorizontalFlip(),
    a.RandomBrightness(limit=0.2, p=0.5),
    a.CoarseDropout(p=0.5, max_holes=20, max_height=8, max_width=8, min_holes=10, min_height=8, min_width=8),
    a.GaussNoise(p=1.0, var_limit=(10.0, 50.0)),
    # a.MotionBlur(p=1.0, blur_limit=(3, 7)),
])

args['validation_augmentation'] = a.Compose([
    a.VerticalFlip(p=0.5),
    a.HorizontalFlip(p=0.5),
])

args['test_augmentation'] = a.Compose([])