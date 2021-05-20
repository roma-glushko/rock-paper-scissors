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

args['train_augmentation'] = a.Compose([
    a.VerticalFlip(p=0.5),
    a.HorizontalFlip(p=0.5),
    a.RandomBrightness(limit=0.1),
    # a.JpegCompression(quality_lower=85, quality_upper=100, p=0.5),
    a.HueSaturationValue(p=0.5, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20))
])

args['validation_augmentation'] = a.Compose([
    a.VerticalFlip(p=0.5),
    a.HorizontalFlip(p=0.5),
])

args['test_augmentation'] = a.Compose([])