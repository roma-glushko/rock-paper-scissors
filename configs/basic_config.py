import albumentations as a

args = {
    'seed': 63815329,

    'train_dataset_path': 'data/rps/rps/',
    'test_dataset_path': 'data/rps-test-set/rps-test-set/',
    'image_size': (300, 300),
    'num_classes': 3,  # no noise class

    'epochs': 10,
    'batch_size': 32,
    'learning_rate': 0.001,
    'feature_extractor': 'MobileNetV2',
}

args['train_augmentation'] = a.ReplayCompose([
    a.VerticalFlip(),
    a.HorizontalFlip(),
    # a.RandomBrightness(limit=0.2),
    a.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, brightness_by_max=False),
    a.CoarseDropout(max_holes=20, max_height=8, max_width=8, min_holes=10, min_height=8, min_width=8),
    a.GaussNoise(p=1.0, var_limit=(10.0, 50.0)),
    # failed augmentations:
    # a.HueSaturationValue(always_apply=False, p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30), val_shift_limit=(-20, 20))
    # a.MotionBlur(p=1.0, blur_limit=(3, 7)),
    # a.ImageCompression(always_apply=False, p=1.0, quality_lower=80, quality_upper=100, compression_type=0)
    # a.ISONoise(always_apply=False, p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
])

args['validation_augmentation'] = a.Compose([])

args['test_augmentation'] = a.Compose([])