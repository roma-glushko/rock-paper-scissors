import albumentations as a

args = {
    'seed': 63815329,

    'train_dataset_path': 'data/full-rps-webcam-train-dataset/',  # 'data/rps/rps/',
    'val_dataset_path': 'data/webcam_val/',  # 'data/webcam_val/', 'data/rps-test-set/rps-test-set/'
    'test_dataset_path': 'data/webcam_test/',
    'image_size': (300, 300),
    'num_classes': 3,  # no noise class

    'epochs': 50,
    'batch_size': 32,
    'learning_rate': 0.001,
    'feature_extractor': 'MobileNetV2',
}

args['train_augmentation'] = a.Compose([
    a.VerticalFlip(),
    a.HorizontalFlip(),
    a.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.1, brightness_by_max=False),
    a.HueSaturationValue(p=1.0, hue_shift_limit=(-20, 20), sat_shift_limit=(-30, 30),
                         val_shift_limit=(-20, 20)),
    a.GaussNoise(p=1.0, var_limit=(10.0, 50.0)),
    a.MotionBlur(p=1.0, blur_limit=(3, 6)),
    a.CoarseDropout(p=0.8, max_holes=50, max_height=10, max_width=10, min_holes=20, min_height=8, min_width=8),
    a.ImageCompression(p=0.5, quality_lower=80, quality_upper=100, compression_type=0),
    a.ISONoise(p=1.0, intensity=(0.1, 0.5), color_shift=(0.01, 0.05)),
    # a.InvertImg(p=0.4),
])