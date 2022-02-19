from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.transforms.functional as F
from src.core.datasets import *
from src.utils.util import normalization_params,normalization_params2
from torchvision.datasets import *
import torchvision

DATASETS = {
    'cub': CUB200,
    'mnist' : MNIST_Wrapper,
    'voc': VOC2012,
    'OpenImages30k' : OpenImages30k,
    'ImageNet' : ImageNet,
    'OxfordPets' : OxfordPets
}

MODES = ['train', 'val']

def build(data_config, logger):
    data_name = data_config['name']
    root = data_config['root']
    batch_size = data_config['batch_size']
    num_workers = data_config['num_workers']
    transform_config = data_config['transform']

    dataloaders = {}
    for mode in MODES:
        if data_name == 'cub':
            transform = compose_transforms(data_name, transform_config, mode)
            dataset = DATASETS[data_name](root, logger, mode, transform=transform,
                                          transform_config=transform_config)
        elif data_name == 'mnist':
            transform  = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Normalize((0.5,), (1.0,))
                    ])
            if mode == 'train':
                dataset = DATASETS[data_name](root, transform=transform, train=True, download=True)
            else:
                dataset = DATASETS[data_name](root, transform=transform, train=False, download=True)

        elif data_name == 'amnist':
            transform = compose_transforms2(data_name, transform_config, mode)
            dataset = DATASETS[data_name](root, logger, mode, transform=transform)

        elif data_name == 'OpenImages30k':
            transform = compose_transforms2(data_name, transform_config, mode)
            dataset = DATASETS[data_name](root, logger, mode, transform=transform)

        elif data_name == 'OxfordPets':
            transform = compose_transforms(data_name, transform_config, mode)
            dataset = DATASETS[data_name](root, logger, mode, transform=transform)

        elif data_name == 'ImageNet':
            transform = compose_transforms(data_name, transform_config, mode)
            #dataset = DATASETS[data_name](root, logger, mode, transform=transform, noise=False)
            dataset = torchvision.datasets.ImageNet('/data/opensets/imagenet-pytorch',
                                                    transform=transform,
                                                    split=mode)

        else:
            train_full = None
            num_full_supervision = data_config.get('num_full_sup', 10582)
            if num_full_supervision < 10582:
                train_full = 'train_full_' + str(num_full_supervision)

            if mode == 'train':
                train_crop_size = transform_config['train_crop_size']
                dataset = DATASETS[data_name](root, crop_size=train_crop_size,
                                              train_full=train_full, scale=True, flip=True)
            elif mode == 'val':
                val_crop_size = transform_config['val_crop_size']
                dataset = DATASETS[data_name](root, crop_size=val_crop_size,
                                              metadata_split='val', scale=False, flip=False)
            elif mode == 'test':
                val_crop_size = transform_config['val_crop_size']
                dataset = DATASETS[data_name](root, crop_size=val_crop_size,
                                              metadata_split='test', scale=False, flip=False)

        shuffle = True if mode == 'train' else False
        dataloaders[mode] = DataLoader(dataset, batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers)
    return dataloaders

def compose_transforms(data_name, transform_config, mode):
    mean, std = normalization_params()
    image_size = transform_config['image_size']
    try:
        crop_size = transform_config['crop_size']
    except:
        crop_size = transform_config['train_crop_size']

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transform

def compose_transforms2(data_name, transform_config, mode):
    #print(transform_config)
    mean, std = normalization_params2()
    image_size = transform_config['image_size']
    crop_size = transform_config['train_crop_size']

    if mode == 'train':
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomCrop(crop_size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    else:
        transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.CenterCrop(crop_size),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)])
    return transform
