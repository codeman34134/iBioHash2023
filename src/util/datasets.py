# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
# --------------------------------------------------------
# References:
# DeiT: https://github.com/facebookresearch/deit
# --------------------------------------------------------

import os
import PIL
from PIL import Image
from timm.data.transforms import _pil_interp

from torchvision import transforms

from timm.data import create_transform
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD

from torchvision.datasets import ImageFolder

def build_dataset(mode, args):
    transform = build_transform(mode, args)
    if mode == 'train':
        fold_name = 'train'
    if mode == 'query':
        fold_name = 'query'
    if mode == 'retrieval':
        fold_name = 'retrieval'
    root = os.path.join(args.data_path, fold_name)
    dataset = ImageFolder(root, transform=transform)
    print(dataset)

    return dataset


def build_transform(is_train, config):
    resize_im = config.DATA.IMG_SIZE > 32
    mean = [0.4702532, 0.48587758, 0.38928695]
    std = [0.19859357, 0.19675725, 0.19619795]
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=config.DATA.IMG_SIZE,
            is_training=True,
            color_jitter=config.AUG.COLOR_JITTER if config.AUG.COLOR_JITTER > 0 else None,
            auto_augment=config.AUG.AUTO_AUGMENT if config.AUG.AUTO_AUGMENT != 'none' else None,
            re_prob=config.AUG.REPROB,
            re_mode=config.AUG.REMODE,
            re_count=config.AUG.RECOUNT,
            interpolation=config.DATA.INTERPOLATION,
            mean=mean,
            std=std
        )
        if not resize_im:
            # replace RandomResizedCropAndInterpolation with
            # RandomCrop
            transform.transforms[0] = transforms.RandomCrop(config.DATA.IMG_SIZE, padding=4)
        return transform

    t = []
    if resize_im:
        if config.TEST.CROP:
            size = int((256 / 224) * config.DATA.IMG_SIZE)
            t.append(
                transforms.Resize(size, interpolation=_pil_interp(config.DATA.INTERPOLATION)),
                # to maintain same ratio w.r.t. 224 images
            )
            t.append(transforms.CenterCrop(config.DATA.IMG_SIZE))
        else:
            t.append(
                transforms.Resize((config.DATA.IMG_SIZE, config.DATA.IMG_SIZE),
                                  interpolation=_pil_interp(config.DATA.INTERPOLATION))
            )

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


import random
import os
from collections import defaultdict, Counter
import torchvision.datasets as datasets
from torch.utils.data import Dataset, random_split

def split_dataset(args):
    # 定义训练集和验证集的比例
    train_ratio = 0.8
    val_ratio = 0.2

    # 加载数据集
    data_dir = args.data_path
    dataset = datasets.ImageFolder(data_dir)

    # 获取类别名称和对应的样本数量
    class_names = dataset.classes
    class_counts =  Counter(dataset.targets)


    # 计算每个类别应该分配到训练集和验证集中的样本数量
    k_fold_class_counts = [{} for _ in range(args.k)]
    for label, count in class_counts.items():
        fold_count = int(count/args.k)
        for i in range(args.k):
            k_fold_class_counts[i][label] = fold_count

    # 随机分配样本到训练集和验证集
    k_fold_data = [[]for _ in range(args.k)]
    for i in range(len(dataset)):
        img_path, label = dataset.imgs[i]
        for j in range(args.k):
            if k_fold_class_counts[j][label] > 0:
                k_fold_data[j].append((img_path, label))
                k_fold_class_counts[j][label] -= 1
                break

    for i in range(args.k):
        k_fold_data[i].sort()

    train_data = []
    val_data = []
    for i in range(args.k):
        if args.val_fold == i:
            val_data += k_fold_data[i]
        else:
            train_data += k_fold_data[i]

    return train_data,val_data

def training_set_only(args):
    # 加载数据集
    data_dir = args.data_path
    dataset = datasets.ImageFolder(data_dir)

    # 获取类别名称和对应的样本数量
    class_names = dataset.classes
    class_counts =  Counter(dataset.targets)


    # 计算每个类别应该分配到训练集和验证集中的样本数量
    k_fold_class_counts = [{} for _ in range(args.k)]
    for label, count in class_counts.items():
        fold_count = int(count/args.k)
        for i in range(args.k):
            k_fold_class_counts[i][label] = fold_count

    # 随机分配样本到训练集和验证集
    k_fold_data = [[]for _ in range(args.k)]
    for i in range(len(dataset)):
        img_path, label = dataset.imgs[i]
        for j in range(args.k):
            if k_fold_class_counts[j][label] > 0:
                k_fold_data[j].append((img_path, label))
                k_fold_class_counts[j][label] -= 1
                break

    for i in range(args.k):
        k_fold_data[i].sort()

    train_data = []
    for i in range(args.k):
        train_data += k_fold_data[i]

    return train_data

def split_dataset_by_class(args):
    # 定义训练集和验证集的比例
    train_ratio = 0.8
    val_ratio = 0.2

    # 加载数据集
    data_dir = args.data_path
    dataset = datasets.ImageFolder(data_dir)

    # 获取类别名称和对应的样本数量
    class_names = dataset.classes
    class_counts =  Counter(dataset.targets)


    # 计算每个类别应该分配到训练集和验证集中的样本数量
    k_fold_class_counts = [{} for _ in range(args.k)]
    fold_class_count = int(len(class_names) / args.k)
    fold_class_counts = [fold_class_count] * args.k
    for j in range(len(class_names)):
        for i in range(args.k):
            if fold_class_counts[i] > 0:
                k_fold_class_counts[i][j] = 400
                fold_class_counts[i] -= 1
                break

    # 随机分配样本到训练集和验证集
    k_fold_data = [[]for _ in range(args.k)]
    for i in range(len(dataset)):
        img_path, label = dataset.imgs[i]
        for j in range(args.k):
            if label in k_fold_class_counts[j].keys() and k_fold_class_counts[j][label] > 0:
                k_fold_data[j].append((img_path, label))
                k_fold_class_counts[j][label] -= 1
                break

    for i in range(args.k):
        k_fold_data[i].sort()

    train_data = []
    val_data = []
    for i in range(args.k):
        if args.val_fold == i:
            val_data += k_fold_data[i]
        else:
            train_data += k_fold_data[i]

    return train_data,val_data

def build_transform_iBotHash(is_train, args):
    mean = [0.4702532,0.48587758,0.38928695]
    std = [0.19859357,0.19675725,0.19619795]
    # train transform
    if is_train:
        # this should always dispatch to transforms_imagenet_train
        transform = create_transform(
            input_size=args.input_size,
            is_training=True,
            color_jitter=args.color_jitter,
            auto_augment=args.aa,
            interpolation='bicubic',
            re_prob=args.reprob,
            re_mode=args.remode,
            re_count=args.recount,
            mean=mean,
            std=std,
        )
        return transform

    # eval transform
    t = []
    if args.input_size <= 224:
        crop_pct = 224 / 256
    else:
        crop_pct = 1.0
    size = int(args.input_size / crop_pct)
    t.append(
        transforms.Resize(size, interpolation=PIL.Image.BICUBIC),  # to maintain same ratio w.r.t. 224 images
    )
    t.append(transforms.CenterCrop(args.input_size))

    t.append(transforms.ToTensor())
    t.append(transforms.Normalize(mean, std))
    return transforms.Compose(t)


# 定义自定义数据集类
class CustomDataset(Dataset):
    def __init__(self, data, transform=None):
        self.data = data
        self.transform = transform

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_path, label = self.data[index]
        img = Image.open(img_path).convert('RGB')
        if self.transform:
            img = self.transform(img)
        return img, label, index