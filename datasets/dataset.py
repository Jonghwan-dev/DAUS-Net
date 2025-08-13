import os
import cv2
import random
import numpy as np
import torch
from scipy import ndimage
from scipy.ndimage import zoom
from torch.utils.data import Dataset

from datasets.omni_dataset import position_prompt_dict
from datasets.omni_dataset import nature_prompt_dict

from datasets.omni_dataset import position_prompt_one_hot_dict
from datasets.omni_dataset import nature_prompt_one_hot_dict
from datasets.omni_dataset import type_prompt_one_hot_dict


def random_horizontal_flip(image, label):
    axis = 1
    image = np.flip(image, axis=axis).copy()
    label = np.flip(label, axis=axis).copy()
    return image, label


def random_rotate(image, label):
    angle = np.random.randint(-20, 20)
    image = ndimage.rotate(image, angle, order=0, reshape=False)
    label = ndimage.rotate(label, angle, order=0, reshape=False)
    return image, label


def _aspect_ratio_resize_pad(image, label, out_h, out_w, pad_value_img=0.0, pad_value_lbl=0):
    """
    Resize keeping aspect ratio, then pad to (out_h, out_w).
    image: HxWx3 float [0..1]
    label: HxW   int/float
    Returns padded (image, label).
    """
    h, w = image.shape[:2]
    scale = min(out_h / h, out_w / w)
    new_h, new_w = int(round(h * scale)), int(round(w * scale))

    # Resize
    image_rs = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    label_rs = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

    # Create padded canvas
    pad_img = np.full((out_h, out_w, 3), pad_value_img, dtype=np.float32)
    pad_lbl = np.full((out_h, out_w), pad_value_lbl, dtype=label_rs.dtype)

    # Center paste
    off_y = (out_h - new_h) // 2
    off_x = (out_w - new_w) // 2
    pad_img[off_y:off_y+new_h, off_x:off_x+new_w, :] = image_rs
    pad_lbl[off_y:off_y+new_h, off_x:off_x+new_w] = label_rs

    return pad_img, pad_lbl


def _maybe_hflip(image, label, p=0.5):
    if random.random() < p:
        image = np.ascontiguousarray(np.flip(image, axis=1))
        label = np.ascontiguousarray(np.flip(label, axis=1))
    return image, label


def _random_rotation(image, label, max_deg=20, p=0.5):
    if random.random() < p:
        angle = random.uniform(-max_deg, max_deg)
        h, w = label.shape
        M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
        # BORDER_CONSTANT with 0 for mask, 0.0 for image background
        image = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=0.0)
        label = cv2.warpAffine(label, M, (w, h), flags=cv2.INTER_NEAREST, borderMode=cv2.BORDER_CONSTANT, borderValue=0)
    return image, label


class RandomGeneratorTM(object):
    """
    Train-time transform: light augments + aspect ratio resize + pad to target size.
    Maintains the same output dict schema and tensor shapes as original RandomGenerator.
    """
    def __init__(self, output_size, hflip_p=0.5, rot_p=0.5, max_rot_deg=20, scale_jitter=(0.9, 1.1),
                 intensity_jitter=0.1, gaussian_noise_std=0.0, blur_p=0.0):
        self.output_size = output_size  # [H, W]
        self.hflip_p = hflip_p
        self.rot_p = rot_p
        self.max_rot_deg = max_rot_deg
        self.scale_jitter = scale_jitter
        self.intensity_jitter = intensity_jitter
        self.gaussian_noise_std = gaussian_noise_std
        self.blur_p = blur_p

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        # Optional scale jitter in original resolution space
        if self.scale_jitter is not None:
            s = random.uniform(self.scale_jitter[0], self.scale_jitter[1])
            if s != 1.0:
                new_h, new_w = max(1, int(round(image.shape[0] * s))), max(1, int(round(image.shape[1] * s)))
                image = cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
                label = cv2.resize(label, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Light spatial augments
        image, label = _maybe_hflip(image, label, p=self.hflip_p)
        image, label = _random_rotation(image, label, max_deg=self.max_rot_deg, p=self.rot_p)

        # Aspect ratio preserving resize + pad to target size
        image, label = _aspect_ratio_resize_pad(image, label, self.output_size[0], self.output_size[1])

        # Intensity jitter (grayscale image in [0,1])
        if self.intensity_jitter is not None and self.intensity_jitter > 0:
            jitter = 1.0 + random.uniform(-self.intensity_jitter, self.intensity_jitter)
            image = np.clip(image * jitter, 0.0, 1.0)
        # Gaussian noise
        if self.gaussian_noise_std is not None and self.gaussian_noise_std > 0:
            noise = np.random.normal(0.0, self.gaussian_noise_std, size=image.shape).astype(np.float32)
            image = np.clip(image + noise, 0.0, 1.0)
        # Optional blur
        if self.blur_p > 0 and random.random() < self.blur_p:
            # kernel size odd and small
            image = cv2.GaussianBlur(image, ksize=(3, 3), sigmaX=0)

        # Convert HxWx3 to grayscale and channel-first (1,H,W)
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.mean(axis=2).astype(np.float32)
        # Binarize mask to {0,255}
        label = (label > 0).astype(np.uint8) * 255
        image = torch.from_numpy(image).unsqueeze(0)  # (1,H,W)
        label = torch.from_numpy(label.astype(np.float32))

        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': sample['type_prompt']}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class ResizePadTM(object):
    """
    Val/Test-time deterministic transform: aspect ratio resize + center pad to target size.
    """
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        image, label = _aspect_ratio_resize_pad(image, label, self.output_size[0], self.output_size[1])

        # Convert HxWx3 to grayscale and channel-first (1,H,W)
        if image.ndim == 3 and image.shape[2] == 3:
            image = image.mean(axis=2).astype(np.float32)
        # Binarize mask to {0,255}
        label = (label > 0).astype(np.uint8) * 255
        image = torch.from_numpy(image).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': sample['type_prompt']}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class RandomGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']

        if random.random() > 0.5:
            image, label = random_horizontal_flip(image, label)
        elif random.random() > 0.5:
            image, label = random_rotate(image, label)
        x, y, _ = image.shape

        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)

        scale = random.uniform(0.8, 1.2)
        image = zoom(image, (scale, scale, 1), order=1)
        label = zoom(label, (scale, scale), order=0)

        x, y, _ = image.shape
        if scale > 1:
            startx = x//2 - (self.output_size[0]//2)
            starty = y//2 - (self.output_size[1]//2)
            image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
            label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]
        else:
            if x > self.output_size[0]:
                startx = x//2 - (self.output_size[0]//2)
                image = image[startx:startx+self.output_size[0], :, :]
                label = label[startx:startx+self.output_size[0], :]
            if y > self.output_size[1]:
                starty = y//2 - (self.output_size[1]//2)
                image = image[:, starty:starty+self.output_size[1], :]
                label = label[:, starty:starty+self.output_size[1]]
            x, y, _ = image.shape
            new_image = np.zeros((self.output_size[0], self.output_size[1], 3))
            new_label = np.zeros((self.output_size[0], self.output_size[1]))
            if x < y:
                startx = self.output_size[0]//2 - (x//2)
                starty = 0
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            else:
                startx = 0
                starty = self.output_size[1]//2 - (y//2)
                new_image[startx:startx+x, starty:starty+y, :] = image
                new_label[startx:startx+x, starty:starty+y] = label
            image = new_image
            label = new_label

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class CenterCropGenerator(object):
    def __init__(self, output_size):
        self.output_size = output_size

    def __call__(self, sample):
        image, label = sample['image'], sample['label']
        if 'type_prompt' in sample:
            type_prompt = sample['type_prompt']
        x, y, _ = image.shape
        if x > y:
            image = zoom(image, (self.output_size[0] / y, self.output_size[1] / y, 1), order=1)
            label = zoom(label, (self.output_size[0] / y, self.output_size[1] / y), order=0)
        else:
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / x, 1), order=1)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / x), order=0)
        x, y, _ = image.shape
        startx = x//2 - (self.output_size[0]//2)
        starty = y//2 - (self.output_size[1]//2)
        image = image[startx:startx+self.output_size[0], starty:starty+self.output_size[1], :]
        label = label[startx:startx+self.output_size[0], starty:starty+self.output_size[1]]

        image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)
        label = torch.from_numpy(label.astype(np.float32))
        if 'type_prompt' in sample:
            sample = {'image': image, 'label': label.long(), 'type_prompt': type_prompt}
        else:
            sample = {'image': image, 'label': label.long()}
        return sample


class USdatasetSeg(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, "imgs", img_name)
        label_path = os.path.join(self.data_dir, "masks", img_name)

        image = cv2.imread(img_path)
        label = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)

        label_info_list = [info.strip().split(":") for info in self.label_info]
        for single_label_info in label_info_list:
            label_index = int(single_label_info[0])
            label_value_in_image = int(single_label_info[2])
            label[label == label_value_in_image] = label_index

        label[label > 0] = 1

        sample = {'image': image/255.0, 'label': label}
        if self.transform:
            sample = self.transform(sample)
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample


class USdatasetCls(Dataset):
    def __init__(self, base_dir, list_dir, split, transform=None, prompt=False):
        self.transform = transform  # using transform in torch!
        self.split = split
        self.sample_list = open(os.path.join(list_dir, self.split+'.txt')).readlines()

        self.data_dir = base_dir
        self.label_info = open(os.path.join(list_dir, "config.yaml")).readlines()
        self.prompt = prompt

    def __len__(self):
        return len(self.sample_list)

    def __getitem__(self, idx):

        img_name = self.sample_list[idx].strip('\n')
        img_path = os.path.join(self.data_dir, img_name)

        image = cv2.imread(img_path)
        label = int(img_name.split("/")[0])

        sample = {'image': image/255.0, 'label': np.zeros(image.shape[:2])}
        if self.transform:
            sample = self.transform(sample)
        sample['label'] = torch.from_numpy(np.array(label))
        sample['case_name'] = self.sample_list[idx].strip('\n')
        if self.prompt:
            dataset_name = img_path.split("/")[-3]
            sample['type_prompt'] = type_prompt_one_hot_dict["whole"]
            sample['nature_prompt'] = nature_prompt_one_hot_dict[nature_prompt_dict[dataset_name]]
            sample['position_prompt'] = position_prompt_one_hot_dict[position_prompt_dict[dataset_name]]
        return sample
