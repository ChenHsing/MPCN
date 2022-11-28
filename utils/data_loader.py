# -*- coding: utf-8 -*-


import cv2
import json
import numpy as np
import logging
import os
import random
import scipy.io
import scipy.ndimage
import sys
import torch.utils.data.dataset
import torch.nn.functional as F

from enum import Enum, unique
from config import cfg


import utils.binvox_rw

BASE_CLASS  = ['02691156','02958343','03001627','03211117','04401088','03691459','04379243']
NOVEL_CLASS = ['02828884','02933112','03636649','04256520','04530566','04090263']
USE_AVERAGE = True


def downsampling(mat_high):
    x = torch.tensor(mat_high)
    x = x.unsqueeze(0)
    x = x.unsqueeze(0)
    x = x.float()
    y1 = F.interpolate(x, size=[32, 32, 32], mode='area')
    y1 = y1[0][0]
    y1 = y1.squeeze().__ge__(0.0001).float()
    y_np = np.array(y1)
    return y_np

@unique
class DatasetType(Enum):
    TRAIN = 0
    TEST = 1
    VAL = 2


class ShapeNetDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""
    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume = self.get_datum(idx)
        rendering_pos = rendering_images.copy()
        if self.transforms:
            rendering_images = self.transforms(rendering_images)
            rendering_pos = self.transforms(rendering_pos)

        return taxonomy_name, sample_name, rendering_images, rendering_pos, volume, volume

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):
        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']
        # Get data of rendering images
        if self.dataset_type == DatasetType.TRAIN:
            selected_rendering_image_paths = [
                rendering_image_paths[i]
                for i in random.sample(range(len(rendering_image_paths)), self.n_views_rendering)
            ]

        else:
            selected_rendering_image_paths = [rendering_image_paths[i] for i in range(self.n_views_rendering)]

        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED).astype(np.float32) / 255.

            if len(rendering_image.shape) < 3:
                logging.error('It seems that there is something wrong with the image file %s' % (image_path))
                sys.exit(2)

            rendering_images.append(rendering_image)
        # Get data of volume
        _, suffix = os.path.splitext(volume_path)
        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                volume = utils.binvox_rw.read_as_3d_array(f)
                volume = volume.data.astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume


# //////////////////////////////// = End of ShapeNetDataset Class Definition = ///////////////////////////////// #
class ShapeNetDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.SHAPENET.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.SHAPENET.VOXEL_PATH
        self.prior_template = cfg.DATASETS.SHAPENET.PRIOR_PATH
        self.shot_num = cfg.CONST.SHOT_NUM

        # Load all taxonomies of the dataset
        with open(cfg.DATASETS.SHAPENET.TAXONOMY_FILE_PATH, encoding='utf-8') as file:
            self.dataset_taxonomy = json.loads(file.read())

    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []

        # Load data for each category
        for taxonomy in self.dataset_taxonomy:
            taxonomy_folder_name = taxonomy['taxonomy_id']

            samples = []
            # print(taxonomy)
            if dataset_type == DatasetType.TRAIN:
                if taxonomy['train'] is not None:
                    samples = taxonomy['train']
            elif dataset_type == DatasetType.TEST:
                samples = taxonomy['test']
            elif dataset_type == DatasetType.VAL:
                samples = taxonomy['val']
            files.extend(self.get_files_of_taxonomy(dataset_type,taxonomy_folder_name, samples))

        return ShapeNetDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, dataset_type,taxonomy_folder_name, samples):
        files_of_taxonomy = []

        for sample_idx, sample_name in enumerate(samples):
            # Get file path of volumes
            volume_file_path = self.volume_path_template % (taxonomy_folder_name, sample_name)
            if taxonomy_folder_name in BASE_CLASS and USE_AVERAGE:
                shot = 'avg'
                prior_file_path = self.prior_template % (shot, taxonomy_folder_name)
            elif taxonomy_folder_name in BASE_CLASS and not USE_AVERAGE:
                shot = self.shot_num
                prior_file_path = []
                for i in range(shot):
                    length = len(samples)
                    randnum = random.randint(0, length - 1)
                    sap_name = samples[randnum]
                    tmp = self.volume_path_template % (taxonomy_folder_name, sap_name)
                    prior_file_path.append(tmp)
            else:
                shot = self.shot_num
                prior_file_path = []
                for i in range(shot):
                    length = len(samples)
                    randnum = random.randint(0,length-1)
                    sap_name = samples[randnum]
                    tmp = self.volume_path_template % (taxonomy_folder_name, sap_name)
                    prior_file_path.append(tmp)
            if not os.path.exists(volume_file_path):
                logging.warn('Ignore sample %s/%s since volume file not exists.' % (taxonomy_folder_name, sample_name))
                continue

            # Get file list of rendering images
            img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, 0)
            img_folder = os.path.dirname(img_file_path)
            total_views = len(os.listdir(img_folder))
            rendering_image_indexes = range(total_views)
            rendering_images_file_path = []
            for image_idx in rendering_image_indexes:
                img_file_path = self.rendering_image_path_template % (taxonomy_folder_name, sample_name, image_idx)
                if not os.path.exists(img_file_path):
                    continue

                rendering_images_file_path.append(img_file_path)

            if len(rendering_images_file_path) == 0:
                logging.warn('Ignore sample %s/%s since image files not exists.' % (taxonomy_folder_name, sample_name))
                continue
            # print('all_len',len(rendering_images_file_path))

            # Append to the list of rendering images
            if dataset_type == DatasetType.TRAIN and False:
                for i in range(24):
                    files_of_taxonomy.append({
                        'taxonomy_name': taxonomy_folder_name,
                        'sample_name': sample_name,
                        'rendering_images': [rendering_images_file_path[i]],
                        'volume': volume_file_path,
                        'prior_shape': prior_file_path,
                    })
            else:
                files_of_taxonomy.append({
                    'taxonomy_name': taxonomy_folder_name,
                    'sample_name': sample_name,
                    'rendering_images': rendering_images_file_path,
                    'volume': volume_file_path,
                    'prior_shape': prior_file_path,
                })


        return files_of_taxonomy


class PascalDataset(torch.utils.data.dataset.Dataset):
    """ShapeNetDataset class used for PyTorch DataLoader"""

    def __init__(self, dataset_type, file_list, n_views_rendering, transforms=None):
        self.dataset_type = dataset_type
        self.file_list = file_list
        self.transforms = transforms
        self.n_views_rendering = n_views_rendering

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        taxonomy_name, sample_name, rendering_images, volume, prior_shape = self.get_datum(idx)
        rendering_pos = rendering_images.copy()
        if self.transforms:
            rendering_images = self.transforms(rendering_images)
            rendering_pos = self.transforms(rendering_pos)

        return taxonomy_name, sample_name, rendering_images,rendering_pos, volume, prior_shape

    def set_n_views_rendering(self, n_views_rendering):
        self.n_views_rendering = n_views_rendering

    def get_datum(self, idx):

        taxonomy_name = self.file_list[idx]['taxonomy_name']
        sample_name = self.file_list[idx]['sample_name']
        rendering_image_paths = self.file_list[idx]['rendering_images']
        volume_path = self.file_list[idx]['volume']
        prior_path = self.file_list[idx]['prior_shape']

        selected_rendering_image_paths = [rendering_image_paths]
        rendering_images = []
        for image_path in selected_rendering_image_paths:
            rendering_image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
            rendering_image = cv2.resize(rendering_image, (224, 224))
            rendering_image = rendering_image.astype(np.float32) / 255.
            if len(rendering_image.shape) < 3:
                rendering_image = np.stack((rendering_image,) * 3, -1)

            rendering_images.append(rendering_image)

        # Get data of volume
        _, suffix = os.path.splitext(volume_path)

        if suffix == '.mat':
            volume = scipy.io.loadmat(volume_path)
            volume = volume['Volume'].astype(np.float32)
        elif suffix == '.binvox':
            with open(volume_path, 'rb') as f:
                binvox_file = utils.binvox_rw.read_as_3d_array(f)
                numpy_file = binvox_file.data.reshape(128, 128, 128).astype('uint8')
                numpy_file = np.array(numpy_file)
                numpy_file = downsampling(numpy_file)
                volume = numpy_file
            with open(volume_path, 'rb') as f:
                binvox_file = utils.binvox_rw.read_as_3d_array(f)
                numpy_file = binvox_file.data.reshape(128, 128, 128).astype('uint8')
                numpy_file = np.array(numpy_file)
                numpy_file = downsampling(numpy_file)
                prior_volume = numpy_file
        elif suffix == '.npy':
            volume = np.load(volume_path).astype(np.float32)
            prior_volume = np.load(prior_path).astype(np.float32)

        return taxonomy_name, sample_name, np.asarray(rendering_images), volume, prior_volume




class PascalDataLoader:
    def __init__(self, cfg):
        self.dataset_taxonomy = None
        self.rendering_image_path_template = cfg.DATASETS.PASCAL3D.RENDERING_PATH
        self.volume_path_template = cfg.DATASETS.PASCAL3D.VOXEL_PATH


    def get_dataset(self, dataset_type, n_views_rendering, transforms=None):
        files = []
        sample = 'train'
        if dataset_type == DatasetType.TRAIN:
            sample = cfg.DATASETS.PASCAL3D.TRAIN

        elif dataset_type == DatasetType.TEST:
            sample = cfg.DATASETS.PASCAL3D.TEST
        elif dataset_type == DatasetType.VAL:
            sample = cfg.DATASETS.PASCAL3D.TEST

        files.extend(self.get_files_of_taxonomy(dataset_type, sample))

        logging.info('Complete collecting files of the dataset. Total files: %d.' % (len(files)))
        return PascalDataset(dataset_type, files, n_views_rendering, transforms)

    def get_files_of_taxonomy(self, dataset_type, taxonomy_folder_name):
        files_of_taxonomy = []
        with open(taxonomy_folder_name, "r") as f:
            for line in f.readlines():
                line = line.strip('\n')
                splits = line.split(' ')
                clz = splits[0]
                filename = splits[1]
                cad = splits[2]
                cad = str(cad).rjust(2, '0')
                img_path = os.path.join(self.rendering_image_path_template, clz, filename)
                vox_path = os.path.join(self.volume_path_template, clz.split('_')[0], cad + '.npy')
                prior_path = os.path.join(self.volume_path_template, clz.split('_')[0], '10-shot.npy')
                volume_file_path = vox_path
                if dataset_type == "Train":
                    prior_file_path = prior_path
                else:
                    prior_file_path = prior_path

                files_of_taxonomy.append({
                    'taxonomy_name': clz,
                    'sample_name': filename,
                    'rendering_images': img_path,
                    'volume': volume_file_path,
                    'prior_shape': prior_file_path,
                })
        return files_of_taxonomy
# /////////////////////////////// = End of PascalDataLoader Class Definition = /////////////////////////////// #

DATASET_LOADER_MAPPING = {
    'ShapeNet': ShapeNetDataLoader,
    'Pascal3D': PascalDataLoader,
}
