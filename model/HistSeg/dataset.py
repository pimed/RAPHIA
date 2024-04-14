"""
@Author: Sulaiman Vesal
Date: Tuesday, 04, 2020


"""
import os.path
import numpy as np
import cv2
import pandas as pd
from skimage.exposure import match_histograms
from matplotlib import pyplot as plt
from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,
    CenterCrop,
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion,
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,
    RandomGamma,
    GaussNoise
)

class ImageProcessor:
    @staticmethod
    def augmentation(image, mask, noise=False, transform=False, clahe=True, r_bright=True, r_gamma=True):
        aug_list = [
            VerticalFlip(p=0.5),
            HorizontalFlip(p=0.5),
            RandomRotate90(p=0.5),
                    ]
        if r_bright:
            aug_list += [RandomBrightnessContrast(p=.5)]
        if r_gamma:
            aug_list += [RandomGamma(p=.5)]
        if clahe:
            aug_list += [CLAHE(p=1., always_apply=True)]
        if noise:
            aug_list += [GaussNoise(p=.5, var_limit=1.)]
        if transform:
            aug_list += [ElasticTransform(p=.5, sigma=1., alpha_affine=20, border_mode=0)]
        aug = Compose(aug_list)

        augmented = aug(image=image, mask=mask)
        image_heavy = augmented['image']
        mask_heavy = augmented['mask']
        return image_heavy, mask_heavy

    @staticmethod
    def split_data(img_path):
        """
        Load train csv file and split the data into train and validation!
        :return:
        """
        df_train = pd.read_csv(img_path)
        ids_train = df_train['histology']
        return ids_train

    @staticmethod
    def crop_volume(vol, crop_size=112):

        """
        :param vol:
        :return:
        """
        return np.array(vol[:,
                        int(vol.shape[1] / 2) - crop_size: int(vol.shape[1] / 2) + crop_size,
                        int(vol.shape[2] / 2) - crop_size: int(vol.shape[2] / 2) + crop_size, ])


class DataGenerator:
    def __init__(self, df,
                 channel="channel_first",
                 apply_noise=False,
                 apply_transform=False,
                 phase="train",
                 apply_online_aug=True,
                 batch_size=16,
                 height=256,
                 width=256,
                 crop_size=0,
                 n_samples=-1,
                 offline_aug=False,
                 toprint=False):
        assert phase == "train" or phase == "valid", r"phase has to be either 'train' or 'valid'"
        assert isinstance(apply_noise, bool), "apply_noise has to be bool"
        assert isinstance(apply_online_aug, bool), "apply_online_aug has to be bool"
        self._data = df
        self._len = len(df)
        self._shuffle_indices = np.arange(len(df))
        self._shuffle_indices = np.random.permutation(self._shuffle_indices)
        self._height, self._width = height, width
        self._apply_aug = apply_online_aug
        self._apply_noise = apply_noise
        self._apply_tranform = apply_transform
        self._crop_size = crop_size
        self._phase = phase
        self._channel = channel
        self._batch_size = batch_size
        self._index = 0
        self._totalcount = 0
        if n_samples == -1:
            self._n_samples = len(df)
        else:
            self._n_samples = n_samples
        self._offline_aug = offline_aug
        self._toprint = toprint

    def __len__(self):
        return self._len

    @property
    def apply_aug(self):
        return self._apply_aug

    @apply_aug.setter
    def apply_aug(self, aug):
        assert isinstance(aug, bool), "apply_aug has to be bool"
        self._apply_aug = aug

    def get_image_paths(self, id):
        if self._phase == "train":
            img_path = '../dataset/histology/{}'.format(id)
            mask_path = '../dataset/mask/{}'.format(id)
        else:
            img_path = '../dataset/histology/{}'.format(id)
            mask_path = '../dataset/mask/{}'.format(id)
        return img_path, mask_path

    def convert(self, img, target_type_min, target_type_max, target_type):
        imin = img.min()
        imax = img.max()

        a = (target_type_max - target_type_min) / (imax - imin)
        b = target_type_max - a * imax
        new_img = (a * img + b).astype(target_type)
        return new_img

    def get_images_masks(self, img_path, mask_path):
        """
        :param img_path:
        :param mask_path:
        :return:
        """
        #img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        img = cv2.imread(img_path)
        # img = cv2.resize(img, (self._width, self._height), interpolation=cv2.INTER_AREA)
        mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
        mask = np.where(mask>1, 1, 0)
        # mask = cv2.resize(mask, (self._width, self._height), interpolation=cv2.INTER_AREA)
        return img, mask

    def one_hot_encod(self, y, num_classes):
        """
        :param y:
        :param num_classes:
        :return:
        """
        """ 1-hot encodes a tensor """

        return np.eye(num_classes, dtype='uint8')[y]

    def __iter__(self):
        # self._index = 0
        self._totalcount = 0
        return self

    def __next__(self):
        # while True:
        # shuffle image names
        x_batch = []
        y_batch = []
        cases = []

        indices = []
        if self._totalcount >= self._n_samples:
            self._totalcount = 0
            raise StopIteration
        for i in range(self._batch_size):
            indices.append(self._index)
            self._index += 1
            self._totalcount += 1
            self._index = self._index % self._len
            if self._totalcount >= self._n_samples:
                break
        ids_train_batch = self._data.iloc[self._shuffle_indices[indices]]

        for _id in ids_train_batch.values:
            img_path, mask_path = self.get_image_paths(id=_id)
            img, mask = self.get_images_masks(img_path=img_path, mask_path=mask_path)
            if self._apply_aug:
                img, mask = ImageProcessor.augmentation(img, mask, noise=self._apply_noise, transform=self._apply_tranform)
            else:
                aug = Compose([CLAHE(always_apply=True)])
                augmented = aug(image=img, mask=mask)
                img, mask = augmented["image"], augmented["mask"]
            x_batch.append(img)
            y_batch.append(mask)
            cases.append(mask_path)
        # min-max batch normalisation
        x_batch = np.array(x_batch, np.float32) / 255.

        if self._crop_size:
            x_batch = ImageProcessor.crop_volume(x_batch, crop_size=self._crop_size // 2)
            y_batch = ImageProcessor.crop_volume(np.array(y_batch), crop_size=self._crop_size // 2)
        if self._channel == "channel_first":
            x_batch = np.moveaxis(x_batch, -1, 1)
        
        y_batch = self.one_hot_encod(np.array(y_batch), num_classes=2)
        y_batch = np.moveaxis(y_batch, source=3, destination=1)

        return x_batch, y_batch, cases


if __name__ == "__main__":
    ids_train = ImageProcessor.split_data("../training_data/train_new.csv")
    ids_valid = ImageProcessor.split_data("../training_data/test_new.csv")
    bs = 16
    num_samples = 1000

    trainA_generator = DataGenerator(df=ids_train,
                                     channel="channel_first",
                                     apply_noise=False,
                                     phase="valid",
                                     apply_online_aug=False,
                                     batch_size=bs,
                                     n_samples=-1)
    img, mask, cases = trainA_generator.__next__()
    print(np.mean(img), np.std(img), img.shape, mask.shape)

    temp = np.argmax(mask, axis=1)
    temp2 = np.moveaxis(img, source=1, destination=3)

    for i in range(16):
        f = plt.figure()
        f.add_subplot(1, 2, 1)
        plt.imshow(temp2[i], cmap='gray'),
        plt.title('Histology Slice')
        f.add_subplot(1, 2, 2)
        plt.imshow(temp2[i], cmap='gray'),
        plt.imshow(temp[i], cmap='jet', alpha=0.5)
        plt.title('Ground Truth Mask')
        plt.show(block=True)
