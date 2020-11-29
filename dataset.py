"""Pascal VOC Dataset Segmentation Dataloader"""

from collections import Counter
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from torch.utils.data import Dataset
from tqdm import tqdm
from PIL import Image
from torchvision import transforms


VOC_CLASSES = ('background',  # always index 0
               'aeroplane', 'bicycle', 'bird', 'boat',
               'bottle', 'bus', 'car', 'cat', 'chair',
               'cow', 'diningtable', 'dog', 'horse',
               'motorbike', 'person', 'pottedplant',
               'sheep', 'sofa', 'train', 'tvmonitor')

NUM_CLASSES = len(VOC_CLASSES) + 1


class PascalVOCDataset(Dataset):
    """Pascal VOC 2007 Dataset"""
    def __init__(self, list_file, img_dir, mask_dir, img_size, is_transform=None):
        self.images = open(list_file, "rt").read().split("\n")[:-1]
        self.is_transform = is_transform

        self.img_extension = ".jpg"
        self.mask_extension = ".png"

        self.image_root_dir = img_dir
        self.mask_root_dir = mask_dir

        self.img_size = img_size if isinstance(img_size, tuple) else (img_size, img_size)
        self.counts = self.__compute_class_probability()
        self.tf = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        name = self.images[index]
        image_path = os.path.join(self.image_root_dir, name + self.img_extension)
        mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

        image = Image.open(image_path)
        mask = Image.open(mask_path)

        if self.is_transform:
            image, mask = self.transform(image, mask)

        # data = {
        #             'image': torch.FloatTensor(image),
        #             'mask' : torch.LongTensor(gt_mask)
        #             }

        return image, mask

    def transform(self, img, lbl):
        if self.img_size == ("same", "same"):
            pass
        else:
            img = img.resize((self.img_size[0], self.img_size[1]))  # uint8 with RGB mode
            lbl = lbl.resize((self.img_size[0], self.img_size[1]))
        img = self.tf(img)
        lbl = torch.from_numpy(np.array(lbl)).long()
        lbl[lbl == 255] = 0
        return img, lbl

    def __compute_class_probability(self):
        counts = dict((i, 0) for i in range(NUM_CLASSES))

        for name in self.images:
            mask_path = os.path.join(self.mask_root_dir, name + self.mask_extension)

            raw_image = Image.open(mask_path).resize((512, 512))
            imx_t = np.array(raw_image).reshape(512*512)
            imx_t[imx_t == 255] = 0

            for i in range(NUM_CLASSES):
                counts[i] += np.sum(imx_t == i)

        return counts

    def get_class_probability(self):
        values = np.array(list(self.counts.values()))
        p_values = values/np.sum(values)

        return torch.Tensor(p_values)


# if __name__ == "__main__":
#     data_root = os.path.join("data", "VOCdevkit", "VOC2007")
#     list_file_path = os.path.join(data_root, "ImageSets", "Segmentation", "train.txt")
#     img_dir = os.path.join(data_root, "JPEGImages")
#     mask_dir = os.path.join(data_root, "SegmentationObject")
#
#     objects_dataset = PascalVOCDataset(list_file=list_file_path,
#                                        img_dir=img_dir,
#                                        mask_dir=mask_dir)
#
#     print(objects_dataset.get_class_probability())
#
#     sample = objects_dataset[0]
#     image, mask = sample['image'], sample['mask']
#
#     image.transpose_(0, 2)
#
#     fig = plt.figure()
#
#     a = fig.add_subplot(1,2,1)
#     plt.imshow(image)
#
#     a = fig.add_subplot(1,2,2)
#     plt.imshow(mask)
#
#     plt.show()

