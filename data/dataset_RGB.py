import albumentations as A
import numpy as np
import torchvision.transforms.functional as F
from torch.utils.data import Dataset
import random


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['jpeg', 'JPEG', 'jpg', 'png', 'JPG', 'PNG', 'gif'])


class DataLoaderTrain(Dataset):
    def __init__(self, rgb_file, img_options=None):
        super(DataLoaderTrain, self).__init__()

        with np.load(rgb_file) as data:
            self.tar_files = data['tar']

        self.img_options = img_options
        self.sizex = len(self.tar_files)  # get the size of target

        self.transform = A.Compose([
            A.Transpose(p=0.3),
            A.Flip(p=0.3),
            A.RandomRotate90(p=0.3),
            A.Rotate(p=0.3),
            A.Resize(height=img_options['h'], width=img_options['w']),
        ])
        
        self.degrade = A.Compose([
            # A.RandomShadow(p=0.5)
            A.RandomShadow(shadow_roi=(0, 0, 1, 1), num_shadows_upper=10, shadow_dimension=15, p=1)
        ])

    def mixup(self, inp_img, tar_img, mode='mixup'):
        mixup_index_ = random.randint(0, self.sizex - 1)

        mixup_tar_img = self.transform(self.tar_files[mixup_index_])['image']
        mixup_inp_img = self.degrade(mixup_tar_img)['image']

        alpha = 0.2
        lam = np.random.beta(alpha, alpha)

        mixup_inp_img = F.to_tensor(mixup_inp_img)
        mixup_tar_img = F.to_tensor(mixup_tar_img)

        if mode == 'mixup':
            inp_img = lam * inp_img + (1 - lam) * mixup_inp_img
            tar_img = lam * tar_img + (1 - lam) * mixup_tar_img
        else:
            img_h, img_w = self.img_options['h'], self.img_options['w']

            cx = np.random.uniform(0, img_w)
            cy = np.random.uniform(0, img_h)

            w = img_w * np.sqrt(1 - lam)
            h = img_h * np.sqrt(1 - lam)

            x0 = int(np.round(max(cx - w / 2, 0)))
            x1 = int(np.round(min(cx + w / 2, img_w)))
            y0 = int(np.round(max(cy - h / 2, 0)))
            y1 = int(np.round(min(cy + h / 2, img_h)))

            inp_img[:, y0:y1, x0:x1] = mixup_inp_img[:, y0:y1, x0:x1]
            tar_img[:, y0:y1, x0:x1] = mixup_tar_img[:, y0:y1, x0:x1]

        return inp_img, tar_img


    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        tar_img = self.tar_files[index_]

        tar_img = self.transform(tar_img)['image']

        inp_img = self.degrade(tar_img)['image']

        inp_img = F.to_tensor(inp_img)
        tar_img = F.to_tensor(tar_img)

        if index_ > 0 and index_ % 3 == 0:
            if random.random() > 0.5:
                inp_img, tar_img = self.mixup(inp_img, tar_img, mode='mixup')
            else:
                inp_img, tar_img = self.mixup(inp_img, tar_img, mode='cutmix')

        return inp_img, tar_img


class DataLoaderVal(Dataset):
    def __init__(self, rgb_file, img_options=None):
        super(DataLoaderVal, self).__init__()

        with np.load(rgb_file) as data:
            self.inp_files = data['inp']
            self.tar_files = data['tar']

        self.img_options = img_options
        self.sizex = len(self.inp_files)  # get the size of target

    def __len__(self):
        return self.sizex

    def __getitem__(self, index):
        index_ = index % self.sizex

        inp_img = F.to_tensor(self.inp_files[index_])
        tar_img = F.to_tensor(self.tar_files[index_])

        return inp_img, tar_img
