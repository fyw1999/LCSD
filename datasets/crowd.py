from PIL import Image
import torch.utils.data as data
import os
import cv2
import time
from glob import glob
import torch
import torchvision.transforms.functional as F
from torchvision import transforms
import random
import numpy as np
import scipy.io as sio
import warnings
warnings.filterwarnings("ignore")

def random_crop(im_h, im_w, crop_h, crop_w):
    res_h = im_h - crop_h
    res_w = im_w - crop_w
    i = random.randint(0, res_h)
    j = random.randint(0, res_w)
    return i, j, crop_h, crop_w


def gen_discrete_map(im_height, im_width, points):
    """
        func: generate the discrete map.
        points: [num_gt, 2], for each row: [width, height]
        """
    discrete_map = np.zeros([im_height, im_width], dtype=np.float32)
    h, w = discrete_map.shape[:2]
    num_gt = points.shape[0]
    if num_gt == 0:
        return discrete_map
    
    # fast create discrete map
    points_np = np.array(points).round().astype(int)
    p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
    p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
    p_index = torch.from_numpy(p_h* im_width + p_w)
    discrete_map = torch.zeros(im_width * im_height).scatter_add_(0, index=p_index, src=torch.ones(im_width*im_height)).view(im_height, im_width).numpy()

    ''' slow method
    for p in points:
        p = np.round(p).astype(int)
        p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
        discrete_map[p[0], p[1]] += 1
    '''
    assert np.sum(discrete_map) == num_gt
    return discrete_map


class Base(data.Dataset):
    def __init__(self, root_path, crop_size, downsample_ratio=8):

        self.root_path = root_path
        self.c_size = crop_size
        self.d_ratio = downsample_ratio
        assert self.c_size % self.d_ratio == 0
        self.dc_size = self.c_size // self.d_ratio
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])

    def __len__(self):
        pass

    def __getitem__(self, item):
        pass

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.c_size
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0]
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()


class Crowd_qnrf(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))
        if method not in ['train', 'val']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)  
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name


class Crowd_nwpu(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.im_list = sorted(glob(os.path.join(self.root_path, '*.jpg')))
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        img_path = self.im_list[item]
        gd_path = img_path.replace('jpg', 'npy')
        img = Image.open(img_path).convert('RGB')
        if self.method == 'train':
            keypoints = np.load(gd_path)
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            keypoints = np.load(gd_path)
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, len(keypoints), name
        elif self.method == 'test':
            img = self.trans(img)
            name = os.path.basename(img_path).split('.')[0]
            return img, name


class Crowd_sh(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        if method not in ['train', 'val']:
            raise Exception("not implement")
        if method == 'train':
            im_list_file = os.path.join(root_path,"train.list")
        elif method == 'val':
            im_list_file = os.path.join(root_path,"test.list")
        with open(im_list_file,'r') as f:
            self.im_list = f.read().split('\n')
        if '' in self.im_list:
            self.im_list.remove('')
        print('number of img: {}'.format(len(self.im_list)))

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        pair_path = self.im_list[item]
        img_path = os.path.join(self.root_path, pair_path.split()[0])
        gd_path = os.path.join(self.root_path, pair_path.split()[1])
        img = Image.open(img_path).convert('RGB')
        keypoints = np.loadtxt(gd_path)

        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val':
            img = self.trans(img)
            name = img_path.split('/')[-1].split('.')[0]
            return img, len(keypoints), name

    def train_transform(self, img, keypoints):
        wd, ht = img.size
        st_size = 1.0 * min(wd, ht)
        # resize the image to fit the crop size
        if st_size < self.c_size:
            rr = 1.0 * self.c_size / st_size
            wd = round(wd * rr)
            ht = round(ht * rr)
            st_size = 1.0 * min(wd, ht)
            img = img.resize((wd, ht), Image.BICUBIC)
            keypoints = keypoints * rr
        assert st_size >= self.c_size, print(wd, ht)
        assert len(keypoints) >= 0
        i, j, h, w = random_crop(ht, wd, self.c_size, self.c_size)
        img = F.crop(img, i, j, h, w)
        if len(keypoints) > 0:
            keypoints = keypoints - [j, i]
            idx_mask = (keypoints[:, 0] >= 0) * (keypoints[:, 0] <= w) * \
                       (keypoints[:, 1] >= 0) * (keypoints[:, 1] <= h)
            keypoints = keypoints[idx_mask]
        else:
            keypoints = np.empty([0, 2])

        gt_discrete = gen_discrete_map(h, w, keypoints)
        down_w = w // self.d_ratio
        down_h = h // self.d_ratio
        gt_discrete = gt_discrete.reshape([down_h, self.d_ratio, down_w, self.d_ratio]).sum(axis=(1, 3))
        assert np.sum(gt_discrete) == len(keypoints)

        if len(keypoints) > 0:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                keypoints[:, 0] = w - keypoints[:, 0] - 1
        else:
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
        gt_discrete = np.expand_dims(gt_discrete, 0)

        return self.trans(img), torch.from_numpy(keypoints.copy()).float(), torch.from_numpy(
            gt_discrete.copy()).float()
        
class Crowd(data.Dataset):
    def __init__(self, root_path, method, downsample_ratio):
        self.root_path = root_path
        self.method = method
        self.downsample_ratio = downsample_ratio
        if method == 'train':
            im_list_file = os.path.join(root_path, "train_data.list")
        elif method == 'test':
            im_list_file = os.path.join(root_path, "test_data.list")
        with open(im_list_file,'r') as f:
            self.im_list = f.read().split('\n')
        if '' in self.im_list:
            self.im_list.remove('')
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'test']:
            raise Exception("not implement")
        
        self.trans = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])


    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        pair_path = self.im_list[item]
        img_path = os.path.join(self.root_path, pair_path.split()[0])
        gd_path = os.path.join(self.root_path, pair_path.split()[1])
        img = cv2.cvtColor(cv2.imread(img_path), cv2.COLOR_BGR2RGB)
        keypoints = np.loadtxt(gd_path)
        if len(keypoints.shape) == 1 and keypoints.shape[0] == 2:
            keypoints = np.expand_dims(keypoints,0)

        img_height = img.shape[0]
        img_width = img.shape[1]
        if img.shape[0] % self.downsample_ratio:
            img_height = round(img.shape[0] / self.downsample_ratio) * self.downsample_ratio

        if img.shape[1] % self.downsample_ratio:
            img_width = round(img.shape[1] / self.downsample_ratio) * self.downsample_ratio
        
        if len(keypoints) > 0:
            keypoints[:,0] = keypoints[:,0] * (img_width/img.shape[1])
            keypoints[:,1] = keypoints[:,1] * (img_height/img.shape[0])

        img = cv2.resize(img, (img_width, img_height), interpolation=cv2.INTER_CUBIC)
        img_tensor = self.trans(img)
        base_name = os.path.basename(img_path)
        name = base_name.split('.')[0]
        return img, img_tensor, keypoints, name

class Crowd_har(Base):
    def __init__(self, root_path, crop_size,
                 downsample_ratio=8,
                 method='train'):
        super().__init__(root_path, crop_size, downsample_ratio)
        self.method = method
        self.root_path = root_path
        if method == 'train':
            im_list_file = os.path.join(root_path,"train_data.list")
        elif method == 'val1':
            im_list_file = os.path.join(root_path,"real_test_data.list")
        elif method == 'val2':
            im_list_file = os.path.join(root_path,"fake_test_data.list")
        elif method == 'test':
            im_list_file = os.path.join(root_path,"test_data.list")
        with open(im_list_file,'r') as f:
            self.im_list = f.read().split('\n')
        if '' in self.im_list:
            self.im_list.remove('')
        print('number of img: {}'.format(len(self.im_list)))

        if method not in ['train', 'val1', 'val2', 'test']:
            raise Exception("not implement")

    def __len__(self):
        return len(self.im_list)

    def __getitem__(self, item):
        pair_path = self.im_list[item]
        img_path = os.path.join(self.root_path, pair_path.split()[0])
        gd_path = os.path.join(self.root_path, pair_path.split()[1])
        img = Image.open(img_path).convert('RGB')
        keypoints = np.loadtxt(gd_path)
        if len(keypoints.shape) == 1 and keypoints.shape[0] == 2:
            keypoints = np.expand_dims(keypoints,0)
        if self.method == 'train':
            return self.train_transform(img, keypoints)
        elif self.method == 'val1' or self.method == 'val2' or self.method == 'test':
            img = self.trans(img)
            name = img_path.split('/')
            scene_name = name[-2]
            base_name = name[-1].split('.')[0]
            name = scene_name + '_' + base_name
            return img, len(keypoints), name
    
    
