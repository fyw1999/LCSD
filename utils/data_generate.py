import os, glob
import cv2
import json
import math
import shutil
import torch
import scipy
import pickle
import random
import numpy as np
from tqdm import tqdm
import scipy.io as io
from PIL import Image
from copy import deepcopy
from itertools import islice
import utils.log_utils as log_utils
from torchvision import transforms
from sortedcontainers import SortedDict
from scipy.ndimage.filters import gaussian_filter 
from models.har_model.util import util as har_util
from models.har_model.models.networks import RainNet
from models.har_model.models.normalize import RAIN
from utils.utils import generate_gaussian_kernels, compute_distances, get_gt_dots, gaussian_filter_density
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator

matplotlib.use('Agg') 

class Har:
    def __init__(self, args):
        self.args = args
        self.device = torch.device("cuda")
        self.har_model = self.load_har_model(args)
        self.har_model = self.har_model.to(self.device)
        self.har_transform_image = transforms.Compose([
            transforms.Resize([args.har_image_size, args.har_image_size]),
            transforms.ToTensor(),
            transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))
        ])
        self.har_transform_mask = transforms.Compose([
            transforms.Resize([args.har_image_size, args.har_image_size]),
            transforms.ToTensor()
        ])
        self.args = args

    def load_har_model(self, args):
        net = RainNet(input_nc=3, 
            output_nc=3, 
            ngf=64, 
            norm_layer=RAIN, 
            use_dropout=True)

        load_path = os.path.join(args.resource_path, "net_G_last.pth")
        assert os.path.exists(load_path), print('%s not exists. Please check the file'%(load_path))
        print(f'loading the model from {load_path}')
        state_dict = torch.load(load_path, map_location='cpu')
        har_util.copy_state_dict(net.state_dict(), state_dict)
        # net.load_state_dict(state_dict)
        return net

    def harmonization(self, images, masks):
        batch_size = self.args.har_batch_size
        har_images = []
        img_height = images[0].height
        img_width = images[0].width
        start = 0
        end = min(batch_size, len(images))
        while end <= len(images) and start != end:
            batch_images = images[start:end]
            batch_masks = masks[start:end]
            batch_images_tensor = [self.har_transform_image(img).unsqueeze(0).to(self.device) for img in batch_images]
            batch_masks_tensor = [self.har_transform_mask(mask).unsqueeze(0).to(self.device) for mask in batch_masks]
            
            batch_images_tensor = torch.cat(batch_images_tensor)
            batch_masks_tensor = torch.cat(batch_masks_tensor)
            batch_har_images_tensor = self.har_model.processImage(batch_images_tensor, batch_masks_tensor, batch_images_tensor)
            #(batch_size, 3,512,512)
            batch_har_images = [har_util.tensor2im(batch_har_images_tensor[i].unsqueeze(0)) 
                                for i in range(batch_har_images_tensor.shape[0])]
            har_images.extend(batch_har_images)
            del batch_images_tensor, batch_masks_tensor, batch_har_images_tensor, batch_har_images
            torch.cuda.empty_cache()
            start = end
            end = min(end + batch_size, len(images))

        images = [Image.fromarray(img).resize((img_width, img_height), Image.ANTIALIAS) for img in har_images]
        return images

class DatasetGenerator:
    def __init__(self, args):
        args.min_ped_num = 0
        args.max_ped_num = 100
        args.num_pattern = 'uniform'
        args.pos_pattern = 'uniform'
        args.harmonization = True
        args.har_batch_size = 16
        args.har_image_size = 512
        args.save_synthetic_dataset = True
        args.ped_source = "GCC"
        args.ped_num = 20
        self.stage = ['train']

        if args.harmonization:
            self.har = Har(args)

        pedestrian_dir = os.path.join(args.resource_path, "pedestrians", args.ped_source)
        with open(os.path.join(pedestrian_dir, "info_json.json"), "r") as f:
            self.pesestrians_info = json.load(f)
        self.pedestrians_all = {}
        pedestrians_path = glob.glob(os.path.join(pedestrian_dir, '*.png'))
        for p in pedestrians_path:
            base_name = os.path.basename(p)
            id = base_name.split(".")[0]
            self.pedestrians_all[id] = cv2.cvtColor(cv2.imread(p), cv2.COLOR_BGR2RGB)
        
        select_pedestrians_keys = random.sample(list(self.pedestrians_all), args.ped_num)
        self.pedestrians = {key: self.pedestrians_all[key] for key in select_pedestrians_keys}

        base_path = os.path.join(args.resource_path, args.dataset, args.scene, "scene.jpg")
        
        self.base_image = cv2.cvtColor(cv2.imread(base_path), cv2.COLOR_BGR2RGB)

        negetive_samples_paths = glob.glob(os.path.join(args.resource_path, "*_negetive_samples", "*.jpg"))
        self.negetive_samples = []
        for negetive_samples_path in negetive_samples_paths:
            img = Image.open(negetive_samples_path).convert("RGB")
            self.negetive_samples.append(img)

        self.args = args
        
        self.bg_area = self.base_image.shape[0]*self.base_image.shape[1]
        self.pre_def_scale = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 110, 120, 130, 140, 150, 160, 170, 180, 190, 200]

    def data_aug(self, image):
        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        color_jitter = transforms.ColorJitter(brightness=(0.5,1.5), contrast=(1), saturation=(0.5,1.5), hue=(-0.1,0.1))

        if random.random() < 0.5:
            image = color_jitter(image)
        image = transforms.RandomGrayscale(p=0.2)(image)

        if random.random() < 0.5:
            image = blurring_image(image)

        return image
    
    def generate(self, iter_num, predicted_distribution, predicted_num, scale_model=None):
        self.dataset_name = '_'.join([self.args.dataset, self.args.scene, self.args.ped_source + "-{}".format(self.args.ped_num),
                                        str(self.args.train_num)])
        if predicted_num is not None:
            self.dataset_name += '_prenum'
        else:
            self.dataset_name += '_{}-{}-{}'.format(str(self.args.min_ped_num), str(self.args.max_ped_num), 'uniform')
        probability_cumulative_array = None
        self.roi_mask = np.ones(self.base_image.shape[0:2], dtype=np.uint8)
        if isinstance(predicted_distribution, np.ndarray):
            predicted_distribution = cv2.resize(predicted_distribution, (self.base_image.shape[1],
                                                                         self.base_image.shape[0]), cv2.INTER_CUBIC)
            map_sum = np.sum(predicted_distribution)
            probability_map = predicted_distribution / map_sum
            probability_array = probability_map.reshape(probability_map.shape[0]*probability_map.shape[1])
            probability_cumulative_array = np.cumsum(probability_array)

            self.args.predicted_distribution =True
            self.dataset_name += '_predis'
        else:
            self.dataset_name += "_global-uniform"
        
        if scale_model != None and scale_model.is_use():
            self.dataset_name += "_fit-scale"
        else:
            self.dataset_name += "_random-scale"

        self.dataset_name += "_" + str(iter_num)
        self.save_base_path = os.path.join(self.args.save_dir, "synthetic_datasets", self.dataset_name)
        if not os.path.exists(self.save_base_path):
            os.makedirs(self.save_base_path)
        logger = log_utils.get_logger(os.path.join(self.save_base_path, 'dataset.log'))
        log_utils.print_config(vars(self.args), logger)
            
        data = {} 

        for method in self.stage:
            if method == 'train':
                pictures_num = self.args.train_num
                tmp_data = method + '_data'
            else:
                pictures_num = self.args.val_num
                tmp_data = method + '_data'
                    
            images = []
            gt_points = []
            gt_boxes = []
            masks = []
            images_head_diameters = []
            for i in range(pictures_num):
                if predicted_num is not None:
                    totoal_pedestrians_num = np.random.randint(self.args.min_ped_num, int(max(predicted_num)) + 1)
                else:
                    totoal_pedestrians_num = np.random.randint(self.args.min_ped_num, self.args.max_ped_num + 1)
                
                image, point, gt_img_boxes, mask, head_diameters = self.generate_one_image(self.base_image.copy(), 
                                                                                           totoal_pedestrians_num,
                                                                                           probability_cumulative_array,
                                                                                           scale_model)
                image = Image.fromarray(image).convert("RGB")
                images.append(image)
                gt_points.append(point)
                gt_boxes.append(gt_img_boxes)
                masks.append(Image.fromarray(mask).convert("1"))
                images_head_diameters.append(head_diameters)
            
            if self.args.harmonization:
                images = self.har.harmonization(images, masks)
        
            gt_density_maps = self.generate_density_map(gt_points, images, 5, images_head_diameters)

            if tmp_data == 'train_data':
                for img in self.negetive_samples:
                    images.append(img)
                    gt_points.append(np.empty((0, 2)))
                    gt_boxes.append(np.empty((0, 5)))
                    width, height = img.size
                    density_map = np.zeros((height, width), dtype=np.float32)
                    gt_density_maps.append(density_map)

            data[tmp_data] = {}
            data[tmp_data]["images"] = images
            data[tmp_data]["gt_points"] = gt_points
            data[tmp_data]["gt_density_maps"] = gt_density_maps
            data[tmp_data]["gt_boxes"] = gt_boxes

            data[tmp_data]['masks'] = []
            for img in images:
                width, height = img.size
                data[tmp_data]['masks'].append(np.ones((height, width), dtype=np.float32))

        if self.args.save_synthetic_dataset:
            self.save_synthetic_dataset(data)
            
        return data, deepcopy(self.base_image)
    
    def generate_one_image(self, base_image, totoal_pedestrians_num, probability_cumulative_array, scale_model):
        position_cache = np.zeros_like(self.base_image[:,:,0], dtype=np.int64)
        dot_cache = np.zeros_like(self.base_image[:,:,0], dtype=np.int64)
        count = 0
        ground_truth = []
        gt_img_boxes = []
        head_diameters = []
        pedestrians_keys = list(self.pedestrians.keys()) # maybe pesestrians_info has more pedestrians than pedestrians
        for i in range(totoal_pedestrians_num):
            head_point, box, index, scale_rate, head_diameter = self.select_position_pedestrian(pedestrians_keys,
                                                                      dot_cache,
                                                                      probability_cumulative_array,
                                                                      scale_model)
            pedestrain_info = self.pesestrians_info[str(index)]
            head_x = pedestrain_info["ann"]["x"]
            head_y = pedestrain_info["ann"]["y"]
            pedestrian = self.pedestrians[str(index)]
            base_image, position_cache, dot_cache = self.paste(base_image, pedestrian, head_point, head_x, head_y, 
                                                    position_cache, scale_rate, dot_cache)
            count += 1
            ground_truth.append(head_point)
            gt_img_boxes.append(box)
            head_diameters.append(head_diameter)


        ground_truth = np.array(ground_truth)
        gt_img_boxes = np.array(gt_img_boxes)
        if ground_truth.shape[0]==0:
            ground_truth = np.empty((0, 2))
            gt_img_boxes = np.empty((0, 5))
        else:
            gt_img_boxes[:,[1,3]] = gt_img_boxes[:,[1,3]]/base_image.shape[1]
            gt_img_boxes[:,[2,4]] = gt_img_boxes[:,[2,4]]/base_image.shape[0]

        position_cache = position_cache.astype(bool)
        position_cache = position_cache.astype(np.uint8)*255

        return base_image, ground_truth, gt_img_boxes, position_cache, head_diameters
    
    def select_position_pedestrian(self, pedestrians_keys, dot_cache, probability_cumulative_array, scale_model):
        base_height = self.base_image.shape[0]
        base_width = self.base_image.shape[1]
        while 1 :
            if isinstance(probability_cumulative_array, np.ndarray):
                u = np.random.rand()
                while u > probability_cumulative_array[-1]: # May be round-off errors
                    u = np.random.rand()
                pos_index = np.where(u <= probability_cumulative_array)[0][0]
                y = pos_index // base_width
                x = pos_index - y*base_width
            else:
                x = np.random.randint(1, base_width)
                y = np.random.randint(1, base_height)

            index = np.random.choice(pedestrians_keys, 1)[0]#update at 2023-1-16 since pedestrians images may less than pedestrians_pooling
            # index = np.random.randint(1, len(pedestrians_pooling) + 1)
            pedestrain_info = self.pesestrians_info[str(index)]
            pedestrian_height = pedestrain_info["height"]
            pedestrian_width = pedestrain_info["width"]
            head_x = pedestrain_info["ann"]["x"]
            head_y = pedestrain_info["ann"]["y"]


            if scale_model != None and scale_model.is_use():
                current_area = scale_model.predict(y)
            else:
                random_scale = random.choice(self.pre_def_scale)
                max_pedestrian_area = int(self.bg_area / random_scale)
                current_area = (y/base_height) * max_pedestrian_area * (2/3)
        
            if current_area < 0:
                continue

            rate = math.sqrt(current_area/(pedestrian_height*pedestrian_width))
            pedestrian_height = int(pedestrian_height*rate)
            pedestrian_width = int(pedestrian_width*rate)
            head_x = int(head_x*rate)
            head_y = int(head_y*rate)
            head_diameter = pedestrian_width/2

            x1  = x - head_x
            y1 = y - head_y
            
            x2 = x1 + pedestrian_width
            y2 = y1 + pedestrian_height
            
            if 0 < x1 < base_width and 0 < x2 < base_width and 0 < y1 < base_height and 0 < y2 < base_height \
                and pedestrian_height != 0 and pedestrian_width != 0 and self.roi_mask[y, x] == 1 \
                and dot_cache[y, x] == 0 and current_area >= 10:
                break
        assert rate != 0
        return [x, y], [0, x1+pedestrian_width/2, y1+pedestrian_height/2, pedestrian_width, pedestrian_height], index, rate, head_diameter
    
    def paste(self, base_image, pedestrian, head_point, head_x, head_y, position_cache, scale_rate, dot_cache):
        p_height = int(pedestrian.shape[0]*scale_rate)
        p_width = int(pedestrian.shape[1]*scale_rate)
        pedestrian =  cv2.resize(pedestrian, (p_width, p_height))
        head_x = int(head_x*scale_rate)
        head_y = int(head_y*scale_rate)
        x = head_point[0]
        y = head_point[1]
        x_left = x - head_x
        y_top = y - head_y
        
        mask = pedestrian.copy()
        mask = mask.astype(bool)
        figure_mask = mask.astype(np.int64)[:,:,0]
        mask = np.invert(mask)
        bk_mask = mask.astype(np.int64)[:,:,0]
        position_mask = np.ones_like(mask[:,:,0])*y
        local_position_cache = position_cache[y_top:y_top+p_height,x_left:x_left+p_width]
        indicate_mask = position_mask - local_position_cache
        indicate_mask[indicate_mask >= 0] = 0
        indicate_mask[indicate_mask < 0] = 1
        indicate_mask = indicate_mask*figure_mask
        indicate_mask = indicate_mask + bk_mask
        indicate_mask = indicate_mask.astype(np.int64)
        
        position_cache[y_top:y_top+p_height,x_left:x_left+p_width] *=  indicate_mask
        invert_indicate_mask = np.invert(indicate_mask.astype(bool))
        invert_indicate_mask = invert_indicate_mask.astype(np.int64)
        position_cache[y_top:y_top+p_height,x_left:x_left+p_width] += invert_indicate_mask*y
        font = cv2.FONT_HERSHEY_SIMPLEX
        dot_cache[y_top:y_top+int(p_width/2), x_left:x_left+p_width] += figure_mask[0:int(p_width/2), :]
        
        # cv2.circle(pedestrian, (head_x,head_y),2,[0,0,255],-1)
        indicate_mask = indicate_mask.reshape(indicate_mask.shape[0], indicate_mask.shape[1],1)
        three_channel_indicate_mask = np.concatenate((indicate_mask, indicate_mask, indicate_mask), axis=2)
        invert_indicate_mask = invert_indicate_mask.reshape(invert_indicate_mask.shape[0], invert_indicate_mask.shape[1], 1)
        three_channel_invert_indicate_mask = np.concatenate((invert_indicate_mask, invert_indicate_mask, invert_indicate_mask), axis=2)
        base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:]*three_channel_indicate_mask
        base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] = base_image[y_top:y_top+p_height,x_left:x_left+p_width,:] + pedestrian*three_channel_invert_indicate_mask
        # base_image = cv2.putText(base_image, str(count)+"("+str(x)+","+str(y)+")", (int(x),int(y)), font, 0.25, (0, 0, 255), 1)
        return base_image, position_cache, dot_cache

    def generate_density_map(self, points, images, sigma_method, images_head_diameters):
        gt_density_maps = []

        precomputed_kernels_path = os.path.join(self.args.save_dir, 'gaussian_kernels.pkl')
        if not os.path.exists(precomputed_kernels_path):
            generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=128, num_sigmas=129)
    
        with open(precomputed_kernels_path, 'rb') as f:
            kernels_dict = pickle.load(f)
            kernels_dict = SortedDict(kernels_dict)

        precomputed_distances_path = os.path.join(self.save_base_path, 'distances_dict.pkl')
        compute_distances(precomputed_distances_path, images, points)
        with open(precomputed_distances_path, 'rb') as f:
            distances_dict = pickle.load(f)
        
        for index, (img, point, img_head_diameters) in enumerate(zip(images, points, images_head_diameters)):
            width, height = img.size
            gt_points = get_gt_dots(point, height, width)
            distances = distances_dict[index]
            density_map = gaussian_filter_density(gt_points, height, width, distances, kernels_dict, img_head_diameters, min_sigma=2, method=sigma_method, const_sigma=15)
            
            gt_density_maps.append(density_map)
        return gt_density_maps

    def generate_gaussian_kernels(self, out_kernels_path='gaussian_kernels.pkl', round_decimals = 3, sigma_threshold = 4, sigma_min=0, sigma_max=20, num_sigmas=801):
        """
        Computing gaussian filter kernel for sigmas in linspace(sigma_min, sigma_max, num_sigmas) and saving 
        them to dict.     
        """
        kernels_dict = dict()
        sigma_space = np.linspace(sigma_min, sigma_max, num_sigmas)
        for sigma in tqdm(sigma_space):
            sigma = np.round(sigma, decimals=round_decimals) 
            kernel_size = np.ceil(sigma*sigma_threshold).astype(np.int)

            img_shape  = (kernel_size*2+1, kernel_size*2+1)
            img_center = (img_shape[0]//2, img_shape[1]//2)

            arr = np.zeros(img_shape)
            arr[img_center] = 1

            arr = gaussian_filter(arr, sigma, mode='constant') 
            kernel = arr / arr.sum()
            kernels_dict[sigma] = kernel
            
        print(f'Computed {len(sigma_space)} gaussian kernels. Saving them to {out_kernels_path}')

        with open(out_kernels_path, 'wb') as f:
            pickle.dump(kernels_dict, f)
        
    def compute_distances(self, out_dist_path, images, points, n_neighbors = 4, leafsize=1024):
        distances_dict = []
        for index, (img, point) in tqdm(enumerate(zip(images, points))):
            width, height = img.size
            non_zero_points = self.get_gt_dots(point, height, width)
            distances = []
            if non_zero_points.shape[0] != 0:
                tree = scipy.spatial.KDTree(non_zero_points.copy(), leafsize=leafsize)  # build kdtree
                distances, _ = tree.query(non_zero_points, k=n_neighbors)  # query kdtree

            distances_dict.append(distances)

        with open(out_dist_path, 'wb') as f:
            pickle.dump(distances_dict, f)
    

    def save_hybrid_dataset(self, hybrid_data, iter_num):
        save_path = os.path.join(self.args.save_dir, "synthetic_datasets", "hybrid_data_{}".format(iter_num))

        
        for key, value in hybrid_data.items():
            images = value['images']
            gt_boxes = value['gt_boxes']
            images_names = value['images_names']
            img_save_path = os.path.join(save_path, key, "images")
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            for index, (img, img_gt_boxes, img_name) in enumerate(zip(images, gt_boxes, images_names)):
                boxes = np.zeros((img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)

                # Create plot
                plt.figure()
                fig, ax = plt.subplots(1)
                img = np.array(img)
                ax.imshow(img)
                
                for x1, y1, x2, y2 in img_xyxy_gt_boxes:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=2, edgecolor='red', facecolor="none")
                    # Add the bbox to the plot
                    ax.add_patch(bbox)

                # Save generated image with detections
                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.savefig(os.path.join(img_save_path, img_name+'.jpg'), bbox_inches="tight", pad_inches=0.0)
                plt.close()


    def save_synthetic_dataset(self, data):
        for key, value  in data.items():
            images = value['images']
            gt_points = value['gt_points']
            gt_density_maps = value['gt_density_maps']
            save_images_path = os.path.join(self.save_base_path, key, 'images')
            if not os.path.exists(save_images_path):
                os.makedirs(save_images_path)

            save_gt_points_path = save_images_path.replace("images", "gt_points")
            if not os.path.exists(save_gt_points_path):
                os.makedirs(save_gt_points_path)
            
            save_gt_density_map_path = save_images_path.replace("images", "gt_density_maps")
            if not os.path.exists(save_gt_density_map_path):
                os.makedirs(save_gt_density_map_path)

            image_txt_pairs = []
            for i, img in enumerate(images):
                img = cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)
                gt_point = np.array(gt_points[i])
                gt_density_map = gt_density_maps[i]

                img_save_path = os.path.join(save_images_path, str(i+1)+".jpg")
                cv2.imwrite(img_save_path, img)

                gt_save_path = img_save_path.replace("images", "gt_points").replace("jpg", "txt")
                np.savetxt(gt_save_path, gt_point, fmt="%d")

                gt_density_map_save_path = img_save_path.replace("images", "gt_density_maps").replace("jpg", "npy")
                np.save(gt_density_map_save_path, gt_density_map)

                image_txt_pairs.append([img_save_path.replace(self.save_base_path + '/', ""),\
                                        gt_save_path.replace(self.save_base_path + '/', ""),
                                        gt_density_map_save_path.replace(self.save_base_path + '/', "")])

            image_txt_pairs = np.array(image_txt_pairs)
            np.savetxt(os.path.join(self.save_base_path, key + '.list'), image_txt_pairs, fmt="%s")
        
    def cp_real_test(self, real_dir):
        if 'cityuhk-x' in real_dir.lower():
            real_image_path = glob.glob(os.path.join(real_dir, 'test_data', 'images', self.args.scene))[0]
        else:
            real_image_path = glob.glob(os.path.join(real_dir, 'test_data', 'images'))[0]
        real_gt_path = real_image_path.replace('images', 'ground_truth_txt')
        real_images = glob.glob(os.path.join(real_image_path, "*.jpg"))
        real_images.extend(glob.glob(os.path.join(real_image_path, "*.png")))
        real_txts = glob.glob(os.path.join(real_gt_path, "*.txt"))
        if os.path.join(real_image_path, "scene.jpg") in real_images:
            real_images.remove(os.path.join(real_image_path, "scene.jpg"))
        if os.path.join(real_gt_path, "statistic.txt") in real_txts:
            real_txts.remove(os.path.join(real_gt_path, "statistic.txt"))
        assert len(real_images) == len(real_txts)
        
        tar_image_path = os.path.join(self.save_base_path, "real_test_data", "images")
        if not os.path.exists(tar_image_path):
            os.makedirs(tar_image_path)
        tar_gt_path = os.path.join(self.save_base_path, "real_test_data", "ground_truth_txt")
        if not os.path.exists(tar_gt_path):
            os.makedirs(tar_gt_path)
        
        real_images.sort()
        real_txts.sort()
        image_txt_pairs = []
        for i in range(len(real_images)):
            real_image = real_images[i]
            real_txt = real_txts[i]
            image_base_name = os.path.basename(real_image)
            txt_base_name = os.path.basename(real_txt)
            assert image_base_name.split('.')[0] == txt_base_name.split('.')[0]
            image_txt_pairs.append([os.path.join(tar_image_path.replace(self.save_base_path + '/',""), image_base_name), \
                os.path.join(tar_gt_path.replace(self.save_base_path + '/',""), txt_base_name)])
            
            shutil.copy(real_image, tar_image_path)
            shutil.copy(real_txt, tar_gt_path)
        image_txt_pairs = np.array(image_txt_pairs)
        np.savetxt(os.path.join(self.save_base_path, "real_test_data.list"), image_txt_pairs, fmt="%s")
    

    def generate_pseudo_detection_data(self, real_data):
        data = {}
        for key, value in real_data.items():
            images = deepcopy(value['images'])
            gt_boxes = deepcopy(value['gt_boxes'])
            data[key] = {}
            data[key]['images'] = []
            data[key]['gt_boxes'] = []
            for index, (img, img_gt_boxes) in enumerate(zip(images, gt_boxes)):#(class,x1,y1,w,h)
                img_gt_boxes[:,[1,3]] = img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                img_gt_boxes[:,[2,4]] = img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                img_gt_boxes = self.xywh2xyxy_np(img_gt_boxes[:,1:])
                img = np.array(img)
                base_img = deepcopy(self.base_image)
                for x1, y1, x2, y2 in img_gt_boxes:
                    base_img[round(y1):round(y2), round(x1):round(x2)] = img[round(y1):round(y2), round(x1):round(x2)]

                new_img = Image.fromarray(base_img)
                data[key]['images'].append(new_img)
            data[key]['gt_boxes'].extend(value['gt_boxes'])
        
        return data
            
    def xywh2xyxy_np(self, x):
        y = np.zeros_like(x)
        y[..., 0] = x[..., 0] - x[..., 2] / 2
        y[..., 1] = x[..., 1] - x[..., 3] / 2
        y[..., 2] = x[..., 0] + x[..., 2] / 2
        y[..., 3] = x[..., 1] + x[..., 3] / 2
        return y

    def compute_scale(self, real_data):
        scale_dict = {}
        images = deepcopy(real_data['train_data']['images'])
        gt_boxes = deepcopy(real_data['train_data']['gt_boxes'])
        for index, (img, img_gt_boxes) in enumerate(zip(images, gt_boxes)):
            img_gt_boxes[:,[1,3]] = img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
            img_gt_boxes[:,[2,4]] = img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
            img_gt_boxes = self.xywh2xyxy_np(img_gt_boxes[:,1:])
            for x1, y1, x2, y2 in img_gt_boxes:
                h = y2 - y1
                w = x2 - x1
                scale = w*h
                scale_dict[y1] = scale
            
        self.scale_dict = SortedDict(scale_dict)
    
    def merge_data(self, com_data, real_data=None):

        merged_data = {}
        for key, value in com_data.items():
            merged_data[key] = {}
            merged_data[key]['images'] = []
            merged_data[key]['gt_points'] = []
            merged_data[key]['masks'] = []
            merged_data[key]['gt_density_maps'] = []
            merged_data[key]['loss_weights'] = []

            com_images = value['images']
            com_gt_points = value['gt_points']
            com_gt_density_maps = value['gt_density_maps']

            merged_data[key]['images'].extend(com_images)
            merged_data[key]['gt_points'].extend(com_gt_points)
            merged_data[key]['gt_density_maps'].extend(com_gt_density_maps)
            merged_data[key]['loss_weights'].extend([1.]*len(com_images))

            if real_data != None:
                real_images = real_data[key]['images']
                real_gt_points = real_data[key]['gt_points']
                real_gt_density_maps = real_data[key]['gt_density_maps']

                merged_data[key]['images'].extend(real_images)
                merged_data[key]['gt_points'].extend(real_gt_points)
                merged_data[key]['gt_density_maps'].extend(real_gt_density_maps)
                merged_data[key]['loss_weights'].extend([self.args.loss_weight]*len(real_images))

            h = self.args.img_height // self.args.downsample_ratio
            w = self.args.img_width // self.args.downsample_ratio
            for i in range(len(com_images)):
                merged_data[key]['masks'].append(np.ones((h, w), dtype=np.float32))
            
            if real_data != None:
                merged_data[key]['masks'].extend(real_data[key]['masks'])
            
        return merged_data
    
    def enhance_data(self, real_data, iter_num):
        new_data = {}
        for key, value in real_data.items():
            images = value['images']
            gt_boxes = value['gt_boxes']
            images_names = value['images_names']

            new_images = []
            new_gt_boxes = []
            new_images_names = []

            new_data[key] = {}
            new_data[key]['images'] = []
            new_data[key]['gt_boxes'] = []
            new_data[key]['images_names'] = []

            for index, (real_img, real_img_gt_boxes, img_name) in enumerate(zip(images, gt_boxes, images_names)):
                boxes = np.zeros((real_img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = real_img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = real_img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                real_img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)

                base_img = deepcopy(self.base_image)
                real_img = np.array(real_img)
                for x1, y1, x2, y2 in real_img_xyxy_gt_boxes:
                    base_img[round(y1):round(y2), round(x1):round(x2)] = deepcopy(real_img[round(y1):round(y2), round(x1):round(x2)])
                
                image = Image.fromarray(base_img).convert("RGB")
                new_images.append(image)
                new_images_names.append("{}".format(img_name))
                new_gt_boxes.append(real_img_gt_boxes)
            
            new_data[key]['images'].extend(new_images)
            new_data[key]['gt_boxes'].extend(new_gt_boxes)
            new_data[key]['images_names'].extend(new_images_names)
        
        return new_data
    
    def enhance_data2(self, real_data, predicted_distribution, predicted_num, iter_num, scale_net):
        probability_cumulative_array = None
        if predicted_num != None:
            predicted_distribution = cv2.resize(predicted_distribution, (self.base_image.shape[1],
                                                                            self.base_image.shape[0]), cv2.INTER_CUBIC)
            map_sum = np.sum(predicted_distribution)
            probability_map = predicted_distribution / map_sum
            mean = np.mean(predicted_num)
            sigma = np.std(predicted_num)
            probability_array = probability_map.reshape(probability_map.shape[0]*probability_map.shape[1])
            probability_cumulative_array = np.cumsum(probability_array)

        new_data = {}

        for key, value in real_data.items():
            if key == 'train_data':
                totoal_picture_num = self.args.train_num
            else:
                totoal_picture_num = self.args.val_num
            images = value['images']
            gt_boxes = value['gt_boxes']
            images_names = value['images_names']

            new_images = []
            new_masks = []
            new_gt_boxes = []
            new_images_names = []

            new_data[key] = {}
            new_data[key]['images'] = []
            new_data[key]['gt_boxes'] = []
            new_data[key]['images_names'] = []

            picture_num = totoal_picture_num // len(images)



            for index, (real_img, real_img_gt_boxes, img_name) in enumerate(zip(images, gt_boxes, images_names)):
                boxes = np.zeros((real_img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = real_img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = real_img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                real_img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)

                base_img = deepcopy(self.base_image)
                gt_boxes_cache = np.zeros(self.base_image.shape[0:2])
                real_img = np.array(real_img)
                for x1, y1, x2, y2 in real_img_xyxy_gt_boxes:
                    base_img[round(y1):round(y2), round(x1):round(x2)] = real_img[round(y1):round(y2), round(x1):round(x2)]
                    gt_boxes_cache[round(y1):round(y2), round(x1):round(x2)] = 1
                
                for i in range(picture_num):
                    totoal_pedestrians_num  = round(np.clip(np.random.normal(mean, sigma, 1), 0, None)[0])
                    image, point, gt_img_boxes, mask, head_diameters = self.generate_one_image(deepcopy(base_img), 
                                                                                            totoal_pedestrians_num, 
                                                                                            predicted_num, 
                                                                                            probability_cumulative_array,
                                                                                            scale_net,
                                                                                            gt_boxes_cache)
                    image = Image.fromarray(image).convert("RGB")
                    mask = Image.fromarray(mask).convert("1")
                    new_images.append(image)
                    new_masks.append(mask)
                    new_images_names.append("{}_{}".format(img_name, str(i)))

                    gt_img_boxes = np.concatenate((real_img_gt_boxes, gt_img_boxes), 0)

                    new_gt_boxes.append(gt_img_boxes)

            if self.args.harmonization:
                new_images = self.har.harmonization(new_images, new_masks)
            
            new_data[key]['images'].extend(new_images)
            new_data[key]['gt_boxes'].extend(new_gt_boxes)
            new_data[key]['images_names'].extend(new_images_names)
        
        self.save_hybrid_dataset(new_data, iter_num)
        
        return new_data
    
    def merge_detection_data(self, com_data, real_data):

        merged_data = {}
        for key, value in com_data.items():
            merged_data[key] = {}
            merged_data[key]['images'] = []
            merged_data[key]['gt_boxes'] = []

            com_images = value['images']
            com_gt_boxes = value['gt_boxes']

            merged_data[key]['images'].extend(com_images)
            merged_data[key]['gt_boxes'].extend(com_gt_boxes)

            if real_data != None:
                real_images = real_data[key]['images']
                real_gt_boxes = real_data[key]['gt_boxes']

                merged_data[key]['images'].extend(real_images)
                merged_data[key]['gt_boxes'].extend(real_gt_boxes)
            
        return merged_data
    
    def filter_outlier(self, outlier_masks, real_data, iter_num):

        filtered_real_data = {}
        for stage, value in real_data.items():

            filtered_real_data[stage] = {}
            filtered_real_data[stage]["images"] = []
            filtered_real_data[stage]["gt_boxes"] = []
            filtered_real_data[stage]["images_names"] = []
            
            if stage in outlier_masks.keys():
                outlier_mask = outlier_masks[stage]
            else:
                outlier_mask = None
            images = deepcopy(value["images"])
            gt_boxes = deepcopy(value["gt_boxes"])
            images_names = deepcopy(value['images_names'])

            count = 0

            for index, (img, img_gt_boxes, img_name) in enumerate(zip(images, gt_boxes, images_names)):
                tmp_gt_boxes = []
                for img_gt_box in img_gt_boxes:
                    if not isinstance(outlier_mask, np.ndarray) or not outlier_mask[count]:
                        tmp_gt_boxes.append(img_gt_box.reshape(1, img_gt_box.shape[0]))
                    count += 1
                
                filtered_real_data[stage]["images"].append(img)
                filtered_real_data[stage]["images_names"].append(img_name)
                if len(tmp_gt_boxes) > 0:
                    filtered_real_data[stage]["gt_boxes"].append(np.concatenate(tmp_gt_boxes, 0))
                else:
                    filtered_real_data[stage]["gt_boxes"].append(np.empty((0, 5)))
        
        return filtered_real_data

    def compared_visual(self, detected_data, filtered_data, enhanced_data, iter_num):
        save_path = os.path.join(self.args.save_dir, "visual_dataset", "compared_dataset_{}".format(iter_num))

        for key, value in detected_data.items():
            detected_images = value['images']
            detected_gt_boxes = value['gt_boxes']
            images_names = value['images_names']

            filtered_images = filtered_data[key]['images']
            filtered_gt_boxes = filtered_data[key]['gt_boxes']

            enhanced_images = enhanced_data[key]['images']
            enhanced_gt_boxes = enhanced_data[key]['gt_boxes']
            
            img_save_path = os.path.join(save_path, key, "images")
            if not os.path.exists(img_save_path):
                os.makedirs(img_save_path)

            for index, (img, img_gt_boxes, img_name) in enumerate(zip(detected_images, detected_gt_boxes, images_names)):
                boxes = np.zeros((img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)

                filtered_img = filtered_images[index]
                filtered_img_gt_boxes = filtered_gt_boxes[index]
                boxes = np.zeros((filtered_img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = filtered_img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = filtered_img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                filtered_img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)

                enhanced_img = enhanced_images[index]
                enhanced_img_gt_boxes = enhanced_gt_boxes[index]
                boxes = np.zeros((enhanced_img_gt_boxes.shape[0], 4))
                boxes[:,[0,2]] = enhanced_img_gt_boxes[:,[1,3]]*self.base_image.shape[1]
                boxes[:,[1,3]] = enhanced_img_gt_boxes[:,[2,4]]*self.base_image.shape[0]
                enhanced_img_xyxy_gt_boxes = self.xywh2xyxy_np(boxes)


                # Create plot
                plt.figure()
                fig, ax = plt.subplots(1, 3)
                img = np.array(img)
                ax[0].imshow(img)
                
                for x1, y1, x2, y2 in img_xyxy_gt_boxes:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='red', facecolor="none")
                    # Add the bbox to the plot
                    ax[0].add_patch(bbox)

                # Save generated image with detections
                ax[0].axis("off")

                filtered_img = np.array(filtered_img)
                ax[1].imshow(filtered_img)
                
                for x1, y1, x2, y2 in filtered_img_xyxy_gt_boxes:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='red', facecolor="none")
                    # Add the bbox to the plot
                    ax[1].add_patch(bbox)

                # Save generated image with detections
                ax[1].axis("off")

                enhanced_img = np.array(enhanced_img)
                ax[2].imshow(enhanced_img)
                
                for x1, y1, x2, y2 in enhanced_img_xyxy_gt_boxes:

                    box_w = x2 - x1
                    box_h = y2 - y1

                    # Create a Rectangle patch
                    bbox = patches.Rectangle((x1, y1), box_w, box_h, linewidth=1, edgecolor='red', facecolor="none")
                    # Add the bbox to the plot
                    ax[2].add_patch(bbox)

                # Save generated image with detections
                ax[2].axis("off")

                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                plt.tight_layout() 

                plt.savefig(os.path.join(img_save_path, img_name+'.jpg'), bbox_inches="tight", pad_inches=0.0, dpi=400)
                plt.close()