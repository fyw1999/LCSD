import math
import sys
import random
import numpy as np
import torch
import torch.nn as nn
from scipy import spatial as ss
from torchvision import transforms
import sys
import scipy
from tqdm import tqdm
import pickle
from itertools import islice
from scipy.ndimage.filters import gaussian_filter 
sys.setrecursionlimit(100000) #  set the recursion depth
# Hungarian method for bipartite graph
def hungarian(matrixTF):
    # matrix to adjacent matrix
    edges = np.argwhere(matrixTF)
    lnum, rnum = matrixTF.shape
    graph = [[] for _ in range(lnum)]
    for edge in edges:
        graph[edge[0]].append(edge[1])

    # deep first search
    match = [-1 for _ in range(rnum)]
    vis = [-1 for _ in range(rnum)]
    def dfs(u):
        for v in graph[u]:
            if vis[v]: continue
            vis[v] = True
            if match[v] == -1 or dfs(match[v]):
                match[v] = u
                return True
        return False

    # for loop
    ans = 0
    for a in range(lnum):
        for i in range(rnum): vis[i] = False
        if dfs(a): ans += 1

    # assignment matrix
    assign = np.zeros((lnum, rnum), dtype=bool)
    for i, m in enumerate(match):
        if m >= 0:
            assign[m, i] = True

    return ans, assign

def filter_pseudo_point_with_local_std(density_map, std, uncertain_thre):
    prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
    std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

    maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0, 0]

    width = maxpool_output.shape[1]

    maxpool_std = nn.functional.max_pool2d(std, 5, 1, 2)[0, 0] * uncertain_thre
    std = std[0, 0]

    nonzero_points = maxpool_output.nonzero()
    nonzero_y = nonzero_points.T[0]
    nonzero_x = nonzero_points.T[1]
    nonzero_idx = nonzero_y*width + nonzero_x
    idx = nonzero_idx[std.view(-1)[nonzero_idx] <= maxpool_std.view(-1)[nonzero_idx]]
    y = torch.div(idx, width, rounding_mode='floor')
    x = idx - y*width

    points = torch.cat([x[:,None], y[:,None]], 1)

    return points

def filter_pseudo_point_with_global_std(density_map, std, uncertain_thre):
    prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
    std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

    maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0, 0]

    width = maxpool_output.shape[1]

    std = std[0, 0]
    std = std / (std.max() + 1e-6)

    nonzero_points = maxpool_output.nonzero()
    nonzero_y = nonzero_points.T[0]
    nonzero_x = nonzero_points.T[1]
    nonzero_idx = nonzero_y*width + nonzero_x
    idx = nonzero_idx[std.view(-1)[nonzero_idx] <= uncertain_thre]
    y = torch.div(idx, width, rounding_mode='floor')
    x = idx - y*width

    points = torch.cat([x[:,None], y[:,None]], 1)

    return points

def pseudo_point_with_std(density_map, std):
    prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
    std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

    maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0, 0]

    std = std[0, 0]
    std = std / (std.max() + 1e-6)

    nonzero_points = maxpool_output.nonzero()
    nonzero_y = nonzero_points.T[0]
    nonzero_x = nonzero_points.T[1]
    points = torch.cat([nonzero_x[:,None], nonzero_y[:,None]], 1)
    uncertainty_map = std
    
    return points, uncertainty_map

def gen_soft_pseudo_point_with_local_std(density_map, std, beta):
    prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
    std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

    maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0, 0]

    width = maxpool_output.shape[1]

    maxpool_std = nn.functional.max_pool2d(std, 5, 1, 2)[0, 0]
    minpool_std = -1 * (nn.functional.max_pool2d(std*(-1), 5, 1, 2)[0, 0])
    std = std[0, 0]

    nonzero_points = maxpool_output.nonzero()
    nonzero_y = nonzero_points.T[0]
    nonzero_x = nonzero_points.T[1]
    nonzero_idx = nonzero_y*width + nonzero_x

    target_value = torch.exp(-1*beta*(std.view(-1)[nonzero_idx] - minpool_std.view(-1)[nonzero_idx]))

    points = torch.cat([nonzero_x[:,None], nonzero_y[:,None]], 1)

    return points, target_value



def pseudo_point_mask(density_map, points, scale_model, downsample_ratio):

        mask = torch.zeros_like(density_map)

        height = density_map.shape[0]
        width = density_map.shape[1]

        for point in points:
            x = point[0].item()
            y = point[1].item()

            if scale_model != None and scale_model.is_use():
                area = scale_model.predict(y) / 2
                if area < 625:
                    area = 625
                r = math.sqrt(area)
                r = r / downsample_ratio
                pad = int(r // 2)
            else:
                pad = 1

            x //= downsample_ratio
            y //= downsample_ratio
            start_y = max(0, y - pad)
            end_y = min(height-1, y + pad + 1)

            start_x = max(0, x - pad)
            end_x = min(width-1, x + pad + 1)

            mask[start_y:end_y, start_x:end_x] = 1.
        
        return mask

def gen_pseudo_point(density_map):
    prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)

    maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
    maxpool_output = torch.eq(maxpool_output, prob_outputs)
    maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
    maxpool_output = maxpool_output[0, 0]
    
    nonzero_points = maxpool_output.nonzero()
    nonzero_y = nonzero_points.T[0]
    nonzero_x = nonzero_points.T[1]

    points = torch.cat([nonzero_x[:,None], nonzero_y[:,None]], 1)

    return points

def eval_loc_F1_point(pred_points, gt_points, max_dist_thresh = 100):
    def compute_metrics(dist_matrix, match_matrix, pred_num, gt_num, sigma):
        for i_pred_p in range(pred_num):
            pred_dist = dist_matrix[i_pred_p, :]
            match_matrix[i_pred_p, :] = pred_dist <= sigma

        tp, assign = hungarian(match_matrix)
        fn_gt_index = np.array(np.where(assign.sum(0) == 0))[0]
        tp_pred_index = np.array(np.where(assign.sum(1) == 1))[0]
        fp_pred_index = np.array(np.where(assign.sum(1) == 0))[0]

        tp = tp_pred_index.shape[0]
        fp = fp_pred_index.shape[0]
        fn = fn_gt_index.shape[0]

        return tp, fp, fn

    # the arrays for tp, fp, fn, precision, recall, and f1 only use the entries from 1 to max_dist_thresh. Do not use index 0.
    tp_class = np.zeros(max_dist_thresh )
    fp_class = np.zeros(max_dist_thresh )
    fn_class = np.zeros(max_dist_thresh )

    for dist_thresh in range(0, max_dist_thresh):

        tp, fp, fn = [0, 0, 0]

        if len(gt_points) == 0 and len(pred_points) != 0:
            fp_pred_index = np.array(range(pred_points.shape[0]))
            fp = fp_pred_index.shape[0]

        if len(gt_points) != 0 and len(pred_points) == 0:
            fn_gt_index = np.array(range(gt_points.shape[0]))
            fn = fn_gt_index.shape[0]

        if len(gt_points) != 0 and len(pred_points) != 0:
            dist_matrix = ss.distance_matrix(pred_points, gt_points, p=2)
            match_matrix = np.zeros(dist_matrix.shape, dtype=bool)
            tp, fp, fn  = compute_metrics(dist_matrix, match_matrix, pred_points.shape[0], gt_points.shape[0], dist_thresh+1)

        # false positive, fp,  remaining points in prediction that were not matched to any point in ground truth
        tp_class[dist_thresh] += tp
        fp_class[dist_thresh] += fp
        fn_class[dist_thresh] += fn

    return tp_class, fp_class, fn_class

def generate_gaussian_kernels(out_kernels_path='gaussian_kernels.pkl', round_decimals = 3, sigma_threshold = 4, sigma_min=0, sigma_max=20, num_sigmas=801):
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
        
def compute_distances(out_dist_path, images, points, n_neighbors = 4, leafsize=1024):
    distances_dict = []
    for index, (img, point) in tqdm(enumerate(zip(images, points))):
        width, height = img.size
        non_zero_points = get_gt_dots(point, height, width)
        distances = []
        if non_zero_points.shape[0] != 0:
            tree = scipy.spatial.KDTree(non_zero_points.copy(), leafsize=leafsize)  # build kdtree
            distances, _ = tree.query(non_zero_points, k=n_neighbors)  # query kdtree

        distances_dict.append(distances)
    
    with open(out_dist_path, 'wb') as f:
        pickle.dump(distances_dict, f)

def compute_distances_online(images, points, n_neighbors = 4, leafsize=1024):
    distances_dict = []
    for index, (img, point) in enumerate(tqdm(zip(images, points))):
        width, height = img.size
        non_zero_points = get_gt_dots(point, height, width)
        distances = []
        if non_zero_points.shape[0] != 0:
            tree = scipy.spatial.KDTree(non_zero_points.copy(), leafsize=leafsize)  # build kdtree
            distances, _ = tree.query(non_zero_points, k=n_neighbors)  # query kdtree

        distances_dict.append(distances)
    
    return distances_dict

def get_gt_dots(gt, img_height, img_width):
    """
    Load Matlab file with ground truth labels and save it to numpy array.
    ** cliping is needed to prevent going out of the array
    """
    gt[:,0] = gt[:,0].clip(0, img_width - 1)
    gt[:,1] = gt[:,1].clip(0, img_height - 1)
    return gt

def gaussian_filter_density(non_zero_points, map_h, map_w, distances=None, kernels_dict=None, img_head_diameters=None, min_sigma=2, method=1, const_sigma=15):
    """
    Fast gaussian filter implementation : using precomputed distances and kernels
    """
    gt_count = non_zero_points.shape[0]
    density_map = np.zeros((map_h, map_w), dtype=np.float32)
    if gt_count == 0:
        return density_map

    for i in range(gt_count):
        point_x, point_y = non_zero_points[i]
        point_x = int(point_x)
        point_y = int(point_y)
        try:
            sigma = compute_sigma(gt_count, point_y, distances[i], img_head_diameters[i] if img_head_diameters!= None else None , min_sigma=min_sigma, method=method, fixed_sigma=const_sigma)
        except Exception as re:
            print(re)
            print(i)
            print(gt_count)
            print(non_zero_points)
            print(distances)
            sys.exit(0)

        closest_sigma = find_closest_key(kernels_dict, sigma)
        kernel = kernels_dict[closest_sigma]
        full_kernel_size = kernel.shape[0]
        kernel_size = full_kernel_size // 2 

        min_img_x = max(0, point_x-kernel_size)
        min_img_y = max(0, point_y-kernel_size)
        max_img_x = min(point_x+kernel_size+1, map_w - 1)
        max_img_y = min(point_y+kernel_size+1, map_h - 1)

        kernel_x_min = kernel_size - point_x if point_x <= kernel_size else 0
        kernel_y_min = kernel_size - point_y if point_y <= kernel_size else 0
        kernel_x_max = kernel_x_min + max_img_x - min_img_x
        kernel_y_max = kernel_y_min + max_img_y - min_img_y

        density_map[min_img_y:max_img_y, min_img_x:max_img_x] += kernel[kernel_y_min:kernel_y_max, kernel_x_min:kernel_x_max]
    return density_map

def find_closest_key(sorted_dict, key):
    """
    Find closest key in sorted_dict to 'key'
    """
    keys = list(islice(sorted_dict.irange(minimum=key), 1))
    keys.extend(islice(sorted_dict.irange(maximum=key, reverse=True), 1))
    return min(keys, key=lambda k: abs(key - k))

def compute_sigma(gt_count, y_cor, distance=None, head_diameter=None, min_sigma=1, method=1, fixed_sigma=15):
    """
    Compute sigma for gaussian kernel with different methods :
    * method = 1 : sigma = (sum of distance to 3 nearest neighbors) / 10
    * method = 2 : sigma = distance to nearest neighbor
    * method = 3 : sigma = fixed value
    ** if sigma lower than threshold 'min_sigma', then 'min_sigma' will be used
    ** in case of one point on the image sigma = 'fixed_sigma'
    """    
    if method == 1:
        if gt_count != 1:
            dist = 0
            count = 0
            for j in range(1, len(distance)):
                if distance[j] != np.inf:
                    dist += distance[j]
                    count += 1
            sigma = (dist/count) * 0.1
        else:
            sigma = fixed_sigma
    elif method == 2:
        if gt_count != 1:
            sigma = distance[1]
        else:
            sigma = fixed_sigma
    elif method == 3:
        sigma = fixed_sigma
    elif method == 4:
        sigma = (13/511)*y_cor + 1009/511
    elif method == 5:
        sigma = head_diameter/3

    if sigma < min_sigma:
        sigma = min_sigma

    return sigma


class Strong_AUG(object):
    def __init__(self):
        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        self.color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        self.blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
    def __call__(self, img):
        if random.random() < 0.8:
            img = self.color_jitter(img)
        img = transforms.RandomGrayscale(p=0.2)(img)

        if random.random() < 0.5:
            img = self.blurring_image(img)
        
        return img

