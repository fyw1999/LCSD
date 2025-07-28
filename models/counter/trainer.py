import os
import cv2
import math
import random
import torch
import numpy as np
import torch.nn as nn
from torch import optim
from PIL import Image
from copy import deepcopy
from datasets.crowd import Crowd
from torchvision import transforms
from scipy import ndimage, spatial
from utils.pytorch_utils import AverageMeter, AverageCategoryMeter
import torchvision.transforms.functional as F
from models.counter.models import vgg19
from models.counter.geomloss import SamplesLoss
from torch.utils.data.dataloader import default_collate
from sortedcontainers import SortedDict
import pickle
from utils.utils import generate_gaussian_kernels, get_gt_dots, gaussian_filter_density, compute_distances_online
from utils.utils import pseudo_point_mask, gen_pseudo_point, eval_loc_F1_point, pseudo_point_with_std
from utils.utils import Strong_AUG
from collections import OrderedDict
import matplotlib
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
os.environ["CUDA_LAUNCH_BLOCKING"] = "1"
class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor
def grid(H, W, stride):
    coodx = torch.arange(0, W, step=stride) + stride / 2
    coody = torch.arange(0, H, step=stride) + stride / 2
    y, x = torch.meshgrid( [  coody / 1, coodx / 1 ] )
    return torch.stack( (x,y), dim=2 ).view(-1,2)

def per_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    s = (x_col[:,:,:,-1] + y_lin[:,:,:,-1]) / 2
    s = s * 0.2 + 0.5
    return (torch.exp(C/s) - 1)

def exp_cost(X, Y):
    x_col = X.unsqueeze(-2)
    y_lin = Y.unsqueeze(-3)
    C = torch.sum((torch.abs(x_col - y_lin)) ** 2, -1)
    C = torch.sqrt(C)
    return (torch.exp(C/scale) - 1.)

matplotlib.use('Agg') 
class CounterTrainer():
    def __init__(self, args, logger):
        # my_ot loss setting
        args.wtv = 0.01
        args.wot = 1
        args.wct = 0.01
        
        # gl setting
        global scale
        args.scale = 0.6
        scale = args.scale
        args.blur = 0.01
        args.scaling = 0.5
        args.cost = 'exp'
        if args.cost == 'exp':
            self.cost = exp_cost
        elif args.cost == 'per':
            self.cost = per_cost
        args.reach = 0.5
        args.p = 1
        args.tau = 0.1
        args.d_point = 'l1'
        args.d_pixel = 'l2'

        # fixed setting
        args.keep_rate = 0.99
        args.batch_sr = 0.9
        args.max_epochs = 50
        args.num_workers = 3
        args.lr = 1e-5
        args.weight_decay = 1e-4
        args.counter = "vgg"
        args.loss = "my_ot"

        self.logger = logger
        self.device = torch.device("cuda")
        self.device_count = torch.cuda.device_count()
        self.images_transforms = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.images_strong_transforms = transforms.Compose([
            Strong_AUG(),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        ])
        self.restore_transform = transforms.Compose([
            DeNormalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            transforms.ToPILImage()])

        if args.counter == 'csrnet':
            self.student_model = CSRNet(args)
            self.teacher_model = CSRNet(args)
        elif args.counter == 'vgg':
            self.student_model = vgg19(args)
            self.teacher_model = vgg19(args)
        elif args.counter == 'sfcn':
            self.student_model = SFCN()
            self.teacher_model = SFCN()
        else:
             raise Exception("invalid counter")

        self.student_model.to(self.device)
        self.teacher_model.to(self.device)
        self.teacher_model.eval()
        self.optimizer = optim.Adam(self.student_model.parameters(), lr=args.lr, 
                                    weight_decay=args.weight_decay)
        
        self.tv_loss = nn.L1Loss(reduction='none').to(self.device)
        self.mae = nn.L1Loss(reduction='none').to(self.device)
        self.criterion = torch.nn.MSELoss(reduction='none').to(self.device)
        self.gl_criterion = SamplesLoss(blur=args.blur, scaling=args.scaling, debias=False, backend='tensorized', cost=self.cost, reach=args.reach, p=args.p)
        self.my_ot = SamplesLoss("sinkhorn", blur=0.01, scaling=0.9)
        #when reduction='mean', the loss is divided by the number of all pixels in batch 

        self.real_stage = ["train", "test"]
        self.real_dataset = {x:Crowd(args.scene_dataset, x, args.downsample_ratio) for x in self.real_stage}
        
        self.real_dataloader = {x:torch.utils.data.DataLoader(self.real_dataset[x],
                                        collate_fn=default_collate,
                                        batch_size=1,
                                        shuffle=False,
                                        num_workers=args.num_workers * self.device_count,
                                        pin_memory=True)
                           for x in self.real_stage}
        
        self.args = args
        self.real_train_images, self.real_test_images = self.read_real_images()
        self.reset_metirc()
        self.global_epoch = 0
        
    
    def read_real_images(self):
        train_list = os.path.join(self.args.scene_dataset, "train_data.list")
        test_list = os.path.join(self.args.scene_dataset, "test_data.list")

        with open(train_list,'r') as f:
            train_file_list = f.read().split('\n')
        if '' in train_file_list:
            train_file_list.remove('')

        with open(test_list,'r') as f:
            test_file_list = f.read().split('\n')
        if '' in test_file_list:
            test_file_list.remove('')

        real_train_images = []
        real_test_images = []
        for pair_path in train_file_list:
            img_path = os.path.join(self.args.scene_dataset, pair_path.split()[0])
            base_name = os.path.basename(img_path)
            name = base_name.split('.')[0]
            real_train_images.append({"image":Image.open(img_path).convert('RGB'),"image_name":name})
        
        for pair_path in test_file_list:
            img_path = os.path.join(self.args.scene_dataset, pair_path.split()[0])
            base_name = os.path.basename(img_path)
            name = base_name.split('.')[0]
            real_test_images.append({"image":Image.open(img_path).convert('RGB'),"image_name":name})
        
        return real_train_images, real_test_images

    def train_epoch(self, data, scale_model, iter_num):
        epoch_loss = AverageMeter()
        epoch_mae = AverageMeter()
        epoch_mse = AverageMeter()
        self.student_model.train()

        if iter_num >= self.args.start_iter:
            com_batch_size = round(self.args.batch_size * self.args.batch_sr)
        else:
            com_batch_size = self.args.batch_size

        start = 0
        end = com_batch_size
        images = deepcopy(data['images'])
        gt_points = deepcopy(data['gt_points'])
        gt_density_maps = deepcopy(data['gt_density_maps'])
        masks = deepcopy(data['masks'])

        shuffle_index = np.random.permutation(np.arange(len(images)))
        images = [images[ind] for ind in shuffle_index]
        gt_points = [gt_points[ind] for ind in shuffle_index]
        gt_density_maps = [gt_density_maps[ind] for ind in shuffle_index]
        masks = [masks[ind] for ind in shuffle_index]
            
        while end <= len(images) and start != end:
            inputs, points, gt_discrete, density_maps, batch_masks, gt_counts, un_maps = self.batch_data(images, gt_points, gt_density_maps, masks, start, end)

            if iter_num >= self.args.start_iter:    
                real_inputs, real_points, real_gt_discrete, real_density_maps, real_batch_masks,\
                      real_gt_counts, real_un_maps = self.batch_real_data(scale_model, iter_num)

                inputs, points, gt_discrete, density_maps, batch_masks, gt_counts, un_maps = \
                self.merge(inputs, points, gt_discrete, density_maps, batch_masks, gt_counts, un_maps,
                        real_inputs, real_points, real_gt_discrete, real_density_maps, real_batch_masks, real_gt_counts, real_un_maps)
            
            inputs = inputs.to(self.device)
            gd_count = np.array(gt_counts, dtype=np.float32)
            points = [p.to(self.device) for p in points]
            un_maps = [u.to(self.device) for u in un_maps]
            gt_discrete = gt_discrete.to(self.device)
            density_maps = density_maps.to(self.device)
            batch_masks = batch_masks.to(self.device)
            N = inputs.size(0)
            shape = (inputs.shape[0], int(inputs.shape[2]/self.args.downsample_ratio), int(inputs.shape[3]/self.args.downsample_ratio))

            with torch.set_grad_enabled(True):
                outputs, outputs_normed = self.student_model(inputs)
                cood_grid = grid(outputs.shape[2], outputs.shape[3], 1).unsqueeze(0) * self.args.downsample_ratio + (self.args.downsample_ratio / 2)
                cood_grid = cood_grid.type(torch.cuda.FloatTensor) / float(self.args.crop_size)
                i = 0
                emd_loss = torch.zeros(shape[0]).to(self.device)
                count_loss_weight = torch.zeros(shape[0]).to(self.device)
                tv_loss_weight = torch.ones(outputs.shape).to(self.device)
                for p in points:
                    if len(p) < 1:
                        gt = torch.zeros((1, shape[1], shape[2])).to(self.device)
                        emd_loss[i] = torch.abs(gt.sum() - outputs_normed[i].sum()) / shape[0]
                    else:
                        uncertainty_map = deepcopy(un_maps[i])
                        p_y = p[:,1]
                        p_x = p[:,0]
                        width = inputs.shape[3]
                        p_idx = (p_y*width + p_x).long()
                        point_uncertainty = uncertainty_map.view(-1)[p_idx]
                        point_weight = torch.exp(-1*self.args.beta*(point_uncertainty)).reshape(1, -1, 1)
                        gt = (torch.ones((1, len(p), 1))/len(p)).to(self.device)
                        cood_points = p.reshape(1, -1, 2) / float(self.args.crop_size) 
                        A = outputs_normed[i].reshape(1, -1, 1)
                        l, F, G = self.my_ot(A, cood_grid, gt, cood_points)
                        C = self.cost(cood_grid, cood_points)
                        PI = torch.exp((F.repeat(1,1,C.shape[2])+G.permute(0,2,1).repeat(1,C.shape[1],1)-C).detach()/0.01**2)*A*gt.permute(0,2,1)
                        PI_SUM = PI.sum(2).reshape(1,-1,1).repeat(1,1,len(p)) + 1e-6
                        
                        pixel_weight = torch.matmul(PI/PI_SUM, point_weight)
                        pixel_weight[torch.isnan(pixel_weight)] = 1.
                        pixel_weight[pixel_weight == 0] = 1.

                        emd_loss[i] = torch.sum(F*A*pixel_weight) + torch.sum(G*gt*point_weight)
                        count_loss_weight[i] = torch.sum(point_weight)
                        down_h = inputs.shape[2] // self.args.downsample_ratio
                        down_w = inputs.shape[3] // self.args.downsample_ratio
                        if i >= com_batch_size: #real data
                            point_weight_map = self.gen_weight_map(inputs.shape[2], inputs.shape[3], p.cpu().numpy(), point_weight.view(-1).cpu().numpy())
                            point_weight_map = point_weight_map.reshape([down_h, self.args.downsample_ratio, down_w, self.args.downsample_ratio]).sum(axis=(1, 3))
                            point_weight_map[point_weight_map == 0] = 1
                        else: #syn data
                            point_weight_map = np.ones([down_h, down_w], dtype=np.float32)
                        point_weight_map = np.expand_dims(point_weight_map, 0)
                        tv_loss_weight[i] = torch.from_numpy(point_weight_map)

                    i += 1

                # Compute counting loss.
                gd_count_tensor = torch.from_numpy(gd_count).float().to(self.device)
                count_loss = ((count_loss_weight + 1e-6) / (gd_count_tensor + 1e-6)) * self.mae((outputs * batch_masks).sum(1).sum(1).sum(1), gd_count_tensor) * self.args.wct
                # Compute TV loss.
                gt_discrete_normed = gt_discrete / (gd_count_tensor.unsqueeze(1).unsqueeze(2).unsqueeze(3) + 1e-6)
                tv_loss = (self.tv_loss(outputs_normed, gt_discrete_normed) * tv_loss_weight * batch_masks).sum(1).sum(1).sum(
                        1) * count_loss_weight * self.args.wtv
                ot_loss = emd_loss * self.args.wot
                loss = ot_loss + count_loss + tv_loss
                if iter_num >= self.args.start_iter:
                    loss = torch.mean(loss[:com_batch_size]) + torch.mean(loss[com_batch_size:]) * self.args.loss_weight
                else: 
                    loss = torch.mean(loss[:com_batch_size])

                self.optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(self.student_model.parameters(), 1)
                self.optimizer.step()

                ##############EMA update teacher
                if iter_num >= self.args.start_iter: 
                    student_model_dict = self.student_model.state_dict()
                    new_teacher_dict = OrderedDict()
                    for key, value in self.teacher_model.state_dict().items():
                        if key in student_model_dict.keys():
                            new_teacher_dict[key] = (
                                student_model_dict[key] *
                                (1 - self.args.keep_rate) + value * self.args.keep_rate
                            )
                        else:
                            raise Exception("{} is not found in student model".format(key))
                    self.teacher_model.load_state_dict(new_teacher_dict)
                else:
                    student_model_dict = self.student_model.state_dict()
                    self.teacher_model.load_state_dict(student_model_dict)
                ##############

                pred_count = torch.sum(outputs.view(N, -1), dim=1).detach().cpu().numpy()
                pred_err = pred_count - gd_count
                epoch_loss.update(loss.item(), N)
                epoch_mse.update(np.mean(pred_err * pred_err), N)
                epoch_mae.update(np.mean(abs(pred_err)), N)
            start = end
            end = min(end + com_batch_size, len(images))
    
    def val_epoch(self, data):
        self.model.eval()  # Set model to evaluate mode
        model_state_dic = self.model.state_dict()
        epoch_res = []
        images = deepcopy(data['images'])
        gt_points = deepcopy(data['gt_points'].copy())
        for index, (img, gt_point) in enumerate(zip(images, gt_points)):
            input = self.images_transforms(img).unsqueeze(0)
            input = input.repeat(4,1,1,1)
            input = input.to(self.device)
            with torch.set_grad_enabled(False):
                output, _ = self.model(input)
            output = torch.mean(output.squeeze(1), 0)
            res = len(gt_point) - torch.sum(output).item()
            epoch_res.append(res)
        epoch_res = np.array(epoch_res)
        mse = np.sqrt(np.mean(np.square(epoch_res)))
        mae = np.mean(np.abs(epoch_res))
    
        if (2.0 * mse + mae) < (2.0 * self.best_mse + self.best_mae):
            self.best_mse = mse
            self.best_mae = mae
            self.logger.info('-'*5 + "best mse {:.2f} mae {:.2f} model epoch {}".format(self.best_mse,
                                                                                        self.best_mae,
                                                                                        self.epoch))
            self.best_state_dict = deepcopy(model_state_dic)

    def load_best(self):
        if self.best_state_dict != None:
            self.student_model.load_state_dict(self.best_state_dict)
    
    def save_model(self, iter_num):
        model_state_dict = self.student_model.state_dict()
        model_save_path = os.path.join(self.args.save_dir, 'models')
        if not os.path.exists(model_save_path):
            os.makedirs(model_save_path)
        torch.save(model_state_dict, os.path.join(model_save_path, 'counter_model_{}.pth'.format(iter_num)))
    
    def reset_metirc(self):
        self.best_mae = np.inf
        self.best_mse = np.inf
        self.best_state_dict = None

    def save_dis(self, pre_dis, iter_num, base_image=None, name=None, recon=False):
        if name == None:
            img_name = "visual_dis"
            if recon:
                img_name = img_name + '_recon'
        else:
            img_name = name
        save_path = os.path.join(self.args.save_dir, "pre_dis", "iter_{}".format(iter_num))
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        vis_img = (pre_dis - pre_dis.min()) / (pre_dis.max() - pre_dis.min() + 1e-5)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        
        if name == None:
            vis_img = cv2.resize(vis_img, (base_image.shape[1], base_image.shape[0]), interpolation=cv2.INTER_LINEAR)
            cv2.imwrite(os.path.join(save_path, img_name + '_no_bg' +'.jpg'), vis_img)
            vis_img = cv2.addWeighted(cv2.cvtColor(base_image, cv2.COLOR_RGB2BGR), 0.4, vis_img, 0.6, 0)
        cv2.imwrite(os.path.join(save_path, img_name+'.jpg'), vis_img)
    
    def save_num(self, pre_num, gt_num, iter_num):
        np_pre_num = np.array(pre_num).reshape(len(pre_num),1)
        np_gt_num = np.array(gt_num).reshape(len(gt_num),1)
        con_gt_pre = np.concatenate([np_pre_num,np_gt_num], 1)
        pre_num_save_path = os.path.join(self.args.save_dir, "pre_num", "iter_{}".format(iter_num))

        if not os.path.exists(pre_num_save_path):
            os.makedirs(pre_num_save_path)
        np.savetxt(os.path.join(pre_num_save_path, "pre_gt_num.txt"), con_gt_pre, fmt="%.2f")

    def predict(self, iter_num, base_image):
        self.student_model.eval()
        loc_100_metrics = {} 

        image_errs = {}
        gt_num = {}
        pre_num = {}
        h = self.args.img_height // self.args.downsample_ratio
        w = self.args.img_width // self.args.downsample_ratio

        images = {}
        images_names = {}
        pseudo_points = {}
        pseudo_density_maps = {}

        for stage in self.real_stage:
            max_dist_thresh = 100
            loc_100_metrics[stage] = \
            {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}

            pre_num[stage] = []
            gt_num[stage] = []
            image_errs[stage] = []
            images[stage] = []
            images_names[stage] = []
            pseudo_points[stage] = []
            pseudo_density_maps[stage] = []
            
            for img, img_tensor, points, name in self.real_dataloader[stage]:
                input = img_tensor.to(self.device)
                count = len(points[0])
                input = input.repeat(4,1,1,1)
                with torch.set_grad_enabled(False):
                    multi_output, _ = self.student_model(input)
                output = torch.mean(multi_output.squeeze(1), 0)

                pseudo_point = None
                pseudo_point = gen_pseudo_point(output)
                
                img_err = count - len(pseudo_point)

                tp_100, fp_100, fn_100 = eval_loc_F1_point(pseudo_point.cpu().numpy(), points[0].cpu().numpy(), max_dist_thresh = 100)
                loc_100_metrics[stage]['tp_100'].update(tp_100)
                loc_100_metrics[stage]['fp_100'].update(fp_100)
                loc_100_metrics[stage]['fn_100'].update(fn_100)

                pre_num[stage].append(len(pseudo_point))
                gt_num[stage].append(count)
                image_errs[stage].append(img_err)

                self.save_dis(output.cpu().numpy(), iter_num, name=stage + name[0])

                img = Image.fromarray(img.squeeze(0).numpy()).convert("RGB")
                images[stage].append(img)
                images_names[stage].append(name[0])
                pseudo_points[stage].append(pseudo_point.cpu().numpy())
                pseudo_density_maps[stage].append(output.cpu().numpy())

        train_image_errs = np.array(image_errs['train'])
        test_image_errs = np.array(image_errs['test'])
        train_mse = np.round(np.sqrt(np.mean(np.square(train_image_errs))), 2)
        train_mae = np.round(np.mean(np.abs(train_image_errs)), 2)
        test_mse = np.round(np.sqrt(np.mean(np.square(test_image_errs))), 2)
        test_mae = np.round(np.mean(np.abs(test_image_errs)), 2)
        self.logger.info('-'*5 + "real train val mae:{}, mse:{}".format(train_mae, train_mse))
        self.logger.info('-'*5 + "real test val mae:{}, mse:{}".format(test_mae, test_mse))

        train_pre_100 = loc_100_metrics['train']['tp_100'].sum / (loc_100_metrics['train']['tp_100'].sum  + loc_100_metrics['train']['fp_100'].sum + 1e-20)
        train_rec_100 = loc_100_metrics['train']['tp_100'].sum / (loc_100_metrics['train']['tp_100'].sum  + loc_100_metrics['train']['fn_100'].sum + 1e-20) # True pos rate
        train_f1_100 = 2 * (train_pre_100 * train_rec_100) / (train_pre_100 + train_rec_100+ 1e-20)

        test_pre_100 = loc_100_metrics['test']['tp_100'].sum / (loc_100_metrics['test']['tp_100'].sum  + loc_100_metrics['test']['fp_100'].sum + 1e-20)
        test_rec_100 = loc_100_metrics['test']['tp_100'].sum / (loc_100_metrics['test']['tp_100'].sum  + loc_100_metrics['test']['fn_100'].sum + 1e-20) # True pos rate
        test_f1_100 = 2 * (test_pre_100 * test_rec_100) / (test_pre_100 + test_rec_100+ 1e-20)

        self.logger.info('-'*5 + "real train val precision:{}, recall:{}, f1-measure:{}".format(train_pre_100.mean().round(2), train_rec_100.mean().round(2), train_f1_100.mean().round(2)))
        self.logger.info('-'*5 + "real test val precision:{}, recall:{}, f1-measure:{}".format(test_pre_100.mean().round(2), test_rec_100.mean().round(2), test_f1_100.mean().round(2)))
        
        # self.save_num(pre_num, gt_num, iter_num)
        
        # self.save_dis(total_dis, iter_num, name="total_dis")

        self.test_visual({"images":images,
                          "images_names":images_names,
                          "gt_points":pseudo_points}, "point_show", iter_num)

        precomputed_kernels_path = os.path.join(self.args.save_dir, 'gaussian_kernels.pkl')
        if not os.path.exists(precomputed_kernels_path):
            generate_gaussian_kernels(precomputed_kernels_path, round_decimals=3, sigma_threshold=4, sigma_min=0, sigma_max=128, num_sigmas=129)

        with open(precomputed_kernels_path, 'rb') as f:
            kernels_dict = pickle.load(f)
            kernels_dict = SortedDict(kernels_dict)

        distances_dict = compute_distances_online(images['train'], pseudo_points['train'])

        base_image = cv2.resize(base_image, (images['train'][0].size[0], images['train'][0].size[1]), cv2.INTER_CUBIC)
        dis = np.zeros((base_image.shape[0], base_image.shape[1]))
        for index, (img, point) in enumerate(zip(images['train'], pseudo_points['train'])):
            width, height = img.size
            gt_point = get_gt_dots(point, height, width)
            distances = distances_dict[index]
            density_map = gaussian_filter_density(gt_point, height, width, distances, kernels_dict, min_sigma=2, method=3, const_sigma=32)
            dis += density_map
        self.save_dis(dis, iter_num, base_image=base_image)
        return dis, pre_num['train']
        
    def visual_std_and_error(self, uncertain_stds, learned_error_maps, images_name, iter_num):
        save_path = os.path.join(self.args.save_dir, "visual_std_and_error", "iter_{}".format(iter_num))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        for index, (std, name) in enumerate(zip(uncertain_stds, images_name)):
            vis_img = (std - std.min()) / (std.max() - std.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, "{}_std".format(name)+'.jpg'), vis_img)

        for index, (error_map, name) in enumerate(zip(learned_error_maps, images_name)):
            vis_img = (error_map - error_map.min()) / (error_map.max() - error_map.min() + 1e-5)
            vis_img = (vis_img * 255).astype(np.uint8)
            vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
            cv2.imwrite(os.path.join(save_path, "{}_error_map".format(name)+'.jpg'), vis_img)


    def ada_blur_dis(self, dis):
        density = np.zeros(dis.shape, dtype=np.float32)
        gt_count = np.count_nonzero(dis)
        if gt_count == 0:
            return density
        
        if gt_count > 3:
            query_num = 4
        else:
            query_num = gt_count
        
        pts = np.array(list(zip(np.nonzero(dis)[1], np.nonzero(dis)[0])))
        leafsize = 2048
        # build kdtree
        tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=query_num)

        for i, pt in enumerate(pts):            
            dist = 0
            distance = distances[i]
            if gt_count != 1:
                for j in range(1, len(distance)):
                    if distance[j] != np.inf:
                        dist += distance[j]
                dist = dist / (query_num-1)
                sigma = dist
            else:
                sigma = np.average(np.array(dis.shape))/2./2. #case: 1 point
            pt2d = np.zeros(dis.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = dis[pt[1],pt[0]]
            density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return density
    
    def reconstruct_density_map(self, gt):
        density = np.zeros(gt.shape, dtype=np.float32)
        gt_count = np.count_nonzero(gt)
        if gt_count == 0:
            return density

        if gt_count > 3:
            query_num = 4
        else:
            query_num = gt_count
        pts = np.array(list(zip(np.nonzero(gt)[1], np.nonzero(gt)[0])))
        leafsize = 2048
        # build kdtree
        tree = spatial.KDTree(pts.copy(), leafsize=leafsize)
        # query kdtree
        distances, locations = tree.query(pts, k=query_num)

        for i, pt in enumerate(pts):            
            dist = 0
            distance = distances[i]
            if gt_count != 1:
                for j in range(1, len(distance)):
                    if distance[j] != np.inf:
                        dist += distance[j]
                dist = dist / (query_num-1)
                sigma = 10 / dist
            else:
                sigma = np.average(np.array(gt.shape))/2./2. #case: 1 point
            pt2d = np.zeros(gt.shape, dtype=np.float32)
            pt2d[pt[1],pt[0]] = sigma / 10
            density += ndimage.filters.gaussian_filter(pt2d, sigma, mode='constant')
        return density
    
    def train(self, data, scale_model, iter_num):
        args = self.args
        for epoch in range(1, args.max_epochs+1):
            self.epoch = epoch
            self.global_epoch += 1
            self.train_epoch(data['train_data'], scale_model, iter_num)
    
    def merge(self, inputs, points, gt_discrete, density_maps, batch_masks, gt_counts, un_maps,
                       real_inputs, real_points, real_gt_discrete, real_density_maps, real_batch_masks, real_gt_counts, real_un_maps):
        
        try:
            return torch.cat([inputs, real_inputs], 0), points + real_points, torch.cat([gt_discrete, real_gt_discrete], 0), \
                torch.cat([density_maps, real_density_maps], 0), torch.cat([batch_masks, real_batch_masks], 0), gt_counts + real_gt_counts, un_maps + real_un_maps
        except Exception:
            raise("cat error!")
    
    def _crop(self, image):
        wd, ht = image.size
        st_size = 1.0 * min(wd, ht)
        assert st_size >= self.args.crop_size, print(wd, ht)
        i, j, h, w = self.random_crop(ht, wd, self.args.crop_size, self.args.crop_size)
        image = F.crop(image, i, j, h, w)
        return image

    def _flip(self, image):
        if random.random() > 0.5:
            image = F.hflip(image)
        return image
    
    def data_aug(self, images):
        kernel_size = int(random.random() * 4.95)
        kernel_size = kernel_size + 1 if kernel_size % 2 == 0 else kernel_size
        blurring_image = transforms.GaussianBlur(kernel_size, sigma=(0.1, 2.0))
        color_jitter = transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=0.25)
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]
        normalize = transforms.Normalize(mean, std)
        to_tensor = transforms.ToTensor()

        weak_aug = normalize(to_tensor(images))

        strong_aug = images
        if random.random() < 0.8:
            strong_aug = color_jitter(strong_aug)
        strong_aug = transforms.RandomGrayscale(p=0.2)(strong_aug)

        if random.random() < 0.5:
            strong_aug = blurring_image(strong_aug)

        strong_aug = normalize(to_tensor(strong_aug))
        return weak_aug, strong_aug

    def augmentation(self, image):
        image = self._crop(image)
        image = self._flip(image)
        image_wk, image_str = self.data_aug(image)
        return image_wk, image_str

    def batch_real_data(self, scale_model, iter_num):
        tensor_batch_images = []
        tensor_batch_gt_points = []
        tensor_batch_gt_discrete = []
        tensor_batch_gt_density_maps = []
        tensor_batch_masks = []

        tensor_batch_gt_counts = []
        tensor_batch_un_maps = []

        batch_images = []
        batch_images_names = []
        batch_gt_points = []
        batch_gt_density_maps = []
        batch_masks = []
        batch_un_maps = []

        real_batch_size = round(self.args.batch_size * (1-self.args.batch_sr))
        selected_images = random.sample(self.real_train_images, real_batch_size)
        for item in selected_images:
            image = item["image"]
            image_name = item["image_name"]
            
            img_height = image.size[1]
            img_width = image.size[0]
            if image.size[1] % self.args.downsample_ratio:
                img_height = round(image.size[1] / self.args.downsample_ratio) * self.args.downsample_ratio

            if image.size[0] % self.args.downsample_ratio:
                img_width = round(image.size[0] / self.args.downsample_ratio) * self.args.downsample_ratio
            
            image = image.resize((img_width, img_height), Image.ANTIALIAS)
            input = self.images_transforms(image).to(self.device).unsqueeze(0)
            input = input.repeat(4,1,1,1)
            with torch.set_grad_enabled(False):
                multi_output, _ = self.teacher_model(input)
                output = torch.mean(multi_output.squeeze(1), 0)
                std = torch.std(multi_output.squeeze(1), 0)

            
            pseudo_point = None
            if self.args.loss == 'my_ot':
                pseudo_point, un_map = pseudo_point_with_std(output, std)
                    
            elif self.args.loss == 'ablation':
                # pseudo_point = torch.round(den2seq(output, scale_factor=8, max_itern=16, ot_scaling=0.75)).long()
                pseudo_point = gen_pseudo_point(output)
                un_map = torch.zeros((input.shape[2], input.shape[3]))
            
            mask = pseudo_point_mask(output, pseudo_point, scale_model, self.args.downsample_ratio)

            batch_images.append(image)
            batch_images_names.append(image_name)
            batch_gt_points.append(pseudo_point.cpu().numpy())
            batch_gt_density_maps.append(output.cpu().numpy())
            batch_masks.append(mask.cpu().numpy())
            batch_un_maps.append(un_map.cpu().numpy())

        self.draw_images_and_pseudo_points({"images":batch_images, 
                                             "masks":batch_masks,
                                             "images_names":batch_images_names, 
                                             "gt_points":batch_gt_points}, "teacher_point_show", iter_num)

        for index, (img, point, gt_density_map) in enumerate(zip(batch_images, batch_gt_points, batch_gt_density_maps)):
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            mask = batch_masks[index]
            uncertainty = batch_un_maps[index]

            assert st_size >= self.args.crop_size, print(wd, ht)
            assert len(point) >= 0
            i, j, h, w = self.random_crop(ht, wd, self.args.crop_size, self.args.crop_size)
            img = F.crop(img, i, j, h, w)
            uncertainty = uncertainty[i:i+h,j:j+w]

            if len(point) > 0:
                point = point - [j, i]
                idx_mask = (point[:, 0] > 0) * (point[:, 0] < w) * \
                        (point[:, 1] > 0) * (point[:, 1] < h)
                point = point[idx_mask]
                tensor_batch_gt_counts.append(len(point))
            else:
                point = np.empty([0, 2])
                tensor_batch_gt_counts.append(0)
            gt_discrete = self.gen_discrete_map(h, w, point)
            down_h = h // self.args.downsample_ratio
            down_w = w // self.args.downsample_ratio
            gt_discrete = gt_discrete.reshape([down_h, self.args.downsample_ratio, down_w, self.args.downsample_ratio]).sum(axis=(1, 3))
            
            i = i // self.args.downsample_ratio
            j = j // self.args.downsample_ratio
            h = h // self.args.downsample_ratio
            w = w // self.args.downsample_ratio
            gt_density_map = gt_density_map[i:i+h,j:j+w]
            mask = mask[i:i+h,j:j+w]

            # assert np.sum(gt_discrete) == len(point)
            
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                gt_density_map = np.fliplr(gt_density_map)
                mask = np.fliplr(mask)
                uncertainty = np.fliplr(uncertainty)
                if len(point) > 0:
                    point[:, 0] = self.args.crop_size - point[:, 0]
        
            gt_discrete = np.expand_dims(gt_discrete, 0)
            tensor_batch_images.append(self.images_strong_transforms(img).unsqueeze(0))
            tensor_batch_gt_points.append(torch.from_numpy(point).float())
            tensor_batch_gt_discrete.append(torch.from_numpy(gt_discrete.copy()).unsqueeze(0))
            tensor_batch_gt_density_maps.append(torch.from_numpy(gt_density_map.copy()).unsqueeze(0).unsqueeze(0))
            tensor_batch_masks.append(torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0))

            tensor_batch_un_maps.append(torch.from_numpy(uncertainty.copy()))

        try:
            tensor_batch_images = torch.cat(tensor_batch_images)
            tensor_batch_gt_discrete = torch.cat(tensor_batch_gt_discrete).float()
            tensor_batch_gt_density_maps = torch.cat(tensor_batch_gt_density_maps)
            tensor_batch_masks = torch.cat(tensor_batch_masks)
        except Exception:
            raise("tensor cat error!")
        
        return tensor_batch_images, tensor_batch_gt_points, tensor_batch_gt_discrete, tensor_batch_gt_density_maps, tensor_batch_masks, tensor_batch_gt_counts, tensor_batch_un_maps

    def batch_data(self, images, gt_points, gt_density_maps, masks, start, end):
        tensor_batch_images = []
        tensor_batch_gt_points = []
        tensor_batch_gt_discrete = []
        tensor_batch_gt_density_maps = []
        tensor_batch_masks = []

        tensor_batch_gt_counts = []

        tensor_un_maps = []

        batch_images = images[start:end]
        batch_gt_points = gt_points[start:end]
        batch_gt_density_maps = gt_density_maps[start:end]
        batch_masks = masks[start:end]

        for index, (img, point, gt_density_map) in enumerate(zip(batch_images, batch_gt_points, batch_gt_density_maps)):
            wd, ht = img.size
            st_size = 1.0 * min(wd, ht)
            mask = batch_masks[index]

            assert st_size >= self.args.crop_size, print(wd, ht)
            assert len(point) >= 0
            i, j, h, w = self.random_crop(ht, wd, self.args.crop_size, self.args.crop_size)
            img = F.crop(img, i, j, h, w)
            gt_density_map = gt_density_map[i:i+h,j:j+w]
            mask = mask[i:i+h,j:j+w]
            if len(point) > 0:
                point = point - [j, i]
                idx_mask = (point[:, 0] > 0) * (point[:, 0] < w) * \
                        (point[:, 1] > 0) * (point[:, 1] < h)
                point = point[idx_mask]
                tensor_batch_gt_counts.append(len(point))
            else:
                point = np.empty([0, 2])
                tensor_batch_gt_counts.append(0)
            gt_discrete = self.gen_discrete_map(h, w, point)
            down_h = h // self.args.downsample_ratio
            down_w = w // self.args.downsample_ratio
            gt_discrete = gt_discrete.reshape([down_h, self.args.downsample_ratio, down_w, self.args.downsample_ratio]).sum(axis=(1, 3))
            gt_density_map = cv2.resize(gt_density_map, (down_w, down_h), interpolation = cv2.INTER_CUBIC)*(self.args.downsample_ratio**2)
            mask = cv2.resize(mask, (down_w, down_h), interpolation = cv2.INTER_CUBIC)

            assert np.sum(gt_discrete) == len(point)
            
            if random.random() > 0.5:
                img = F.hflip(img)
                gt_discrete = np.fliplr(gt_discrete)
                gt_density_map = np.fliplr(gt_density_map)
                mask = np.fliplr(mask)
                if len(point) > 0:
                    point[:, 0] = self.args.crop_size - point[:, 0] - 1

                    
            gt_discrete = np.expand_dims(gt_discrete, 0)
            tensor_batch_images.append(self.images_transforms(img).unsqueeze(0))
            tensor_batch_gt_points.append(torch.from_numpy(point).float())
            tensor_batch_gt_discrete.append(torch.from_numpy(gt_discrete.copy()).unsqueeze(0))
            tensor_batch_gt_density_maps.append(torch.from_numpy(gt_density_map.copy()).unsqueeze(0).unsqueeze(0))
            tensor_batch_masks.append(torch.from_numpy(mask.copy()).unsqueeze(0).unsqueeze(0))

            tensor_un_maps.append(torch.zeros((self.args.crop_size, self.args.crop_size)))
        
        tensor_batch_images = torch.cat(tensor_batch_images)
        tensor_batch_gt_discrete = torch.cat(tensor_batch_gt_discrete).float()
        tensor_batch_gt_density_maps = torch.cat(tensor_batch_gt_density_maps)
        tensor_batch_masks = torch.cat(tensor_batch_masks)
        return tensor_batch_images, tensor_batch_gt_points, tensor_batch_gt_discrete, tensor_batch_gt_density_maps, tensor_batch_masks, tensor_batch_gt_counts, tensor_un_maps

    def random_crop(self, im_h, im_w, crop_h, crop_w):
        res_h = im_h - crop_h
        res_w = im_w - crop_w
        i = random.randint(0, res_h)
        j = random.randint(0, res_w)
        return i, j, crop_h, crop_w

    def gen_discrete_map(self, im_height, im_width, points):
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
        assert np.sum(discrete_map) == num_gt

        ''' slow method
        for p in points:
            p = np.round(p).astype(int)
            p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
            discrete_map[p[0], p[1]] += 1
        '''
        return discrete_map
    
    def gen_weight_map(self, im_height, im_width, points, target_value):
        """
            func: generate the discrete map.
            points: [num_gt, 2], for each row: [width, height]
            """
        weight_map = np.zeros([im_height, im_width], dtype=np.float32)
        h, w = weight_map.shape[:2]
        num_gt = points.shape[0]
        if num_gt == 0:
            return weight_map
        
        # fast create discrete map
        points_np = np.array(points).round().astype(int)
        p_h = np.minimum(points_np[:, 1], np.array([h-1]*num_gt).astype(int))
        p_w = np.minimum(points_np[:, 0], np.array([w-1]*num_gt).astype(int))
        p_index = torch.from_numpy(p_h* im_width + p_w)

        assert len(points) == len(target_value)
        weight_map = torch.zeros(im_width * im_height).scatter_(0, index=p_index, src=torch.from_numpy(target_value)).view(im_height, im_width).numpy()
        # assert np.sum(weight_map) == np.sum(target_value)


        ''' slow method
        for p in points:
            p = np.round(p).astype(int)
            p[0], p[1] = min(h - 1, p[1]), min(w - 1, p[0])
            discrete_map[p[0], p[1]] += 1
        '''
        return weight_map
    

    
    def gen_pseudo_point_with_global_std(self, density_map, std):
        prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
        std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

        maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
        maxpool_output = torch.eq(maxpool_output, prob_outputs)
        maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
        maxpool_output = maxpool_output[0, 0]

        height = maxpool_output.shape[0]
        width = maxpool_output.shape[1]

        maxpool_std = nn.functional.max_pool2d(std, 5, 1, 2)[0, 0] * (self.args.uncertain_thre)
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
    
    def gen_pseudo_point_with_std2(self, density_map, std, pede_num):
        prob_outputs = nn.functional.upsample_bilinear(density_map.unsqueeze(0).unsqueeze(0), scale_factor=8)
        std = nn.functional.upsample_bilinear(std.unsqueeze(0).unsqueeze(0), scale_factor=8)

        maxpool_output = nn.functional.max_pool2d(prob_outputs, 5, 1, 2)
        maxpool_output = torch.eq(maxpool_output, prob_outputs)
        maxpool_output = maxpool_output.type(torch.cuda.FloatTensor) * prob_outputs
        maxpool_output = maxpool_output[0, 0]

        height = maxpool_output.shape[0]
        width = maxpool_output.shape[1]

        std = std[0, 0]

        nonzero_points = maxpool_output.nonzero()
        nonzero_y = nonzero_points.T[0]
        nonzero_x = nonzero_points.T[1]
        nonzero_idx = nonzero_y*width + nonzero_x # one-dimension idx
        nonzero_std = std.view(-1)[nonzero_idx]
        _, idx = torch.sort(nonzero_std, 0)
        idx = idx[:min(pede_num, len(idx))]
        filtered_nonzero_idx = nonzero_idx[idx]
        y = torch.div(filtered_nonzero_idx, width, rounding_mode='floor')
        x = filtered_nonzero_idx - y*width

        points = torch.cat([x[:,None], y[:,None]], 1)

        return points

    def get_uncertain_gt(self, data):
        self.model.eval()
        uncertain_data = {}
        for stage, stage_value in data.items():
            uncertain_data[stage] = {}
            uncertain_data[stage]["images"] = []
            uncertain_data[stage]["gt_error_maps"] = []

            images = deepcopy(stage_value['images'])
            gt_density_maps = deepcopy(stage_value['gt_density_maps'])
            for index, (img, gt_density_map) in enumerate(zip(images, gt_density_maps)):
                input = self.images_transforms(img).unsqueeze(0)
                input = input.repeat(4,1,1,1)
                input = input.to(self.device)
                with torch.set_grad_enabled(False):
                    output, _ = self.model(input)
                output = torch.mean(output.squeeze(1), 0)
                uncertain_gt = np.abs(output.cpu().numpy() - gt_density_map)
                uncertain_data[stage]["images"].append(img)
                uncertain_data[stage]["gt_error_maps"].append(uncertain_gt)
    
        return uncertain_data
    
    def visual_gt_error_map(self, data):
        pass
    
    def teacher_psdudo_points(self, data, iter_num):
        images = deepcopy(data['train_data']['images'])
        images_names = deepcopy(data['train_data']['images_names'])
        gt_points = deepcopy(data['train_data']['gt_points'])

        save_path = os.path.join(self.args.save_dir, "teacher_points", "iter_{}".format(iter_num))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for index, (img, img_name, img_points) in enumerate(zip(images, images_names, gt_points)):
            img = self.restore_transform(img)
            # Create plot
            plt.figure()
            fig, ax = plt.subplots(1)
            ax.imshow(img)

            for x, y in img_points.cpu():
                # Create a Rectangle patch
                point = patches.Circle((x, y), 2, linewidth=2, facecolor="red")
                # Add the bbox to the plot
                ax.add_patch(point)

            plt.axis("off")
            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            output_path = os.path.join(save_path, f"{img_name}.png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0)
            plt.close()

    def draw_images_and_pseudo_points(self, data, path, iter_num):
        images = deepcopy(data['images'])
        masks = deepcopy(data['masks'])
        images_names = deepcopy(data['images_names'])
        gt_points = deepcopy(data['gt_points'])

        save_path = os.path.join(self.args.save_dir, path, "iter_{}".format(iter_num))
        
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        
        for index, (img, mask, img_name, img_points) in enumerate(zip(images, masks, images_names, gt_points)):
            img = np.array(img)
            # Create plot
            plt.figure()
            fig, ax = plt.subplots(1,2)
            ax[0].imshow(img)

            for x, y in img_points:
                # Create a Rectangle patch
                point = patches.Circle((x, y), 2, linewidth=2, facecolor="red")
                # Add the bbox to the plot
                ax[0].add_patch(point)

            ax[0].axis("off")
            ax[1].imshow(mask)

            plt.gca().xaxis.set_major_locator(NullLocator())
            plt.gca().yaxis.set_major_locator(NullLocator())
            plt.tight_layout()
            output_path = os.path.join(save_path, f"{img_name}.png")
            plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0, dpi=400)
            plt.close()
        
    def test_visual(self, data, path, iter_num):
        images = deepcopy(data['images'])
        images_names = deepcopy(data['images_names'])
        gt_points = deepcopy(data['gt_points'])

        save_path = os.path.join(self.args.save_dir, path, "iter_{}".format(iter_num))
        if not os.path.exists(save_path):
            os.makedirs(save_path)
        plt.figure()
        for key in images.keys():
            for index, (img, img_name, img_points) in enumerate(zip(images[key], images_names[key], gt_points[key])):
                img = np.array(img)
                # Create plot
                
                fig, ax = plt.subplots(1)
                ax.imshow(img)

                for x, y in img_points:
                    # Create a Rectangle patch
                    point = patches.Circle((x, y), 2, linewidth=2, facecolor="red")
                    # Add the bbox to the plot
                    ax.add_patch(point)

                plt.axis("off")
                plt.gca().xaxis.set_major_locator(NullLocator())
                plt.gca().yaxis.set_major_locator(NullLocator())
                output_path = os.path.join(save_path, "{}_{}.jpg".format(key, img_name))
                plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0, dpi=400)
                plt.clf()
                plt.close()
