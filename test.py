import argparse
import torch
import os
import numpy as np
from models.counter.models import vgg19
from datasets.crowd import Crowd
from torch.utils.data.dataloader import default_collate
from utils.utils import gen_pseudo_point, eval_loc_F1_point
from utils.pytorch_utils import AverageCategoryMeter
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.ticker import NullLocator
import torch.nn.functional as F
parser = argparse.ArgumentParser(description='Test ')
parser.add_argument('--dataset', default='Mall',
                    help='dataset name')
parser.add_argument('--scene', default='scene_001', type=str, 
                    help='scene name, for Mall and UCSD, set this to scene_001')
parser.add_argument('--resource-path', type=str, default=r'/data/fyw/dataset/crowdcount/weakly-supervised-scene-specific/resources/',
                        help='path for pedestrians info and background images')
parser.add_argument('--scene-dataset', default='mall_800_1200',
                        help='real data path')
parser.add_argument('--model-path', type=str, \
                    default='save/Mall/scene_001/0728-153342/models/counter_model_3.pth',
                    help='saved model path')
parser.add_argument('--test-name', type=str, default='check',
                    help='dataset name: qnrf, nwpu, sha, shb')
parser.add_argument('--pred-density-map-path', type=str, default='test_save',
                    help='save predicted density maps when pred-density-map-path is not empty.')
parser.add_argument('--device', default='1', help='assign device')


args = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = args.device  # set vis gpu
device = torch.device('cuda')

model_path = args.model_path
dataset_path = os.path.join(args.resource_path, args.dataset, args.scene, args.scene_dataset)
dataset = Crowd(dataset_path, 'test', 8)
dataloader = torch.utils.data.DataLoader(dataset, 1, shuffle=False, collate_fn=default_collate,
                                         num_workers=3, pin_memory=False)

if args.pred_density_map_path:
    args.pred_density_map_path = os.path.join(args.pred_density_map_path, args.test_name)
    import cv2
    if not os.path.exists(args.pred_density_map_path):
        os.makedirs(args.pred_density_map_path)

model = vgg19(args)
model.to(device)
model.load_state_dict(torch.load(model_path, device))
model.eval()
image_errs = []
countt = 0
max_dist_thresh = 100
loc_100_metrics = {'tp_100': AverageCategoryMeter(max_dist_thresh), 'fp_100': AverageCategoryMeter(max_dist_thresh), 'fn_100': AverageCategoryMeter(max_dist_thresh)}
for img, inputs, points, name in dataloader:
    inputs = inputs.to(device)
    count = len(points[0])
    assert inputs.size(0) == 1, 'the batch size should equal to 1'
    inputs = inputs.repeat(4,1,1,1)
    with torch.set_grad_enabled(False):
        multi_output, _ = model(inputs)
        output = torch.mean(multi_output.squeeze(1), 0)
        std = torch.std(multi_output.squeeze(1), 0)
        weight = torch.exp(std)
    pseudo_point = None
    pseudo_point = gen_pseudo_point(output)
    img_err = count - torch.sum(output).item()
    print("img_name:{}, error:{}, gt:{}, pre:{}, order:{}".format(name, img_err, count, len(pseudo_point), countt))
    image_errs.append(img_err)

    tp_100, fp_100, fn_100 = eval_loc_F1_point(pseudo_point.cpu().numpy(), points[0].cpu().numpy(), max_dist_thresh = max_dist_thresh)
    loc_100_metrics['tp_100'].update(tp_100)
    loc_100_metrics['fp_100'].update(fp_100)
    loc_100_metrics['fn_100'].update(fn_100)
    if args.pred_density_map_path:
        vis_img = F.upsample_bilinear(output.unsqueeze(0).unsqueeze(0), scale_factor=8)[0, 0].cpu().numpy()
        # normalize density map values from 0 to 1, then map it to 0-255.
        vis_img = (vis_img - vis_img.min()) / (vis_img.max() - vis_img.min() + 1e-5)
        vis_img = (vis_img * 255).astype(np.uint8)
        vis_img = cv2.applyColorMap(vis_img, cv2.COLORMAP_JET)
        cv2.imwrite(os.path.join(args.pred_density_map_path, str(name[0]) + '.png'), vis_img)

        img = img.squeeze(0)
        plt.figure()
        fig, ax = plt.subplots(1)
        ax.imshow(img)

        for x, y in pseudo_point:
            # Create a Rectangle patch
            point = patches.Circle((x.item(), y.item()), 2, linewidth=2, facecolor="red")
            # Add the bbox to the plot
            ax.add_patch(point)

        ax.axis("off")

        plt.gca().xaxis.set_major_locator(NullLocator())
        plt.gca().yaxis.set_major_locator(NullLocator())
        plt.tight_layout()
        output_path = os.path.join(args.pred_density_map_path, str(name[0]) + '_show.png')
        plt.savefig(output_path, bbox_inches="tight", pad_inches=0.0, dpi=400)
        plt.close()

    countt += 1

image_errs = np.array(image_errs)
mse = np.sqrt(np.mean(np.square(image_errs)))
mae = np.mean(np.abs(image_errs))
print('{}: mae {}, mse {}\n'.format(model_path, mae, mse))
pre_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fp_100'].sum + 1e-20)
rec_100 = loc_100_metrics['tp_100'].sum / (loc_100_metrics['tp_100'].sum  + loc_100_metrics['fn_100'].sum + 1e-20) # True pos rate
f1_100 = 2 * (pre_100 * rec_100) / (pre_100 + rec_100+ 1e-20)
print('avg precision_overall', pre_100.mean())
print('avg recall_overall',    rec_100.mean())
print('avg F1_overall',        f1_100.mean())
print(countt)
