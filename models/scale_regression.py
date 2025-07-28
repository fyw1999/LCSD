import json
import os, glob
import numpy as np
from copy import deepcopy
import matplotlib.pyplot as plt
from sklearn.linear_model import RANSACRegressor, LinearRegression

class ScalenetTrainer():
    def __init__(self, args, logger):
        self.logger = logger
        self.args = args
        self.regr = None
        self.read_real_scale()

    def read_real_scale(self):
        scales = []
        y_positions = []
        real_json_file_path = glob.glob(os.path.join(self.args.scene_dataset, "train_data", "gt_bbox_json", "*.json"))
        for file_path in real_json_file_path:
            with open(file_path) as f:
                file = json.load(f)
            shapes = file["shapes"]
            for data in shapes:
                points = np.array(data["points"])
                x1, y1, x2, y2 = points[0][0], points[0][1], points[1][0], points[1][1]
                w = x2 - x1
                h = y2 - y1
                scale = w*h
                y_position = y1 + h*0.1
                scales.append(scale)
                y_positions.append(y_position)

        self.real_scales = np.array(scales)[:,None]
        self.real_y_positions = np.array(y_positions)[:,None]

        self.real_regr = LinearRegression()
        self.real_regr.fit(self.real_y_positions, self.real_scales)

    def is_use(self):
        return self.regr != None
    
    def predict(self, y_position):
        input = np.array(y_position, dtype=np.float32).reshape(-1, 1)
        pre_scale = self.regr.predict(input)[0,0]
        return pre_scale
    
    def linear_fit(self, data, iter_num):
        outlier_masks = {}
        y_positions = deepcopy(data["train_data"]["y_positions"])
        scales = deepcopy(data["train_data"]['scales'])
        
        y_positions = np.concatenate(y_positions, 0)
        scales = np.concatenate(scales, 0)

        self.regr = None

        if len(y_positions) > 10:
            self.regr = RANSACRegressor(random_state=42)
            if np.any(np.isnan(y_positions)) or np.any(np.isnan(scales)):
                print(y_positions)
                print(scales)
            self.regr.fit(y_positions, scales)

            self.save_fig(y_positions, scales, iter_num)

            train_inlier_mask = self.regr.inlier_mask_
            train_outlier_mask = ~train_inlier_mask

            outlier_masks["train_data"] = train_outlier_mask

        
        # fit test
        y_positions = deepcopy(data["test_data"]["y_positions"])
        scales = deepcopy(data["test_data"]['scales'])
        
        y_positions = np.concatenate(y_positions, 0)
        scales = np.concatenate(scales, 0)

        self.test_regr = None

        if len(y_positions) > 10:
            self.test_regr = RANSACRegressor(random_state=42)
            self.test_regr.fit(y_positions, scales)

            test_inlier_mask = self.test_regr.inlier_mask_
            test_outlier_mask = ~test_inlier_mask

            outlier_masks["test_data"] = test_outlier_mask
        
        return outlier_masks

    def save_fig(self, y_positions, scales, iter_num):
        plt.scatter(y_positions, scales, color='#C4D2EA', label='prediction', s=10)
        plt.plot(y_positions, self.regr.predict(y_positions), color='#7BABDD', label='prediction_fit')

        plt.scatter(self.real_y_positions, self.real_scales, color='#E5ACA8', label='real', s=10)
        plt.plot(self.real_y_positions, self.real_regr.predict(self.real_y_positions), color='#C83231', label='real_fit')

        plt.gcf().subplots_adjust(left=0.2)
        plt.gcf().subplots_adjust(bottom=0.15)
        plt.legend(loc=9)
        plt.grid()
        plt.xlabel("position", fontsize=15)
        plt.ylabel("scale", fontsize=15)
        plt.title("Scale Visualization", fontsize=15)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        
        save_path = os.path.join(self.args.save_dir, "linear_fit_fig", "iter_{}".format(iter_num))

        if not os.path.exists(save_path):
            os.makedirs(save_path)
            
        plt.savefig(os.path.join(save_path, 'plt_show.jpg'), dpi=500)



