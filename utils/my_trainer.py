import os
import torch
from datetime import datetime
import utils.log_utils as log_utils
from models.scale_regression import ScalenetTrainer
from utils.data_generate import DatasetGenerator
from models.counter.trainer import CounterTrainer
from models.detector.trainer import DetectorTrainer


class MyTrainer:
    def __init__(self, args):
        self.args = args
    
    def setup(self):
        args = self.args
        if args.dataset.lower() == 'cityuhk-x':
            args.img_height = 384
            args.img_width = 512
            args.crop_size = 256
        elif args.dataset.lower() == 'mall':
            args.img_height = 480
            args.img_width = 640
            args.crop_size = 256
        elif args.dataset.lower() == 'ucsd':
            args.img_height = 158
            args.img_width = 238
            args.crop_size = 128

        args.downsample_ratio = 8
        
        sub_dir = datetime.strftime(datetime.now(), '%m%d-%H%M%S')  # prepare saving path
        self.save_dir = os.path.join(args.save_dir, args.dataset, args.scene, sub_dir)
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self.logger = log_utils.get_logger(os.path.join(self.save_dir, 'train.log'))
        args.save_dir = self.save_dir
        args.scene_dataset = os.path.join(args.resource_path, args.dataset, args.scene, args.scene_dataset)
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            self.device_count = torch.cuda.device_count()
            assert self.device_count == 1
            self.logger.info('using {} gpus'.format(self.device_count))
        else:
            raise Exception("gpu is not available")

        self.data_generator = DatasetGenerator(args)
        self.counter = CounterTrainer(args, self.logger)
        self.detector = DetectorTrainer(args, self.logger)
        self.scale_net = ScalenetTrainer(args, self.logger)

        log_utils.print_config(vars(args), self.logger)

    def train(self):
        pre_dis = None
        pre_num = None
        scale_model = None
        for i in range(1, self.args.iterative_num+1):
            self.logger.info("Iter:{}/{}".format(i, self.args.iterative_num))
            # generate a synthetic dataset 
            com_data, base_image = self.data_generator.generate(i, 
                                                                pre_dis, 
                                                                pre_num, 
                                                                scale_model)
            # train counter
            self.counter.train(com_data, scale_model, i)
            self.counter.save_model(i)

            #train detector
            com_data, _ = self.data_generator.generate(i, pre_dis, pre_num)
            self.detector.train(com_data)
            self.detector.save_model(i)

            _, scale_data = self.detector.predict(i)
            self.scale_net.linear_fit(scale_data, i)
            scale_model = self.scale_net
            
            pre_dis, pre_num = self.counter.predict(i, base_image)